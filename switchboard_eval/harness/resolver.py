#!/usr/bin/env python3
"""Entity resolution: a spoken device name -> one Home Assistant entity, or nothing.

Generic by design (no per-home aliases or rules):
  1. domain filter   the intent decides which HA domains are eligible (lights.* -> light, lock.* -> lock, ...)
  2. normalize       lowercase, drop punctuation, leading determiners ("the", "my"), trailing courtesy
  3. score           char n-gram cosine against the entity's friendly name AND its object_id words
                     (robust to ASR noise: "desk land" ~ "desk lamp"); exact name match = 1.0
  4. decide          accept only if score >= THRESHOLD and it beats the runner-up by MARGIN;
                     otherwise "unresolved" (a first-class outcome: the router should not act on a device
                     command whose device doesn't exist, e.g. an upstream misroute handing over "me laugh")
THRESHOLD/MARGIN are chosen on a tuning split and checked on held-out strings and a second registry
(resolver_eval.py).

Usage (smoke): resolver.py registry.json "desk lamp" device.on
"""
import json, re, sys

THRESHOLD = 0.5
MARGIN = 0.05
DETERMINERS = re.compile(r"^(the|my|our|a|an|all the|all)\s+")
COURTESY = re.compile(r"\s+(please|thanks|thank you|now)$")
ANY = None  # every domain
DOMAINS_FOR_INTENT = {
    "device.on": ["light", "switch", "fan", "media_player", "input_boolean", "scene", "script", "climate",
                  "humidifier", "vacuum"],
    "device.off": ["light", "switch", "fan", "media_player", "input_boolean", "climate", "humidifier", "vacuum"],
    "device.stop": ["media_player", "vacuum", "fan", "cover"],
    "device.status": ANY,
    "lights.brightness": ["light"], "lights.color": ["light"],
    "cover.open": ["cover"], "cover.close": ["cover"],
    "lock.lock": ["lock"], "lock.unlock": ["lock"], "lock.status": ["lock"],
    "thermostat.set": ["climate"],
    "media.play": ["media_player"], "media.pause": ["media_player"], "media.resume": ["media_player"],
    "media.next": ["media_player"], "media.previous": ["media_player"],
    "volume.up": ["media_player"], "volume.down": ["media_player"], "volume.set": ["media_player"],
    "volume.mute": ["media_player"], "volume.unmute": ["media_player"],
}


# Device-type words in the spoken name narrow the eligible domains (general HA knowledge, not per-home):
# "lamp in the bedroom" is not the bedroom speaker; "garage light" is not the garage door.
TYPE_WORDS = [
    (re.compile(r"\b(light|lights|lamp|lamps|bulb|bulbs|sconce|chandelier)\b"), {"light", "switch"}),
    (re.compile(r"\b(fan|fans)\b"), {"fan", "switch"}),
    (re.compile(r"\b(speaker|speakers|tv|television|player|radio)\b"), {"media_player"}),
    (re.compile(r"\b(door|doors)\b"), {"lock", "cover"}),
    (re.compile(r"\b(window|windows|blind|blinds|shade|shades|curtain|curtains|garage)\b"), {"cover"}),
    (re.compile(r"\b(thermostat|heat|heating|ac|air conditioning)\b"), {"climate"}),
]
SLUG_WEIGHT = 0.9  # entity_id slugs go stale when users rename entities; the friendly name is authoritative


def type_domains(text):
    hits = [doms for rx, doms in TYPE_WORDS if rx.search(text)]
    return set.intersection(*hits) if hits else None


def normalize(text):
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower().replace("'", ""))
    t = re.sub(r"\s+", " ", t).strip()
    t = COURTESY.sub("", t)
    return DETERMINERS.sub("", t).strip()


class EntityResolver:
    def __init__(self, entities):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.entities = [e for e in entities if e.get("entity_id")]
        self.names = [normalize(e.get("name") or "") for e in self.entities]
        self.slugs = [normalize(e["entity_id"].split(".", 1)[1].replace("_", " ")) for e in self.entities]
        self.domains = [e["entity_id"].split(".", 1)[0] for e in self.entities]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(self.names + self.slugs)
        self.N, self.S = self.vec.transform(self.names), self.vec.transform(self.slugs)

    def rank(self, text, intent=None):
        q = normalize(text)
        if not q:
            return []
        allowed = DOMAINS_FOR_INTENT.get(intent, ANY) if intent else ANY
        typed = type_domains(q)
        if typed is not None:
            allowed = typed if allowed is ANY else (set(allowed) & typed) or set(allowed)
        qv = self.vec.transform([q])
        by_name, by_slug = (self.N @ qv.T).toarray().ravel(), (self.S @ qv.T).toarray().ravel()
        out = []
        for i, e in enumerate(self.entities):
            if allowed is not ANY and self.domains[i] not in allowed:
                continue
            if q == self.names[i]:
                score = 1.0
            else:
                score = max(by_name[i], SLUG_WEIGHT * (1.0 if q == self.slugs[i] else by_slug[i]))
            out.append((float(score), e["entity_id"]))
        return sorted(out, reverse=True)

    def resolve(self, text, intent=None, threshold=THRESHOLD, margin=MARGIN):
        """(entity_id or None, score, reason)."""
        ranked = self.rank(text, intent)
        if not ranked:
            return None, 0.0, "no candidates"
        top, runner = ranked[0], (ranked[1][0] if len(ranked) > 1 else 0.0)
        if top[0] < threshold:
            return None, top[0], "below threshold"
        if top[0] < 1.0 and top[0] - runner < margin:
            return None, top[0], "ambiguous"
        return top[1], top[0], "ok"


def load_registry(path):
    return json.load(open(path))


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    r = EntityResolver(load_registry(argv[0]))
    intent = argv[2] if len(argv) > 2 else None
    print(r.resolve(argv[1], intent), r.rank(argv[1], intent)[:3])


if __name__ == "__main__":
    main(sys.argv[1:])
