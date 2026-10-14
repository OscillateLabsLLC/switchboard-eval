#!/usr/bin/env python3
"""Stage B: pick the device's SKILL LABEL for a predicted canonical intent, and build the skill-native match_data.

Dispatch in OVOS needs `skill_id:intent_name` plus the data the skill expects; canonical intents are for sharing
and scoring. Many canonical intents have several labels on a device (Hub: settings.change has 29), so choosing
among them is part of routing. Uses only THIS device's declarations:
  Adapt labels      evaluate the parsed rule like Adapt: every `require` vocab present (whole-word), each `one_of`
                    satisfied; score = required + optional vocab matched. match_data = {VocabName: matched text},
                    the keys Adapt-based skills read (message.data["Location"]); regex entities fill named groups.
  Padatious labels  template match (full, then loose); score = literal words; match_data = slot values under the
                    template's own slot names.
  neither matches   char-ngram similarity to the label's examples (keyword combos / expansions), no match_data.
A single label is trivial. Deterministic.

Usage (smoke): labels.py devices/speaker/manifest.json weather.current "what is the weather in seattle"
"""
from switchboard_eval.paths import ROOT
import json, os, re, sys

HERE = ROOT
sys.path.insert(0, HERE)
from switchboard_eval.harness.run_ovos import build_map
from switchboard_eval.harness.slots import SlotExtractor, literal_words, named_template_regex


def vkey(name):
    return re.sub(r"[-_]", "", name.lower())


def phrase_regex(phrases):
    alts = sorted({re.escape(p.lower()) for p in phrases if p.strip()}, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(alts) + r")\b") if alts else None


class AdaptRule:
    def __init__(self, definition):
        self.steps = definition["steps"]
        self.vocab = {k: phrase_regex(v) for k, v in (definition.get("vocab") or {}).items()}
        self.regex = {k: [re.compile(p, re.I) for p in v] for k, v in (definition.get("regex") or {}).items()}

    def _find(self, name, utt):
        key = vkey(name)
        rx = self.vocab.get(key)
        if rx:
            m = rx.search(utt)
            if m:
                return m.group(1)
        for pat in self.regex.get(key, []):
            m = pat.search(utt)
            if m:
                return next((v for g, v in m.groupdict().items() if vkey(g) == key and v), None)
        return None

    def evaluate(self, utt):
        """(score, match_data) if the rule fires, else None."""
        data, score = {}, 0
        for method, args in self.steps:
            if method == "require":
                v = self._find(args[0], utt)
                if v is None:
                    return None
                data[args[0]], score = v, score + 2
            elif method == "one_of":
                hits = [(a, self._find(a, utt)) for a in args]
                hits = [(a, v) for a, v in hits if v is not None]
                if not hits:
                    return None
                data[hits[0][0]], score = hits[0][1], score + 2
            elif method == "optionally":
                v = self._find(args[0], utt)
                if v is not None:
                    data[args[0]], score = v, score + 1
        return score, data


class LabelSelector:
    def __init__(self, manifest, schema, examples):
        from sklearn.feature_extraction.text import TfidfVectorizer
        label_map = build_map(schema)
        self.by_canon, self.adapt, self.templates, ex_text = {}, {}, {}, {}
        for sid, defs in json.load(open(manifest))["definitions"].items():
            for d in defs if isinstance(defs, list) else []:
                label = f"{sid}:{d['intent_name']}"
                canon = label_map.get(label, (None,))[0]
                if not canon:
                    continue
                self.by_canon.setdefault(canon, []).append(label)
                if d["method"] == "adapt":
                    self.adapt[label] = AdaptRule(d["definition"])
                else:
                    self.templates[label] = [(literal_words(l), re.compile(r"^\s*" + named_template_regex(l) + r"\s*[?.!]*$"),
                                              re.compile(r"(?:^|\s)" + named_template_regex(l) + r"\s*[?.!]*$"))
                                             for l in d["definition"].get("samples") or [] if _compiles(l)]
        for line in open(examples):
            e = json.loads(line)
            ex_text.setdefault(e["label"], []).append(re.sub(r"\{+[a-z_]+\}+", " ", e["example"]).lower())
        self.ex_text = {l: " ".join(v[:200]) for l, v in ex_text.items()}
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(list(self.ex_text.values()) or ["x"])

    def select(self, canon, utterance):
        """(label or None, match_data, how)."""
        labels = self.by_canon.get(canon, [])
        if not labels:
            return None, {}, "no label"
        utt = utterance.lower().strip()
        scored = []
        for label in labels:
            if label in self.adapt:
                hit = self.adapt[label].evaluate(utt)
                if hit:
                    scored.append((hit[0], label, hit[1], "adapt rule"))
            else:
                for mode in ("full", "loose"):
                    best = None
                    for lit, full, loose in self.templates.get(label, []):
                        m = full.match(utt) if mode == "full" else loose.search(utt)
                        if m and (best is None or lit > best[0]):
                            best = (lit, {g.split("__")[0]: v.strip() for g, v in m.groupdict().items() if v})
                    if best:
                        scored.append((best[0] + (1 if mode == "full" else 0), label, best[1], f"template {mode}"))
                        break
        if scored:
            # ties: the more specific match wins (fewer slots filled: "tell me a joke" beats "tell me a joke [about {q}]")
            score, label, data, how = max(scored, key=lambda s: (s[0], -len(s[2]), s[1]))
            return label, data, how
        if len(labels) == 1:
            return labels[0], {}, "only label"
        q = self.vec.transform([utt])
        sims = [(float((self.vec.transform([self.ex_text.get(l, "")]) @ q.T).toarray()[0, 0]), l) for l in labels]
        return max(sims)[1], {}, "nearest examples"


def _compiles(line):
    try:
        re.compile(named_template_regex(line))
        return True
    except (ValueError, re.error):
        return False


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    s = LabelSelector(os.path.join(HERE, argv[0]), os.path.join(HERE, "schema/intents.yaml"),
                      os.path.join(HERE, "devices/speaker/examples.jsonl" if "speaker" in argv[0] else "devices/desktop/examples.jsonl"))
    print(s.select(argv[1], argv[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
