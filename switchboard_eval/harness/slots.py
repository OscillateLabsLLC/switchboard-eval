#!/usr/bin/env python3
"""Slot extraction from the skills' own templates (architecture step 3, first pass).

The exact layer already matched a template like "turn on {entity}"; the slot value is just the text the
wildcard covered. For classifier picks, the predicted intent's own templates are tried as a LOOSE matcher
(search, not full match), so "could you turn on the desk lamp please" still yields entity=desk lamp.
LLM tool arguments already carry slots (router.py keeps those).

Template slot names map to canonical slots (schema/intents.yaml) via per-label renames; failing that, a template
slot whose name is a canonical slot of the intent is kept; failing that, if the intent has exactly one
canonical slot and the template one slot, they correspond. Unmappable slots are dropped (never guessed).

Usage (smoke): slots.py "turn on the desk lamp" device.on [--manifest devices/speaker/manifest.json]
"""
from switchboard_eval.paths import ROOT
import json, os, re, sys

HERE = ROOT
sys.path.insert(0, HERE)
import yaml
from switchboard_eval.inputs.harvest_examples import expand
from switchboard_eval.harness.run_ovos import build_map

SLOT = re.compile(r"\{+([a-z_]+)\}+")
TRAILING_COURTESY = re.compile(r"\s+(please|thanks|thank you)$")


def named_template_regex(line):
    """Like harvest_examples.template_regex, but slots become named groups (repeats get __N suffixes)."""
    pat, i, seen = "", 0, {}
    line = line.lower().replace("{{", "{").replace("}}", "}")
    while i < len(line):
        ch = line[i]
        if ch == "{":
            j = line.index("}", i)
            name = line[i + 1:j]
            seen[name] = seen.get(name, 0) + 1
            pat += f"(?P<{name}__{seen[name]}>.+?)"
            i = j + 1
            continue
        pat += {"(": "(?:", ")": ")", "[": "(?:", "]": ")?", "|": "|", " ": r"\s*"}.get(ch, re.escape(ch))
        i += 1
    return pat


def literal_words(line):
    return min(len(SLOT.sub(" ", e).split()) for e in expand(line)[:50]) if line else 0


class SlotExtractor:
    def __init__(self, manifest, schema, shared_manifests=()):
        """shared_manifests: other devices' manifests, used for SLOTS only (Neon's Adapt-only alerts have no
        templates; the OVOS alerts templates describe the same canonical intents)."""
        self.label_map = build_map(schema)
        self.canon_slots = {c: spec.get("slots") or [] for c, spec in yaml.safe_load(open(schema))["intents"].items()}
        self.by_canon = {}
        for sid, defs in (kv for m in (manifest, *shared_manifests) for kv in json.load(open(m))["definitions"].items()):
            for d in defs if isinstance(defs, list) else []:
                if d.get("method") != "template":
                    continue
                label = f"{sid}:{d['intent_name']}"
                canon, ren = self.label_map.get(label, (None, {}))
                for line in d["definition"].get("samples") or []:
                    if not SLOT.search(line):
                        continue
                    try:
                        body = named_template_regex(line)
                        full = re.compile(r"^\s*" + body + r"\s*[?.!]*$")
                        # loose: extra words allowed BEFORE the template; the end stays anchored, so a
                        # trailing slot runs to the end (courtesy words are trimmed in _canonical)
                        loose = re.compile(r"(?:^|\s)" + body + r"\s*[?.!]*$")
                    except (ValueError, re.error):
                        continue
                    if canon:
                        self.by_canon.setdefault(canon, []).append((literal_words(line), full, loose, ren))

    def extract(self, canon, utterance, loose=True):
        """Canonical slots for `canon` from the best-matching template (most literal words), or {}."""
        utt = utterance.lower().strip()
        for mode in ("full", "loose") if loose else ("full",):
            best = None
            for lit, full, lrx, ren in self.by_canon.get(canon, []):
                m = (full.match(utt) if mode == "full" else lrx.search(utt))
                if m and (best is None or lit > best[0]):
                    best = (lit, m, ren)
            if best:
                return self._canonical(canon, best[1], best[2])
        return {}

    def _canonical(self, canon, match, renames):
        slots, allowed = {}, self.canon_slots.get(canon, [])
        raw = {}
        for g, v in match.groupdict().items():
            v = TRAILING_COURTESY.sub("", (v or "").strip())
            if v:
                raw.setdefault(g.split("__")[0], v)
        for name, value in raw.items():
            target = renames.get(name) or (name if name in allowed else None)
            if target is None and len(allowed) == 1 and len(raw) == 1:
                target = allowed[0]
            if target:
                slots[target] = value
        return slots


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    manifest = argv[argv.index("--manifest") + 1] if "--manifest" in argv else "devices/desktop/manifest.json"
    ex = SlotExtractor(os.path.join(HERE, manifest), os.path.join(HERE, "schema/intents.yaml"))
    print(ex.extract(argv[1], argv[0]))


if __name__ == "__main__":
    main(sys.argv[1:])
