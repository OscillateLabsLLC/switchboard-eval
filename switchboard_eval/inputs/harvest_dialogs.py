#!/usr/bin/env python3
"""Decline-class negatives from skills' own .dialog files: real text the assistant SPEAKS,
which its microphone can pick back up (self-echo). Derived, not generated.

Drops lines that normalize to an eval utterance (leakage) and lines that match an exact-layer
template (they would teach the decline class to swallow real requests).

Usage: harvest_dialogs.py ROOT [ROOT ...] [--templates devices/desktop/manifest.json] [--out devices/desktop/dialog_negatives.jsonl]
"""
from switchboard_eval.paths import ROOT
import glob, itertools, json, os, re, string, sys

sys.path.insert(0, ROOT)
from switchboard_eval.inputs.harvest_examples import clean, expand

MAX_PER_LINE = 5  # dialog alternations explode; a few variants per line is plenty


def norm(s):
    return re.sub(r"\s+", " ", s.lower().translate(str.maketrans("", "", string.punctuation))).strip()


def dialog_files(root, lang="en-us"):
    pats = [f"**/locale/{lang}/**/*.dialog", f"**/dialog/{lang}/**/*.dialog"]
    return sorted({f for p in pats for f in glob.glob(os.path.join(root, p), recursive=True)})


def main(argv):
    roots = [a for a in argv if not a.startswith("--") and not a.endswith((".json", ".jsonl"))]
    out = argv[argv.index("--out") + 1] if "--out" in argv else "devices/desktop/dialog_negatives.jsonl"
    evals = {norm(json.loads(l)["utterance"]) for l in open("data/dataset.jsonl")}
    templates = []
    if "--templates" in argv:
        from switchboard_eval.inputs.harvest_live import harvest
        _, templates = harvest(argv[argv.index("--templates") + 1], "en-us")
    seen, kept, leaked, template_hits, files = set(), [], 0, 0, 0
    for root in roots:
        for path in dialog_files(root):
            files += 1
            for line in open(path, encoding="utf-8", errors="ignore"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for ex in itertools.islice(expand(line), MAX_PER_LINE):
                    text = norm(re.sub(r"\{+[^}]*\}+", " ", clean(ex)))  # slot values unknown: drop them
                    if len(text.split()) < 2 or text in seen:
                        continue
                    seen.add(text)
                    if text in evals:
                        leaked += 1
                        continue
                    if any(rx.match(text) for _, rx, _ in templates):
                        template_hits += 1
                        continue
                    kept.append({"label": "synthetic:__decline__", "skill_id": "dialog",
                                 "intent": "__decline__", "example": text, "slots": [],
                                 "source": "dialog", "kind": "assistant_echo",
                                 "file": os.path.relpath(path, root)})
    with open(out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"{files} dialog files -> {len(kept)} negatives -> {out} "
          f"(dropped {leaked} eval matches, {template_hits} exact-template matches)")


if __name__ == "__main__":
    main(sys.argv[1:])
