#!/usr/bin/env python3
"""Harvest training examples from a live stack's INTENT-4 manifest dump.

Same output format as harvest_examples.py, but the source is devices/desktop/manifest.json
(dump_introspection.py), so it matches the skills the live baseline actually
runs instead of the repos in fetch_skills.sh.

With --tag-dataset, overwrites each row's `seen_in_templates` using the LIVE
templates. Tag against the same source the compared systems trained on, or the
seen/unseen split measures the wrong thing.

Usage: harvest_live.py devices/desktop/manifest.json [--out devices/desktop/examples.jsonl]
                       [--tag-dataset data/dataset.jsonl] [--lang en-us]
"""
from switchboard_eval.paths import ROOT
import itertools, json, os, re, sys

sys.path.insert(0, ROOT)
from switchboard_eval.inputs.harvest_examples import MAX_PER_LINE, clean, expand, template_regex


def harvest(dump_path, lang):
    examples, templates = [], []
    for skill_id, defs in json.load(open(dump_path))["definitions"].items():
        if not isinstance(defs, list):
            continue
        for d in defs:
            if d["lang"].lower() != lang or d["method"] != "template":
                continue
            label = f"{skill_id}:{d['intent_name']}"
            for line in d["definition"].get("samples") or []:
                templates.append((label, template_regex(line), line))
                for ex in itertools.islice(expand(line), MAX_PER_LINE):
                    ex = clean(ex)
                    if ex:
                        examples.append({"label": label, "skill_id": skill_id,
                                         "intent": d["intent_name"], "example": ex,
                                         "slots": sorted(set(re.findall(r"\{+([a-z_]+)\}+", ex)))})
    return examples, templates


def literal_words(line):
    """Fewest literal (non-slot) words in any expansion of a template. 'play {query}' -> 1."""
    return min(len(re.sub(r"\{+[a-z_]+\}+", " ", ex).split()) for ex in expand(line))


def tag(path, templates):
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        u = r["utterance"].lower()
        r["seen_in_templates"] = sorted({lab for lab, rx, _ in templates if rx.match(u)})
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    hits = sum(bool(r["seen_in_templates"]) for r in rows)
    print(f"tagged {path}: {hits}/{len(rows)} rows match a live template")


def main(argv):
    if not argv:
        sys.exit(__doc__)
    out = argv[argv.index("--out") + 1] if "--out" in argv else "devices/desktop/examples.jsonl"
    lang = (argv[argv.index("--lang") + 1] if "--lang" in argv else "en-us").lower()
    examples, templates = harvest(argv[0], lang)
    seen = set()
    with open(out, "w") as f:
        for e in examples:
            k = (e["label"], e["example"].lower())
            if k not in seen:
                seen.add(k)
                f.write(json.dumps(e) + "\n")
    print(f"{len(seen)} examples across {len({e['label'] for e in examples})} intents -> {out}")
    if "--tag-dataset" in argv:
        tag(argv[argv.index("--tag-dataset") + 1], templates)


if __name__ == "__main__":
    main(sys.argv[1:])
