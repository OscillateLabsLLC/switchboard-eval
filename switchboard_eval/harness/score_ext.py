#!/usr/bin/env python3
"""Score predictions on the external negatives slice (data/negatives_ext.jsonl; gold is always None).

Reports false accept per source and per pipeline stage, then what the accepted rows triggered.
This is a stress slice, separate from the bar's criterion 4 (household speech).

Usage: score_ext.py data/negatives_ext.jsonl preds1.jsonl [preds2.jsonl ...] [--show N]
"""
import json, os, sys
from collections import Counter, defaultdict


def report(rows, preds, show):
    by_source, by_stage, acted = defaultdict(lambda: [0, 0]), Counter(), []
    for r in rows:
        p = preds[r["id"]]
        hit = p.get("intent") is not None
        by_source[r["source"]][0] += hit
        by_source[r["source"]][1] += 1
        if hit:
            by_stage[p.get("stage")] += 1
            acted.append((r, p))
    total = sum(n for _, n in by_source.values())
    print(f"  false accept {100 * len(acted) / total:5.1f}%  ({len(acted)}/{total})   " + "   ".join(
        f"{s} {100 * a / n:.1f}%" for s, (a, n) in sorted(by_source.items())))
    print("  accepted by stage: " + ", ".join(f"{k} {v}" for k, v in by_stage.most_common()))
    print("  triggered: " + ", ".join(f"{k} {v}" for k, v in
                                      Counter(p.get("llm_tool") or p["intent"] for _, p in acted).most_common(8)))
    for r, p in acted[:show]:
        print(f"    {r['source']:11} {r['utterance'][:52]!r:55} -> {p.get('llm_tool') or p['intent']}")


def main(argv):
    show = int(argv[argv.index("--show") + 1]) if "--show" in argv else 0
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--show")]
    if len(args) < 2:
        sys.exit(__doc__)
    rows = [json.loads(l) for l in open(args[0])]
    for path in args[1:]:
        print(os.path.basename(path).removesuffix(".jsonl"))
        report(rows, {p["id"]: p for p in map(json.loads, open(path))}, show)


if __name__ == "__main__":
    main(sys.argv[1:])
