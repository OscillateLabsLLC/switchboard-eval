#!/usr/bin/env python3
"""Evaluate resolver.py without fitting one house.

  home   data/resolver/resolver_gold_home.tsv vs the home's registry (data/hub/ha_states.json, gitignored), split by hash of the
         spoken string into TUNE and TEST. THRESHOLD/MARGIN are swept on TUNE only.
  demo   data/resolver/resolver_gold_demo.tsv vs Home Assistant's demo registry (data/resolver/ha_demo_registry.json): a different house,
         never used for selection.
Selection rule (fixed before looking): maximize correct - 2 * wrong_actions, where a wrong action is acting on
the wrong entity or resolving when the gold is NONE (acting is worse than declining).

Usage: resolver_eval.py   (bench venv)
"""
from switchboard_eval.paths import ROOT
import hashlib, json, os, sys

HERE = ROOT
sys.path.insert(0, HERE)
from switchboard_eval.harness.resolver import EntityResolver

THRESHOLDS = [round(0.3 + 0.05 * i, 2) for i in range(11)]
MARGINS = [0.0, 0.02, 0.05, 0.1]


def load_gold(path):
    rows = []
    for line in open(os.path.join(HERE, path)):
        if line.strip() and not line.startswith("#"):
            spoken, intent, gold = line.split("#")[0].rstrip().split("\t")[:3]
            rows.append((spoken, intent, None if gold == "NONE" else gold))
    return rows


def split(rows):
    h = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2
    return [r for r in rows if h(r[0]) == 0], [r for r in rows if h(r[0]) == 1]


def evaluate(resolver, rows, threshold, margin):
    stats = {"correct": 0, "wrong_entity": 0, "false_resolve": 0, "missed": 0, "n": len(rows)}
    errors = []
    for spoken, intent, gold in rows:
        got, score, _ = resolver.resolve(spoken, intent, threshold, margin)
        if got == gold:
            stats["correct"] += 1
            continue
        kind = "false_resolve" if gold is None else ("missed" if got is None else "wrong_entity")
        stats[kind] += 1
        errors.append((kind, spoken, gold, got, round(score, 2)))
    stats["utility"] = stats["correct"] - 2 * (stats["wrong_entity"] + stats["false_resolve"])
    return stats, errors


def line(name, s):
    return (f"{name:28} correct {s['correct']}/{s['n']}  wrong entity {s['wrong_entity']}  "
            f"acted on nothing {s['false_resolve']}  missed {s['missed']}")


def main():
    home = EntityResolver(json.load(open(os.path.join(HERE, "data", "hub", "ha_states.json"))))
    demo = EntityResolver(json.load(open(os.path.join(HERE, "data/resolver/ha_demo_registry.json"))))
    tune, test = split(load_gold("data/resolver/resolver_gold_home.tsv"))
    best = max(((evaluate(home, tune, t, m)[0]["utility"], -t, m, t) for t in THRESHOLDS for m in MARGINS))
    threshold, margin = best[3], best[2]
    print(f"selected on home TUNE (n={len(tune)}): threshold={threshold} margin={margin}")
    for name, resolver, rows in (("home TUNE", home, tune), ("home TEST (held out)", home, test),
                                 ("demo house (never tuned)", demo, load_gold("data/resolver/resolver_gold_demo.tsv"))):
        s, errs = evaluate(resolver, rows, threshold, margin)
        print(line(name, s))
        for e in errs:
            print("     ", e)


if __name__ == "__main__":
    main()
