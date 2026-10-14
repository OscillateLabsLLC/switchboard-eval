#!/usr/bin/env python3
"""Score candidates against the go/no-go bar in CLAUDE.md, versus stock OVOS.

Usage: bar.py data/dataset.jsonl schema/intents.yaml stock.jsonl cand1.jsonl [cand2.jsonl ...] [--split dev|test]
Default split is test. Select on dev; report test once.

Criteria measured here (2 = determinism and 6 = mini PC latency are separate runs):
  1 magic phrases: rows whose gold intent OWNS a matching live template, >= 98%
  3 real use: Hub and Mac real rows, McNemar p < 0.05 vs stock
  4 false accept on negatives (near_miss, ood, self_echo, background_speech, context_reply), <= 25%
  5 high-risk wrong executions == 0 (confirm policy in score.py)
"""
from switchboard_eval.paths import ROOT
import json, os, sys

import yaml

sys.path.insert(0, ROOT)
from switchboard_eval.harness.score import correct as scored_correct
from switchboard_eval.harness.score import load, mcnemar, pop_split, score
from switchboard_eval.harness.serves import STOCK_DEVICE, equivalences

MAGIC_BAR, FA_BAR, P_BAR = 98.0, 25.0, 0.05
NEG_TAGS = {"self_echo", "background_speech", "context_reply"}


def slices(gold, label2canon):
    def owners(r):
        return {label2canon.get(l) for l in r.get("seen_in_templates", [])} - {None}
    head = [r for r in gold.values() if r["category"] not in ("compound", "followup")]
    return {
        "magic": [r for r in head if isinstance(r["intent"], str) and r["intent"] in owners(r)],
        "hub": [r for r in head if r.get("source") == "neon_hub"],
        "mac": [r for r in head if r["category"] == "real" and r.get("source") != "neon_hub"],
        "neg": [r for r in head if r["intent"] is None and (
            r["category"] in ("near_miss", "ood") or set(r.get("tags", [])) & NEG_TAGS)],
    }


def pct(pred, rows, ok):
    return 100 * sum(ok(pred.get(r["id"], {}), r) for r in rows) / len(rows)


def device_of(name):
    """The device whose skills produced a predictions file: from its name; stock ran on the Mac's."""
    if name.startswith("stock"):
        return STOCK_DEVICE
    return next((d for d in ("hub", "mac") if f"__{d}__" in name), None)


def served(name):
    device = device_of(name)
    return equivalences(device) if device else frozenset()


def report(name, pred, stock, gold, s):
    mine = served(name)
    correct = lambda p, r: scored_correct(r, p, mine)
    acted = lambda p, r: p.get("intent") is not None
    magic = pct(pred, s["magic"], correct)
    fa = pct(pred, s["neg"], acted)
    hx = len(score(gold, pred, sweep=False)["high_risk_wrong_executions"])
    print(f"\n{name}")
    print(f"  1 magic phrases   {magic:5.1f}%  (n={len(s['magic'])})   {'PASS' if magic >= MAGIC_BAR else 'FAIL'}")
    for key, label in (("hub", "real Hub"), ("mac", "real Mac")):
        rows = s[key]
        if not rows:
            continue
        a, b, p = mcnemar({r["id"]: r for r in rows}, pred, stock, mine, served("stock"))
        acc = pct(pred, rows, correct)
        verdict = "PASS" if p < P_BAR and a > b else "FAIL"
        print(f"  3 {label:<14} {acc:5.1f}%  (n={len(rows)}) p={p:.3g} ({a}+/{b}-)   {verdict}")
    print(f"  4 false accept    {fa:5.1f}%  (n={len(s['neg'])})   {'PASS' if fa <= FA_BAR else 'FAIL'}")
    print(f"  5 high-risk exec  {hx}            {'PASS' if hx == 0 else 'FAIL'}")


def main(argv):
    split, argv = pop_split(argv)
    split = split or "test"
    if len(argv) < 4:
        sys.exit(__doc__)
    gold = {i: g for i, g in load(argv[0]).items() if g.get("split") == split}
    label2canon = {}
    for canon, spec in yaml.safe_load(open(argv[1]))["intents"].items():
        for e in spec.get("ovos") or []:
            label2canon[e if isinstance(e, str) else e["label"]] = canon
    s = slices(gold, label2canon)
    stock = load(argv[2])
    print(f"split={split}  rows={len(gold)}  (criteria 2 determinism, 6 latency: separate runs)")
    report("stock OVOS (reference)", stock, stock, gold, s)
    for path in argv[3:]:
        report(os.path.basename(path).removesuffix(".jsonl"), load(path), stock, gold, s)


if __name__ == "__main__":
    main(sys.argv[1:])
