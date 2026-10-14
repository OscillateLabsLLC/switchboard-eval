#!/usr/bin/env python3
"""Score intent predictions against data/dataset.jsonl. Stdlib only.

Predictions JSONL, one row per dataset id:
  {"id": "u0001", "intent": "lights.on" | null, "slots": {...},
   "confidence": 0.87 | null, "latency_ms": 12.3 | null}
  intent null = system declined / no match.

Usage:
  score.py data/dataset.jsonl preds.jsonl              # report
  score.py data/dataset.jsonl preds_a.jsonl preds_b.jsonl   # report both + McNemar
  add --json for machine-readable output
  add --split dev|test to score one half: choose systems on dev, report test once
"""
import json, math, re, string, sys
from collections import defaultdict

SEPARATE = {"compound", "followup"}          # excluded from headline numbers
RISK_ORDER = ["none", "low", "medium", "high"]
# Decision policy (owner decision 2026-10-11): intents at these risk levels always CONFIRM
# and never auto-execute. A wrong prediction of one costs a confirm prompt, not an action.
CONFIRM_RISKS = {"high"}


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path)) if r}


def norm(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation.replace(":", "")))
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^the ", "", s)


def correct(gold, pred, served=frozenset()):
    """Headline correctness. Compound: partial credit if pred is any gold intent. `served`: (predicted, gold) pairs
    the predicting device's skill handlers serve (serves.py), counted right."""
    g, p = gold["intent"], pred.get("intent")
    if isinstance(g, list):
        return p in g
    return p == g or (p, g) in served


def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def mcnemar(gold, a, b, served_a=frozenset(), served_b=frozenset()):
    """Exact two-sided McNemar on headline items. Returns (a_only, b_only, p)."""
    a_only = b_only = 0
    for i, g in gold.items():
        if g["category"] in SEPARATE:
            continue
        ca, cb = correct(g, a.get(i, {}), served_a), correct(g, b.get(i, {}), served_b)
        a_only += ca and not cb
        b_only += cb and not ca
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n * 2
    return a_only, b_only, min(1.0, p)


def score(gold, pred, sweep=True):
    rep = {"missing_predictions": [i for i in gold if i not in pred]}
    head = [(g, pred.get(i, {})) for i, g in gold.items() if g["category"] not in SEPARATE]

    ok = sum(correct(g, p) for g, p in head)
    rep["headline_accuracy"] = pct(ok, len(head))
    rep["headline_n"] = len(head)
    # ~95% CI half-width, normal approx; differences inside this are noise
    acc = ok / len(head) if head else 0
    rep["ci95_halfwidth_pts"] = round(196 * math.sqrt(acc * (1 - acc) / len(head)), 1) if head else None

    ind = [(g, p) for g, p in head if g["intent"] is not None]
    nul = [(g, p) for g, p in head if g["intent"] is None]
    rep["in_domain_accuracy"] = pct(sum(correct(g, p) for g, p in ind), len(ind))
    rep["in_domain_false_reject"] = pct(sum(p.get("intent") is None for g, p in ind), len(ind))
    rep["false_accept_rate"] = pct(sum(p.get("intent") is not None for g, p in nul), len(nul))

    # High-risk: any prediction of a high-risk intent that wasn't the gold intent.
    # Under CONFIRM_RISKS these become wrong confirm prompts (annoyance, minimize);
    # only the ones the policy would execute count as wrong executions (blocker).
    intent_risk = {g["intent"]: g["risk"] for g in gold.values() if isinstance(g["intent"], str)}
    rep["high_risk_wrong_fires"] = [
        {"id": i, "utterance": g["utterance"], "gold": g["intent"], "pred": pred[i]["intent"]}
        for i, g in gold.items()
        if i in pred and intent_risk.get(pred[i].get("intent")) == "high"
        and pred[i].get("intent") != g["intent"]
    ]
    rep["high_risk_wrong_executions"] = [
        e for e in rep["high_risk_wrong_fires"] if intent_risk.get(e["pred"]) not in CONFIRM_RISKS
    ]

    # Slots: exact match on normalized surface spans, only where intent was right
    s_ok = s_n = 0
    for g, p in head:
        if g["score_slots"] and isinstance(g["intent"], str) and g["slots"] and correct(g, p):
            s_n += 1
            gs = {k: norm(v) for k, v in g["slots"].items()}
            ps = {k: norm(v) for k, v in (p.get("slots") or {}).items() if k in gs}
            s_ok += gs == ps
    rep["slot_exact_match"] = pct(s_ok, s_n)
    rep["slot_n"] = s_n

    cats = defaultdict(lambda: [0, 0])
    for i, g in gold.items():
        c = cats[g["category"]]
        c[0] += correct(g, pred.get(i, {}))
        c[1] += 1
    rep["by_category"] = {k: {"acc": pct(a, n), "n": n} for k, (a, n) in sorted(cats.items())}

    if any("seen_in_templates" in g for g in gold.values()):
        sv = defaultdict(lambda: [0, 0])
        for g, p in head:
            k = "seen" if g.get("seen_in_templates") else "unseen"
            sv[k][0] += correct(g, p)
            sv[k][1] += 1
        rep["template_coverage"] = {k: {"acc": pct(a, n), "n": n} for k, (a, n) in sorted(sv.items())}

    lat = [p["latency_ms"] for p in pred.values() if p.get("latency_ms") is not None]
    rep["latency_ms"] = {"p50": percentile(lat, .5), "p95": percentile(lat, .95), "n": len(lat)}

    if sweep and any(p.get("confidence") is not None for p in pred.values()):
        rows = []
        for t in [x / 20 for x in range(0, 20)]:
            def thr(p):
                c = p.get("confidence")
                return p.get("intent") if c is None or c >= t else None
            acc_ = [(g, {**p, "intent": thr(p)}) for g, p in ind]
            nul_ = [(g, {**p, "intent": thr(p)}) for g, p in nul]
            accepted = [(g, p) for g, p in acc_ if p["intent"] is not None]
            rows.append({
                "threshold": t,
                "coverage": pct(len(accepted), len(acc_)),
                "precision_when_acting": pct(sum(correct(g, p) for g, p in accepted), len(accepted)),
                "false_accept_rate": pct(sum(p["intent"] is not None for g, p in nul_), len(nul_)),
            })
        rep["threshold_sweep"] = rows

    rep["errors"] = [
        {"id": i, "cat": g["category"], "utterance": g["utterance"],
         "gold": g["intent"], "pred": pred.get(i, {}).get("intent")}
        for i, g in gold.items() if not correct(g, pred.get(i, {}))
    ]
    return rep


def print_report(name, r):
    print(f"\n=== {name} ===")
    print(f"headline accuracy   {r['headline_accuracy']}%  (n={r['headline_n']}, ±{r['ci95_halfwidth_pts']} pts 95% CI)")
    print(f"in-domain accuracy  {r['in_domain_accuracy']}%   false reject {r['in_domain_false_reject']}%")
    print(f"false accept rate   {r['false_accept_rate']}%   (should-not-act items)")
    print(f"slot exact match    {r['slot_exact_match']}%  (n={r['slot_n']})")
    print(f"latency ms          p50={r['latency_ms']['p50']} p95={r['latency_ms']['p95']}")
    hx = r["high_risk_wrong_executions"]
    print(f"HIGH-RISK WRONG EXECUTIONS: {len(hx)}   (blocker; policy confirms {sorted(CONFIRM_RISKS)})")
    for e in hx:
        print(f"   !! {e['utterance']!r} -> {e['pred']} (gold {e['gold']})")
    hr = r["high_risk_wrong_fires"]
    print(f"high-risk wrong confirm prompts: {len(hr)}   (annoyance; minimize)")
    for e in hr:
        print(f"   ?  {e['utterance']!r} -> {e['pred']} (gold {e['gold']})")
    if "template_coverage" in r:
        tc = r["template_coverage"]
        print("seen vs unseen (does a skill template already match the utterance?):")
        for k in ("seen", "unseen"):
            if k in tc:
                print(f"   {k:<7} {tc[k]['acc']}%  n={tc[k]['n']}")
    print("by category:")
    for k, v in r["by_category"].items():
        tag = "  (separate)" if k in SEPARATE else ""
        print(f"   {k:<11} {str(v['acc']):>5}%  n={v['n']}{tag}")
    if "threshold_sweep" in r:
        print("threshold sweep (thr / coverage / precision when acting / false accept):")
        for s in r["threshold_sweep"]:
            print(f"   {s['threshold']:.2f}  {s['coverage']}  {s['precision_when_acting']}  {s['false_accept_rate']}")
    if r["missing_predictions"]:
        print(f"WARNING: {len(r['missing_predictions'])} dataset ids have no prediction")


def pop_split(argv):
    """Remove `--split dev|test` from argv; return (split or None, argv)."""
    if "--split" not in argv:
        return None, argv
    i = argv.index("--split")
    return argv[i + 1], argv[:i] + argv[i + 2:]


def load_gold(path, split=None):
    """Select on dev, report on test once. split=None scores every row."""
    gold = load(path)
    return {i: g for i, g in gold.items() if g.get("split") == split} if split else gold


def main(argv):
    as_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    split, argv = pop_split(argv)
    if len(argv) not in (2, 3):
        sys.exit(__doc__)
    gold = load_gold(argv[0], split)
    preds = [load(p) for p in argv[1:]]
    reps = {p: score(gold, d) for p, d in zip(argv[1:], preds)}
    if len(preds) == 2:
        a_only, b_only, p = mcnemar(gold, *preds)
        cmp = {"a_only_correct": a_only, "b_only_correct": b_only, "mcnemar_p": round(p, 4)}
    if as_json:
        out = {"reports": reps}
        if len(preds) == 2:
            out["comparison"] = cmp
        print(json.dumps(out, indent=2))
        return
    for name, r in reps.items():
        print_report(name, r)
    if len(preds) == 2:
        print(f"\n=== {argv[1]} vs {argv[2]} ===")
        print(f"A right/B wrong: {a_only}   B right/A wrong: {b_only}   McNemar p={cmp['mcnemar_p']}")
        print("p < 0.05: the difference is probably real. Otherwise treat it as noise.")


if __name__ == "__main__":
    main(sys.argv[1:])
