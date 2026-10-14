#!/usr/bin/env python3
"""Phase 3 addressee rules on top of the frozen pipeline, evaluated on the go/no-go bar.

Recombines existing predictions (no model calls), so sweeps are instant and deterministic:
  phase1   exact layer + classifier (preds/v2__exact_...)
  llm      frozen LLM fallback, full routing (preds/v2__llm_strict_gate0.9__{DEV,TEST})
  clarify  same + clarify tool (preds/v2__llm_clarify_gate0.9__{DEV,TEST})
Rules (context from data/context/{hub,mac}_row_context.jsonl; rows without it are COLD: never spoke, no echo):
  echo     echo_run >= K words shared with recent assistant speech -> decline, before anything else
  route    Phase 1 declines: low-confidence ones always go to the LLM; decline-CLASS ones go
           never / only when the assistant spoke within ACTIVE_S / always
  clarify  the clarify-enabled LLM answer is used never / only when active / always

Usage: context_policy.py [--split dev] [--sweep]
       context_policy.py --split dev --echo 4 --active 30 --route active --clarify active
"""
from switchboard_eval.paths import ROOT
import argparse, json, os, sys

HERE = ROOT
sys.path.insert(0, HERE)
from switchboard_eval.harness.bar import slices, pct
from switchboard_eval.harness.score import load, mcnemar, score

PHASE1 = "preds/v4__exact_ocp__char__logreg__syn_dlg__t0.5.jsonl"
LLM = "preds/v4__llm_strict_gate0.9__{}.jsonl"
CLARIFY = "preds/v2__llm_clarify_gate0.9__{}.jsonl"
COLD = {"gap_s": None, "asked": False, "echo_run": 0}


def load_inputs(split):
    def opt(path):
        p = os.path.join(HERE, path.format(split.upper()))
        return load(p) if os.path.exists(p) else None
    ctx = {r["row_id"]: r for dev in ("hub", "mac") if os.path.exists(os.path.join(HERE, f"data/context/{dev}_row_context.jsonl"))
           for r in map(json.loads, open(os.path.join(HERE, f"data/context/{dev}_row_context.jsonl")))}
    return opt(PHASE1), opt(LLM), opt(CLARIFY), ctx  # the Phase 1 file is private-only; the public tree has none


def active(c, active_s):
    return c["gap_s"] is not None and c["gap_s"] < active_s


def decide(rid, phase1, llm, clarify, ctx, echo, active_s, route, use_clarify):
    c, p = ctx.get(rid, COLD), phase1[rid]
    if echo and c["echo_run"] >= echo:
        return {**p, "intent": None, "stage": "rule_echo"}
    if p["intent"] is not None or p.get("stage") == "exact":
        return p
    decline_class = (p.get("top3") or [""])[0] == "__decline__"
    if decline_class and not (route == "always" or (route == "active" and active(c, active_s))):
        return p
    wants_clarify = use_clarify == "always" or (use_clarify == "active" and active(c, active_s))
    source = clarify if wants_clarify and clarify is not None else llm
    return source.get(rid, p) if source is not None else p


def evaluate(gold, s, stock, preds):
    correct = lambda p, r: p.get("intent") == r["intent"]
    hub_a, hub_b, hub_p = mcnemar({r["id"]: r for r in s["hub"]}, preds, stock)
    return {"magic": pct(preds, s["magic"], correct), "hub": pct(preds, s["hub"], correct), "hub_p": hub_p,
            "mac": pct(preds, s["mac"], correct),
            "fa": pct(preds, s["neg"], lambda p, r: p.get("intent") is not None),
            "hx": len(score(gold, preds, sweep=False)["high_risk_wrong_executions"])}


def setup(split):
    import yaml
    gold = {i: g for i, g in load(os.path.join(HERE, "data/dataset.jsonl")).items() if g.get("split") == split}
    label2canon = {}
    for canon, spec in yaml.safe_load(open(os.path.join(HERE, "schema/intents.yaml")))["intents"].items():
        for e in spec.get("ovos") or []:
            label2canon[e if isinstance(e, str) else e["label"]] = canon
    return gold, slices(gold, label2canon), load(os.path.join(HERE, "data/baselines/baseline_stock.jsonl"))


def row(name, m):
    return (f"{name:44} magic {m['magic']:5.1f}  Hub {m['hub']:5.1f} (p={m['hub_p']:.1g})  "
            f"Mac {m['mac']:5.1f}  FA {m['fa']:5.1f}  hx {m['hx']}")


def run(split, echo, active_s, route, use_clarify, inputs, ctx_setup):
    phase1, llm, clarify, ctx = inputs
    gold, s, stock = ctx_setup
    preds = {rid: decide(rid, phase1, llm, clarify, ctx, echo, active_s, route, use_clarify) for rid in gold}
    return evaluate(gold, s, stock, preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--echo", type=int, default=0)
    ap.add_argument("--active", type=float, default=30)
    ap.add_argument("--route", default="never", choices=["never", "active", "always"])
    ap.add_argument("--clarify", default="never", choices=["never", "active", "always"])
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    inputs, cs = load_inputs(a.split), setup(a.split)
    if not a.sweep:
        print(row(f"echo={a.echo} active={a.active:g}s route={a.route} clarify={a.clarify}",
                  run(a.split, a.echo, a.active, a.route, a.clarify, inputs, cs)))
        return
    print(f"split={a.split}  (route/clarify 'active' = assistant spoke within the window)")
    for name, args in [("phase1 only (no LLM)", None),
                       ("frozen: route=never  (upstream default)", (0, 30, "never", "never")),
                       ("frozen: route=always (opt-in)", (0, 30, "always", "never"))]:
        m = evaluate(cs[0], cs[1], cs[2], inputs[0]) if args is None else run(a.split, *args, inputs, cs)
        print(row(name, m))
    for echo in (0, 3, 4, 5, 6):
        print(row(f"echo>={echo}, route=never", run(a.split, echo, 30, "never", "never", inputs, cs)))
    for active_s in (15, 30, 60, 120, 300):
        for route, clar in (("active", "never"), ("active", "active"), ("never", "active")):
            print(row(f"echo>=4 active<{active_s:g}s route={route} clarify={clar}",
                      run(a.split, 4, active_s, route, clar, inputs, cs)))


if __name__ == "__main__":
    main()
