#!/usr/bin/env python3
"""Regression check: router.py must reproduce the committed reference predictions exactly, row by row.

References are the router's own outputs under the adopted config (2026-10-12, preds/v5__<device>__<route>__<SPLIT>):
device class sets, slot-masked synthetic examples, Mac phrases shared into the Hub router.
LLM answers come from cache/llm_cache.jsonl, so this makes no model calls for rows already measured.
Earlier references (preds/v2..v4, bench.py + llm_fallback.py) document the pre-adoption pipeline.

Usage: check_router.py [--quick]   (--quick: dev only)   (bench venv)
"""
from switchboard_eval.paths import ROOT
import json, os, sys

HERE = ROOT
sys.path.insert(0, HERE)
from switchboard_eval.harness import context_policy as cp
from switchboard_eval.harness.router import DeviceConfig, Router
from switchboard_eval.harness.score import load


def check(device, route, split):
    ref_path = os.path.join(HERE, f"preds/v5__{device}__{route}__{split.upper()}.jsonl")
    ref = load(ref_path)
    router = Router(DeviceConfig.hub(route_decline_class=route) if device == "hub"
                    else DeviceConfig(route_decline_class=route))
    gold, _, _ = cp.setup(split)
    _, _, _, ctx = cp.load_inputs(split)
    bad = []
    for rid, r in gold.items():
        c = ctx.get(rid, cp.COLD)
        got = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
        if got.get("intent") != ref[rid].get("intent"):
            bad.append((rid, r["utterance"], ref[rid].get("intent"), got.get("intent")))
    print(f"{device} {route} {split}: {len(gold) - len(bad)}/{len(gold)} identical")
    for b in bad[:8]:
        print("   MISMATCH", b)
    return not bad


def main():
    splits = ["dev"] if "--quick" in sys.argv else ["dev", "test"]
    ok = all([check(d, r, s) for d in ("mac", "hub") for r in ("always", "active") for s in splits])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
