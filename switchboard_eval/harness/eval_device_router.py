#!/usr/bin/env python3
"""Score a device router configuration on data/dataset.jsonl with the go/no-go bar (same slices as bar.py).

The Mac numbers so far came from the Mac router; the Hub router (its own skills) needs its own
measurement before any Hub replay number means anything.

Usage: eval_device_router.py --device hub|mac [--split dev] [--route always|active|never] [--no-llm] [--out preds.jsonl]
"""
from switchboard_eval.paths import ROOT
import argparse, json, os, sys

HERE = ROOT
sys.path.insert(0, HERE)
from switchboard_eval.harness import context_policy as cp
from switchboard_eval.harness.router import DeviceConfig, Router


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="hub", choices=["hub", "mac"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--route", default="always", choices=["always", "active", "never"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--llm-context", action="store_true", help="experiment: conversation context for the LLM")
    ap.add_argument("--legacy", action="store_true",
                    help="pre-2026-10-12 config: whole-schema classes, unmasked synthetic, no shared phrases")
    a = ap.parse_args()
    kw = {"route_decline_class": a.route, "llm": not a.no_llm, "llm_context": a.llm_context}
    if a.legacy:
        neg = "devices/speaker/dialog_negatives.jsonl" if a.device == "hub" else "devices/desktop/dialog_negatives.jsonl"
        kw.update(device_classes_only=False, shared_examples="", synthetic=f"skills/synthetic_examples.jsonl,{neg}")
    router = Router(DeviceConfig.hub(**kw) if a.device == "hub" else DeviceConfig(**kw))
    gold, slices, stock = cp.setup(a.split)
    _, _, _, ctx = cp.load_inputs(a.split)
    last_speak = {}
    for dev in ("hub", "mac"):
        path = os.path.join(HERE, "data", dev, "row_last_speak.json")
        if os.path.exists(path):
            last_speak.update(json.load(open(path)))
    preds = {}
    for rid, r in gold.items():
        c = ctx.get(rid)  # rows without logs get NO timing keys, so nothing is claimed about them
        context = {"gap_s": c["gap_s"], "echo_run": c["echo_run"], "last_speak": last_speak.get(rid)} if c else {}
        preds[rid] = {"id": rid, **router.route(r["utterance"], context)}
    if a.out:
        with open(a.out, "w") as f:
            for p in preds.values():
                f.write(json.dumps(p, default=str) + "\n")
    print(cp.row(f"{a.device} router, route={a.route}{' no-llm' if a.no_llm else ''}{' LEGACY' if a.legacy else ''}{' +context' if a.llm_context else ''}, {a.split}",
                 cp.evaluate(gold, slices, stock, preds)))


if __name__ == "__main__":
    main()
