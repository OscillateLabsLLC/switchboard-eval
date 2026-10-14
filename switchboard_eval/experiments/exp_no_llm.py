#!/usr/bin/env python3
"""Rung 0 of the small-model ladder: the bar with NO LLM, against the shipped default (v6, Qwen3-14B-AWQ on the GPU host).

What an upstream default without any LLM would ship. Same routers as the gate (packaged Switchboard, v6 defaults:
catch-all yield 0.8), the LLM step removed, so whatever the classifier passes is a pass (no intent). Each device's
router on its own traffic, as the bar decides; false accepts on the shared negative rows. Writes
preds/nollm__<device>__active__<SPLIT>.jsonl so bar.py can score it next to the v6 references.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_no_llm <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys


KEYS = ("intent", "stage", "confidence", "top3", "decline_class")


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp

    harness = ROOT
    os.chdir(harness)
    for device in ("mac", "hub"):
        router = ch.build_router(harness, device, "active")
        router.llm = None
        for split in ("dev", "test"):
            gold, _, _ = cp.setup(split)
            _, _, _, ctx = cp.load_inputs(split)
            out = os.path.join(harness, "preds", f"nollm__{device}__active__{split.upper()}.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for rid, r in gold.items():
                    c = ctx.get(rid, cp.COLD)
                    d = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                    f.write(json.dumps({"id": rid, **{k: d[k] for k in KEYS if k in d}}) + "\n")
            print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
