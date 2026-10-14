#!/usr/bin/env python3
"""Reference predictions v6 = v5 + the word-boundary matcher + the catch-all yield at 0.8, for switchboard's gate.

The yield comes from exp_guard_rules.py's own implementation (it replaces the exact decision outright), NOT from the
packaged ExactStage, so scripts/check_harness.py comparing the package to these files checks the package against the
rule as measured. Same record shape as the v5 references. LLM cache misses are asked of the LLM server and cached.

v7 (2026-10-14) = v6 with float32 classifier features (float32 only; BENCH "Full benchmark: float32"): the
same script, run on a float32 switchboard checkout, writing preds/v7__*.

Usage (from intent-eval, in .venv-bench, switchboard checkout with the word-boundary matcher):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.refs_v6 <switchboard>/scripts [v7]
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys

from switchboard_eval.experiments.exp_guard_rules import guard
from switchboard_eval.experiments.exp_mac_draft import build


YIELD = 0.8
KEYS = ("intent", "stage", "confidence", "top3", "decline_class")


def main(scripts_dir, version="v6"):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp
    from ovos_switchboard.core import templates
    from ovos_switchboard.core.schema import Schema

    if templates.WORD_GAP == r"\s*":
        raise SystemExit("needs the word-boundary matcher")
    if version == "v7":
        import numpy as np
        from ovos_switchboard.core.classifier import _vectorizers
        if any(v.dtype is not np.float32 for v in _vectorizers()):
            raise SystemExit("v7 references need a float32 switchboard checkout")
    harness = ROOT
    os.chdir(harness)
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    for device in ("mac", "hub"):
        for route in ("always", "active"):
            router = guard(build(ch, harness, device, schema, route=route), None, YIELD)
            for split in ("dev", "test"):
                gold, _, _ = cp.setup(split)
                _, _, _, ctx = cp.load_inputs(split)
                out = os.path.join(harness, "preds", f"{version}__{device}__{route}__{split.upper()}.jsonl")
                with open(out, "w", encoding="utf-8") as f:
                    for rid, r in gold.items():
                        c = ctx.get(rid, cp.COLD)
                        d = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                        f.write(json.dumps({"id": rid, **{k: d[k] for k in KEYS if k in d}}) + "\n")
                print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
