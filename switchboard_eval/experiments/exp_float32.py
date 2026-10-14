#!/usr/bin/env python3
"""Full benchmark for float32 classifier features (2026-10-14).

float32 TF-IDF features make the fit run in float32: training peak -16%, time -33% at 465 intents
(exp_memory_float32.py), but a slightly different model. Rule fixed before results: adopt only if (1) no bar
criterion flips PASS -> FAIL on dev or test, (2) no significant loss vs float64 (paired McNemar p >= 0.05 or float32
better) on any real-traffic slice, (3) determinism stays 0 flips, (4) non-LLM p95 latency not worse beyond noise.

This script: the packaged router built from a float32 switchboard checkout, LLM = Qwen3-14B-AWQ on the GPU host (cache
first, misses asked and cached), every device x decline policy x split, scored against the v6 (float64) references.
Writes preds/f32__<device>__<route>__<SPLIT>.jsonl.

Usage (from intent-eval, in .venv-bench):
    PYTHONPATH=<float32 switchboard> python -m switchboard_eval.experiments.exp_float32 <float32 switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys

from switchboard_eval.experiments.exp_ext_active import LLM_URL

KEYS = ("intent", "stage", "confidence", "top3", "decline_class")


def router_for(ch, harness, device, route):
    from ovos_switchboard.adapters.openai_llm import LLMClient, LLMEndpoint
    router = ch.build_router(harness, device, route)

    def prompt(name):
        with open(os.path.join(harness, "prompts", name), encoding="utf-8") as f:
            return f.read().strip()

    router.llm.llm = LLMClient(LLMEndpoint(url=LLM_URL, model=ch.HARNESS_MODEL, timeout_s=30),
                               system_prompt=prompt("fallback_strict.txt"), gate_prompt=prompt("addressee_gate.txt"),
                               cache_path=os.path.join(harness, "cache/llm_cache.jsonl"))
    return router


def compare(cp, score_mod, split, device, route):
    """bar metrics for f32 and v6, and McNemar f32 vs v6 on each real slice."""
    gold, slices, stock = cp.setup(split)
    f32 = score_mod.load(f"preds/f32__{device}__{route}__{split.upper()}.jsonl")
    v6 = score_mod.load(f"preds/v6__{device}__{route}__{split.upper()}.jsonl")
    out = {"f32": cp.evaluate(gold, slices, stock, f32), "v6": cp.evaluate(gold, slices, stock, v6)}
    for key in ("hub", "mac"):
        rows = {r["id"]: r for r in slices[key]}
        wins, losses, p = score_mod.mcnemar(rows, f32, v6)
        out[f"mcnemar_{key}"] = {"f32_only_right": wins, "v6_only_right": losses, "p": p}
    changed = [rid for rid in gold if f32.get(rid, {}).get("intent") != v6.get(rid, {}).get("intent")]
    out["changed"] = [(rid, gold[rid]["utterance"], gold[rid]["intent"], v6[rid].get("intent"), f32[rid].get("intent"))
                      for rid in changed]
    return out


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp
    from switchboard_eval.harness import score as score_mod

    harness = ROOT
    os.chdir(harness)
    results = {}
    for device in ("mac", "hub"):
        for route in ("active", "always"):
            router = router_for(ch, harness, device, route)
            for split in ("dev", "test"):
                gold, _, _ = cp.setup(split)
                _, _, _, ctx = cp.load_inputs(split)
                with open(f"preds/f32__{device}__{route}__{split.upper()}.jsonl", "w", encoding="utf-8") as f:
                    for rid, r in gold.items():
                        c = ctx.get(rid, cp.COLD)
                        d = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                        f.write(json.dumps({"id": rid, **{k: d[k] for k in KEYS if k in d}}) + "\n")
                results[f"{device}/{route}/{split}"] = compare(cp, score_mod, split, device, route)
                print(json.dumps({f"{device}/{route}/{split}": results[f"{device}/{route}/{split}"]}, default=str),
                      flush=True)
    with open("preds/exp_float32.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
