#!/usr/bin/env python3
"""Measurement: the shipped build on the external negatives stress slice (data/negatives_ext.jsonl, 600 overheard sentences).

The 2026-10-12 numbers (1.3% without the LLM, 14.3% with it) predate the `active` decline policy and the v6 build;
the ~2.5% for "route only below-threshold rows" was an estimate. Here, the shipped Hub router (check_harness), every
sentence cold (a false wake with the assistant silent: gap None), three variants:
  nollm    the LLM step off
  active   Qwen3-14B-AWQ on the GPU host, route_decline_class "active" (shipped default)
  always   the same LLM, "always" (the opt-in policy): should land near the old 14.3%, which checks the instrument
Cache misses go to the LLM server and are cached in cache/llm_cache.jsonl. Writes preds/ext__<variant>.jsonl for score_ext.py.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_ext_active <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys

LLM_URL = os.environ.get("INTENT_EVAL_LLM_URL", "http://localhost:8080/v1/chat/completions")
KEYS = ("intent", "stage", "confidence", "decline_class", "llm_tool")


def router_for(ch, harness, variant):
    from ovos_switchboard.adapters.openai_llm import LLMClient, LLMEndpoint
    from ovos_switchboard.core.options import ClassifierOptions
    router = ch.build_router(harness, "hub", "always" if variant == "always" else "active")
    if variant == "nollm":
        router.llm = None
        return router

    def prompt(name):
        with open(os.path.join(harness, "prompts", name), encoding="utf-8") as f:
            return f.read().strip()

    router.llm.llm = LLMClient(LLMEndpoint(url=LLM_URL, model=ch.HARNESS_MODEL, timeout_s=30),
                               system_prompt=prompt("fallback_strict.txt"), gate_prompt=prompt("addressee_gate.txt"),
                               cache_path=os.path.join(harness, "cache/llm_cache.jsonl"))
    assert router.classifier.options.route_decline_class == ClassifierOptions(
        route_decline_class="always" if variant == "always" else "active").route_decline_class
    return router


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch

    harness = ROOT
    os.chdir(harness)
    with open("data/negatives_ext.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for variant in ("nollm", "active", "always"):
        router = router_for(ch, harness, variant)
        out = os.path.join("preds", f"ext__{variant}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                d = router.route(r["utterance"], {"gap_s": None, "echo_run": 0})
                f.write(json.dumps({"id": r["id"], **{k: d[k] for k in KEYS if k in d}}) + "\n")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
