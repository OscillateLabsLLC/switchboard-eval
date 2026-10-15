#!/usr/bin/env python3
"""Latency per predictions file, from the latency_ms that run_ovos.py records (the live query path).

For a live Switchboard run, two subsets too: rows decided by the exact or classifier step, and rows decided by the
LLM step. (Rows the LLM passed on to a fallback also paid for the LLM call; they're in the total.)

Usage: python -m switchboard_eval.harness.latency preds1.jsonl [preds2.jsonl ...]
"""
import json
import os
import sys

FAST_STAGES = ("ovos-switchboard-exact-pipeline-plugin", "ovos-switchboard-classifier-pipeline-plugin")
LLM_STAGE = "ovos-switchboard-llm-pipeline-plugin"


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q / 100 * len(values)))]


def line(name, values):
    return f"{name:40} {len(values):4} rows   median {pct(values, 50):7.1f} ms   p95 {pct(values, 95):7.1f} ms"


def main(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        name = os.path.basename(path).removesuffix(".jsonl")
        print(line(name, [r["latency_ms"] for r in rows]))
        if any(str(r.get("stage", "")).startswith("ovos-switchboard") for r in rows):
            fast = [r["latency_ms"] for r in rows if r.get("stage") in FAST_STAGES]
            slow = [r["latency_ms"] for r in rows if r.get("stage") == LLM_STAGE]
            print(line("  decided by exact or classifier", fast))
            print(line("  decided by the LLM step", slow))


if __name__ == "__main__":
    main(sys.argv[1:])
