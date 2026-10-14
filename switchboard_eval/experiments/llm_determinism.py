#!/usr/bin/env python3
"""Criterion 2 for the LLM path: does the frozen Phase 2 config give the same answer twice?

Re-runs every routed row (dev + test) with NO cache, several times, and compares each run to
the original cached run (cache/llm_cache.jsonl). Sequential runs test temperature-0 repeatability.
A concurrent run (--workers N) tests vLLM batch nondeterminism: concurrent requests change
batch composition, which changes kernel numerics, which can flip close calls. Live use shares
the GPU host with other workloads, so the concurrent run is the realistic one.

Reports flips in: tool chosen, tool arguments, gate decision at --gate, and final intent.

Usage: llm_determinism.py [--runs seq1:1,seq2:1,par8:8] [--gate 0.9]   (bench venv)
Writes each run's raw responses to llm_runs/<tag>.jsonl (a fresh file per run).
"""
from switchboard_eval.paths import ROOT
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = ROOT
sys.path.insert(0, HERE)
from switchboard_eval.harness.llm_fallback import LLMRouter, should_route
from switchboard_eval.harness.shortlist import Shortlist

PREDS = "preds/exact_ocp__char__logreg__syn_dlg__t0.5.jsonl"
SYSTEM_FILE, GATE_FILE, K = "prompts/fallback_strict.txt", "prompts/addressee_gate.txt", 12


def routed_utterances():
    rows = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "data/dataset.jsonl")))}
    preds = map(json.loads, open(os.path.join(HERE, PREDS)))
    return [(p["id"], rows[p["id"]]["utterance"]) for p in preds if should_route(p, None)]


def run(shortlist, cache_path, items, workers):
    router = LLMRouter(shortlist, cache_path, open(os.path.join(HERE, SYSTEM_FILE)).read().strip())
    gate = open(os.path.join(HERE, GATE_FILE)).read().strip()

    def one(item):
        rid, utt = item
        res, g = router.route(utt, K), router.p_addressed(utt, gate)
        return rid, {"tool": res["tool"], "args": res["args"], "p": g["p_yes"], "cached": res["cached"]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, items))


def compare(name, base, other, gate):
    keys = sorted(base)
    tool = [k for k in keys if base[k]["tool"] != other[k]["tool"]]
    args = [k for k in keys if base[k]["tool"] == other[k]["tool"] and base[k]["args"] != other[k]["args"]]
    passes = lambda r: r["p"] >= gate
    gated = [k for k in keys if passes(base[k]) != passes(other[k])]
    final = lambda r: r["tool"] if passes(r) else "<declined>"
    flips = [k for k in keys if final(base[k]) != final(other[k])]
    dp = max(abs(base[k]["p"] - other[k]["p"]) for k in keys)
    print(f"{name}: tool {len(tool)}, args {len(args)}, gate {len(gated)}, FINAL {len(flips)}/{len(keys)}"
          f"  (max |dP| {dp:.4f})")
    for k in sorted(set(tool + args + gated)):
        print(f"   {k}: {base[k]['tool']} {base[k]['args']} P={base[k]['p']:.3f}  ->  "
              f"{other[k]['tool']} {other[k]['args']} P={other[k]['p']:.3f}")
    return len(flips)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="seq1:1,seq2:1,par8:8")
    ap.add_argument("--gate", type=float, default=0.9)
    a = ap.parse_args()
    items = routed_utterances()
    shortlist = Shortlist(os.path.join(HERE, "devices/desktop/tools.json"), os.path.join(HERE, "devices/desktop/examples.jsonl"))
    base = run(shortlist, os.path.join(HERE, "cache/llm_cache.jsonl"), items, 1)
    missing = [k for k, v in base.items() if not v["cached"]]
    print(f"{len(items)} routed rows; baseline from cache/llm_cache.jsonl ({len(missing)} were not cached)")
    os.makedirs(os.path.join(HERE, "llm_runs"), exist_ok=True)
    for spec in a.runs.split(","):
        tag, workers = spec.split(":")
        path = os.path.join(HERE, "llm_runs", f"{tag}.jsonl")
        if os.path.exists(path):
            sys.exit(f"{path} exists; pick a new run tag so the run is uncached")
        compare(f"{tag} (workers={workers})", base, run(shortlist, path, items, int(workers)), a.gate)


if __name__ == "__main__":
    main()
