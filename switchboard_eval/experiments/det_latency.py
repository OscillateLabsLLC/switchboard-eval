#!/usr/bin/env python3
"""Bar items 2 and 6 for the packaged Switchboard router, offline.

  run <out.json>         one fresh build per device (new process = "restart"; the core Router never uses the OVOS
                         runtime's model cache, so the classifier retrains every time), every dev+test row routed on
                         the active route. LLM answers from the harness cache only; a cache miss routes that row with
                         the LLM step off. Writes {device/split/id: [intent, label, stage]}.
  compare <a> <b> [...]  rows whose (intent, label, stage) differ between runs.
  latency                Mac router, LLM off, warm: per-stage and whole-path times over dev+test rows and the archived
                         real Mac transcripts.

Usage (from intent-eval, in .venv-bench):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.det_latency <switchboard>/scripts run preds/det_run1.json
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.det_latency <switchboard>/scripts compare preds/det_run1.json preds/det_run2.json
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.det_latency <switchboard>/scripts latency [--rows-only]
"""
from switchboard_eval.paths import ROOT
import json
import os
import statistics
import sys
import time


def route(router, utterance, ctx):
    """The decision, with the LLM step off for a harness-cache miss."""
    try:
        return router.route(utterance, ctx), False
    except RuntimeError:
        llm, router.llm = router.llm, None
        try:
            return router.route(utterance, ctx), True
        finally:
            router.llm = llm


def run(ch, cp, harness, out):
    """One build per device; every dev+test decision."""
    decisions, misses = {}, 0
    for device in ("mac", "hub"):
        router = ch.build_router(harness, device, "active")
        for split in ("dev", "test"):
            gold, _, _ = cp.setup(split)
            _, _, _, ctx = cp.load_inputs(split)
            for rid, r in gold.items():
                c = ctx.get(rid, cp.COLD)
                d, miss = route(router, r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                misses += miss
                decisions[f"{device}/{split}/{rid}"] = [d.get("intent"), d.get("label"), d.get("stage")]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(decisions, f, sort_keys=True)
    print(f"{out}: {len(decisions)} decisions, {misses} LLM cache misses routed with the LLM off")


def compare(paths):
    """Rows whose decision differs across runs."""
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    keys = set().union(*runs)
    flips = [k for k in sorted(keys) if len({json.dumps(r.get(k)) for r in runs}) > 1]
    print(f"{len(paths)} runs, {len(keys)} rows, {len(flips)} flips")
    for k in flips:
        print(f"  {k}: " + " | ".join(json.dumps(r.get(k)) for r in runs))


def pct(values, q):
    """The q-th percentile (nearest rank), in ms."""
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q / 100 * len(s)) - 1))]


def latency(ch, cp, harness, rows_only=False):
    """Per-stage and whole-path timings, LLM off, after one warm-up pass. rows_only: dev+test rows without the
    archived transcripts, for hosts that shouldn't hold them (the mini PC run, 2026-10-14)."""
    t0 = time.perf_counter()
    router = ch.build_router(harness, "mac", "active")
    print(f"build (classifier trained) {time.perf_counter() - t0:.1f} s")
    router.llm = None
    utterances = []
    for split in ("dev", "test"):
        gold, _, _ = cp.setup(split)
        utterances += [r["utterance"] for r in gold.values()]
    if not rows_only:
        with open(os.path.join(harness, "data", "mac", "context.jsonl"), encoding="utf-8") as f:
            utterances += [json.loads(line)["text"] for line in f]
    ctx = {"gap_s": 1e9, "echo_run": 0}
    stages = {"echo": router.echo, "exact": router.exact, "classifier": router.classifier}
    if getattr(router, "prefix", None):  # PR #5; absent on main
        stages["prefix"] = router.prefix
    for u in utterances:  # warm-up
        router.route(u, ctx)
    times = {name: [] for name in [*stages, "route (whole path)"]}
    for u in utterances:
        for name, stage in stages.items():
            t0 = time.perf_counter()
            stage.decide(u, ctx)
            times[name].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        router.route(u, ctx)
        times["route (whole path)"].append((time.perf_counter() - t0) * 1000)
    print(f"{len(utterances)} utterances, warm, LLM off (ms)")
    print(f"{'stage':20} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7} {'mean':>7}")
    for name, v in times.items():
        print(f"{name:20} {pct(v, 50):7.2f} {pct(v, 95):7.2f} {pct(v, 99):7.2f} {max(v):7.2f} "
              f"{statistics.mean(v):7.2f}")


def main(argv):
    sys.path.insert(0, argv[0])
    sys.path.insert(0, ROOT)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp

    harness = ROOT
    os.chdir(harness)
    if argv[1] == "run":
        run(ch, cp, harness, argv[2])
    elif argv[1] == "compare":
        compare(argv[2:])
    else:
        latency(ch, cp, harness, rows_only="--rows-only" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
