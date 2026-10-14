#!/usr/bin/env python3
"""Small-model ladder: the bar with the LLM step on a local model (LM Studio), against v6 (Qwen3-14B-AWQ on the GPU host)
and rung 0 (no LLM, exp_no_llm.py).

LM Studio returns no logprobs, ignores vLLM's chat_template_kwargs and doesn't enforce tool_choice=required, so this
client differs from the shipped one in three ways: the addressee gate reads a plain-text yes/no answer (p_yes 1.0 or
0.0: binary, uncalibrated); the think switch (Qwen3's "/no_think") is appended to the user message; and a text answer
that is exactly one offered tool's name counts as that tool call (salvaged; counted in the output). Prompts, tools, shortlist and routers are the
gate's. Each model gets its own cache file (llm_cache_<model>.jsonl), so the 14B cache stays untouched and reruns
are free. Writes preds/llm-<model>__<device>__active__<SPLIT>.jsonl for bar.py and prints LLM-stage latency.

Usage (from intent-eval, in .venv-bench, LM Studio serving <model>):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_small_llm <switchboard>/scripts <model> [--url URL] [--think-switch STR] [--reasoning-none]
        [--k N]         skill tools offered per utterance (shipped 12); preds files get a __kN tag when N != 12
        [--cache PATH]  a different cache file: a fresh path times every call (and re-asks every question)
"""
from switchboard_eval.paths import ROOT
import json
import os
import statistics
import sys

from ovos_switchboard.adapters.openai_llm import LLMClient, LLMEndpoint, parse_tool_call


LM_STUDIO = "http://localhost:1234/v1/chat/completions"
KEYS = ("intent", "stage", "confidence", "top3", "decline_class")


class LocalClient(LLMClient):
    """The shipped client, with a text-answer gate and a think switch in the user message."""

    def __init__(self, *args, think_switch="", reasoning_none=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.think_switch, self.calls = think_switch, []
        # LM Studio's own switch, for models that ignore "/no_think" (Qwen3.6)
        self.extra = {"reasoning_effort": "none"} if reasoning_none else {}

    def route(self, utterance, tools):
        # the shipped request body, so answers cached before the salvage rule stay valid
        body = {"model": self.endpoint.model, "temperature": 0, "max_tokens": self.endpoint.route_max_tokens,
                "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False}, "tools": tools,
                **self.extra,
                "messages": [{"role": "system", "content": self.system_prompt},
                             {"role": "user", "content": utterance + self.think_switch}]}
        names = {t["function"]["name"] for t in tools}
        res = self._cached(body, lambda r: parse_tool_call_or_name(r, names), {"n_tools": len(tools)})
        self.calls.append(("route", res["ms"], res.get("cached")))
        return res

    def p_addressed(self, utterance):
        body = {"model": self.endpoint.model, "temperature": 0, "max_tokens": 8, **self.extra,
                "messages": [{"role": "system", "content": self.gate_prompt},
                             {"role": "user", "content": utterance + self.think_switch}]}
        res = self._cached(body, parse_text_yes_no)
        self.calls.append(("gate", res["ms"], res.get("cached")))
        return res


def parse_tool_call_or_name(r, names):
    """A tool call; or, when the server doesn't enforce tool_choice=required (LM Studio), a text answer that is
    exactly one offered tool's name ("ignore", "<ignore />"). Anything else is unusable."""
    try:
        return parse_tool_call(r)
    except ValueError:
        text = (r["choices"][0]["message"].get("content") or "").strip().strip("<>/ .\n").lower()
        if text in names:
            return {"tool": text, "args": {}, "salvaged": True}
        raise


def parse_text_yes_no(r):
    """p_yes 1.0 / 0.0 from a "yes"/"no" text answer; fail on anything else."""
    words = (r["choices"][0]["message"].get("content") or "").strip().lower().split()
    first = words[0].strip(".,!") if words else ""
    if first not in ("yes", "no"):
        raise ValueError(f"gate answer is not yes/no: {r['choices'][0]['message'].get('content')!r}")
    return {"p_yes": 1.0 if first == "yes" else 0.0}


def build(ch, harness, device, client, k):
    """check_harness.build_router with the LLM swapped for `client` and `k` skill tools offered."""
    from ovos_switchboard.core.options import LLMOptions
    router = ch.build_router(harness, device, "active")
    router.llm.llm = client
    router.llm.options = LLMOptions(gate=router.llm.options.gate, shortlist_k=k)
    return router


def main(argv):
    scripts_dir, model = argv[0], argv[1]
    url = argv[argv.index("--url") + 1] if "--url" in argv else LM_STUDIO
    think = argv[argv.index("--think-switch") + 1] if "--think-switch" in argv else " /no_think"
    reasoning_none = "--reasoning-none" in argv
    k = int(argv[argv.index("--k") + 1]) if "--k" in argv else 12
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp

    harness = ROOT
    os.chdir(harness)

    def prompt(name):
        with open(os.path.join(harness, "prompts", name), encoding="utf-8") as f:
            return f.read().strip()

    tag = model.replace("/", "_")
    cache = argv[argv.index("--cache") + 1] if "--cache" in argv else os.path.join(harness, f"cache/llm_cache_{tag}.jsonl")
    client = LocalClient(LLMEndpoint(url=url, model=model, timeout_s=60),
                         system_prompt=prompt("fallback_strict.txt"), gate_prompt=prompt("addressee_gate.txt"),
                         cache_path=cache, think_switch=think,
                         reasoning_none=reasoning_none)
    failures = 0
    for device in ("mac", "hub"):
        router = build(ch, harness, device, client, k)
        for split in ("dev", "test"):
            gold, _, _ = cp.setup(split)
            _, _, _, ctx = cp.load_inputs(split)
            out = os.path.join(harness, "preds", f"llm-{tag}{'' if k == 12 else f'__k{k}'}__{device}__active__{split.upper()}.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for rid, r in gold.items():
                    c = ctx.get(rid, cp.COLD)
                    try:
                        d = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                    except RuntimeError:  # LLM unavailable or unusable answer: the plugin declines
                        failures += 1
                        d = {"intent": None, "stage": "llm_unavailable"}
                    f.write(json.dumps({"id": rid, **{k: d[k] for k in KEYS if k in d}}) + "\n")
            print(f"wrote {out}", flush=True)
    fresh = {kind: [ms for k, ms, cached in client.calls if k == kind and not cached] for kind in ("route", "gate")}
    for kind, ms in fresh.items():
        if ms:
            q = statistics.quantiles(ms, n=20) if len(ms) > 20 else [max(ms)] * 19
            print(f"{model} {kind}: n={len(ms)} p50={statistics.median(ms):.0f} ms p95={q[18]:.0f} ms")
    salvaged = sum(1 for e in client.cache.values() if e.get("salvaged"))
    print(f"{model}: unusable LLM answers (declined): {failures}; tool calls salvaged from text: {salvaged}")


if __name__ == "__main__":
    main(sys.argv[1:])
