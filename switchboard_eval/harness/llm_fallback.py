#!/usr/bin/env python3
"""Phase 2: send rows the stack declined to an LLM with manifest-derived tools.

Reads a bench.py predictions file. Exact-layer rows are never touched (design goal 1).
Classifier rows that were declined (intent null) — or, with --below X, accepted with
confidence < X — go to the LLM with the shortlisted tools (shortlist.py) and
tool_choice=required, so it must pick a tool, `ignore` included. The tool call maps back
to a canonical intent for scoring: tool -> OVOS label (devices/desktop/tools.json) -> schema/intents.yaml.
Unmapped labels stay `__unmapped__:label` (they would fire a skill, so they count as acting).

Only --split rows are routed (default dev), so test stays unconsulted until a milestone.
Responses are cached in --cache keyed by (model, prompt, utterance, tools): reruns are
free and deterministic.

Usage: llm_fallback.py --preds preds/X.jsonl --out preds/X__llm.jsonl
                       [--below 0.7] [--hint] [--k 12] [--split dev|test|all]
"""
from switchboard_eval.paths import ROOT
import argparse, hashlib, json, math, os, sys, threading, time, urllib.request

sys.path.insert(0, ROOT)
from switchboard_eval.harness.run_ovos import build_map
from switchboard_eval.harness.shortlist import Shortlist

# The OpenAI-compatible LLM server; the references used Qwen3-14B-AWQ on a 24 GB GPU.
URL = os.environ.get("INTENT_EVAL_LLM_URL", "http://localhost:8080/v1/chat/completions")
MODEL = "Qwen/Qwen3-14B-AWQ"
SYSTEM_PROMPT = (
    "You route one transcribed utterance from a home voice assistant's microphone to exactly one "
    "tool. Transcripts may contain speech-recognition errors; route by the most plausible meaning. "
    "The microphone wakes on a wake word, but false wakes are common: if the speech is addressed to "
    "another person, is a remark rather than a request, or sounds like TV, radio, or the assistant's "
    "own voice, call ignore. Call unsupported only for a request clearly made to the assistant that "
    "no tool can perform.")
GATE_TOP_LOGPROBS = 10
# clarify SPEAKS, so it is not None: it counts as a false accept on negatives and as wrong (never
# correct) on requests. Report "asked" separately; bar.py's numbers stay conservative.
CLARIFY = "__clarify__"
META_CANON = {"ask_knowledge": "knowledge.ask", "stop": "stop", "unsupported": None, "ignore": None,
              "clarify": CLARIFY}


class LLMRouter:
    def __init__(self, shortlist, cache_path, system_prompt=SYSTEM_PROMPT, think=False, url=URL, model=MODEL):
        self.shortlist, self.url, self.model, self.cache_path = shortlist, url, model, cache_path
        self.system_prompt, self.think = system_prompt, think
        self.write_lock = threading.Lock()  # llm_determinism.py calls route() from worker threads
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                e = json.loads(line)
                self.cache[e["key"]] = e

    @staticmethod
    def _user_message(utterance, note):
        return f"{note}\nUtterance: {utterance}" if note else utterance

    def route(self, utterance, k, extra_names=(), note=None):
        tools = self.shortlist.tools(utterance, k)
        names = {t["function"]["name"] for t in tools}
        tools += [self.shortlist.by_name[n] for n in extra_names if n not in names]
        body = {"model": self.model, "temperature": 0, "max_tokens": 2000 if self.think else 200,
                "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": self.think},
                "tools": tools,
                "messages": [{"role": "system", "content": self.system_prompt},
                             {"role": "user", "content": self._user_message(utterance, note)}]}
        return self._cached(body, _parse_tool_call, {"n_tools": len(tools)})

    def p_addressed(self, utterance, gate_prompt, note=None):
        """P(yes) to 'is this directed at the assistant?', from first-token logprobs."""
        body = {"model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True,
                "top_logprobs": GATE_TOP_LOGPROBS, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": gate_prompt},
                             {"role": "user", "content": self._user_message(utterance, note)}]}
        return self._cached(body, _parse_yes_no)

    def _cached(self, body, parse, extra=None):
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return {**self.cache[key], "cached": True}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=120))
        entry = {"key": key, **parse(r), "ms": round((time.perf_counter() - t0) * 1000, 1),
                 "prompt_tokens": r["usage"]["prompt_tokens"], **(extra or {})}
        with self.write_lock, open(self.cache_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            self.cache[key] = entry
        return {**entry, "cached": False}


def _parse_tool_call(r):
    calls = r["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        raise RuntimeError(f"tool_choice=required returned no tool call: {r['choices'][0]['message']}")
    fn = calls[0]["function"]
    return {"tool": fn["name"], "args": json.loads(fn["arguments"] or "{}")}


def _parse_yes_no(r):
    """Renormalize over yes/no variants in the top logprobs; fail loud if neither appears."""
    mass = {"yes": 0.0, "no": 0.0}
    for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        word = t["token"].strip(" _").lower()
        if word in mass:
            mass[word] += math.exp(t["logprob"])
    if not sum(mass.values()):
        raise RuntimeError(f"gate answer has no yes/no token: {r['choices'][0]}")
    return {"p_yes": mass["yes"] / (mass["yes"] + mass["no"])}


def to_intent(tool, args, tool_labels, label_map):
    """(canonical intent or None, slots) for a tool call."""
    if tool in META_CANON:
        return META_CANON[tool], {k: v for k, v in args.items() if isinstance(v, str)}
    label = tool_labels[tool]
    canon, ren = label_map.get(label, (f"__unmapped__:{label}", {}))
    return canon, {ren.get(k, k): v for k, v in args.items() if isinstance(v, str)}


def hint_tools(pred, canon2tools):
    """Tools behind the classifier's top-3 canonical guesses, added to the shortlist."""
    return [n for c in pred.get("top3") or [] for n in canon2tools.get(c, [])]


def should_route(pred, below):
    if pred.get("stage") != "classifier":
        return False
    return pred["intent"] is None or (below is not None and (pred.get("confidence") or 0) < below)


def run(a):
    tools_doc = json.load(open(a.tools))
    label_map = build_map(a.schema)
    canon2tools = {}
    for name, label in tools_doc["labels"].items():
        canon = label_map.get(label, (None, {}))[0]
        if canon:
            canon2tools.setdefault(canon, []).append(name)
    system_prompt = open(a.system_file).read().strip() if a.system_file else SYSTEM_PROMPT
    router = LLMRouter(Shortlist(a.tools, a.examples), a.cache, system_prompt, a.think)
    gate_prompt = open(a.gate_file).read().strip() if a.gate is not None else None
    rows = {r["id"]: r for r in map(json.loads, open(a.dataset))}
    out, log = [], []
    for p in map(json.loads, open(a.preds)):
        r = rows[p["id"]]
        if (a.split != "all" and r.get("split") != a.split) or not should_route(p, a.below):
            out.append(p)
            continue
        res = router.route(r["utterance"], a.k, hint_tools(p, canon2tools) if a.hint else ())
        intent, slots = to_intent(res["tool"], res["args"], tools_doc["labels"], label_map)
        extra, stage, ms = {}, "llm", res["ms"]
        if gate_prompt:
            g = router.p_addressed(r["utterance"], gate_prompt)
            extra, ms = {"p_addressed": round(g["p_yes"], 4)}, ms + g["ms"]
            if g["p_yes"] < a.gate:
                intent, slots, stage = None, {}, "llm_gate"
        out.append({**p, "intent": intent, "slots": slots, "stage": stage, "pre_llm_intent": p["intent"],
                    "llm_tool": res["tool"], "latency_ms": round(p["latency_ms"] + ms, 1), **extra})
        log.append((r, p, res, intent))
    with open(a.out, "w") as f:
        for p in out:
            f.write(json.dumps(p) + "\n")
    return log


def summarize(log, out_path):
    fresh = sorted(res["ms"] for _, _, res, _ in log if not res["cached"])
    pick = lambda q: fresh[min(len(fresh) - 1, int(q * len(fresh)))] if fresh else None
    tools = {}
    for _, _, res, _ in log:
        kind = res["tool"] if res["tool"] in META_CANON else "skill tool"
        tools[kind] = tools.get(kind, 0) + 1
    print(f"routed {len(log)} rows to the LLM ({len(fresh)} fresh calls, rest cached) -> {out_path}")
    if fresh:
        print(f"fresh-call latency p50 {pick(0.5):.0f} ms, p95 {pick(0.95):.0f} ms")
    print("tool kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(tools.items(), key=lambda x: -x[1])))
    fixed = sum(i == r["intent"] and p["intent"] != r["intent"] for r, p, _, i in log)
    broke = sum(i != r["intent"] and p["intent"] == r["intent"] for r, p, _, i in log)
    print(f"vs pre-LLM prediction: {fixed} rows fixed, {broke} rows broken")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--schema", default="schema/intents.yaml")
    ap.add_argument("--tools", default="devices/desktop/tools.json")
    ap.add_argument("--examples", default="devices/desktop/examples.jsonl")
    ap.add_argument("--cache", default="cache/llm_cache.jsonl")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--below", type=float, default=None,
                    help="also route accepted classifier rows with confidence below this")
    ap.add_argument("--hint", action="store_true", help="add the classifier's top-3 intents' tools")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--system-file", default=None, help="system prompt variant (default: SYSTEM_PROMPT)")
    ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking before the tool call")
    ap.add_argument("--gate", type=float, default=None,
                    help="decline routed rows whose P(addressed to the assistant) is below this")
    ap.add_argument("--gate-file", default="prompts/addressee_gate.txt")
    a = ap.parse_args()
    summarize(run(a), a.out)


if __name__ == "__main__":
    main()
