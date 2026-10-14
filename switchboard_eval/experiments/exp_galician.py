#!/usr/bin/env python3
"""Galician end to end: Switchboard built for a gl-es device, scored on real-utterance test rows.

Test rows: data/gl/dataset_gl.jsonl, drafted by Claude (no native speaker was available, so it used its own
judgement, 2026-10-14): NOT native-speaker reviewed; rows flagged "uncertain" are reported separately. Rows are
natural paraphrases, not template text, so they test generalisation rather than the exact layer.

Device: skills/manifests/gl-es.json (the installed skills' gl-ES templates), LLM tools derived from the same definitions
(build_tools(..., "gl-es")), decline class trained on the installed skills' gl-ES .dialog lines, no synthetic rows
(they're English). Cold context for every row (nobody spoke recently). Prompts are the English ones.
LLM: none, or a local model through exp_small_llm.LocalClient (LM Studio).

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_galician [--model qwen3-1.7b] [--reasoning-none] [--think-switch STR]
"""
from switchboard_eval.paths import ROOT
import glob
import json
import os
import sys

from switchboard_eval.experiments.exp_small_llm import LM_STUDIO, LocalClient
from ovos_switchboard.adapters.openai_llm import LLMEndpoint
from ovos_switchboard.core.device import Device, harvest_dialog_negatives
from ovos_switchboard.core.router import Router
from ovos_switchboard.core.schema import Schema
from ovos_switchboard.core.tools import build_tools


LANG = "gl-es"
SITE = os.path.expanduser("~/.venvs/ovos/lib/python3.11/site-packages")


def dialog_lines(lang):
    """Every installed skill's .dialog lines for `lang`."""
    lines = []
    for path in glob.glob(f"{SITE}/*/locale/*/**/*.dialog", recursive=True):
        if path.split("/locale/")[1].split("/")[0].lower() == lang:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return lines


def score(rows, preds):
    """right / wrong / pass on requests, false accepts on non-commands."""
    req = [r for r in rows if r["intent"]]
    neg = [r for r in rows if not r["intent"]]
    right = sum(preds[r["id"]].get("intent") == r["intent"] for r in req)
    passed = sum(not preds[r["id"]].get("intent") for r in req)
    fa = sum(bool(preds[r["id"]].get("intent")) for r in neg)
    return {"requests": len(req), "right_pct": round(100 * right / len(req), 1),
            "wrong_pct": round(100 * (len(req) - right - passed) / len(req), 1),
            "pass_pct": round(100 * passed / len(req), 1), "negatives": len(neg),
            "false_accept_pct": round(100 * fa / len(neg), 1)}


def main(argv):
    harness = ROOT
    model = argv[argv.index("--model") + 1] if "--model" in argv else None
    with open(os.path.join(harness, "skills", "manifests", f"{LANG}.json"), encoding="utf-8") as f:
        defs = json.load(f)["definitions"]
    with open(os.path.join(harness, "data", "gl", "dataset_gl.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    device = Device.from_definitions(defs, LANG)
    negatives = harvest_dialog_negatives(dialog_lines(LANG), defs, LANG)
    client = None
    if model:
        def prompt(name):
            with open(os.path.join(harness, "prompts", name), encoding="utf-8") as fp:
                return fp.read().strip()
        client = LocalClient(LLMEndpoint(url=LM_STUDIO, model=model, timeout_s=60),
                             system_prompt=prompt("fallback_strict.txt"), gate_prompt=prompt("addressee_gate.txt"),
                             cache_path=os.path.join(harness, f"cache/llm_cache_{model.replace('/', '_')}.jsonl"),
                             think_switch=argv[argv.index("--think-switch") + 1] if "--think-switch" in argv
                             else " /no_think", reasoning_none="--reasoning-none" in argv)
    router = Router(device, schema, build_tools(defs, LANG), synthetic=negatives, llm=client)
    preds = {}
    for r in rows:
        try:
            preds[r["id"]] = router.route(r["utterance"], {"gap_s": None, "echo_run": 0})
        except RuntimeError:
            preds[r["id"]] = {"intent": None, "stage": "llm_unavailable"}
    sure = [r for r in rows if not r.get("uncertain")]
    result = {"lang": LANG, "llm": model or "none", "dialog_negatives": len(negatives),
              "all": score(rows, preds), "excluding_uncertain": score(sure, preds)}
    print(json.dumps(result))
    for r in rows:
        p = preds[r["id"]]
        mark = "ok " if p.get("intent") == r["intent"] else "XX "
        print(f"  {mark}{r['id']} {r['utterance']!r:55} gold {r['intent']} -> {p.get('intent')} ({p.get('stage')})")
    with open(os.path.join(harness, "preds", "exp_galician.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({**result, "preds": {k: v.get("intent") for k, v in preds.items()}}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
