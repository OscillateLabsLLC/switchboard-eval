#!/usr/bin/env python3
"""Measurement: OVOS's end-to-end golden utterances, scored at the skill-intent level.

data/ovos_gold/en-US.jsonl is the en-US test split of OpenVoiceOS/ovos-intents (v6.1): utterances from the skills'
own end-to-end tests, each labeled `<skill_id>:<intent_name>`. A prediction is right only if it names exactly that
skill intent (stricter than the canonical intents the rest of the harness scores).

Rows are those whose label the device has installed, and each is tagged:
  template  one of the gold intent's own templates matches the utterance (what the exact step exists for)
  unseen    none does: the utterance has to be generalized to

Switchboard: the harness router for the device, every row cold (no "assistant just spoke"), the LLM step off and
on (cache first; misses asked of INTENT_EVAL_LLM_URL and cached). Stock OVOS variants: live runs of run_ovos.py
--stock / --without on the same file, read from data/baselines/ovos_gold/ when present.

Writes preds/ovos_gold__<device>__{nollm,llm}.jsonl and prints right / wrong / no answer per system and tag.
Usage (from the repo root): PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_ovos_gold \\
    <switchboard>/scripts [device]
"""
import json
import os
import sys

from switchboard_eval.experiments.exp_float32 import router_for
from switchboard_eval.paths import DEVICE_DIRS, ROOT

GOLD = os.path.join(ROOT, "data", "ovos_gold", "en-US.jsonl")
STOCK_DIR = os.path.join(ROOT, "data", "baselines", "ovos_gold")


def device_templates(device):
    """{label: [compiled template regex]} for the device's installed intents."""
    from ovos_switchboard.core.device import Device
    from ovos_switchboard.core.templates import template_regex
    dev = Device.from_files(os.path.join(ROOT, DEVICE_DIRS[device], "manifest.json"),
                            os.path.join(ROOT, DEVICE_DIRS[device], "examples.jsonl"))
    out = {}
    for label, line in dev.template_entries():
        out.setdefault(label, []).append(template_regex(line))
    with open(os.path.join(ROOT, DEVICE_DIRS[device], "manifest.json"), encoding="utf-8") as f:
        labels = {f"{s}:{d['intent_name']}" for s, defs in json.load(f)["definitions"].items() for d in defs}
    return out, labels


def gold_rows(device):
    templates, labels = device_templates(device)
    with open(GOLD, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["label"] in labels]
    for r in rows:
        r["tag"] = "template" if any(rx.match(r["utterance"]) for rx in templates.get(r["label"], [])) else "unseen"
    return rows


def route_all(router, rows, out_path):
    cold = {"gap_s": None, "echo_run": 0}
    preds = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            try:
                d = router.route(r["utterance"], cold)
            except RuntimeError:  # the LLM server answered without a usable tool call: live, the step declines
                d = {"intent": None, "stage": "llm_unusable"}
            preds[r["id"]] = {"label": d.get("label") if d.get("intent") else None, "stage": d.get("stage")}
            f.write(json.dumps({"id": r["id"], **preds[r["id"]]}) + "\n")
    return preds


def stock_preds(name):
    path = os.path.join(STOCK_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {p["id"]: {"label": p.get("label"), "stage": p.get("stage")} for p in map(json.loads, f)}


def report(name, preds, rows):
    parts = []
    for tag in ("all", "template", "unseen"):
        sel = [r for r in rows if tag == "all" or r["tag"] == tag]
        right = sum(preds[r["id"]]["label"] == r["label"] for r in sel)
        none = sum(preds[r["id"]]["label"] is None for r in sel)
        parts.append(f"{tag} {100 * right / len(sel):5.1f}% right, {100 * (len(sel) - right - none) / len(sel):4.1f}% "
                     f"wrong, {100 * none / len(sel):4.1f}% none (n={len(sel)})")
    print(f"{name:26} " + " | ".join(parts), flush=True)


def main(scripts_dir, device="mac"):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch  # pylint: disable=import-error  # Switchboard's scripts
    os.chdir(ROOT)
    rows = gold_rows(device)
    print(f"{len(rows)} gold rows on the {device} device: "
          f"{sum(r['tag'] == 'template' for r in rows)} match one of their intent's templates, "
          f"{sum(r['tag'] == 'unseen' for r in rows)} don't", flush=True)
    for name in ("stock", "stock_padatious_only", "stock_padatious_adapt", "stock_m2v_only", "stock_adapt_only",
                 "switchboard_live_desktop"):
        preds = stock_preds(name)
        if preds:
            report(name, preds, rows)
    router = router_for(ch, ROOT, device, "active")
    llm = router.llm
    for variant, step in (("nollm", None), ("llm", llm)):
        router.llm = step
        preds = route_all(router, rows, os.path.join(ROOT, "preds", f"ovos_gold__{device}__{variant}.jsonl"))
        report(f"switchboard_{variant}", preds, rows)
        unusable = sum(p["stage"] == "llm_unusable" for p in preds.values())
        if unusable:
            print(f"  {unusable} rows where the LLM server returned no usable tool call (declined, as live)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
