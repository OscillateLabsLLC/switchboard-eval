#!/usr/bin/env python3
"""Measurement: all 1,007 OVOS golden utterances (en-US), live, with every skill the set uses installed.

exp_ovos_gold scores the 683 rows whose skill the desktop has. For this run the 18 missing skills were installed on
the desktop at the commits OpenVoiceOS/ovos-intents v6.1 pins (ovos-m2v-pipeline train/sources.yaml at e0b2f4e);
the application launcher was already installed. alerts, volume and naptime were first upgraded to the same pins
(and kept); date-time stayed at its installed version, because the pinned commit needs NumPy 2 and the desktop's
OVOS runs on 1.26. Other skills the desktop already had stayed at their installed versions.

Every system here is a live run through the intent query path (run_ovos.py), read from
data/baselines/ovos_gold_full/: OVOS's default pipeline and each matcher alone (--stock / --without), and the
desktop's Switchboard pipeline (1.1.1, with and without its LLM step). Scored by exact skill intent. Rows are
tagged template / unseen against the live install's templates (manifest.json in the same folder, dumped from the
desktop with dump_introspection.py during the run).

Headline: the rows whose intent the live install registers, which any system could answer. The rest are counted
by reason and scored nowhere: mark1-ctrl refuses to load on anything but a Mark 1, the application launcher
registers nothing on macOS, and the desktop's date-time predates intent renames the dataset's labels use.
Usage (from the repo root): python -m switchboard_eval.experiments.exp_ovos_gold_full
"""
import collections
import json
import os

from switchboard_eval.experiments.exp_ovos_gold import report
from switchboard_eval.paths import ROOT

GOLD = os.path.join(ROOT, "data", "ovos_gold", "en-US.jsonl")
RUNS = os.path.join(ROOT, "data", "baselines", "ovos_gold_full")
SYSTEMS = ("stock", "stock_padatious_only", "stock_padatious_adapt", "stock_m2v_only", "stock_adapt_only",
           "switchboard_nollm", "switchboard_llm")


def tag_rows(rows):
    """template if one of the row's own intent's live templates matches it, else unseen."""
    from ovos_switchboard.core.device import Device
    from ovos_switchboard.core.templates import template_regex
    with open(os.path.join(RUNS, "manifest.json"), encoding="utf-8") as f:
        device = Device.from_definitions(json.load(f)["definitions"])
    templates = {}
    for label, line in device.template_entries():
        templates.setdefault(label, []).append(template_regex(line))
    for r in rows:
        r["tag"] = "template" if any(rx.match(r["utterance"]) for rx in templates.get(r["label"], [])) else "unseen"


def registered_labels():
    with open(os.path.join(RUNS, "manifest.json"), encoding="utf-8") as f:
        defs = json.load(f)["definitions"]
    return {f"{s}:{d['intent_name']}" for s, ds in defs.items() for d in ds}, set(defs)


def main():
    with open(GOLD, encoding="utf-8") as f:
        every = [json.loads(line) for line in f]
    labels, skills = registered_labels()
    missing = collections.Counter("skill not loaded" if r["label"].split(":")[0] not in skills
                                  else "intent not registered" for r in every if r["label"] not in labels)
    rows = [r for r in every if r["label"] in labels]
    tag_rows(rows)
    print(f"{len(every)} gold rows; {len(rows)} have their intent registered on the desktop "
          f"({sum(r['tag'] == 'template' for r in rows)} match one of their intent's templates, "
          f"{sum(r['tag'] == 'unseen' for r in rows)} don't); not answerable: {dict(missing)}")
    for name in SYSTEMS:
        path = os.path.join(RUNS, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            preds = {p["id"]: {"label": p.get("label"), "stage": p.get("stage")} for p in map(json.loads, f)}
        report(name, preds, rows)


if __name__ == "__main__":
    main()
