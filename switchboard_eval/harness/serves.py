#!/usr/bin/env python3
"""Which wrong-looking predictions are right on a given device: schema/intents.yaml `serves` turned into (predicted, gold)
pairs per device. A pair counts when EVERY label the device has for the predicted intent also serves the gold
intent, so a prediction proves the serving handler ran without knowing the dispatched label. That also makes the rule
apply to stock OVOS, whose predictions carry no label (stock ran on the Mac's skills).

    python -m switchboard_eval.harness.serves   # prints the pairs per device
"""
from switchboard_eval.paths import ROOT
import json
import os

import yaml

HERE = ROOT
MANIFESTS = {"mac": "devices/desktop/manifest.json", "hub": "devices/speaker/manifest.json"}
STOCK_DEVICE = "mac"


def device_labels(device):
    with open(os.path.join(HERE, MANIFESTS[device]), encoding="utf-8") as f:
        defs = json.load(f)["definitions"]
    return {f"{skill}:{d['intent_name']}" for skill, ds in defs.items() for d in (ds if isinstance(ds, list) else [])}


def equivalences(device):
    """{(predicted canonical, gold canonical)} that count as right on `device`."""
    with open(os.path.join(HERE, "schema/intents.yaml"), encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    serves, have = doc.get("serves") or {}, device_labels(device)
    pairs = set()
    for canon, spec in doc["intents"].items():
        labels = {e if isinstance(e, str) else e["label"] for e in spec.get("ovos") or []} & have
        if not labels:
            continue
        for gold in set.intersection(*(set(serves.get(label, [])) for label in labels)):
            pairs.add((canon, gold))
    return frozenset(pairs)


if __name__ == "__main__":
    for device in MANIFESTS:
        print(device, sorted(equivalences(device)))
