#!/usr/bin/env python3
"""Audit of the exact layer against what skills declare and what nobody meant as a command. No review needed.

Design goal 1: a phrase the exact layer dispatches is always right. Two deterministic checks per device:

  templates  every expansion of every registered template, slots filled with a neutral value, through the exact
             layer. Expected: it reaches its own intent. A collision (2+ intents match, so the classifier decides) is
             by design but listed, since each one is a routing decision nobody made. "unclaimed" means the template
             is too wildcard-heavy to claim the exact layer (min_literal), also by design.
  negatives  speech that is not a command: the assistant's own dialog files and spoken lines (speak.log), meeting
             speech (data/negatives_ext.jsonl, AMI), and dataset rows whose gold is "no intent". ANY exact dispatch is a
             finding: an open slot or a too-short template claiming speech it doesn't own.

Writes preds/exact_audit_<device>.jsonl (one row per finding) and prints a summary.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exact_audit <switchboard>/scripts [--show N] [--min-literal N]
"""
from switchboard_eval.paths import ROOT
import collections
import json
import os
import re
import sys

from ovos_switchboard.core.device import Device, read_jsonl
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.schema import Schema
from ovos_switchboard.core.templates import SLOT_TOKEN, expansions, template_regex


SLOT_FILLER = "the blue box"
SPEAK = re.compile(r"Speak: (.+?)\s*$")
DIALOG = {"mac": "devices/desktop/dialog_negatives.jsonl", "hub": "devices/speaker/dialog_negatives.jsonl"}


def exact_layer(ch, harness, device, min_literal=1):
    """The device's ExactLayer, built as check_harness builds the router's."""
    spec = ch.DEVICES[device]

    def path(name):
        return os.path.join(harness, name)

    dev = Device.from_files(path(spec["manifest"]), path(spec["examples"]),
                            shared_examples=[path(p) for p in spec.get("shared_examples", [])],
                            shared_manifests=[path(p) for p in spec.get("shared_manifests", [])])
    return ExactLayer(dev, Schema.load(path("schema/intents.yaml")), min_literal), dev


def matching_labels(layer, utterance):
    """Every label whose claimed templates match: the detail behind ExactLayer.route."""
    return sorted({label for label, rx in layer.templates if rx.match(utterance)})


def audit_templates(layer, dev):
    """Findings for template expansions that don't reach their own intent."""
    claimed = {rx.pattern for _, rx in layer.templates}
    findings, counts, seen = [], collections.Counter(), set()
    for label, line in dev.template_entries():
        own = layer.schema.canon_or_unmapped(label)
        line_claimed = template_regex(line).pattern in claimed
        for e in expansions(line):
            utt = SLOT_TOKEN.sub(SLOT_FILLER, e).lower()
            if (label, utt) in seen:
                continue
            seen.add((label, utt))
            routed = layer.route(utt)
            if routed == own:
                counts["own"] += 1
                continue
            if not line_claimed:
                counts["unclaimed"] += 1
                continue
            labels = matching_labels(layer, utt)
            kind = "collision" if routed is None and len(labels) > 1 else "other"
            counts[kind] += 1
            owners = sorted({layer.schema.canon_or_unmapped(lb) for lb in labels})
            findings.append({"check": "templates", "kind": kind, "utterance": utt, "label": label, "template": line,
                             "routed": routed, "owners": owners, "labels": labels})
    return findings, counts


def negative_corpora(harness, device):
    """{corpus: [utterance]} of speech that is not a command to this device."""
    corpora = {"dialog": [r["example"] for r in read_jsonl(os.path.join(harness, DIALOG[device]))],
               "ext_ami": [r["utterance"] for r in read_jsonl(os.path.join(harness, "data/negatives_ext.jsonl"))]}
    with open(os.path.join(harness, "data", device, "speak.log"), encoding="utf-8", errors="replace") as f:
        corpora["speak_log"] = [m.group(1) for m in map(SPEAK.search, f) if m]
    rows = read_jsonl(os.path.join(harness, "data/dataset.jsonl"))
    corpora["gold_none"] = [r["utterance"] for r in rows if r.get("intent") is None]
    return corpora


def audit_negatives(layer, corpora):
    """Findings for non-command speech the exact layer dispatches."""
    findings, counts = [], collections.Counter()
    for corpus, utterances in corpora.items():
        for utt in dict.fromkeys(u.lower().strip() for u in utterances if u.strip()):
            counts[(corpus, "total")] += 1
            routed = layer.route(utt)
            if routed is None:
                continue
            counts[(corpus, "dispatched")] += 1
            labels = matching_labels(layer, utt)
            templates = [line for label, line in layer._entries  # pylint: disable=protected-access
                         if label in labels and template_regex(line).match(utt)]
            findings.append({"check": "negatives", "kind": corpus, "utterance": utt, "routed": routed,
                             "labels": labels, "templates": templates[:3]})
    return findings, counts


def report(device, t_findings, t_counts, n_findings, n_counts, show):
    """Print the summary for one device."""
    total = sum(t_counts.values())
    print(f"\n== {device}: templates: {total} expansions; " + ", ".join(f"{k} {v}" for k, v in t_counts.items()))
    pairs = collections.Counter(tuple(f["owners"]) for f in t_findings if f["kind"] == "collision")
    print(f"  collisions: {len(pairs)} distinct intent sets")
    for owners, n in pairs.most_common(show):
        print(f"    x{n:<4} {' | '.join(owners)}")
    for f in [f for f in t_findings if f["kind"] == "other"][:show]:
        print(f"    OTHER {f['utterance']!r} ({f['label']}) -> {f['routed']}")
    corpora = sorted({c for c, _ in n_counts})
    print(f"== {device}: negatives: " + ", ".join(
        f"{c} {n_counts[(c, 'dispatched')]}/{n_counts[(c, 'total')]}" for c in corpora))
    by_label = collections.Counter(f["routed"] for f in n_findings)
    for routed, n in by_label.most_common(show):
        examples = [f["utterance"] for f in n_findings if f["routed"] == routed][:3]
        print(f"    x{n:<4} {routed}: {examples}")


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch

    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 15
    min_literal = int(sys.argv[sys.argv.index("--min-literal") + 1]) if "--min-literal" in sys.argv else 1
    harness = ROOT
    for device in ("mac", "hub"):
        layer, dev = exact_layer(ch, harness, device, min_literal)
        t_findings, t_counts = audit_templates(layer, dev)
        n_findings, n_counts = audit_negatives(layer, negative_corpora(harness, device))
        report(device, t_findings, t_counts, n_findings, n_counts, show)
        suffix = "" if min_literal == 1 else f"_ml{min_literal}"
        out = os.path.join(harness, "preds", f"exact_audit_{device}{suffix}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for row in t_findings + n_findings:
                f.write(json.dumps(row) + "\n")
        print(f"  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
