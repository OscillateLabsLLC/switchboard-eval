#!/usr/bin/env python3
"""Measurement: does the right canonical intent reach the right handler?

Gold is canonical, so a dispatch to the wrong handler inside the right canonical (Neon weather's HandleHowWindy vs
HandleHumidity, both weather.current) scores as correct. Seen rows carry `seen_in_templates`, the skill labels whose
templates match the row: a partial label-level gold. For every row the v5 router gets canonically RIGHT and that has
an acceptable label under the gold canonical, the dispatched label is checked against it:
  right    the label is one of the acceptable labels
  sibling  the label is a different intent of the same skill as an acceptable label (the grouping failure)
  other    another skill (usually a duplicate install or a renamed skill id: not a handler error)
No behavior change and no model calls (cached LLM answers only).

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_handler_choice <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import collections
import os
import sys

import yaml


def labels_by_canon(harness):
    """{canonical: set of skill labels} from schema/intents.yaml."""
    with open(os.path.join(harness, "schema/intents.yaml"), encoding="utf-8") as f:
        intents = yaml.safe_load(f)["intents"]
    return {c: {e["label"] if isinstance(e, dict) else e for e in v.get("ovos") or []} for c, v in intents.items()}


def verdict(label, acceptable):
    """right / sibling / other for one dispatched label."""
    if label in acceptable:
        return "right"
    skill = label.rpartition(":")[0]
    return "sibling" if any(a.rpartition(":")[0] == skill for a in acceptable) else "other"


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    sys.path.insert(0, ROOT)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp

    harness = ROOT
    os.chdir(harness)
    canon_labels = labels_by_canon(harness)
    for device in ("mac", "hub"):
        router = ch.build_router(harness, device, "active")
        counts, by_stage, others, siblings = collections.Counter(), collections.Counter(), collections.Counter(), []
        for split in ("dev", "test"):
            gold, _, _ = cp.setup(split)
            _, _, _, ctx = cp.load_inputs(split)
            for rid, r in gold.items():
                golds = set(r["intent"]) if isinstance(r["intent"], list) else {r["intent"]}
                c = ctx.get(rid, cp.COLD)
                d = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                if d.get("intent") not in golds or not d.get("label"):
                    continue
                acceptable = set(r.get("seen_in_templates") or []) & canon_labels.get(d["intent"], set())
                if not acceptable:
                    continue
                v = verdict(d["label"], acceptable)
                counts[v] += 1
                by_stage[(d["stage"], v)] += 1
                if v == "other":
                    others[(d["label"], tuple(sorted(acceptable)))] += 1
                if v == "sibling":
                    siblings.append((split, rid, r["utterance"], r["intent"], d["stage"], d.get("label_how"),
                                     d["label"], sorted(acceptable)))
        total = sum(counts.values())
        print(f"{device}: {total} canonically right rows with label gold: " +
              ", ".join(f"{k} {counts[k]}" for k in ("right", "sibling", "other")))
        print("  by stage: " + ", ".join(f"{s}/{v} {n}" for (s, v), n in sorted(by_stage.items())))
        for split, rid, utt, canon, stage, how, label, acc in siblings:
            print(f"  SIBLING {split} {rid} {utt!r} [{canon}, {stage}, {how}] -> {label}; expected one of {acc}")
        for (label, acc), n in others.most_common():
            print(f"  OTHER x{n} {label}; expected one of {list(acc)}")


if __name__ == "__main__":
    main(sys.argv[1])
