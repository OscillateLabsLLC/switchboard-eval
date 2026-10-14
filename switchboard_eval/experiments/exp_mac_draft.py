#!/usr/bin/env python3
"""Experiment: the Mac mapping draft (mac_mapping_proposals.yaml) on real Mac traffic and on the bar.

Single variable: the schema. Baseline is schema/intents.yaml; the variant merges the draft in (labels appended to existing
canonicals, NEW canonicals created with the risk and description from their comment). Two readouts:
  real   every archived Mac transcript (data/mac/context.jsonl) through the Mac router, both schemas; each changed
         decision is listed for a by-hand verdict (these rows have no gold)
  bar    the go/no-go bar on dev and test, Mac and Hub routers, active route
The LLM step is off for the real readout (no model calls): "classifier pass" means the utterance would go to the LLM,
so a mapping's expected win shows up as pass -> dispatch. The bar uses cached LLM answers; a row the draft sends to
the LLM that the cache lacks is asked of the LLM server and cached, as llm_fallback.py does.

Usage (from intent-eval, in .venv-bench, with the switchboard checkout):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_mac_draft <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import copy
import json
import os
import re
import sys

import yaml


DRAFT = "mac_mapping_proposals.yaml"
LLM_URL = os.environ.get("INTENT_EVAL_LLM_URL", "http://localhost:8080/v1/chat/completions")
NEW_CANON = re.compile(r'^(\S+):\s*#\s*NEW,\s*(low|medium|high)\??(?:\s*\([^)]*\))?:\s*"([^"]+)"')


def merged_schema_doc(harness):
    """schema/intents.yaml with the draft's mappings merged in."""
    with open(os.path.join(harness, "schema/intents.yaml"), encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    with open(os.path.join(harness, DRAFT), encoding="utf-8") as f:
        text = f.read()
    new = {m.group(1): (m.group(2), m.group(3)) for m in map(NEW_CANON.match, text.splitlines()) if m}
    merged = copy.deepcopy(doc)
    for canon, labels in yaml.safe_load(text).items():
        if canon not in merged["intents"]:
            if canon not in new:
                raise ValueError(f"{canon}: not in schema/intents.yaml and no NEW risk/description comment")
            risk, description = new[canon]
            merged["intents"][canon] = {"risk": risk, "slots": [], "description": description, "ovos": []}
        merged["intents"][canon].setdefault("ovos", []).extend(labels)
    return doc, merged


def build(ch, harness, device, schema, llm=True, route="active"):
    """check_harness.build_router with the schema swapped and the LLM optionally off. The catch-all yield (switchboard
    feat/catch-all-yield) is pinned off where the package has it, so these experiments keep measuring v5's exact layer;
    exp_guard_rules.py applies its own rule on top."""
    from ovos_switchboard.core import options
    from ovos_switchboard.core.device import Device, read_jsonl
    from ovos_switchboard.core.router import Router, RouterConfig

    spec = ch.DEVICES[device]

    def path(name):
        return os.path.join(harness, name)

    dev = Device.from_files(path(spec["manifest"]), path(spec["examples"]),
                            shared_examples=[path(p) for p in spec.get("shared_examples", [])],
                            shared_manifests=[path(p) for p in spec.get("shared_manifests", [])])
    synthetic = [r for p in spec["synthetic"] for r in read_jsonl(path(p))]
    with open(path(spec["tools"]), encoding="utf-8") as f:
        tools_doc = json.load(f)
    client = None
    if llm:
        from ovos_switchboard.adapters.openai_llm import LLMClient, LLMEndpoint

        def prompt(name):
            with open(path(f"prompts/{name}"), encoding="utf-8") as f:
                return f.read().strip()

        # cache misses go to the LLM server and are cached, as llm_fallback.py does
        client = LLMClient(LLMEndpoint(url=LLM_URL, model=ch.HARNESS_MODEL, timeout_s=30),
                           system_prompt=prompt("fallback_strict.txt"), gate_prompt=prompt("addressee_gate.txt"),
                           cache_path=path("cache/llm_cache.jsonl"))
    config = {"classifier": options.ClassifierOptions(route_decline_class=route)}
    if hasattr(options, "ExactOptions"):
        config["exact"] = options.ExactOptions(catch_all_yield=0)
    return Router(dev, schema, tools_doc, synthetic=synthetic, llm=client, config=RouterConfig(**config))


def describe(d):
    """One decision as text."""
    if not d.get("intent"):
        return f"{d.get('stage')} pass" if d.get("stage") != "echo" else "echo decline"
    return f"{d['stage']} -> {d['intent']} ({d.get('label')}, {d.get('confidence', '')})"


def real_readout(ch, harness, schemas):
    """Changed decisions on archived real Mac transcripts."""
    with open(os.path.join(harness, "data", "mac", "context.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    routers = {name: build(ch, harness, "mac", s, llm=False) for name, s in schemas.items()}
    changed = []
    for r in rows:
        ctx = {"gap_s": r["gap_s"] if r["gap_s"] is not None else 1e9, "echo_run": r["echo_run"]}
        b, n = (routers[k].route(r["text"], ctx) for k in ("base", "draft"))
        if (b.get("intent"), b.get("label"), b.get("stage")) != (n.get("intent"), n.get("label"), n.get("stage")):
            changed.append((r["text"], describe(b), describe(n)))
    print(f"real Mac transcripts: {len(rows)}, changed decisions: {len(changed)}")
    for text, b, n in changed:
        print(f"  {text!r}\n      base:  {b}\n      draft: {n}")


def bar_readout(ch, cp, harness, schemas):
    """The go/no-go bar per split and device, both schemas."""
    for split in ("dev", "test"):
        gold, slices, stock = cp.setup(split)
        _, _, _, ctx = cp.load_inputs(split)
        for device in ("mac", "hub"):
            cells, preds = [], {}
            for name, schema in schemas.items():
                router = build(ch, harness, device, schema)
                preds[name] = {}
                for rid, r in gold.items():
                    c = ctx.get(rid, cp.COLD)
                    preds[name][rid] = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
                m = cp.evaluate(gold, slices, stock, preds[name])
                cells.append(f"{name} {m['magic']:5.1f}/{m[device]:5.1f}/{m['fa']:4.1f}/{m['hx']}")
            print(f"bar {split:4} {device}  magic/own/FA/hx  " + "   ".join(cells))
            for rid, r in gold.items():
                b, n = preds["base"][rid], preds["draft"][rid]
                if b.get("intent") != n.get("intent"):
                    print(f"  {rid} {r['utterance']!r} gold {r['intent']} ({r.get('category')})\n"
                          f"      base:  {describe(b)}\n      draft: {describe(n)}")


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    sys.path.insert(0, ROOT)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp
    from ovos_switchboard.core.schema import Schema

    harness = ROOT
    os.chdir(harness)
    base, draft = merged_schema_doc(harness)
    schemas = {"base": Schema(base), "draft": Schema(draft)}
    if "--bar-only" not in sys.argv:
        real_readout(ch, harness, schemas)
    bar_readout(ch, cp, harness, schemas)


if __name__ == "__main__":
    main(sys.argv[1])
