#!/usr/bin/env python3
"""The intent router as one library: route(utterance, context) -> decision.

Composes the stages measured in BENCH.md, in waterfall order. Early stages are tuned for
precision (a wrong accept never reaches a fallback); later stages recover recall.
  1. echo guard     context.echo_run >= ECHO_RUN_MIN -> decline (the assistant hearing itself)
  2. exact layer    a declared template phrase with a unique owner
  3. classifier     char-ngram TF-IDF + logistic regression; __decline__ class; threshold
  4. prefix         slot-free template (>= N literal words) + trailing speech (optional)
  5. LLM fallback   shortlisted manifest tools + addressee gate. Low-confidence declines always go;
                    rows the decline CLASS won go per `route_decline_class` (never/active/always)

A device is a manifest in devices/desktop/manifest.json shape plus training examples, so the same code
serves the Mac (INTENT-4 dump) and the Neon Hub (manifest built from its installed skills).

Usage (smoke): router.py "turn on the desk lamp" [--no-llm]
"""
from switchboard_eval.paths import ROOT
import json, os, sys
from dataclasses import dataclass

HERE = ROOT
sys.path.insert(0, HERE)
import numpy as np
from switchboard_eval.experiments.bench import DECLINE, CharEncoder, ExactLayer, LogRegHead, load_synthetic, load_training
from switchboard_eval.harness.llm_fallback import LLMRouter, to_intent
from switchboard_eval.harness.run_ovos import build_map
from switchboard_eval.harness.shortlist import Shortlist
from switchboard_eval.harness.slots import SlotExtractor
from switchboard_eval.harness.labels import LabelSelector

ECHO_RUN_MIN = 4
ACTIVE_S = 120


@dataclass
class DeviceConfig:
    manifest: str = "devices/desktop/manifest.json"
    examples: str = "devices/desktop/examples.jsonl"
    synthetic: str = "skills/synthetic_examples_slotted.jsonl,devices/desktop/dialog_negatives.jsonl"  # slot values masked
    schema: str = "schema/intents.yaml"
    tools: str = "devices/desktop/tools.json"
    threshold: float = 0.5
    prefix_min_literal: int = 0
    route_decline_class: str = "active"          # upstream default (CLAUDE.md, 2026-10-12)
    llm: bool = True
    system_prompt_file: str = "prompts/fallback_strict.txt"
    gate_prompt_file: str = "prompts/addressee_gate.txt"
    gate: float = 0.9
    shortlist_k: int = 12
    cache: str = "cache/llm_cache.jsonl"
    # Adopted 2026-10-12 (BENCH.md): train only intents THIS device can execute; optionally borrow
    # declared phrases for them from other devices' example files.
    device_classes_only: bool = True
    shared_examples: str = ""
    shared_manifests: str = ""  # other devices' manifests, for slot templates only (never the exact layer)
    # Experiment (2026-10-12): tell the LLM and the gate when the assistant last spoke and what it said.
    llm_context: bool = False

    def path(self, name):
        return os.path.join(HERE, getattr(self, name))

    @classmethod
    def hub(cls, **overrides):
        """Neon Hub: manifest built from its installed skills (build_hub_manifest.py), its own dialog negatives."""
        preset = dict(manifest="devices/speaker/manifest.json", examples="devices/speaker/examples.jsonl",
                      synthetic="skills/synthetic_examples_slotted.jsonl,devices/speaker/dialog_negatives.jsonl", tools="devices/speaker/tools.json",
                      shared_examples="devices/desktop/examples.jsonl",  # Mac phrases fix Adapt-only intents (weather)
                      shared_manifests="devices/desktop/manifest.json")  # Mac templates give Adapt-only intents slots
        return cls(**{**preset, **overrides})


class Router:
    def __init__(self, cfg=None):
        self.cfg = cfg or DeviceConfig()
        c = self.cfg
        texts, labels = self._training_data()
        self.encoder = CharEncoder().fit(texts)
        self.head = LogRegHead().fit(self.encoder.encode(texts), labels)
        self.exact = ExactLayer(c.path("manifest"), c.path("schema"))
        self.labels = LabelSelector(c.path("manifest"), c.path("schema"), c.path("examples"))
        self.slots = SlotExtractor(c.path("manifest"), c.path("schema"),
                                   [os.path.join(HERE, m) for m in filter(None, c.shared_manifests.split(","))])
        if c.prefix_min_literal:
            self.exact.enable_prefix(c.path("manifest"), "en-us", c.prefix_min_literal)
        self.llm = None
        if c.llm:
            self.tools_doc = json.load(open(c.path("tools")))
            self.label_map = build_map(c.path("schema"))
            system_prompt = open(c.path("system_prompt_file")).read().strip()
            self.llm = LLMRouter(Shortlist(c.path("tools"), c.path("examples")), c.path("cache"), system_prompt)
            self.gate_prompt = open(c.path("gate_prompt_file")).read().strip()
            if c.llm_context:  # context prompts ONLY when there is context; otherwise the strict prompts above
                ctx_prompt = open(os.path.join(HERE, "prompts/fallback_context.txt")).read().strip()
                self.llm_ctx = LLMRouter(self.llm.shortlist, c.path("cache"), ctx_prompt)
                self.gate_prompt_ctx = open(os.path.join(HERE, "prompts/addressee_gate_context.txt")).read().strip()

    def _training_data(self):
        c = self.cfg
        texts, labels, _, _ = load_training(c.path("examples"), c.path("schema"))
        if c.synthetic:
            s_texts, s_labels = load_synthetic(",".join(os.path.join(HERE, p) for p in c.synthetic.split(",")))
            texts, labels = texts + s_texts, labels + s_labels
        if not c.device_classes_only:
            return texts, labels
        executable = self._executable_intents()
        for path in filter(None, c.shared_examples.split(",")):
            o_texts, o_labels, _, _ = load_training(os.path.join(HERE, path), c.path("schema"), use_descriptions=False)
            texts, labels = texts + o_texts, labels + o_labels
        keep = [(t, l) for t, l in zip(texts, labels)
                if l == DECLINE or l in executable or (l.startswith("__unmapped__") and l in self._own_unmapped)]
        return [t for t, _ in keep], [l for _, l in keep]

    def _executable_intents(self):
        """Canonical intents with a registered handler on this device, plus the ones every device serves
        through pipelines rather than intents (fallback/common query answers questions; stop pipeline)."""
        label_map = build_map(self.cfg.path("schema"))
        dump = json.load(open(self.cfg.path("manifest")))["definitions"]
        own = {f"{sid}:{d['intent_name']}" for sid, ds in dump.items() if isinstance(ds, list) for d in ds}
        self._own_unmapped = {f"__unmapped__:{l}" for l in own if l not in label_map}
        return {label_map[l][0] for l in own if l in label_map and label_map[l][0]} | {"knowledge.ask", "stop"}

    def route(self, utterance, context=None):
        """context: {"gap_s": seconds since the assistant last spoke or None, "echo_run": int}.
        Returns the decision plus, for dispatch, `label` (skill_id:intent_name) and `match_data` (the skill's own
        slot / vocab names)."""
        return self._with_dispatch(self._route(utterance, context), utterance)

    def _with_dispatch(self, d, utterance):
        intent = d.get("intent")
        if not intent:
            return d
        if d.get("stage") in ("llm", "llm_gate") and d.get("llm_tool") in self.tools_doc["labels"]:
            label = self.tools_doc["labels"][d["llm_tool"]]
            if not label.startswith("meta:"):
                return {**d, "label": label, "match_data": d.get("tool_args", {})}
            return d
        if intent.startswith("__unmapped__:"):
            return {**d, "label": intent.split(":", 1)[1], "match_data": {}}
        label, data, how = self.labels.select(intent, utterance)
        return {**d, "label": label, "match_data": data, "label_how": how}

    def _route(self, utterance, context=None):
        context = context or {}
        utt = utterance.lower()
        if context.get("echo_run", 0) >= ECHO_RUN_MIN:
            return {"intent": None, "stage": "echo"}
        owner = self.exact.route(utt)
        if owner:
            return {"intent": owner, "stage": "exact", "confidence": 1.0,
                    "slots": self.slots.extract(owner, utt, loose=False)}
        probs = self.head.scores(self.encoder.encode([utt]))[0]
        order = np.argsort(-probs)
        top, conf = self.head.classes[order[0]], float(probs[order[0]])
        decision = {"intent": top, "stage": "classifier", "confidence": round(conf, 4),
                    "top3": [self.head.classes[j] for j in order[:3]]}
        if not (top.startswith("__unmapped__") or top == DECLINE or conf < self.cfg.threshold):
            return {**decision, "slots": self.slots.extract(top, utt)}
        decision["intent"] = None
        if self.cfg.prefix_min_literal:
            prefixed = self.exact.prefix_route(utt)
            if prefixed:
                return {**decision, "intent": prefixed, "stage": "prefix", "slots": self.slots.extract(prefixed, utt)}
        if self.llm and self._llm_eligible(top == DECLINE, context):
            return self._llm(utterance, decision, self._context_note(context))
        return decision

    def _llm_eligible(self, decline_class, context):
        if not decline_class:
            return True
        policy = self.cfg.route_decline_class
        gap = context.get("gap_s")
        return policy == "always" or (policy == "active" and gap is not None and gap < ACTIVE_S)

    def _context_note(self, context):
        """None when context is off or unknown (no logs): never invent "the assistant hasn't spoken"."""
        if not self.cfg.llm_context or "gap_s" not in context:
            return None
        gap, said = context.get("gap_s"), (context.get("last_speak") or "").strip()
        if gap is None or gap >= ACTIVE_S:
            return "Context: the assistant has not spoken in the last two minutes."
        return f"Context: the assistant last spoke {gap:.0f} seconds ago" + (f' and said: "{said[:200]}"' if said else ".")

    def _llm(self, utterance, decision, note=None):
        llm, gate_prompt = (self.llm_ctx, self.gate_prompt_ctx) if note else (self.llm, self.gate_prompt)
        res = llm.route(utterance, self.cfg.shortlist_k, note=note)
        intent, slots = to_intent(res["tool"], res["args"], self.tools_doc["labels"], self.label_map)
        gate = llm.p_addressed(utterance, gate_prompt, note=note)
        out = {**decision, "intent": intent, "slots": slots, "stage": "llm", "llm_tool": res["tool"], "tool_args": res["args"],
               "p_addressed": round(gate["p_yes"], 4), "llm_ms": res["ms"] + gate["ms"]}
        if gate["p_yes"] < self.cfg.gate:
            out.update(intent=None, slots={}, stage="llm_gate")
        return out


def main(argv):
    if not argv:
        sys.exit(__doc__)
    r = Router(DeviceConfig(llm="--no-llm" not in argv))
    print(json.dumps(r.route(argv[0]), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
