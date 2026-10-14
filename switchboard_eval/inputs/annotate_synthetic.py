#!/usr/bin/env python3
"""Mark slot VALUES in synthetic examples, so training can drop them like template {slots}.

Why: synthetic rows are filled-in sentences ("how do you say thank you in arabic"). Without spans the
classifier learns the slot value ("thank you") as evidence for the intent, and "thank you" alone routes
to translate.phrase. Templates never had this problem: their {slot} placeholders are stripped.

For each synthetic row of an intent with slots (schema/intents.yaml), the LLM returns {slot: exact substring or
null} under a JSON schema; every value is VERIFIED to be a substring of the example (else dropped and
reported). Writes skills/synthetic_examples_slotted.jsonl (original rows + "slot_values" + "masked") and
synthetic_slots_review.txt for human review (CLAUDE.md: synthetic training data must be reviewed).
skills/synthetic_examples.jsonl is not modified.

Usage: annotate_synthetic.py   (bench venv; LLM server)
"""
from switchboard_eval.paths import ROOT
import json, os, re, sys

HERE = ROOT
sys.path.insert(0, HERE)
import yaml
from switchboard_eval.harness.llm_fallback import LLMRouter

SRC, OUT = "skills/synthetic_examples.jsonl", "skills/synthetic_examples_slotted.jsonl"
REVIEW = "synthetic_slots_review.txt"
PROMPT = ("You annotate one voice-assistant request. For each slot, return the EXACT substring of the request "
          "that fills it (copy it character for character), or null if the request doesn't contain one. "
          "Return only the JSON object.")


def annotate(router, example, intent, spec):
    slots = spec["slots"]
    body = {"model": router.model, "temperature": 0, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "slots", "schema": {
                "type": "object", "required": slots, "additionalProperties": False,
                "properties": {s: {"type": ["string", "null"]} for s in slots}}}},
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": f"Intent: {intent} ({spec['description']})\n"
                                                     f"Slots: {', '.join(slots)}\nRequest: {example}"}]}
    res = router._cached(body, lambda r: {"slots": json.loads(r["choices"][0]["message"]["content"])})
    return res["slots"]


def mask(example, values):
    out = example
    for v in sorted((v for v in values.values() if v), key=len, reverse=True):
        out = out.replace(v, " ")
    return re.sub(r"\s+", " ", out).strip()


def main():
    schema = yaml.safe_load(open(os.path.join(HERE, "schema/intents.yaml")))["intents"]
    router = LLMRouter(None, os.path.join(HERE, "cache/llm_cache.jsonl"), PROMPT)
    rows = [json.loads(l) for l in open(os.path.join(HERE, SRC))]
    review, rejected = [], 0
    for r in rows:
        spec = schema.get(r["intent"], {})
        if not spec.get("slots"):
            continue
        ex = r["example"].lower()
        raw = annotate(router, ex, r["intent"], spec)
        values = {k: (v.lower() if v else None) for k, v in raw.items()}
        for k, v in list(values.items()):
            if v and v not in ex:
                values[k], rejected = None, rejected + 1
                review.append(f"REJECTED (not a substring) {r['intent']}: {ex!r} {k}={v!r}")
        r["slot_values"], r["masked"] = values, mask(ex, values)
        review.append(f"{r['intent']:18} {ex!r:60} -> {r['masked']!r}")
    with open(os.path.join(HERE, OUT), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    header = ["# Synthetic examples with slot values masked out (training drops them, like template {slots}).",
              "# Check that the masked text keeps the intent's wording and drops only the filled-in values.", ""]
    open(os.path.join(HERE, REVIEW), "w").write("\n".join(header + review) + "\n")
    print(f"annotated {sum('masked' in r for r in rows)} rows, {rejected} values rejected -> {OUT}, {REVIEW}")


if __name__ == "__main__":
    main()
