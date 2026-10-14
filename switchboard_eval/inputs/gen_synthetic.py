#!/usr/bin/env python3
"""Task 8: generate candidate phrasings for canonical intents with no training examples.

Candidates are written for HUMAN REVIEW. Never copy them into data/dataset.jsonl
(train/test leakage). The model is never shown the eval set, and any candidate
that normalizes to an eval utterance is dropped and reported.

Usage: gen_synthetic.py [--examples devices/desktop/examples.jsonl] [--out skills/synthetic_examples.jsonl]
                        [--url http://localhost:8080/v1] [--n 25]
"""
import argparse, json, os, re, string, sys, urllib.request

import yaml

MODEL = "Qwen/Qwen3-14B-AWQ"
TEMPERATURE = 0.9
TIMEOUT_S = 180

PROMPT = """You write training data for a home voice assistant's intent classifier.

Target intent: {name}
Meaning: {description}
Slots it can carry: {slots}

Other intents exist and must NOT be confused with this one:
{siblings}

Write {n} different things a person might say out loud to trigger ONLY the target intent.
Vary them: short commands, polite requests, casual speech, indirect phrasing, and
phrasings that fill different slots with realistic values. Every one must clearly mean
the target intent and nothing else. Lowercase, no numbering, no quotes.
Reply with a JSON array of strings and nothing else."""


def norm(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def uncovered_intents(schema, examples_path):
    label2canon = {}
    for canon, spec in schema.items():
        for e in spec.get("ovos") or []:
            label2canon[e if isinstance(e, str) else e["label"]] = canon
    covered = {label2canon.get(json.loads(l)["label"]) for l in open(examples_path)}
    return [c for c in schema if c not in covered]


def ask(url, prompt):
    body = json.dumps({
        "model": MODEL, "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{url}/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError(f"no JSON array in reply: {text[:200]!r}")
    return [s for s in json.loads(match.group(0)) if isinstance(s, str)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="devices/desktop/examples.jsonl")
    ap.add_argument("--schema", default="schema/intents.yaml")
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--out", default="skills/synthetic_examples.jsonl")
    ap.add_argument("--url", default=os.environ.get("INTENT_EVAL_LLM_URL", "http://localhost:8080/v1/chat/completions").rsplit("/chat/completions", 1)[0])
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()

    schema = yaml.safe_load(open(a.schema))["intents"]
    eval_utts = {norm(json.loads(l)["utterance"]) for l in open(a.dataset)}
    targets = uncovered_intents(schema, a.examples)
    print(f"{len(targets)} intents with no training examples: {', '.join(targets)}")

    written, leaked = 0, []
    with open(a.out, "w") as f:
        for name in targets:
            spec = schema[name]
            siblings = "\n".join(f"- {k}: {v['description']}" for k, v in schema.items() if k != name)
            prompt = PROMPT.format(name=name, description=spec["description"],
                                   slots=", ".join(spec.get("slots") or []) or "none",
                                   siblings=siblings, n=a.n)
            try:
                phrases = ask(a.url, prompt)
            except Exception as e:
                print(f"FAIL {name}: {e}", file=sys.stderr)
                continue
            seen = set()
            for p in phrases:
                k = norm(p)
                if not k or k in seen:
                    continue
                seen.add(k)
                if k in eval_utts:
                    leaked.append((name, p))
                    continue
                f.write(json.dumps({"label": f"synthetic:{name}", "skill_id": "synthetic",
                                    "intent": name, "example": k, "slots": [],
                                    "source": "synthetic"}) + "\n")
                written += 1
            print(f"ok   {name}: {len(seen)} unique")
    print(f"wrote {written} candidates to {a.out}")
    for name, p in leaked:
        print(f"DROPPED (matches an eval utterance): {name}: {p!r}")


if __name__ == "__main__":
    main()
