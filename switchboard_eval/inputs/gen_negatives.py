#!/usr/bin/env python3
"""Decline lever 2: generate utterances the assistant should NOT act on, for human review.

Writes negative_review.txt (grouped by kind; delete lines you disagree with). After review,
convert with --convert to negative_examples.jsonl ("intent": "__decline__"), which
bench.py --synthetic trains as a decline class. Never copy these into data/dataset.jsonl.

Usage: gen_negatives.py [--n 60] [--url http://localhost:8080/v1]
       gen_negatives.py --convert
"""
from switchboard_eval.paths import ROOT
import argparse, json, os, sys

sys.path.insert(0, ROOT)
from switchboard_eval.inputs.gen_synthetic import ask, norm

KINDS = {
    "family_chatter": "things family members say to EACH OTHER at home, overheard by a voice assistant "
                      "in the room: parents and kids talking about chores, homework, dinner, screen time",
    "tv_dialogue": "lines of dialogue from TV shows, movies, cartoons, or video games playing in the room",
    "assistant_echo": "sentences a voice assistant itself says out loud (answers, confirmations, apologies), "
                      "which its own microphone might pick up",
    "bare_reply": "one- to three-word replies people say mid-conversation that are not commands: "
                  "agreements, numbers, thanks, fillers, hesitations",
    "mention_not_request": "sentences that MENTION devices, timers, music, doors, weather, or the assistant "
                           "but are NOT requests to do anything: complaints, stories, questions about how "
                           "something works, and negated commands like 'don't turn that on'",
}
PROMPT = """Write {n} different examples of: {desc}.
They are transcribed speech: lowercase, no punctuation needed, realistic and varied in length.
None of them may be a request for a voice assistant to DO something.
Reply with a JSON array of strings and nothing else."""
REVIEW, OUT = "negative_review.txt", "negative_examples.jsonl"


def generate(a):
    evals = {norm(json.loads(l)["utterance"]) for l in open(a.dataset)}
    dropped = 0
    with open(REVIEW, "w") as f:
        f.write("# Decline-class training phrases (assistant should NOT act). DELETE any line that IS\n"
                "# a real request, or that you otherwise don't want. Keep the ## headers.\n")
        for kind, desc in KINDS.items():
            phrases, seen = ask(a.url, PROMPT.format(n=a.n, desc=desc)), set()
            f.write(f"\n## {kind}\n")
            for p in phrases:
                k = norm(p)
                if not k or k in seen:
                    continue
                seen.add(k)
                if k in evals:
                    dropped += 1
                    continue
                f.write(k + "\n")
            print(f"ok   {kind}: {len(seen)}")
    print(f"wrote {REVIEW}; dropped {dropped} that matched eval utterances")


def convert():
    kind, n = None, 0
    with open(OUT, "w") as f:
        for line in open(REVIEW):
            line = line.strip()
            if not line or line.startswith("# "):
                continue
            if line.startswith("## "):
                kind = line[3:]
                continue
            f.write(json.dumps({"label": "synthetic:__decline__", "skill_id": "synthetic",
                                "intent": "__decline__", "example": norm(line), "slots": [],
                                "source": "synthetic", "kind": kind, "reviewed": True}) + "\n")
            n += 1
    print(f"wrote {n} reviewed negatives to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--url", default=os.environ.get("INTENT_EVAL_LLM_URL", "http://localhost:8080/v1/chat/completions").rsplit("/chat/completions", 1)[0])
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--convert", action="store_true")
    a = ap.parse_args()
    convert() if a.convert else generate(a)


if __name__ == "__main__":
    main()
