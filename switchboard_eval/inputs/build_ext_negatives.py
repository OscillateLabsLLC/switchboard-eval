#!/usr/bin/env python3
"""External negatives: speech never addressed to the assistant, from public corpora.

A SEPARATE stress slice (data/negatives_ext.jsonl), never mixed into data/dataset.jsonl and never part
of the bar's criterion 4 (that stays on household speech). Every row has gold intent None.

Sources (raw files in data/external/, gitignored; see SOURCES for licenses):
  ami          AMI Meeting Corpus manual transcripts: one segment = one speaker turn
  magpie       MAGPIE idioms, spoken-BNC genre, idiomatic uses whose first word also starts a
               live skill template ("play it by ear", "turn a blind eye"): near-misses. The verb
               set comes from devices/desktop/manifest.json, not a hand list
  dailydialog  DailyDialog single turns (written chit-chat; CC BY-NC-SA: private eval only)

Text is put in Neon Hub STT form (lowercase, no punctuation, apostrophes dropped), 2-20 words,
deduplicated against data/dataset.jsonl, PER_SOURCE rows per source with a fixed seed.

Usage: build_ext_negatives.py [--out data/negatives_ext.jsonl] [--per-source 200]
"""
from switchboard_eval.paths import ROOT
import argparse, glob, json, os, random, re, sys, xml.etree.ElementTree as ET

HERE = ROOT
EXT = os.path.join(HERE, "data", "external")
sys.path.insert(0, HERE)
from switchboard_eval.inputs.harvest_examples import expand

SEED, MIN_WORDS, MAX_WORDS = 20261012, 2, 20
NITE_ID = "{http://nite.sourceforge.net/}id"
SOURCES = {
    "ami": "AMI Meeting Corpus, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/",
    "magpie": "MAGPIE (Haagsma et al. 2020), CC BY 4.0, https://github.com/hslh/magpie-corpus",
    "dailydialog": "DailyDialog (Li et al. 2017) via ConvLab, CC BY-NC-SA 4.0",
}


def stt_form(text):
    text = text.lower().replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()


def ami_turns():
    for seg_path in sorted(glob.glob(os.path.join(EXT, "ami", "x", "segments", "*.segments.xml"))):
        # Index EVERY element: segment ranges often end on a punctuation token or a vocal sound.
        elems = list(ET.parse(seg_path.replace("segments", "words")).getroot())
        pos = {e.get(NITE_ID): i for i, e in enumerate(elems)}
        for seg in ET.parse(seg_path).getroot().iter("segment"):
            for child in seg:
                ids = re.findall(r"id\(([^)]+)\)", child.get("href", ""))
                if ids and ids[0] in pos and ids[-1] in pos:
                    span = elems[pos[ids[0]]:pos[ids[-1]] + 1]
                    yield " ".join(e.text or "" for e in span if e.tag == "w" and e.get("punc") != "true")


def template_first_words():
    from switchboard_eval.inputs.harvest_live import harvest
    _, templates = harvest(os.path.join(HERE, "devices/desktop/manifest.json"), "en-us")
    firsts = set()
    for _, _, line in templates:
        for ex in expand(line)[:50]:
            w = re.sub(r"\{+[a-z_]+\}+", " ", ex).split()
            if w:
                firsts.add(w[0].lower())
    return firsts


def magpie_sentences():
    verbs = template_first_words()
    for line in open(os.path.join(EXT, "magpie", "MAGPIE_filtered_split_random.jsonl")):
        r = json.loads(line)
        if r["genre"].startswith("S") and r["label"] == "i" and r["idiom"].split()[0].lower() in verbs:
            yield re.sub(r" (n't|'[a-z]+)\b", r"\1", r["context"][2])  # BNC splits "did n't", "they 'll"


def dailydialog_turns():
    for d in json.load(open(os.path.join(EXT, "dailydialog", "x", "data", "dialogues.json"))):
        for t in d["turns"]:
            yield t["utterance"]


def sample(texts, n, taken, rng):
    pool = sorted({t for t in map(stt_form, texts) if MIN_WORDS <= len(t.split()) <= MAX_WORDS} - taken)
    picked = rng.sample(pool, min(n, len(pool)))
    taken.update(picked)
    return picked, len(pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "data/negatives_ext.jsonl"))
    ap.add_argument("--per-source", type=int, default=200)
    a = ap.parse_args()
    rng = random.Random(SEED)
    taken = {stt_form(json.loads(l)["utterance"]) for l in open(os.path.join(HERE, "data/dataset.jsonl"))}
    rows = []
    for source, gen in (("ami", ami_turns), ("magpie", magpie_sentences), ("dailydialog", dailydialog_turns)):
        picked, pool = sample(gen(), a.per_source, taken, rng)
        print(f"{source}: {len(picked)} sampled from {pool} eligible")
        rows += [{"id": f"x{len(rows) + i:04d}", "utterance": u, "intent": None, "slots": {},
                  "category": "ext_negative", "lang": "en-us", "risk": "none", "score_slots": False,
                  "split": "ext", "source": source, "tags": []} for i, u in enumerate(picked)]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
