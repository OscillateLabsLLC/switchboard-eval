#!/usr/bin/env python3
"""Pick the K tools most similar to an utterance, so the LLM prompt stays small.

All 149 manifest tools are ~12.1k prompt tokens (~81/tool), over the GPU host's 12,288 context,
so a shortlist is required, not an optimization. Similarity = max char-ngram cosine to any
of a tool's harvested template expansions (devices/desktop/examples.jsonl) or its description.
Meta tools are always appended. Deterministic: same inputs, same shortlist.

Usage (smoke): shortlist.py devices/desktop/tools.json devices/desktop/examples.jsonl "turn on the desk lamp" [--k 12]
"""
import json, re, sys
import numpy as np

SLOT = re.compile(r"\{+[a-z_]+\}+")
DEFAULT_K = 12


class Shortlist:
    def __init__(self, tools_path, examples_path):
        from sklearn.feature_extraction.text import TfidfVectorizer
        doc = json.load(open(tools_path))
        self.by_name = {t["function"]["name"]: t for t in doc["tools"]}
        self.meta = [n for n, lab in doc["labels"].items() if lab.startswith("meta:")]
        label2name = {lab: n for n, lab in doc["labels"].items()}
        texts, owners = [], []
        for line in open(examples_path):
            e = json.loads(line)
            if e["label"] in label2name:
                texts.append(re.sub(r"\s+", " ", SLOT.sub(" ", e["example"])).strip().lower())
                owners.append(label2name[e["label"]])
        for n, t in self.by_name.items():
            if n not in self.meta:
                texts.append(t["function"]["description"].lower())
                owners.append(n)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(texts)
        self.X = self.vec.transform(texts)
        self.owners = np.array(owners)
        self.names = sorted(set(owners))
        self.idx = {n: np.where(self.owners == n)[0] for n in self.names}

    def rank(self, utterance):
        sims = (self.X @ self.vec.transform([utterance.lower()]).T).toarray().ravel()
        best = {n: float(sims[i].max()) for n, i in self.idx.items()}
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))

    def tools(self, utterance, k=DEFAULT_K):
        top = [n for n, _ in self.rank(utterance)[:k]]
        return [self.by_name[n] for n in top + self.meta]


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    k = int(argv[argv.index("--k") + 1]) if "--k" in argv else DEFAULT_K
    s = Shortlist(argv[0], argv[1])
    for n, score in s.rank(argv[2])[:k]:
        print(f"{score:.3f}  {n}")


if __name__ == "__main__":
    main(sys.argv[1:])
