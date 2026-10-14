#!/usr/bin/env python3
"""Phase 1 offline bench: train on harvested skill examples, predict the eval set.

No OVOS required. Writes predictions in score.py format, so any candidate can be
compared against the live baseline:
    bench.py --encoder char --head logreg --out preds_char.jsonl
    score.py data/dataset.jsonl data/baselines/baseline.jsonl preds_char.jsonl

Training data = skills/examples.jsonl (what skills declare) + one pseudo-example per
canonical intent built from its description (schema-guided, Google SGD style).
Canonical intents with no harvested examples (mostly Adapt-based Neon skills
and unbacked intents like lock.*) get the description only. Their rows measure
zero-shot ability, and the report shows how many eval rows that affects.

Encoders (add your own by subclassing Encoder):
  char     TF-IDF over char 2-5 grams + word 1-2 grams. No downloads. The floor.
  st:NAME  sentence-transformers model, e.g. st:BAAI/bge-small-en-v1.5,
           st:intfloat/multilingual-e5-small, st:sentence-transformers/all-MiniLM-L6-v2
  m2v:NAME model2vec static model, e.g. m2v:minishlab/potion-base-8M
Heads:
  proto    cosine to per-intent centroid (what M2V prototype mode does)
  knn      max cosine to any single example
  logreg   trained logistic regression head (the SetFit-style upgrade, minus
           contrastive fine-tuning of the body)
"""
from switchboard_eval.paths import ROOT
import argparse, json, re, sys, time
import numpy as np
import yaml


# ---------- data ----------

DECLINE = "__decline__"  # training class for "should not act"; predicted -> intent None


def load_synthetic(paths):
    """Comma-separated files of rows that carry their canonical intent (or DECLINE) directly."""
    rows = [json.loads(l) for p in paths.split(",") for l in open(p)]
    # "masked" (annotate_synthetic.py) drops slot values, the way template {slots} are stripped
    return [r.get("masked", r["example"]).lower() for r in rows], [r["intent"] for r in rows]


def load_training(examples_path, schema_path, use_descriptions=True):
    schema = yaml.safe_load(open(schema_path))["intents"]
    label2canon = {}
    for canon, spec in schema.items():
        for e in spec.get("ovos") or []:
            label2canon[e if isinstance(e, str) else e["label"]] = canon
    texts, labels, unmapped = [], [], set()
    for line in open(examples_path):
        e = json.loads(line)
        canon = label2canon.get(e["label"])
        if canon is None:
            unmapped.add(e["label"])
            canon = "__unmapped__:" + e["label"]   # kept as a class: a real competitor
        t = re.sub(r"\{+[a-z_]+\}+", " ", e["example"]).strip()
        t = re.sub(r"\s+", " ", t)
        if t:
            texts.append(t.lower())
            labels.append(canon)
    covered = set(labels)
    if use_descriptions:
        for canon, spec in schema.items():
            texts.append(spec["description"].lower())
            labels.append(canon)
    return texts, labels, covered, sorted(unmapped)


# ---------- encoders ----------

class Encoder:
    def fit(self, texts):
        return self

    def encode(self, texts):
        raise NotImplementedError


class CharEncoder(Encoder):
    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_union
        self.v = make_union(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z0-9']+"),
        ).fit(texts)
        return self

    def encode(self, texts):
        from sklearn.preprocessing import normalize
        return normalize(self.v.transform(texts))


class STEncoder(Encoder):
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.prefix = "query: " if "e5" in name else ""
        self.m = SentenceTransformer(name, device="cpu")

    def encode(self, texts):
        return self.m.encode([self.prefix + t for t in texts], normalize_embeddings=True,
                             batch_size=64, show_progress_bar=False)


class M2VEncoder(Encoder):
    def __init__(self, name):
        from model2vec import StaticModel
        self.m = StaticModel.from_pretrained(name)

    def encode(self, texts):
        x = self.m.encode(texts)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def make_encoder(spec):
    if spec == "char":
        return CharEncoder()
    kind, _, name = spec.partition(":")
    return {"st": STEncoder, "m2v": M2VEncoder}[kind](name)


# ---------- heads ----------

def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


class ProtoHead:
    def fit(self, X, y):
        X = _dense(X)
        self.classes = sorted(set(y))
        ya = np.array(y)
        C = np.stack([X[ya == c].mean(0) for c in self.classes])
        self.C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        return self

    def scores(self, X):
        return _dense(X) @ self.C.T


class KnnHead:
    def fit(self, X, y):
        self.X, self.y = _dense(X), np.array(y)
        self.classes = sorted(set(y))
        self.idx = {c: np.where(self.y == c)[0] for c in self.classes}
        return self

    def scores(self, X):
        S = _dense(X) @ self.X.T
        return np.stack([S[:, self.idx[c]].max(1) for c in self.classes], axis=1)


class LogRegHead:
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced").fit(X, y)
        self.classes = list(self.m.classes_)
        return self

    def scores(self, X):
        return self.m.predict_proba(X)


HEADS = {"proto": ProtoHead, "knn": KnnHead, "logreg": LogRegHead}


# ---------- exact layer (design goal 1) ----------

class ExactLayer:
    """Deterministic template match that runs BEFORE the classifier.

    A phrase matching templates of exactly one owner routes to that owner, even an
    unmapped skill: it is a declared phrase. A collision between 2+ owners returns
    None so the classifier decides (design goal 2)."""

    def __init__(self, dump_path, schema_path, lang="en-us", min_literal=1):
        import os
        sys.path.insert(0, ROOT)
        from switchboard_eval.inputs.harvest_live import harvest, literal_words
        _, templates = harvest(dump_path, lang)
        # Wildcard-heavy templates ('play {query}') still train the classifier but don't claim
        # the exact layer: they're catch-alls, not magic phrases.
        self.templates = [(lab, rx) for lab, rx, line in templates if literal_words(line) >= min_literal]
        schema = yaml.safe_load(open(schema_path))["intents"]
        self.label2canon = {}
        for canon, spec in schema.items():
            for e in spec.get("ovos") or []:
                self.label2canon[e if isinstance(e, str) else e["label"]] = canon

    def route(self, utterance):
        owners = {self.label2canon.get(lab, f"__unmapped__:{lab}")
                  for lab, rx in self.templates if rx.match(utterance)}
        return owners.pop() if len(owners) == 1 else None

    def enable_prefix(self, dump_path, lang, min_literal):
        """Trailing speech: someone keeps talking before VAD cuts off ("make me laugh hey dad ...").
        Slot-free templates of >= min_literal words also match as a PREFIX followed by more speech.
        Slot templates are excluded: a trailing {slot} already absorbs extra words."""
        from switchboard_eval.inputs.harvest_live import harvest, literal_words
        _, templates = harvest(dump_path, lang)
        self.prefix_templates = [
            (lab, re.compile(rx.pattern.removesuffix(r"\s*[?.!]*$") + r"[\s?.!,]+\S"))
            for lab, rx, line in templates if "{" not in line and literal_words(line) >= min_literal]

    def prefix_route(self, utterance):
        owners = {self.label2canon.get(lab, f"__unmapped__:{lab}")
                  for lab, rx in self.prefix_templates if rx.match(utterance)}
        return owners.pop() if len(owners) == 1 else None


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="char")
    ap.add_argument("--head", default="logreg", choices=HEADS)
    ap.add_argument("--threshold", type=float, default=None,
                    help="below this confidence, predict null (decline). Default: emit raw "
                         "confidence and read score.py's threshold sweep instead")
    ap.add_argument("--examples", default="skills/examples.jsonl")
    ap.add_argument("--schema", default="schema/intents.yaml")
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--no-descriptions", action="store_true")
    ap.add_argument("--synthetic", default=None,
                    help="comma-separated training files with canonical intents "
                         "(reviewed skills/synthetic_examples.jsonl, devices/desktop/dialog_negatives.jsonl)")
    ap.add_argument("--templates", default=None,
                    help="devices/desktop/manifest.json: enable the exact layer in front of the classifier")
    ap.add_argument("--min-literal", type=int, default=1,
                    help="exact layer ignores templates with fewer literal (non-slot) words")
    ap.add_argument("--prefix-min-literal", type=int, default=0,
                    help="for rows the classifier declines, match slot-free templates of >= N literal "
                         "words as a prefix followed by more speech (trailing speech). 0 = off")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    texts, labels, covered, unmapped = load_training(a.examples, a.schema, not a.no_descriptions)
    if a.synthetic:
        s_texts, s_labels = load_synthetic(a.synthetic)
        texts, labels = texts + s_texts, labels + s_labels
        covered |= set(s_labels) - {DECLINE}
    rows = [json.loads(l) for l in open(a.dataset)]

    t0 = time.perf_counter()
    enc = make_encoder(a.encoder).fit(texts)
    X = enc.encode(texts)
    head = HEADS[a.head]().fit(X, labels)
    exact = ExactLayer(a.templates, a.schema, min_literal=a.min_literal) if a.templates else None
    if a.prefix_min_literal:
        exact.enable_prefix(a.templates, "en-us", a.prefix_min_literal)
    train_s = time.perf_counter() - t0

    preds = []
    for r in rows:
        t = time.perf_counter()
        utt = r["utterance"].lower()
        routed = exact.route(utt) if exact else None
        if routed:
            ms = (time.perf_counter() - t) * 1000
            preds.append({"id": r["id"], "intent": routed, "slots": {}, "confidence": 1.0,
                          "latency_ms": round(ms, 2), "stage": "exact"})
            continue
        s = head.scores(enc.encode([utt]))[0]
        ms = (time.perf_counter() - t) * 1000
        i = int(np.argmax(s))
        intent, conf = head.classes[i], float(s[i])
        if intent.startswith("__unmapped__") or intent == DECLINE or \
                (a.threshold is not None and conf < a.threshold):
            intent = None
        stage = "classifier"
        if intent is None and a.prefix_min_literal:
            prefixed = exact.prefix_route(utt)
            if prefixed:
                intent, stage = prefixed, "prefix"
        preds.append({"id": r["id"], "intent": intent, "slots": {}, "confidence": round(conf, 4),
                      "latency_ms": round(ms, 2), "stage": stage,
                      "top3": [head.classes[j] for j in np.argsort(-s)[:3]]})

    with open(a.out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    def gold_set(r):
        g = r["intent"]
        return set(g) if isinstance(g, list) else ({g} if g else set())
    in_dom = [r for r in rows if gold_set(r)]
    zero_shot = [r for r in in_dom if not gold_set(r) & covered]
    print(f"encoder={a.encoder} head={a.head}  train {len(texts)} texts / {len(set(labels))} classes in {train_s:.1f}s")
    print(f"eval rows whose gold intent has NO harvested examples (description-only, zero-shot): "
          f"{len(zero_shot)}/{len(in_dom)}")
    if unmapped:
        print(f"{len(unmapped)} harvested labels unmapped (kept as competitor classes): {', '.join(unmapped[:6])}"
              + (" ..." if len(unmapped) > 6 else ""))
    split = {"covered": [0, 0], "zero-shot": [0, 0]}
    P = {p["id"]: p for p in preds}
    for r in rows:
        g = r["intent"]
        if not isinstance(g, str) or r["category"] in ("compound", "followup"):
            continue
        k = "covered" if g in covered else "zero-shot"
        split[k][1] += 1
        split[k][0] += P[r["id"]]["intent"] == g
    print("in-domain accuracy by training coverage (no threshold): " + ", ".join(
        f"{k} {100 * c / n:.0f}% (n={n})" for k, (c, n) in split.items() if n))
    print(f"wrote {a.out}. Slots are not predicted (intent-only bench), so ignore slot EM.")


if __name__ == "__main__":
    main()
