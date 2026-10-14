#!/usr/bin/env python3
"""Harvest Padatious-style .intent templates from cloned skill repos.

Two outputs:
  1. skills/examples.jsonl: expanded training examples per `skill_id:intent`
     (the Phase 1 prototype/training side; this is what skills already declare).
  2. With --tag-dataset: marks each eval row whose utterance a template already
     matches (`seen_in_templates`). score.py then reports seen vs unseen
     accuracy separately. Unseen is the real generalization number. Seen rows
     measure whether a matcher can reproduce its own training data.

Never move harvested examples INTO the eval set: that is train/test leakage.

Adapt-only skills (.voc files, no .intent) are listed but not harvested.
Their examples have to come from INTENT-4 introspection on a live stack.

Usage:
  harvest_examples.py SKILLS_DIR [--out skills/examples.jsonl] [--tag-dataset data/dataset.jsonl] [--lang en-us]
SKILLS_DIR holds one cloned repo per subdirectory.
"""
import glob, itertools, json, os, re, sys

MAX_PER_LINE = 200


def skill_id_for(repo):
    txt = ""
    for f in ("setup.py", "pyproject.toml"):
        p = os.path.join(repo, f)
        if os.path.exists(p):
            txt += open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r'["\']([a-z0-9_.-]+\.(?:openvoiceos|neongeckocom|oscillatelabsllc))["\']\s*=', txt)
    if m:
        return m.group(1)
    name = re.search(r'SKILL_NAME\s*=\s*["\']([^"\']+)', txt)
    ep = re.search(r'PLUGIN_ENTRY_POINT\s*=\s*f?["\']\{SKILL_NAME\}\.([a-z]+)=', txt)
    if name and ep:
        return f"{name.group(1)}.{ep.group(1)}"
    return None


def _split_top(s, sep="|"):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def expand(t):
    """Expand (a|b|), [optional] groups. {slots} are left in place."""
    for i, ch in enumerate(t):
        if ch in "([":
            close = ")" if ch == "(" else "]"
            depth = 0
            for j in range(i, len(t)):
                if t[j] in "([":
                    depth += 1
                elif t[j] in ")]":
                    depth -= 1
                    if depth == 0:
                        break
            inner = t[i + 1:j]
            opts = _split_top(inner) if ch == "(" else [inner, ""]
            if close == "]" and "|" in inner:
                opts = _split_top(inner) + [""]
            out = []
            for o in opts:
                out.extend(expand(t[:i] + o + t[j + 1:]))
            return out
    return [t]


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def template_regex(line):
    """Template -> regex matching any expansion, slots as wildcards."""
    pat = ""
    i = 0
    line = line.lower().replace("{{", "{").replace("}}", "}")
    while i < len(line):
        ch = line[i]
        if ch == "{":
            j = line.index("}", i)
            pat += r"(?:.+?)"
            i = j + 1
            continue
        if ch == "(":
            pat += "(?:"
        elif ch == ")":
            pat += ")"
        elif ch == "[":
            pat += "(?:"
        elif ch == "]":
            pat += ")?"
        elif ch == "|":
            pat += "|"
        elif ch == " ":
            pat += r"\s*"
        else:
            pat += re.escape(ch)
        i += 1
    return re.compile(r"^\s*" + pat + r"\s*[?.!]*$")


def harvest(skills_dir, lang="en-us"):
    examples, templates, adapt_only = [], [], []
    for repo in sorted(glob.glob(os.path.join(skills_dir, "*/"))):
        sid = skill_id_for(repo) or os.path.basename(repo.rstrip("/"))
        files = [f for f in glob.glob(os.path.join(repo, "**", "*.intent"), recursive=True)
                 if f"/{lang}/" in f.lower()]
        vocs = [f for f in glob.glob(os.path.join(repo, "**", "*.voc"), recursive=True)
                if f"/{lang}/" in f.lower()]
        if not files:
            if vocs:
                adapt_only.append(sid)
            continue
        for f in files:
            intent = os.path.basename(f)[:-len(".intent")]
            label = f"{sid}:{intent}"
            for line in open(f, encoding="utf-8", errors="ignore"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                templates.append((label, template_regex(line)))
                for ex in itertools.islice(expand(line), MAX_PER_LINE):
                    ex = clean(ex)
                    if ex:
                        slots = sorted(set(re.findall(r"\{+([a-z_]+)\}+", ex)))
                        examples.append({"label": label, "skill_id": sid, "intent": intent,
                                         "example": ex, "slots": slots})
    return examples, templates, adapt_only


def main(argv):
    if not argv:
        sys.exit(__doc__)
    skills_dir = argv[0]
    out = argv[argv.index("--out") + 1] if "--out" in argv else "skills/examples.jsonl"
    lang = argv[argv.index("--lang") + 1] if "--lang" in argv else "en-us"
    ex, tpls, adapt_only = harvest(skills_dir, lang)
    seen = set()
    with open(out, "w") as f:
        for e in ex:
            k = (e["label"], e["example"].lower())
            if k not in seen:
                seen.add(k)
                f.write(json.dumps(e) + "\n")
    labels = sorted({e["label"] for e in ex})
    print(f"{len(seen)} examples across {len(labels)} intents -> {out}")
    if adapt_only:
        print("Adapt-only (not harvestable from templates):", ", ".join(adapt_only))

    if "--tag-dataset" in argv:
        path = argv[argv.index("--tag-dataset") + 1]
        rows = [json.loads(l) for l in open(path)]
        hits = 0
        for r in rows:
            u = r["utterance"].lower()
            r["seen_in_templates"] = sorted({lab for lab, rx in tpls if rx.match(u)})
            hits += bool(r["seen_in_templates"])
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"tagged {path}: {hits}/{len(rows)} rows already match a skill template")


if __name__ == "__main__":
    main(sys.argv[1:])
