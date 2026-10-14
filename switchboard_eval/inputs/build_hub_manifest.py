#!/usr/bin/env python3
"""Build the Neon Hub's intent manifest from its installed skills (the Hub has no INTENT-4).

Only skills that actually registered intents on the live Hub are included (manifest.json from
dump_hub_manifest.py): blacklist IDs that don't match installed IDs don't block anything, so the
live registrations, not the config text, define what runs.

  Padatious  intents named in the live padatious manifest; samples from the skill's .intent file
  Adapt      parsed from the skill's Python source with ast (never imported or executed):
             @intent_handler(IntentBuilder("Name").require(..).optionally(..).one_of(..))
             vocabulary from the skill's .voc files; keyword_examples = deterministic combinations
             (one phrase per required vocab, in builder order, plus each one_of choice)

Writes devices/speaker/manifest.json (devices/desktop/manifest.json shape; Adapt entries use method "adapt") and
devices/speaker/examples.jsonl (template expansions + Adapt keyword examples, harvest_live format).

Usage: build_hub_manifest.py
"""
from switchboard_eval.paths import ROOT
import ast, itertools, json, os, re, sys

HERE = ROOT
sys.path.insert(0, HERE)
from switchboard_eval.inputs.harvest_examples import MAX_PER_LINE, clean, expand

SKILLS = os.path.join(HERE, "data", "hub", "skills")
MAX_KEYWORD_EXAMPLES = 40
CHAIN_METHODS = {"require", "optionally", "one_of"}


def vkey(name):
    """Adapt vocab names vs file stems differ in case and separators ("RelativeDay" / relative-day.voc)."""
    return re.sub(r"[-_]", "", name.lower())


def lang_files(pkg_dir, suffix):
    """{stem_lower: path} for en-us resource files anywhere under the package."""
    out = {}
    for root, _, files in os.walk(pkg_dir):
        if "en-us" not in root.lower().replace("_", "-"):
            continue
        for f in files:
            if f.endswith(suffix):
                out.setdefault(f[: -len(suffix)].lower(), os.path.join(root, f))
    return out


def well_formed(line, sid, intent):
    """Skip (and report) template lines our matcher can't compile, e.g. an unclosed '{name'."""
    from switchboard_eval.inputs.harvest_examples import template_regex
    try:
        template_regex(line)
        return True
    except ValueError:
        print(f"  SKIPPED malformed template in {sid}:{intent}: {line!r}")
        return False


def read_lines(path):
    return [l.strip() for l in open(path, encoding="utf-8", errors="ignore") if l.strip() and not l.startswith("#")]


def builder_chain(node):
    """IntentBuilder("X").require("a").one_of("b","c")... -> (name, [(method, [args])]) or None."""
    steps = []
    while isinstance(node, ast.Call):
        func = node.func
        args = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if isinstance(func, ast.Name) and func.id == "IntentBuilder":
            return (args[0] if args else None), list(reversed(steps))
        if isinstance(func, ast.Attribute):
            if func.attr in CHAIN_METHODS:
                steps.append((func.attr, args))
            node = func.value
            continue
        return None
    return None


def adapt_intents(pkg_dir):
    found = {}
    for root, _, files in os.walk(pkg_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(root, f), encoding="utf-8", errors="ignore").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    chain = builder_chain(node)
                    # ast.walk also visits every nested partial chain; keep the longest (the full builder)
                    if chain and chain[0] and len(chain[1]) > len(found.get(chain[0], [])):
                        found[chain[0]] = chain[1]
    return found


def keyword_examples(steps, vocab):
    required, choices = [], []
    for method, args in steps:
        if method == "require" and args:
            required.append(vocab.get(vkey(args[0]), [args[0].replace("_", " ")])[:6])
        elif method == "one_of":
            choices.append([p for a in args for p in vocab.get(vkey(a), [a.replace("_", " ")])[:3]])
    slots = required + choices
    if not slots:
        return []
    return [" ".join(combo) for combo in itertools.islice(itertools.product(*slots), MAX_KEYWORD_EXAMPLES)]


def load_vocab(pkg_dir):
    vocab = {}
    for stem, path in lang_files(pkg_dir, ".voc").items():
        phrases = []
        for line in read_lines(path):
            phrases += [clean(e) for e in itertools.islice(expand(line), MAX_PER_LINE) if clean(e)]
        vocab[vkey(stem)] = list(dict.fromkeys(phrases))
    return vocab


def build():
    live = json.load(open(os.path.join(HERE, "data", "hub", "manifest.json")))
    padatious = live["intent.service.padatious.manifest"]["intents"]
    vocab_types = {v.get("entity_type") or "" for v in live["intent.service.adapt.vocab.manifest"]["vocab"]}
    skill_ids = json.load(open(os.path.join(SKILLS, "skill_ids.json")))
    definitions, examples, report = {}, [], {}
    for sid, pkg in sorted(skill_ids.items()):
        pkg_dir = os.path.join(SKILLS, pkg)
        mangled = re.sub(r"[^a-zA-Z0-9]", "_", sid)
        defs = []
        intent_files = lang_files(pkg_dir, ".intent")
        for name in dict.fromkeys(p.split(":", 1)[1] for p in padatious if p.startswith(sid + ":")):  # live manifest can repeat a name
            stem = name.removesuffix(".intent")
            path = intent_files.get(stem.lower())
            samples = [l for l in (read_lines(path) if path else []) if well_formed(l, sid, stem)]
            defs.append({"skill_id": sid, "intent_name": stem, "lang": "en-us", "method": "template",
                         "definition": {"samples": samples}})
        if any(t.startswith(mangled) for t in vocab_types):
            vocab = load_vocab(pkg_dir)
            for name, steps in sorted(adapt_intents(pkg_dir).items()):
                kws = keyword_examples(steps, vocab)
                names = {vkey(a) for _, args in steps for a in args}
                regex = {g: [] for g in names}
                for path in lang_files(pkg_dir, ".rx").values():  # Adapt regex entities: (?P<name>...) groups
                    for line in read_lines(path):
                        for g in re.findall(r"\(\?P<(\w+)>", line):
                            if vkey(g) in regex:
                                regex[vkey(g)].append(line)
                defs.append({"skill_id": sid, "intent_name": name, "lang": "en-us", "method": "adapt",
                             "definition": {"steps": steps, "keyword_examples": kws,
                                            # phrases per referenced vocab (Sluice evaluates Adapt rules with these)
                                            "vocab": {n: vocab[n] for n in sorted(names) if n in vocab},
                                            "regex": {n: rx for n, rx in sorted(regex.items()) if rx}}})
                examples += [{"label": f"{sid}:{name}", "skill_id": sid, "intent": name, "example": k,
                              "slots": [], "source": "adapt_keywords"} for k in kws]
        if defs:
            definitions[sid] = defs
            report[sid] = (sum(d["method"] == "template" for d in defs), sum(d["method"] == "adapt" for d in defs))
    for sid, defs in ocp_definitions().items():
        definitions[sid] = defs
        report[sid] = (len(defs), 0)
    return definitions, examples, report


def ocp_definitions():
    """OCP's templates are pipeline-private (no manifest on any core): read them from the copied package,
    with the same labels as the Mac (dump_introspection.OCP_INTENTS)."""
    from switchboard_eval.inputs.dump_introspection import OCP_INTENTS, OCP_SKILL_IDS
    base = os.path.join(SKILLS, "ocp_pipeline", "locale", "en-us")
    out = {}
    for fname, name in OCP_INTENTS.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            sid = OCP_SKILL_IDS.get(name, "ocp")
            out.setdefault(sid, []).append({"skill_id": sid, "intent_name": name, "lang": "en-us",
                                            "method": "template", "definition": {"samples": read_lines(path)}})
    return out


def main():
    definitions, adapt_examples, report = build()
    intents = [{k: d[k] for k in ("skill_id", "intent_name", "lang", "method")} for ds in definitions.values() for d in ds]
    json.dump({"intents": intents, "definitions": definitions},
              open(os.path.join(HERE, "devices/speaker/manifest.json"), "w"), indent=2)
    from switchboard_eval.inputs.harvest_live import harvest
    template_examples, _ = harvest(os.path.join(HERE, "devices/speaker/manifest.json"), "en-us")
    seen, n = set(), 0
    with open(os.path.join(HERE, "devices/speaker/examples.jsonl"), "w") as f:
        for e in template_examples + adapt_examples:
            k = (e["label"], e["example"].lower())
            if k not in seen:
                seen.add(k)
                f.write(json.dumps(e) + "\n")
                n += 1
    for sid, (t, a) in sorted(report.items()):
        print(f"  {sid:45} padatious {t:3}  adapt {a:3}")
    print(f"{len(definitions)} skills, {sum(t for t, _ in report.values())} padatious + "
          f"{sum(a for _, a in report.values())} adapt intents -> devices/speaker/manifest.json; {n} examples -> devices/speaker/examples.jsonl")


if __name__ == "__main__":
    main()
