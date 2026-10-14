#!/usr/bin/env python3
"""A device manifest for any language, built from the installed skills' locale files.

Same shape as devices/desktop/manifest.json (the live English manifest), so the router builds from it unchanged. Every language
comes from the same inputs (the skills installed in a venv), which is what makes cross-language comparisons fair: the
live manifest only ever has the running language. Templates only (`*.intent`); Adapt vocab is not read.

Usage: <ovos venv python> build_lang_manifest.py <lang> <out.json>     e.g. gl-es skills/manifests/gl-es.json
       ... build_lang_manifest.py <lang> <out.json> --repos <dir> [--orgs Org1,Org2] [--no-installed]
                                  [--exclude skill_id,skill_id]   (skills that are never installed together)
--repos adds skills cloned under <dir> as <Org>__<repo> folders, read statically (no code is imported or run), with
the skill_id from their packaging metadata or "<repo>.<org>"; installed skills win on duplicate skill_ids.
"""
import glob
import json
import os
import re
import sys
from importlib.metadata import entry_points


SKILL_GROUPS = ("ovos.plugin.skill", "opm.skill")


def installed_skills():
    """{skill_id: package directory} for every installed skill plugin."""
    import importlib.util
    skills = {}
    for group in SKILL_GROUPS:
        for ep in entry_points(group=group):
            module = ep.value.split(":")[0]
            spec = importlib.util.find_spec(module.split(".")[0])
            if spec and spec.submodule_search_locations:
                skills[ep.name] = list(spec.submodule_search_locations)[0]
    return skills


def samples(path):
    """Template lines of one .intent file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


ENTRY_POINT = re.compile(r"""["']?([a-z0-9][\w.-]*\.[a-z0-9_-]+)["']?\s*=\s*["']?[\w.]+:\w+""", re.I)


def repo_skill_id(repo_dir, org, repo):
    """The skill_id a cloned skill would register: its plugin entry point name, else "<repo>.<org>"."""
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        path = os.path.join(repo_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if "skill" in text:
                m = ENTRY_POINT.search(text)
                if m:
                    return m.group(1).lower()
    return f"{repo}.{org}".lower()


def cloned_skills(root, orgs=None):
    """{skill_id: repo directory} for <Org>__<repo> clones under `root`."""
    skills = {}
    for d in sorted(glob.glob(os.path.join(root, "*__*"))):
        org, repo = os.path.basename(d).split("__", 1)
        if orgs is None or org in orgs:
            skills.setdefault(repo_skill_id(d, org, repo), d)
    return skills


def intent_files(base_dir, lang):
    """`lang`'s .intent files anywhere under `base_dir`: locale/<lang>/ (current) or vocab/<lang>/ (Mycroft-era)."""
    found = []
    for kind in ("locale", "vocab"):
        for folder in glob.glob(os.path.join(base_dir, "**", kind, "*"), recursive=True):
            if os.path.basename(folder).lower() == lang.lower() and "/test" not in folder:
                found += glob.glob(os.path.join(folder, "**", "*.intent"), recursive=True)
    return sorted(set(found))


TEMPLATE_REPOS = ("template", "cookiecutter")
DUPLICATE_SHARE = 0.8  # a clone whose intents are mostly already present is the same skill under another id


def manifest(lang, repos=None, orgs=None, installed=True, exclude=()):
    """{"definitions": {skill_id: [definition, ...]}} for `lang`. Installed skills first; a cloned skill is skipped
    when it is a skill template repo or a duplicate of one already added (same intents under another skill_id)."""
    sources = list(installed_skills().items()) if installed else []
    if repos:
        sources += [(sid, d) for sid, d in cloned_skills(repos, orgs).items()
                    if not any(t in os.path.basename(d).lower() for t in TEMPLATE_REPOS)]
    definitions, seen = {}, set()
    for skill_id, base_dir in sources:
        if skill_id in definitions or skill_id in exclude:
            continue
        defs = skill_definitions(skill_id, base_dir, lang)
        sigs = {(d["intent_name"], tuple(d["definition"]["samples"][:20])) for d in defs}
        if defs and len(sigs & seen) < DUPLICATE_SHARE * len(sigs):
            definitions[skill_id] = defs
            seen |= sigs
    return {"definitions": dict(sorted(definitions.items()))}


def skill_definitions(skill_id, base_dir, lang):
    """One skill's template definitions for `lang`."""
    defs = []
    for path in intent_files(base_dir, lang):
        name = os.path.basename(path).removesuffix(".intent")
        lines = samples(path)
        if lines:
            defs.append({"skill_id": skill_id, "intent_name": name, "lang": lang, "method": "template",
                         "definition": {"skill_id": skill_id, "intent_name": name, "lang": lang, "samples": lines}})
    return defs


def main(argv):
    lang, out = argv[0], argv[1]
    repos = argv[argv.index("--repos") + 1] if "--repos" in argv else None
    orgs = set(argv[argv.index("--orgs") + 1].split(",")) if "--orgs" in argv else None
    exclude = set(argv[argv.index("--exclude") + 1].split(",")) if "--exclude" in argv else set()
    doc = manifest(lang.lower(), os.path.expanduser(repos) if repos else None, orgs, "--no-installed" not in argv,
                   exclude)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    n = sum(len(v) for v in doc["definitions"].values())
    lines = sum(len(d["definition"]["samples"]) for v in doc["definitions"].values() for d in v)
    print(f"{lang}: {len(doc['definitions'])} skills, {n} intents, {lines} template lines -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
