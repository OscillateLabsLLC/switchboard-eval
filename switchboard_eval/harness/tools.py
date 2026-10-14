#!/usr/bin/env python3
"""Phase 2: LLM tool definitions derived ONLY from the INTENT-4 manifest dump.

No skill changes, no hand-written descriptions: everything comes from what a skill
already registers (skill_id, intent_name, templates, slot names, slot blacklists).
schema/intents.yaml is deliberately NOT read here; upstream has no canonical schema, so the
tools must stand on manifest data alone. schema/intents.yaml maps tool calls back for scoring.

Per registration:
  name         short, API-safe (^[a-zA-Z0-9_-]{1,64}$); `labels` maps it back to skill_id:intent
  description  humanized intent name + skill + up to MAX_EXAMPLES template phrasings
  parameters   one string per {slot}; required iff every template guarantees the slot;
               slot_blacklist becomes "never one of: ..."
Plus META_TOOLS for paths the manifest can't express (knowledge, stop, ignore, unsupported).

Usage: tools.py devices/desktop/manifest.json [--out devices/desktop/tools.json] [--review tools_review.md] [--lang en-us] [--clarify]
"""
from switchboard_eval.paths import ROOT
import json, os, re, sys

sys.path.insert(0, ROOT)
from switchboard_eval.inputs.harvest_examples import expand

MAX_EXAMPLES = 4
MAX_EXPANSIONS_PER_LINE = 50
SLOT = re.compile(r"\{+([a-z_]+)\}+")
SKILL_AFFIXES = re.compile(r"^(ovos-skill-|skill-ovos-|skill-)|\.(openvoiceos|oscillatelabsllc|neongeckocom)$")

META_TOOLS = {
    "ask_knowledge": ("A question or conversational request directed at the assistant that no other "
                      "tool handles: facts, explanations, advice, chit-chat, quotes.",
                      {"question": "the request, as spoken"}, ["question"]),
    "stop": ("Stop whatever the assistant is currently doing: speaking, playing media, an "
             "active timer or alarm sound.", {}, []),
    "unsupported": ("A command clearly directed at the assistant that none of the other tools can "
                    "perform. The assistant will say it can't do that.",
                    {"request": "the request, as spoken"}, ["request"]),
    "ignore": ("The speech was NOT directed at the assistant: background conversation, TV or "
               "radio, the assistant hearing its own voice, or noise. The assistant stays silent.",
               {}, []),
}
# Opt-in (--clarify): measured as a variant, so the frozen devices/desktop/tools.json stays unchanged.
CLARIFY_TOOL = ("clarify", "The speech is directed at the assistant, but the request is too ambiguous or "
                "garbled to act on. The assistant asks one short clarifying question.",
                {"question": "the clarifying question to ask"}, ["question"])


def short_skill(skill_id):
    return re.sub(r"[^a-zA-Z0-9]+", "_", SKILL_AFFIXES.sub("", skill_id)).strip("_")


def humanize(intent_name):
    """'what_time_is_it' / 'HandleCurrentWeather' -> 'what time is it' / 'current weather'."""
    words = re.sub(r"[._-]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", intent_name)).lower().split()
    if len(words) > 1 and words[-1] == "intent":
        words = words[:-1]
    if len(words) > 1 and words[0] == "handle":  # Neon Adapt handler naming
        words = words[1:]
    return " ".join(words)


def guaranteed_slots(template):
    """Slots present in EVERY expansion: [optional] groups add none, (a|b) adds the intersection."""
    out, i = set(), 0
    while i < len(template):
        ch = template[i]
        if ch in "([":
            j = _matching_close(template, i)
            if ch == "(":
                branches = [guaranteed_slots(b) for b in _split_branches(template[i + 1:j])]
                out |= set.intersection(*branches)
            i = j + 1
            continue
        m = SLOT.match(template, i)
        if m:
            out.add(m.group(1))
            i = m.end()
            continue
        i += 1
    return out


def _matching_close(t, start):
    depth = 0
    for j in range(start, len(t)):
        depth += t[j] in "(["
        depth -= t[j] in ")]"
        if depth == 0:
            return j
    raise ValueError(f"unbalanced template: {t!r}")


def _split_branches(inner):
    parts, depth, cur = [], 0, ""
    for ch in inner:
        depth += ch in "(["
        depth -= ch in ")]"
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def example_phrasings(samples):
    """Up to MAX_EXAMPLES phrasings, slots shown as <slot>: start from the shortest expansion,
    then greedily add the candidate sharing the fewest words with those already picked
    (near-duplicates like 'go to next song' / 'go to next track' waste the description)."""
    candidates = {}
    for line in samples:
        for e in expand(line)[:MAX_EXPANSIONS_PER_LINE]:
            e = SLOT.sub(lambda m: f"<{m.group(1)}>", re.sub(r"\s+", " ", e).strip())
            if e:
                candidates.setdefault(e.lower(), e)
    pool = sorted(candidates.values(), key=lambda e: (len(e), e))
    picked = pool[:1]
    while pool and len(picked) < min(MAX_EXAMPLES, len(pool)):
        rest = [e for e in pool if e not in picked]
        picked.append(min(rest, key=lambda e: (max(_overlap(e, p) for p in picked), len(e))))
    return picked


def _overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / len(wa | wb)


def tool_for(skill_id, intent_name, definition):
    samples = definition.get("samples") or []
    slots = sorted({s for line in samples for s in SLOT.findall(line)})
    required = sorted(set.intersection(*(guaranteed_slots(l) for l in samples))) if samples else []
    blacklist = definition.get("slot_blacklist") or {}
    examples = "; ".join(f'"{e}"' for e in example_phrasings(samples))
    props = {}
    for s in slots:
        desc = s.replace("_", " ")
        if blacklist.get(s):
            desc += ". Never one of: " + ", ".join(blacklist[s])
        props[s] = {"type": "string", "description": desc}
    return {"type": "function", "function": {
        "name": f"{short_skill(skill_id)}__{re.sub(r'[^a-zA-Z0-9]+', '_', intent_name)}"[:64],
        "description": f"{humanize(intent_name).capitalize()} ({short_skill(skill_id).replace('_', ' ')} skill). "
                       f"Example requests: {examples}",
        "parameters": {"type": "object", "properties": props, "required": required}}}


def adapt_tool_for(skill_id, intent_name, definition):
    """Adapt intents declare keywords, not phrases or slots: describe them by keyword combinations."""
    examples = "; ".join(f'"{e}"' for e in (definition.get("keyword_examples") or [])[:MAX_EXAMPLES])
    return {"type": "function", "function": {
        "name": f"{short_skill(skill_id)}__{re.sub(r'[^a-zA-Z0-9]+', '_', intent_name)}"[:64],
        "description": f"{humanize(intent_name).capitalize()} ({short_skill(skill_id).replace('_', ' ')} skill). "
                       f"Triggered by keywords such as: {examples}",
        "parameters": {"type": "object", "properties": {}, "required": []}}}


def meta_tool(name, desc, params, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "required": required,
                       "properties": {k: {"type": "string", "description": v} for k, v in params.items()}}}}


def build(dump_path, lang="en-us", clarify=False):
    tools, labels = [], {}
    for skill_id, defs in json.load(open(dump_path))["definitions"].items():
        for d in defs if isinstance(defs, list) else []:
            if d["lang"].lower() != lang or d["method"] not in ("template", "adapt"):
                continue
            make = adapt_tool_for if d["method"] == "adapt" else tool_for
            t = make(skill_id, d["intent_name"], d["definition"])
            name = t["function"]["name"]
            if name in labels:
                raise ValueError(f"tool name collision: {name} for {labels[name]} and {skill_id}:{d['intent_name']}")
            labels[name] = f"{skill_id}:{d['intent_name']}"
            tools.append(t)
    metas = list(META_TOOLS.items()) + ([(CLARIFY_TOOL[0], CLARIFY_TOOL[1:])] if clarify else [])
    for name, spec in metas:
        tools.append(meta_tool(name, *spec))
        labels[name] = f"meta:{name}"
    return tools, labels


def review_markdown(tools, labels):
    lines = ["# Auto-generated LLM tools (from the INTENT-4 manifest only)", "",
             f"{len(tools)} tools. Regenerate with `python -m switchboard_eval.harness.tools devices/desktop/manifest.json`.", "",
             "| tool | OVOS label | params (* = required) | description |", "|---|---|---|---|"]
    for t in tools:
        f = t["function"]
        req = set(f["parameters"]["required"])
        params = ", ".join(p + ("*" if p in req else "") for p in f["parameters"]["properties"]) or "—"
        lines.append(f"| `{f['name']}` | `{labels[f['name']]}` | {params} | {f['description'].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main(argv):
    if not argv:
        sys.exit(__doc__)
    opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
    tools, labels = build(argv[0], opt("--lang", "en-us").lower(), "--clarify" in argv)
    out, review = opt("--out", "devices/desktop/tools.json"), opt("--review", "tools_review.md")
    json.dump({"tools": tools, "labels": labels}, open(out, "w"), indent=2)
    open(review, "w").write(review_markdown(tools, labels))
    print(f"{len(tools)} tools ({len(META_TOOLS)} meta) -> {out}, review table -> {review}")


if __name__ == "__main__":
    main(sys.argv[1:])
