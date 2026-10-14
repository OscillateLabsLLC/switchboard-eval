#!/usr/bin/env python3
"""Experiment: guard rules for catch-all templates in the exact layer (follow-up to exact_audit.py, exp_exact_guard.py).

A catch-all is a template with a slot and fewer than 2 literal words ("play {query}", "call {contact}",
"timer [for] {duration}", "{schedkind} later"). Every non-command exact dispatch in the audit came from one.
Variables, on top of the word-boundary matcher (switchboard fix/exact-guard):
  capK    a catch-all's slot matches at most K words. Daniel McKnight (Neon): Padatious, undocumented, matches only
          one word into {contact}, which is why Neon's "Call {contact}" mostly works there.
  yieldT  a catch-all match steps aside (exact passes, the rest of the pipeline decides) when the classifier ranks
          a DIFFERENT intent (or the decline class) first with confidence >= T. Lead from exp_exact_guard: catch-all
          steals had classifier confidence 0.95-0.99 elsewhere; real catch-all commands didn't.
Slot-free templates and templates with 2+ literal words are untouched.

Selection rule, fixed before results: highest dev own-device accuracy averaged over mac and hub, subject to dev magic
>= base and dev FA <= base on BOTH devices and zero high-risk executions; ties -> fewer non-command exact dispatches.
Test is then reported for base and the chosen variant. All dev cells are printed for transparency.

Readouts: bar per split/device (active route; LLM cache misses asked of the LLM server and cached), non-command exact
dispatches (exact_audit corpora), changed decisions on the 508 real Mac transcripts (LLM off) for the chosen variant.

Usage (from intent-eval, in .venv-bench, switchboard checkout on fix/exact-guard or later):
    PYTHONPATH=<switchboard> python -m switchboard_eval.experiments.exp_guard_rules <switchboard>/scripts
"""
from switchboard_eval.paths import ROOT
import json
import os
import sys

from switchboard_eval.experiments.exact_audit import negative_corpora
from switchboard_eval.experiments.exp_mac_draft import build, describe


CAPS = (None, 1, 2, 3)
YIELDS = (None, 0.8, 0.9, 0.95)
ANY_SLOT = "(?:.+?)"


def capped(regex, k):
    """`regex` with each slot limited to k words."""
    import re
    return re.compile(regex.pattern.replace(ANY_SLOT, r"(?:\S+(?:\s+\S+){0,%d})" % (k - 1)))


def split_templates(layer, cap):
    """(strong, catch_all) claimed templates; catch-all slots capped to `cap` words."""
    from ovos_switchboard.core.templates import literal_words, template_regex
    strong, catch = [], []
    for label, line in layer._entries:  # pylint: disable=protected-access
        n = literal_words(line)
        if n < 1:
            continue
        rx = template_regex(line)
        if "{" in line and n < 2:
            catch.append((label, capped(rx, cap) if cap else rx))
        else:
            strong.append((label, rx))
    return strong, catch


def guard(router, cap, tau):
    """Replace the router's exact decision with the guarded one."""
    from ovos_switchboard.core.stages import passed, resolved
    from ovos_switchboard.core.verdict import Outcome  # noqa: F401  (imported for parity with stages)
    layer = router.exact.exact
    strong, catch = split_templates(layer, cap)
    schema = layer.schema

    def decide(utterance, _context=None):
        u = utterance.lower()
        hit_strong = {schema.canon_or_unmapped(lb) for lb, rx in strong if rx.match(u)}
        owners = hit_strong | {schema.canon_or_unmapped(lb) for lb, rx in catch if rx.match(u)}
        if len(owners) != 1:
            return passed("exact", "no unique template owner")
        owner = owners.pop()
        if not hit_strong and tau is not None:
            top, conf = router.classifier.classifier.rank(u)[0]
            if top != owner and conf >= tau:
                return passed("exact", "catch-all yields", details={"yield_to": top, "confidence": conf})
        return resolved("exact", owner, utterance, router.resolver, loose_slots=False, confidence=1.0)

    router.exact.decide = decide
    return router


def name(cap, tau):
    """Variant name."""
    parts = ([f"cap{cap}"] if cap else []) + ([f"yield{tau}"] if tau else [])
    return "+".join(parts) or "base"


def bar_cells(ch, cp, harness, schema, split, routers):
    """{variant: {device: metrics}} and {variant: {device: preds}} for one split."""
    gold, slices, stock = cp.setup(split)
    _, _, _, ctx = cp.load_inputs(split)
    metrics, preds = {}, {}
    for cap in CAPS:
        for tau in YIELDS:
            v = name(cap, tau)
            metrics[v], preds[v] = {}, {}
            for device, router in routers.items():
                guard(router, cap, tau)
                p = {rid: router.route(r["utterance"], {"gap_s": ctx.get(rid, cp.COLD)["gap_s"],
                                                        "echo_run": ctx.get(rid, cp.COLD)["echo_run"]})
                     for rid, r in gold.items()}
                metrics[v][device], preds[v][device] = cp.evaluate(gold, slices, stock, p), p
    return gold, metrics, preds


def negatives(harness, routers):
    """{variant: {device: non-command exact dispatches}}."""
    corpora = {d: negative_corpora(harness, d) for d in routers}
    out = {}
    for cap in CAPS:
        for tau in YIELDS:
            v = name(cap, tau)
            out[v] = {}
            for device, router in routers.items():
                guard(router, cap, tau)
                utts = {u.lower().strip() for c in corpora[device].values() for u in c if u.strip()}
                out[v][device] = sorted(u for u in utts if router.exact.decide(u).intent)
    return out


def choose(dev, negs):
    """The selection rule (module docstring)."""
    base = dev["base"]
    ok = [v for v, m in dev.items() if all(m[d]["magic"] >= base[d]["magic"] and m[d]["fa"] <= base[d]["fa"]
                                           and m[d]["hx"] == 0 for d in m)]
    return max(ok, key=lambda v: (sum(dev[v][d][d] for d in dev[v]) / len(dev[v]),
                                  -sum(len(negs[v][d]) for d in negs[v])))


def fmt(m, device):
    """magic/own/FA/hx."""
    return f"{m['magic']:5.1f}/{m[device]:5.1f}/{m['fa']:4.1f}/{m['hx']}"


def main(scripts_dir):
    sys.path.insert(0, scripts_dir)
    import check_harness as ch
    from switchboard_eval.harness import context_policy as cp
    from ovos_switchboard.core import templates
    from ovos_switchboard.core.schema import Schema

    if templates.WORD_GAP == r"\s*":
        raise SystemExit("needs the word-boundary matcher (switchboard fix/exact-guard)")
    harness = ROOT
    os.chdir(harness)
    schema = Schema.load(os.path.join(harness, "schema/intents.yaml"))
    routers = {d: build(ch, harness, d, schema) for d in ("mac", "hub")}

    negs = negatives(harness, routers)
    _, dev, _ = bar_cells(ch, cp, harness, schema, "dev", routers)
    print("DEV  variant              mac magic/own/FA/hx        hub magic/own/FA/hx        non-command exact mac/hub")
    for v, m in dev.items():
        print(f"     {v:20} {fmt(m['mac'], 'mac'):26} {fmt(m['hub'], 'hub'):26} "
              f"{len(negs[v]['mac'])}/{len(negs[v]['hub'])}")
    chosen = choose(dev, negs)
    print(f"\nchosen on dev: {chosen}")

    gold, test, tpreds = bar_cells(ch, cp, harness, schema, "test", routers)
    for v in ("base", chosen):
        print(f"TEST {v:20} {fmt(test[v]['mac'], 'mac'):26} {fmt(test[v]['hub'], 'hub'):26}")
    for device in routers:
        for rid, r in gold.items():
            b, n = tpreds["base"][device][rid], tpreds[chosen][device][rid]
            if b.get("intent") != n.get("intent"):
                print(f"  test {device} {rid} {r['utterance']!r} gold {r['intent']}\n"
                      f"      base:   {describe(b)}\n      chosen: {describe(n)}")
    print(f"\nnon-command exact dispatches, chosen ({chosen}):")
    for device in routers:
        print(f"  {device}: {negs[chosen][device]}")
    print(f"removed vs base: { {d: sorted(set(negs['base'][d]) - set(negs[chosen][d])) for d in routers} }")

    real(harness, routers["mac"], chosen)


def real(harness, router, chosen):
    """Changed decisions on the archived real Mac transcripts, LLM off, base vs chosen."""
    with open(os.path.join(harness, "data", "mac", "context.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    llm, router.llm = router.llm, None
    variants = {"base": (None, None), chosen: parse(chosen)}
    out = {}
    for v, (cap, tau) in variants.items():
        guard(router, cap, tau)
        out[v] = [router.route(r["text"], {"gap_s": r["gap_s"] if r["gap_s"] is not None else 1e9,
                                           "echo_run": r["echo_run"]}) for r in rows]
    router.llm = llm
    print(f"\nreal Mac transcripts ({len(rows)}), base vs {chosen}, LLM off:")
    for r, b, n in zip(rows, out["base"], out[chosen]):
        if (b.get("intent"), b.get("stage")) != (n.get("intent"), n.get("stage")):
            print(f"  {r['text']!r}\n      base:   {describe(b)}\n      chosen: {describe(n)}")


def parse(v):
    """(cap, tau) from a variant name."""
    cap = tau = None
    for part in v.split("+"):
        if part.startswith("cap"):
            cap = int(part[3:])
        elif part.startswith("yield"):
            tau = float(part[5:])
    return cap, tau


if __name__ == "__main__":
    main(sys.argv[1])
