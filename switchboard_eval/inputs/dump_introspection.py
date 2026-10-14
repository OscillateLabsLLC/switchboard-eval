#!/usr/bin/env python3
"""Dump the live INTENT-4 manifest (ovos.intent.list + per-skill ovos.intent.describe).

Read-only: both topics are pull-queries served by ovos-core's IntentManifest
(verified in 3.7.4a1). The manifest is in-memory and only holds registrations
broadcast after core started, so restart skills/core first if counts look short.

Usage: dump_introspection.py devices/desktop/manifest.json [--host H] [--lang en-us]
Requires: pip install ovos-bus-client
"""
import json, sys

TIMEOUT_S = 15
# Pipeline plugins register intents privately, so the INTENT-4 manifest never lists them
# (verified for OCP in core 3.7.4a1). Their .intent files are still declared phrases.
# file -> the label OCP emits on the probe path (must match schema/intents.yaml mappings).
OCP_INTENTS = {"play.intent": "play", "pause.intent": "pause", "next.intent": "next",
               "prev.intent": "prev", "resume.intent": "resume", "media_stop.intent": "stop"}
OCP_SKILL_IDS = {"stop": "ovos.common_play"}  # media_stop dispatches as ovos.common_play:stop


def ocp_definitions(lang="en-us"):
    """OCP's own templates in manifest shape, keyed by the skill_id each label uses."""
    import os
    import ocp_pipeline
    base = os.path.join(os.path.dirname(ocp_pipeline.__file__), "locale", lang)
    out = {}
    for fname, name in OCP_INTENTS.items():
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        samples = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        sid = OCP_SKILL_IDS.get(name, "ocp")
        out.setdefault(sid, []).append({
            "skill_id": sid, "intent_name": name, "lang": lang, "method": "template",
            "session_id": "plugin", "definition": {"samples": samples, "source": f"ocp_pipeline/{fname}"}})
    return out


def query(bus, topic, data):
    from ovos_bus_client.message import Message
    resp = bus.wait_for_response(Message(topic, data), reply_type=f"{topic}.response",
                                 timeout=TIMEOUT_S)
    if resp is None:
        sys.exit(f"no reply to {topic} {data} within {TIMEOUT_S}s")
    return resp.data


def main():
    from ovos_bus_client import MessageBusClient  # lazy: OCP_INTENTS is reused without the bus client
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "127.0.0.1"
    lang = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else None
    out = args[0]

    bus = MessageBusClient(host=host)
    bus.run_in_thread()
    bus.connected_event.wait(10)

    listing = query(bus, "ovos.intent.list", {"lang": lang} if lang else {})
    if not listing.get("ok"):
        sys.exit(f"ovos.intent.list failed: {listing}")
    skills = sorted({i["skill_id"] for i in listing["intents"]})

    described = {}
    for skill_id in skills:
        reply = query(bus, "ovos.intent.describe",
                      {"skill_id": skill_id, **({"lang": lang} if lang else {})})
        described[skill_id] = reply.get("definitions") if reply.get("ok") else reply
    bus.close()

    for sid, defs in ocp_definitions().items():
        described[sid] = defs  # not in the manifest; see OCP_INTENTS
    json.dump({"intents": listing["intents"], "definitions": described},
              open(out, "w"), indent=2, default=str)
    methods = {}
    for i in listing["intents"]:
        methods[i["method"]] = methods.get(i["method"], 0) + 1
    print(f"{len(listing['intents'])} registrations across {len(skills)} skills {methods} -> {out}")


if __name__ == "__main__":
    main()
