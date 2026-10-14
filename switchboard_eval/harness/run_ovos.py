#!/usr/bin/env python3
"""Run data/dataset.jsonl through a live OVOS stack WITHOUT executing intents.

Uses the intent *query* path (intent.service.intent.get -> .reply), which asks
the pipeline what it would match and does not dispatch to skills. Never
benchmark by emitting recognizer_loop:utterance: that path executes handlers,
and this dataset contains "unlock the front door" and "open the garage".

Verified against ovos-core 3.7.4a1 source (IntentService.handle_get_intent):
  - Topics and reply shape match QUERY/REPLY and extract() below. Still
    confirm with --debug on a live bus.
  - The probe walks the FULL session pipeline (stop, converse, fallback,
    common_query, persona...), calling each match() directly. In 3.7.4a1,
    stop/converse/fallback match() only ping skills and never dispatch.
  - It differs from live handle_utterance: no utterance transformers, no
    match timeout, no blacklist / required-slot / context-gate checks, no
    multilingual retry. The baseline can disagree with live use on those.
  - It uses the default session, so active-skill state on the live device
    leaks into converse/stop results. Run with the device idle.

--stock runs the probe on ovos-config's shipped default pipeline via a named
session, instead of whatever pipeline the device is configured with. The live
config is not touched. Use it to measure "out of the box" OVOS.

Usage: run_ovos.py data/dataset.jsonl schema/intents.yaml preds_ovos.jsonl [--host H] [--debug] [--stock]
Requires: pip install ovos-bus-client pyyaml
"""
import json, sys, time, yaml

QUERY, REPLY = "intent.service.intent.get", "intent.service.intent.reply"
# Keys handle_get_intent (ovos-core 3.7.4a1) adds next to the flattened match_data.
REPLY_META = {"intent_name", "intent_service", "skill_id", "handler"}
NON_SLOT_KEYS = REPLY_META | {"utterance", "utterances", "lang", "target", "__tags__",
                              "conf", "confidence"}


def build_map(schema):
    """label -> (canonical intent, {ovos_slot: canonical_slot}); declined labels map to None."""
    doc = yaml.safe_load(open(schema))
    m = {label: (None, {}) for label in doc.get("declined") or []}
    for canon, spec in doc["intents"].items():
        for e in spec.get("ovos") or []:
            label, ren = (e, {}) if isinstance(e, str) else (e["label"], e.get("slots") or {})
            m[label] = (canon, ren)
            m[label + ".intent"] = (canon, ren)  # legacy topic name vs INTENT-4 canonical
    return m


def extract(reply_data):
    """Pull (ovos_label, slots, confidence, stage) out of the reply.

    ovos-core 3.7.4a1 replies {"intent": {**match_data, intent_name, intent_service,
    skill_id, handler}, "utterance": ...}, or {"intent": None} on no match.
    Slots are flattened into the intent dict, not nested.
    """
    intent = (reply_data or {}).get("intent")
    if not intent:
        return None, {}, None, None
    label = intent.get("intent_name")
    slots = {k: v for k, v in intent.items()
             if isinstance(v, str) and k not in NON_SLOT_KEYS}
    conf = intent.get("conf", intent.get("confidence"))
    conf = conf if isinstance(conf, (int, float)) else None
    return label, slots, conf, intent.get("intent_service")


def ask(bus, utterance, context, timeout):
    """Send one probe and wait for ITS reply only.

    Every probe carries a unique context id, which core's message.reply() preserves.
    Without this, a late reply to a timed-out probe gets recorded as the NEXT row's answer
    (observed after a core restart), silently misaligning the run.
    """
    import threading, uuid
    from ovos_bus_client.message import Message
    req_id = str(uuid.uuid4())
    got, box = threading.Event(), {}

    def on_reply(msg):
        if msg.context.get("intent_eval_req") == req_id:
            box["data"] = msg.data
            got.set()

    bus.on(REPLY, on_reply)
    try:
        bus.emit(Message(QUERY, {"utterance": utterance, "lang": "en-us"},
                         {**context, "intent_eval_req": req_id}))
        got.wait(timeout)
    finally:
        bus.remove(REPLY, on_reply)
    return box.get("data")


def stock_pipeline():
    """The intent pipeline ovos-config ships in its default mycroft.conf."""
    import os, re
    import ovos_config
    path = os.path.join(os.path.dirname(ovos_config.__file__), "mycroft.conf")
    text = re.sub(r"(?m)^\s*//.*$", "", open(path).read())  # mycroft.conf allows // comments
    return json.loads(text)["intents"]["pipeline"]


def main():
    from ovos_bus_client import MessageBusClient  # lazy: build_map is reused from the bench venv
    args =[a for a in sys.argv[1:] if not a.startswith("--")]
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "127.0.0.1"
    debug = "--debug" in sys.argv
    context = {}
    if "--stock" in sys.argv:
        context = {"session": {"session_id": "intent-eval-stock", "lang": "en-us",
                               "pipeline": stock_pipeline()}}
        print("stock pipeline:", context["session"]["pipeline"])
    dataset, schema, out = args[:3]
    mapping = build_map(schema)
    if not mapping:
        print("WARNING: no ovos: labels in schema/intents.yaml; every match will be unmapped")

    bus = MessageBusClient(host=host)
    bus.run_in_thread()
    bus.connected_event.wait(10)

    unmapped, timeouts = {}, 0
    with open(out, "w") as f:
        for n, line in enumerate(open(dataset)):
            row = json.loads(line)
            t0 = time.perf_counter()
            data = ask(bus, row["utterance"], context, timeout=15)
            ms = (time.perf_counter() - t0) * 1000
            if debug and n < 5:
                print(json.dumps(data, indent=2, default=str))
            if data is None:
                timeouts += 1
            label, slots, conf, stage = extract(data)
            canon, ren = mapping.get(label, (None, {})) if label else (None, {})
            if label and label not in mapping:
                unmapped[label] = unmapped.get(label, 0) + 1
                canon = f"__unmapped__:{label}"
            slots = {ren.get(k, k): v for k, v in slots.items()}
            f.write(json.dumps({"id": row["id"], "intent": canon, "slots": slots,
                                "confidence": conf, "latency_ms": round(ms, 1),
                                "stage": stage}) + "\n")
    bus.close()
    if timeouts:
        print(f"WARNING: {timeouts} probes timed out (recorded as no match)")
    if unmapped:
        print("Unmapped OVOS labels (add to schema/intents.yaml `ovos:` lists, then rerun):")
        for k, v in sorted(unmapped.items(), key=lambda x: -x[1]):
            print(f"  {v:>3}  {k}")


if __name__ == "__main__":
    main()
