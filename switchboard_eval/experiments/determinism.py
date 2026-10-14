#!/usr/bin/env python3
"""Criterion 2: does stock OVOS route the same phrase the same way across restarts?

Runs the --stock baseline after each of:
  A1, A2  plain ovos-core restart (Padatious intent cache kept)
  B1, B2  restart with the intent cache moved aside, forcing Padatious to RETRAIN
          (what happens after any skill install/update)
The original cache is moved back at the end, even on failure. Then reports rows whose
predicted intent differs between runs.

--settle N waits N seconds after Padatious first answers, so a background recompile
(padaos "serving last compiled state") finishes. Without it, B runs mix two effects:
transient answers during compile, and a genuinely different retrained model.

Usage: determinism.py [--runs A1,A2,B1,B2] [--settle 180]   (run with the OVOS venv python)
"""
from switchboard_eval.paths import ROOT
import json, os, shutil, subprocess, sys, time

HERE = ROOT
CACHE = os.path.expanduser("~/.local/share/mycroft/intent_cache")
LABEL = f"gui/{os.getuid()}/com.ovos.service"
READY_PROBE_TIMEOUT_S = 900
READY_PHRASE = "what time is it"


def restart_core():
    subprocess.run(["launchctl", "kickstart", "-k", LABEL], check=True)


def wait_until_padatious_answers():
    """Ready = the stock session routes a known phrase through Padatious."""
    from ovos_bus_client import MessageBusClient
    from ovos_bus_client.message import Message
    sys.path.insert(0, HERE)
    from switchboard_eval.harness.run_ovos import stock_pipeline
    ctx = {"session": {"session_id": "intent-eval-ready", "lang": "en-us", "pipeline": stock_pipeline()}}
    deadline = time.time() + READY_PROBE_TIMEOUT_S
    while time.time() < deadline:
        try:
            bus = MessageBusClient()
            bus.run_in_thread()
            if bus.connected_event.wait(5):
                r = bus.wait_for_response(Message("intent.service.intent.get",
                                                  {"utterance": READY_PHRASE, "lang": "en-us"}, ctx),
                                          reply_type="intent.service.intent.reply", timeout=10)
                bus.close()
                stage = ((r.data if r else {}).get("intent") or {}).get("intent_service", "")
                if "padatious" in stage:
                    return
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError(f"Padatious never answered '{READY_PHRASE}' within {READY_PROBE_TIMEOUT_S}s")


def run_stock(tag):
    out = os.path.join(HERE, f"data/baselines/baseline_stock_{tag}.jsonl")
    subprocess.run([sys.executable, os.path.join(HERE, "run_ovos.py"), os.path.join(HERE, "data/dataset.jsonl"),
                    os.path.join(HERE, "schema/intents.yaml"), out, "--stock"], check=True,
                   stdout=subprocess.DEVNULL)
    return out


def flips(paths):
    runs = [{r["id"]: r["intent"] for r in map(json.loads, open(p))} for p in paths]
    gold = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "data/dataset.jsonl")))}
    return [(i, gold[i]["utterance"], [run[i] for run in runs])
            for i in runs[0] if len({run[i] for run in runs}) > 1]


def main():
    runs = (sys.argv[sys.argv.index("--runs") + 1] if "--runs" in sys.argv else "A1,A2,B1,B2").split(",")
    settle = int(sys.argv[sys.argv.index("--settle") + 1]) if "--settle" in sys.argv else 0
    aside = CACHE + f".intent-eval-backup-{int(time.time())}"
    moved_original = False
    outs = {}
    try:
        for tag in runs:
            if tag.startswith("B"):
                if not moved_original:
                    shutil.move(CACHE, aside)
                    moved_original = True
                elif os.path.exists(CACHE):
                    shutil.rmtree(CACHE)  # a cache THIS script trained in the previous B run
            t0 = time.time()
            restart_core()
            time.sleep(15)
            wait_until_padatious_answers()
            time.sleep(settle)
            outs[tag] = run_stock(tag)
            print(f"{tag}: ready+run in {time.time() - t0:.0f}s -> {outs[tag]}", flush=True)
    finally:
        if moved_original:
            if os.path.exists(CACHE):
                shutil.rmtree(CACHE)
            shutil.move(aside, CACHE)
            restart_core()
            print("original intent cache restored; core restarted", flush=True)

    groups = {"A (restart only)": [outs[t] for t in runs if t.startswith("A")],
              "B (retrain)": [outs[t] for t in runs if t.startswith("B")],
              "all runs": list(outs.values())}
    for name, paths in groups.items():
        if len(paths) < 2:
            continue
        f = flips(paths)
        print(f"\n{name}: {len(f)} rows flipped across {len(paths)} runs")
        for i, utt, preds in f[:25]:
            print(f"  {utt[:44]!r:46} {preds}")


if __name__ == "__main__":
    main()
