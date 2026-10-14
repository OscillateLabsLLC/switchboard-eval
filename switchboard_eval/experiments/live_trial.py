#!/usr/bin/env python3
"""LIVE trial driver. WARNING: this EXECUTES skill handlers on the running OVOS stack, exactly as if spoken.

Breaks the harness guardrail on purpose and only with the owner's explicit consent per run (Mike, 2026-10-13: "I give
you consent to pass the intents on my behalf"; lamps, timers and alarms allowed if reverted). Never put high-risk
phrases (unlock, open the garage, shut down, clear data) in a script.

Sends each utterance as the listener does (recognizer_loop:utterance, default session), waits for the assistant to
finish speaking (or a quiet timeout), and prints what OVOS matched and said, with three timings from the send:
  decision  ovos.intent.matched (the pipeline's choice, Switchboard included)
  text      the first `speak` message (adds the skill handler)
  audio     recognizer_loop:audio_output_start (adds text-to-speech)
Switchboard's own per-step times are in its decision log.

Usage: ~/.venvs/ovos/bin/python -m switchboard_eval.experiments.live_trial <utterances.txt> [--repeat N] [--out results.jsonl]
       (one utterance per line; '#' lines are comments)
"""
import json
import statistics
import sys
import threading
import time

from ovos_bus_client import MessageBusClient
from ovos_bus_client.message import Message


QUIET_S = 6.0       # no speech starting within this long after an utterance: move on
SPEECH_MAX_S = 45.0  # cap per utterance (a long joke or forecast)
SETTLE_S = 2.0       # after speech ends, before the next utterance


class Watcher:
    """Collects what happens on the bus for the current utterance, timed from the send."""

    def __init__(self, bus):
        self.events, self.t0, self.audio_at = [], 0.0, None
        self.speaking, self.done = threading.Event(), threading.Event()
        for topic in ("ovos.intent.matched", "complete_intent_failure", "speak"):
            bus.on(topic, self._record(topic))
        bus.on("recognizer_loop:audio_output_start", self._on_audio_start)
        bus.on("recognizer_loop:audio_output_end", lambda _m: self.done.set())

    def _record(self, topic):
        def handler(message):
            self.events.append((topic, message.data, self._ms()))
        return handler

    def _on_audio_start(self, _message):
        if self.audio_at is None:
            self.audio_at = self._ms()
        self.speaking.set()

    def _ms(self):
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def reset(self):
        """Forget the previous utterance and start the clock."""
        self.events.clear()
        self.audio_at = None
        self.speaking.clear()
        self.done.clear()
        self.t0 = time.perf_counter()

    def wait(self):
        """Until speech ends, or nothing was said within QUIET_S."""
        if self.speaking.wait(QUIET_S):
            self.done.wait(SPEECH_MAX_S)
        time.sleep(SETTLE_S)

    def result(self, utterance):
        """The utterance's outcome and timings."""
        matched = next(((d, t) for topic, d, t in self.events if topic == "ovos.intent.matched"), (None, None))
        spoke = next((t for topic, _, t in self.events if topic == "speak"), None)
        data, decision_ms = matched
        return {"utterance": utterance, "intent": (data or {}).get("intent_name"),
                "pipeline": (data or {}).get("pipeline_id"), "decision_ms": decision_ms, "text_ms": spoke,
                "audio_ms": self.audio_at, "said": [d.get("utterance") for topic, d, _ in self.events
                                                    if topic == "speak"]}


def describe(r):
    """Lines for one result."""
    step = (r["pipeline"] or "").removeprefix("ovos-").removesuffix("-pipeline-plugin")
    lines = [f"matched: {r['intent'] or 'nothing'} via {step or '-'}  "
             f"[decision {r['decision_ms']} ms, text {r['text_ms']} ms, audio {r['audio_ms']} ms]"]
    return lines + [f"said: {s!r}" for s in r["said"]]


def summarize(results):
    """Median timings per utterance."""
    print("\n\nmedians (ms from send)        decision      text     audio   n  step")
    by = {}
    for r in results:
        by.setdefault(r["utterance"], []).append(r)
    for utterance, rs in by.items():
        def med(key, rows=rs):
            vals = [r[key] for r in rows if r[key] is not None]
            return f"{statistics.median(vals):8.0f}" if vals else "       -"
        steps = sorted({(r["pipeline"] or "-").removeprefix("ovos-").removesuffix("-pipeline-plugin") for r in rs})
        print(f"{utterance[:30]:30} {med('decision_ms')}  {med('text_ms')}  {med('audio_ms')}  {len(rs):2}  "
              f"{','.join(steps)}")


def main(argv):
    path = argv[0]
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 1
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    with open(path, encoding="utf-8") as f:
        utterances = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    bus = MessageBusClient()
    bus.run_in_thread()
    bus.connected_event.wait(10)
    watcher, results = Watcher(bus), []
    for round_no in range(repeat):
        for utterance in utterances:
            print(f"\n>>> {utterance}  [{time.strftime('%H:%M:%S')}, round {round_no + 1}/{repeat}]", flush=True)
            watcher.reset()
            bus.emit(Message("recognizer_loop:utterance", {"utterances": [utterance], "lang": "en-US"},
                             {"session": {"session_id": "default"}}))
            watcher.wait()
            results.append({**watcher.result(utterance), "round": round_no + 1})
            for line in describe(results[-1]):
                print(f"    {line}", flush=True)
    bus.close()
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    summarize(results)


if __name__ == "__main__":
    main(sys.argv[1:])
