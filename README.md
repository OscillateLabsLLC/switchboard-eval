# switchboard-eval

The evaluation harness for [Switchboard](https://github.com/OscillateLabsLLC/switchboard), a measured replacement
for OVOS intent matching. Everything Switchboard's docs claim was measured here; this repo lets you check it.

## What's in it

- `data/dataset.jsonl`: 517 labeled utterances, split into dev and test. 259 were written by hand to probe specific
  cases (paraphrases, near misses, misheard speech, things not meant for the assistant). 258 are real requests and
  overheard speech from one household's two devices: a Neon smart speaker and an OVOS desktop.
- `data/negatives_ext.jsonl`: 400 sentences never addressed to an assistant, from public corpora, to measure how
  often it acts on overheard speech.
- `devices/`: what each device has installed (its skills' intents, example phrases, LLM tools).
- `schema/intents.yaml`: the shared intents every device's skills map onto, with risk levels.
- `switchboard_eval/`: the scoring code (`harness/`), the experiments (`experiments/`), and the scripts that
  regenerate the inputs (`inputs/`).
- `cache/llm_cache.jsonl`: every LLM answer the results used, so they reproduce without a GPU.
- `preds/`: reference predictions. `RESULTS.md`: the numbers they give.

## Reproduce

```sh
uv venv && uv pip install -e .
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml \
    data/baselines/baseline_stock.jsonl preds/v7__hub__active__TEST.jsonl --split test
```

Switchboard's own regression check rebuilds its router from this repo and must reproduce `preds/` exactly:
`python <switchboard>/scripts/check_harness.py .`

Each experiment's docstring says what it measures and how to run it. Experiments that need an LLM read
`INTENT_EVAL_LLM_URL` (an OpenAI-compatible server; the results used Qwen3-14B-AWQ) and use the cache first.

## About the data

- The real rows were reviewed before publishing. Rows with names, overheard conversation or private details were
  removed (3 rows), as were 2 that target a skill that isn't public. In 8 others, names were replaced
  (`"edited": true`). Those 8 are slightly easier than what was actually said, and stock OVOS's predictions for
  them were made on the original wording.
- The stock OVOS baselines (`data/baselines/`) come from a live OVOS stack and can't be regenerated without one
  (`switchboard_eval/harness/run_ovos.py`, which uses the query path and never runs a skill).
- 200 more non-command sentences from DailyDialog were also measured. Their license (CC BY-NC-SA) doesn't allow
  redistribution here; `switchboard_eval/inputs/build_ext_negatives.py` rebuilds them locally.
- `negatives_ext.jsonl` includes sentences from the AMI Meeting Corpus and the MAGPIE idiom corpus, both under
  CC BY 4.0.

## License

Code: Apache-2.0. The AMI and MAGPIE sentences keep their CC BY 4.0 licenses.
