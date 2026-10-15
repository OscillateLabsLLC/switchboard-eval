# switchboard-eval

The test bench behind [Switchboard](https://github.com/OscillateLabsLLC/switchboard), a replacement for OVOS intent
matching. Every number in Switchboard's docs was measured here, and you can rerun it.

## What's in it

- `data/dataset.jsonl`: 517 labeled utterances, split into dev and test. 259 were written to probe specific cases
  (rephrasings, near misses, misheard speech, speech not meant for the assistant). 258 are real, from one
  household's Neon smart speaker and OVOS desktop.
- `data/negatives_ext.jsonl`: 400 sentences from public corpora that were never addressed to an assistant, to
  measure how often it acts on overheard speech.
- `devices/`: what each device has installed: its skills' intents and phrases, and the LLM tools made from them.
- `schema/intents.yaml`: the shared list of intents that every device's skills map onto, with risk levels.
- `switchboard_eval/`: the scoring code (`harness/`), the experiments (`experiments/`), and the scripts that
  rebuild the inputs (`inputs/`).
- `cache/llm_cache.jsonl`: every LLM answer the results used, so they reproduce without a GPU.
- `preds/`: the reference predictions. `RESULTS.md`: the numbers they give. `BENCH.md`: every experiment, including
  the ones that failed.

## Rerun it

```sh
uv venv && uv pip install -e .
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml \
    data/baselines/baseline_stock.jsonl preds/v7__hub__active__TEST.jsonl --split test
```

Switchboard's own check rebuilds its router from this repo and must reproduce `preds/` exactly:
`python <switchboard>/scripts/check_harness.py .`

Each experiment's docstring says what it measures and how to run it. Experiments that call an LLM read
`INTENT_EVAL_LLM_URL` (any OpenAI-compatible server; the results used Qwen3-14B-AWQ) and check the cache first.

## About the data

- **The real rows were reviewed before publishing.** Three were removed (names, overheard conversation), and two
  more that use a skill that isn't public. In 8 rows, names were swapped for others (marked `"edited": true`); those
  are a little easier than what was really said, and stock OVOS's predictions for them were made on the original
  wording.
- **Stock OVOS's predictions** (`data/baselines/`) came from a live OVOS install and need one to regenerate
  (`switchboard_eval/harness/run_ovos.py`, which only queries and never runs a skill).
- **DailyDialog:** 200 more overheard sentences came from it, but its license (CC BY-NC-SA) doesn't allow sharing
  them here. `switchboard_eval/inputs/build_ext_negatives.py` rebuilds them locally.

## License

Code: Apache-2.0. The sentences from the AMI Meeting Corpus and the MAGPIE idiom corpus in `negatives_ext.jsonl`
keep their CC BY 4.0 licenses.
