# Results

Switchboard's default configuration with Qwen3-14B-AWQ, on this repo's rows. Rerun the commands to check them.

Switchboard's docs report the full set, including the rows withheld here, so their numbers differ slightly: on the speaker's test split, 84.4% right (64 rows) there, 86.7% (60 rows) here; stock OVOS 42.2% and 45.0%. For overheard speech the docs also count 200 DailyDialog sentences (2.5% of 600 acted on); here 2.0% of 400.

## Neon smart speaker, dev split

```sh
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml data/baselines/baseline_stock.jsonl preds/v7__hub__active__DEV.jsonl --split dev
```
```
split=dev  rows=245  (criteria 2 determinism, 6 latency: separate runs)

stock OVOS (reference)
  1 magic phrases    91.2%  (n=102)   FAIL
  3 real Hub        50.0%  (n=70) p=1 (0+/0-)   FAIL
  3 real Mac        72.3%  (n=65) p=1 (0+/0-)   FAIL
  4 false accept     90.6%  (n=32)   FAIL
  5 high-risk exec  0            PASS

v7__hub__active__DEV
  1 magic phrases    91.2%  (n=102)   FAIL
  3 real Hub        81.4%  (n=70) p=2.74e-05 (25+/3-)   PASS
  3 real Mac        84.6%  (n=65) p=0.0386 (10+/2-)   PASS
  4 false accept     25.0%  (n=32)   PASS
  5 high-risk exec  0            PASS
```

## OVOS desktop, dev split

```sh
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml data/baselines/baseline_stock.jsonl preds/v7__mac__active__DEV.jsonl --split dev
```
```
split=dev  rows=245  (criteria 2 determinism, 6 latency: separate runs)

stock OVOS (reference)
  1 magic phrases    91.2%  (n=102)   FAIL
  3 real Hub        50.0%  (n=70) p=1 (0+/0-)   FAIL
  3 real Mac        72.3%  (n=65) p=1 (0+/0-)   FAIL
  4 false accept     90.6%  (n=32)   FAIL
  5 high-risk exec  0            PASS

v7__mac__active__DEV
  1 magic phrases    99.0%  (n=102)   PASS
  3 real Hub        62.9%  (n=70) p=0.122 (18+/9-)   FAIL
  3 real Mac        87.7%  (n=65) p=0.00635 (11+/1-)   PASS
  4 false accept     15.6%  (n=32)   PASS
  5 high-risk exec  0            PASS
```

## Neon smart speaker, test split

```sh
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml data/baselines/baseline_stock.jsonl preds/v7__hub__active__TEST.jsonl --split test
```
```
split=test  rows=272  (criteria 2 determinism, 6 latency: separate runs)

stock OVOS (reference)
  1 magic phrases    96.1%  (n=103)   FAIL
  3 real Hub        45.0%  (n=60) p=1 (0+/0-)   FAIL
  3 real Mac        71.4%  (n=63) p=1 (0+/0-)   FAIL
  4 false accept    100.0%  (n=39)   FAIL
  5 high-risk exec  0            PASS

v7__hub__active__TEST
  1 magic phrases    98.1%  (n=103)   PASS
  3 real Hub        86.7%  (n=60) p=1.62e-06 (27+/2-)   PASS
  3 real Mac        90.5%  (n=63) p=0.000488 (12+/0-)   PASS
  4 false accept     12.8%  (n=39)   PASS
  5 high-risk exec  0            PASS
```

## OVOS desktop, test split

```sh
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml data/baselines/baseline_stock.jsonl preds/v7__mac__active__TEST.jsonl --split test
```
```
split=test  rows=272  (criteria 2 determinism, 6 latency: separate runs)

stock OVOS (reference)
  1 magic phrases    96.1%  (n=103)   FAIL
  3 real Hub        45.0%  (n=60) p=1 (0+/0-)   FAIL
  3 real Mac        71.4%  (n=63) p=1 (0+/0-)   FAIL
  4 false accept    100.0%  (n=39)   FAIL
  5 high-risk exec  0            PASS

v7__mac__active__TEST
  1 magic phrases   100.0%  (n=103)   PASS
  3 real Hub        71.7%  (n=60) p=0.00154 (20+/4-)   PASS
  3 real Mac        88.9%  (n=63) p=0.00739 (13+/2-)   PASS
  4 false accept     12.8%  (n=39)   PASS
  5 high-risk exec  0            PASS
```

## Overheard speech (AMI and MAGPIE sentences, every one a false wake)

```sh
python -m switchboard_eval.harness.score_ext data/negatives_ext.jsonl preds/ext__nollm.jsonl preds/ext__active.jsonl preds/ext__always.jsonl
```
```
ext__nollm
  false accept   1.5%  (6/400)   ami 2.0%   magpie 1.0%
  accepted by stage: classifier 5, exact 1
  triggered: alerts.list 1, assistant.about 1, knowledge.ask 1, date.current 1, device.stop 1, time.current 1
ext__active
  false accept   2.0%  (8/400)   ami 3.0%   magpie 1.0%
  accepted by stage: classifier 5, llm 2, exact 1
  triggered: ask_knowledge 2, alerts.list 1, assistant.about 1, knowledge.ask 1, date.current 1, device.stop 1, time.current 1
ext__always
  false accept   7.8%  (31/400)   ami 8.5%   magpie 7.0%
  accepted by stage: llm 25, classifier 5, exact 1
  triggered: ask_knowledge 22, alerts.list 1, camera__TakePicIntent 1, assistant.about 1, knowledge.ask 1, date.current 1, fallback_llm__email_chat_history 1, easter_eggs__sing_intent 1
```

## Per matcher

How each of OVOS's default matchers does on its own, next to Switchboard. Each OVOS row is the pipeline OVOS ships
(ovos-config's default `mycroft.conf`) with the other matchers removed, so stop, converse, OCP and the fallbacks
are the same in every row. All ran on the live OVOS desktop through the intent query path, which asks what would
match and never runs a skill (`switchboard_eval/harness/run_ovos.py --stock` and `--without`). The first file is
the reference for the p-values.

- Switchboard's accuracy rows (`nollm__*`, `v7__*`) come from this harness, as everywhere else, with each device's
  router scored on its own requests: the speaker rows from `__hub__` files, the desktop rows from `__mac__` files.
- Model2Vec appears only in its high-confidence tier in the default pipeline, so "Model2Vec alone" is Model2Vec as
  shipped, not tuned for use on its own.
- Adapt alone scores low because most intents here declare phrase templates (Padatious) but no Adapt keywords.
- The latency rows are the same live runs. `switchboard_live_desktop` is the desktop's own Switchboard pipeline
  (with a 14B LLM); its accuracy isn't reported because the query path gives every row a cold context (the
  assistant never just spoke), unlike the per-row context the harness replays.

### Summary (test split)

| pipeline | real speaker requests right | real desktop requests right | declared phrases right | non-commands acted on |
|---|---|---|---|---|
| OVOS default (Adapt + Padatious + Model2Vec) | 45.0% | 71.4% | 96.1% | 100.0% |
| Padatious alone | 45.0% | 65.1% | 96.1% | 100.0% |
| Padatious + Adapt | 45.0% | 65.1% | 96.1% | 100.0% |
| Model2Vec alone | 20.0% | 46.0% | 32.0% | 100.0% |
| Adapt alone | 20.0% | 14.3% | 7.8% | 100.0% |
| **Switchboard, no LLM** | 78.3% | 79.4% | 94.2% | 10.3% |
| **Switchboard + 14B LLM** | 86.7% | 88.9% | 98.1% | 12.8% |

For Switchboard, declared phrases is the lower of the two devices and non-commands the higher.

| live desktop | requests | median | 95th percentile |
|---|---|---|---|
| OVOS default | 517 | 16.1 ms | 36.4 ms |
| Padatious alone | 517 | 19.1 ms | 35.6 ms |
| Padatious + Adapt | 517 | 19.3 ms | 35.4 ms |
| Model2Vec alone | 517 | 11.3 ms | 15.4 ms |
| Adapt alone | 517 | 11.8 ms | 18.5 ms |
| Switchboard, all requests | 517 | 9.3 ms | 1134.3 ms |
| Switchboard, decided without the LLM | 401 | 6.7 ms | 21.3 ms |
| Switchboard, decided by the 14B LLM | 64 | 1043.2 ms | 1239.5 ms |

### test split

```sh
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml data/baselines/matchers/stock.jsonl data/baselines/matchers/stock_padatious_only.jsonl data/baselines/matchers/stock_padatious_adapt.jsonl data/baselines/matchers/stock_m2v_only.jsonl data/baselines/matchers/stock_adapt_only.jsonl preds/nollm__hub__active__TEST.jsonl preds/nollm__mac__active__TEST.jsonl preds/v7__hub__active__TEST.jsonl preds/v7__mac__active__TEST.jsonl --split test
```
```
split=test  rows=272  (criteria 2 determinism, 6 latency: separate runs)

stock OVOS (reference)
  1 magic phrases    96.1%  (n=103)   FAIL
  3 real Hub        45.0%  (n=60) p=1 (0+/0-)   FAIL
  3 real Mac        71.4%  (n=63) p=1 (0+/0-)   FAIL
  4 false accept    100.0%  (n=39)   FAIL
  5 high-risk exec  0            PASS

stock_padatious_only
  1 magic phrases    96.1%  (n=103)   FAIL
  3 real Hub        45.0%  (n=60) p=1 (0+/0-)   FAIL
  3 real Mac        65.1%  (n=63) p=0.125 (0+/4-)   FAIL
  4 false accept    100.0%  (n=39)   FAIL
  5 high-risk exec  0            PASS

stock_padatious_adapt
  1 magic phrases    96.1%  (n=103)   FAIL
  3 real Hub        45.0%  (n=60) p=1 (0+/0-)   FAIL
  3 real Mac        65.1%  (n=63) p=0.125 (0+/4-)   FAIL
  4 false accept    100.0%  (n=39)   FAIL
  5 high-risk exec  0            PASS

stock_m2v_only
  1 magic phrases    32.0%  (n=103)   FAIL
  3 real Hub        20.0%  (n=60) p=0.00408 (5+/20-)   FAIL
  3 real Mac        46.0%  (n=63) p=0.00154 (4+/20-)   FAIL
  4 false accept    100.0%  (n=39)   FAIL
  5 high-risk exec  0            PASS

stock_adapt_only
  1 magic phrases     7.8%  (n=103)   FAIL
  3 real Hub        20.0%  (n=60) p=0.00408 (5+/20-)   FAIL
  3 real Mac        14.3%  (n=63) p=1.71e-08 (4+/40-)   FAIL
  4 false accept    100.0%  (n=39)   FAIL
  5 high-risk exec  0            PASS

nollm__hub__active__TEST
  1 magic phrases    94.2%  (n=103)   FAIL
  3 real Hub        78.3%  (n=60) p=8.8e-05 (23+/3-)   PASS
  3 real Mac        82.5%  (n=63) p=0.0654 (9+/2-)   FAIL
  4 false accept      7.7%  (n=39)   PASS
  5 high-risk exec  0            PASS

nollm__mac__active__TEST
  1 magic phrases    97.1%  (n=103)   FAIL
  3 real Hub        66.7%  (n=60) p=0.0106 (18+/5-)   PASS
  3 real Mac        79.4%  (n=63) p=0.267 (9+/4-)   FAIL
  4 false accept     10.3%  (n=39)   PASS
  5 high-risk exec  0            PASS

v7__hub__active__TEST
  1 magic phrases    98.1%  (n=103)   PASS
  3 real Hub        86.7%  (n=60) p=1.62e-06 (27+/2-)   PASS
  3 real Mac        90.5%  (n=63) p=0.000488 (12+/0-)   PASS
  4 false accept     12.8%  (n=39)   PASS
  5 high-risk exec  0            PASS

v7__mac__active__TEST
  1 magic phrases   100.0%  (n=103)   PASS
  3 real Hub        71.7%  (n=60) p=0.00154 (20+/4-)   PASS
  3 real Mac        88.9%  (n=63) p=0.00739 (13+/2-)   PASS
  4 false accept     12.8%  (n=39)   PASS
  5 high-risk exec  0            PASS
```

### dev split

```sh
python -m switchboard_eval.harness.bar data/dataset.jsonl schema/intents.yaml data/baselines/matchers/stock.jsonl data/baselines/matchers/stock_padatious_only.jsonl data/baselines/matchers/stock_padatious_adapt.jsonl data/baselines/matchers/stock_m2v_only.jsonl data/baselines/matchers/stock_adapt_only.jsonl preds/nollm__hub__active__DEV.jsonl preds/nollm__mac__active__DEV.jsonl preds/v7__hub__active__DEV.jsonl preds/v7__mac__active__DEV.jsonl --split dev
```
```
split=dev  rows=245  (criteria 2 determinism, 6 latency: separate runs)

stock OVOS (reference)
  1 magic phrases    93.1%  (n=102)   FAIL
  3 real Hub        50.0%  (n=70) p=1 (0+/0-)   FAIL
  3 real Mac        75.4%  (n=65) p=1 (0+/0-)   FAIL
  4 false accept     90.6%  (n=32)   FAIL
  5 high-risk exec  0            PASS

stock_padatious_only
  1 magic phrases    92.2%  (n=102)   FAIL
  3 real Hub        52.9%  (n=70) p=0.688 (4+/2-)   FAIL
  3 real Mac        69.2%  (n=65) p=0.125 (0+/4-)   FAIL
  4 false accept     90.6%  (n=32)   FAIL
  5 high-risk exec  0            PASS

stock_padatious_adapt
  1 magic phrases    92.2%  (n=102)   FAIL
  3 real Hub        51.4%  (n=70) p=1 (4+/3-)   FAIL
  3 real Mac        69.2%  (n=65) p=0.125 (0+/4-)   FAIL
  4 false accept     90.6%  (n=32)   FAIL
  5 high-risk exec  0            PASS

stock_m2v_only
  1 magic phrases    32.4%  (n=102)   FAIL
  3 real Hub        24.3%  (n=70) p=0.000121 (2+/20-)   FAIL
  3 real Mac        43.1%  (n=65) p=0.000104 (4+/25-)   FAIL
  4 false accept     84.4%  (n=32)   FAIL
  5 high-risk exec  0            PASS

stock_adapt_only
  1 magic phrases     9.8%  (n=102)   FAIL
  3 real Hub        25.7%  (n=70) p=0.00232 (6+/23-)   FAIL
  3 real Mac        16.9%  (n=65) p=5.1e-09 (4+/42-)   FAIL
  4 false accept     84.4%  (n=32)   FAIL
  5 high-risk exec  0            PASS

nollm__hub__active__DEV
  1 magic phrases    88.2%  (n=102)   FAIL
  3 real Hub        75.7%  (n=70) p=0.000534 (22+/4-)   PASS
  3 real Mac        76.9%  (n=65) p=1 (6+/5-)   FAIL
  4 false accept     25.0%  (n=32)   PASS
  5 high-risk exec  0            PASS

nollm__mac__active__DEV
  1 magic phrases    97.1%  (n=102)   FAIL
  3 real Hub        55.7%  (n=70) p=0.572 (16+/12-)   FAIL
  3 real Mac        81.5%  (n=65) p=0.344 (7+/3-)   FAIL
  4 false accept     15.6%  (n=32)   PASS
  5 high-risk exec  0            PASS

v7__hub__active__DEV
  1 magic phrases    91.2%  (n=102)   FAIL
  3 real Hub        81.4%  (n=70) p=2.74e-05 (25+/3-)   PASS
  3 real Mac        84.6%  (n=65) p=0.146 (9+/3-)   FAIL
  4 false accept     25.0%  (n=32)   PASS
  5 high-risk exec  0            PASS

v7__mac__active__DEV
  1 magic phrases    99.0%  (n=102)   PASS
  3 real Hub        62.9%  (n=70) p=0.122 (18+/9-)   FAIL
  3 real Mac        87.7%  (n=65) p=0.0386 (10+/2-)   PASS
  4 false accept     15.6%  (n=32)   PASS
  5 high-risk exec  0            PASS
```

### Latency (live desktop, query path)

```sh
python -m switchboard_eval.harness.latency data/baselines/matchers/stock.jsonl data/baselines/matchers/stock_padatious_only.jsonl data/baselines/matchers/stock_padatious_adapt.jsonl data/baselines/matchers/stock_m2v_only.jsonl data/baselines/matchers/stock_adapt_only.jsonl data/baselines/matchers/switchboard_live_desktop.jsonl
```
```
stock                                     517 rows   median    16.1 ms   p95    36.4 ms
stock_padatious_only                      517 rows   median    19.1 ms   p95    35.6 ms
stock_padatious_adapt                     517 rows   median    19.3 ms   p95    35.4 ms
stock_m2v_only                            517 rows   median    11.3 ms   p95    15.4 ms
stock_adapt_only                          517 rows   median    11.8 ms   p95    18.5 ms
switchboard_live_desktop                  517 rows   median     9.3 ms   p95  1134.3 ms
  decided by exact or classifier          401 rows   median     6.7 ms   p95    21.3 ms
  decided by the LLM step                  64 rows   median  1043.2 ms   p95  1239.5 ms
```
