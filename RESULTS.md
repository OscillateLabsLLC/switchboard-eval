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
