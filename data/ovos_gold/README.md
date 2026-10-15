# OVOS golden utterances

These files are the test split of [OpenVoiceOS/ovos-intents](https://huggingface.co/datasets/OpenVoiceOS/ovos-intents)
(tag v6.1), which OpenVoiceOS publishes under Apache-2.0. The split holds the end-to-end golden utterances from the
OVOS skills' own tests, each labeled `<skill_id>:<intent_name>`. The dataset's work was funded by the NGI0 Commons
Fund / NLnet under grant agreement No 101135429 (European Commission, Next Generation Internet).

| file | rows | ids |
|---|---|---|
| `en-US.jsonl` | 1,007 | `g0000`... |
| `ca-ES.jsonl` | 109 | `c0000`... |
| `gl-ES.jsonl` | 110 | `l0000`... |
| `fr-FR.jsonl` | 100 | `f0000`... |

Changes from the dataset:

- Each row has an id.
- `utterance` is lowercased, the way speech recognition delivers it.
- The dataset's text is kept in `original`.

`data/baselines/ovos_gold/` holds live runs of the en-US file on the OVOS desktop. They went through the intent query
path, which asks what would match and never runs a skill (`run_ovos.py --stock` and `--without`). Two kinds of
pipeline ran: OVOS's default and each matcher alone, and the desktop's own Switchboard pipeline. Only rows whose
skill is installed on the desktop are scored.
