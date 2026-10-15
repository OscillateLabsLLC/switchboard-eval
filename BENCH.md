# Phase 1 bench (Task 7) — 2026-10-11

Trained on `examples_live.jsonl`: 31,266 examples / 139 intents expanded from the live stack's
INTENT-4 registrations (`harvest_live.py introspection.json`), plus one description sentence per
canonical intent. The eval set is `dataset.jsonl` (259 rows). `seen_in_templates` is tagged against
the same live templates. Reproduce with `./run_grid.sh`, then `compare.py`.

**Latency was measured on the MacBook (Apple Silicon), not the mini PC.** Treat the latency
column as relative only. Task 7 asks for mini PC numbers, and those still need a run there.

## Results vs live baseline (McNemar p; +/- = rows only the candidate / only the baseline got right)

| system | headline % | unseen % | false accept % | high-risk fires | first thr with FA <= 20% | latency p50/p95 ms | McNemar p |
|---|---|---|---|---|---|---|---|
| **live baseline** | 38.2 | 22.2 | 93.3 | 0 | never | 83 / 644 | — |
| char + proto | 50.6 | 37.7 | 62.2 | 10 | 0.35: 47% cov / 83% prec | 0.7 / 0.8 | 0.003 |
| char + knn | 50.2 | 36.4 | 73.3 | 8 | 0.55: 56% / 85% | 39 / 41 | 0.0008 |
| **char + logreg** | **56.2** | 40.7 | 62.2 | 6 | 0.55: 53% / 89% | 0.6 / 0.7 | 7e-7 |
| bge-small + proto | 52.6 | 42.0 | 66.7 | 6 | 0.80: 64% / 76% | 6.4 / 6.9 | 0.0005 |
| bge-small + knn | 51.0 | 38.3 | 71.1 | 5 | 0.85: 52% / 79% | 7.2 / 7.7 | 0.0005 |
| **bge-small + logreg** | **56.6** | 44.4 | 64.4 | 4 | 0.45: 67% / 77% | 7.0 / 7.5 | 2e-6 |
| e5-small + proto | 47.4 | 35.8 | 64.4 | 6 | 0.90: 53% / 78% | 6.8 / 7.2 | 0.025 |
| e5-small + knn | 49.0 | 37.0 | 66.7 | 6 | 0.90: 60% / 77% | 7.5 / 7.8 | 0.004 |
| e5-small + logreg | 49.8 | 35.2 | 71.1 | 6 | 0.35: 60% / 77% | 7.3 / 7.7 | 0.002 |
| MiniLM-L6 + proto | 53.4 | **45.1** | 62.2 | 7 | 0.60: 48% / 84% | 3.8 / 4.2 | 0.0006 |
| MiniLM-L6 + knn | 51.0 | 37.7 | 73.3 | 4 | 0.65: 53% / 81% | 4.2 / 4.5 | 0.0005 |
| MiniLM-L6 + logreg | 54.2 | 39.5 | 84.4 | 6 | 0.60: 56% / 82% | 3.9 / 4.2 | 9e-6 |
| potion-8M + proto | 47.4 | 37.0 | 77.8 | 10 | 0.65: 40% / 74% | 0.0 / 0.1 | 0.033 |
| potion-8M + knn | 49.8 | 34.0 | 77.8 | 8 | 0.75: 49% / 80% | 0.6 / 0.6 | 0.0008 |
| potion-8M + logreg | 51.4 | 35.2 | 84.4 | 7 | 0.80: 39% / 90% | 0.1 / 0.1 | 0.0003 |
| OVOS M2V + proto | 47.0 | 37.0 | 80.0 | 11 | 0.60: 52% / 84% | 0.1 / 0.1 | 0.026 |
| OVOS M2V + knn | 48.6 | 36.4 | 73.3 | 3 | 0.70: 55% / 80% | 5.7 / 5.9 | 0.003 |
| OVOS M2V + logreg | 49.0 | 34.0 | 82.2 | 3 | 0.85: 33% / 87% | 0.1 / 0.1 | 0.002 |

## What it says

1. **Every configuration beats the live stack significantly** (all p < 0.05): about +12 to +18
   headline and +12 to +23 unseen.
2. **Every configuration is blocked by the high-risk rule.** All 18 runs fire lock.unlock,
   cover.open, or system.shutdown wrongly at least 3 times. The cause is data, not the encoder.
   Those intents have only one description sentence, so anything containing "garage", "shut",
   or "unlock" collapses onto them. Examples: "don't open the garage" → cover.open,
   "shut up" → system.shutdown, "how do i unlock my phone" → lock.unlock. Task 8's synthetic
   examples, plus contrastive negatives for these intents, come next.
3. **The best two are tied.** bge-small+logreg and char+logreg give McNemar p = 1.0 against each
   other (23 vs 22 discordant rows). On this data a 33M-parameter encoder buys nothing over TF-IDF
   char n-grams, which run ~10x faster and need no model download.
4. **The OVOS M2V model is not the "upstream instead" winner.** With the same trained head,
   bge-small+logreg beats OVOS M2V+logreg (p = 0.008, 33 vs 14). Task 9's within-noise trigger
   does not fire against the best system. Char+logreg is also an option that could go upstream.
5. **No configuration reaches ~90% precision with useful coverage.** At false accept <= 20%,
   the best is ~53-67% coverage at 77-89% precision.
6. (see Task 8 below) 80 of 214 in-domain rows have no live training examples (locks, covers, thermostat, Music
   Assistant, meal plan, monster scanner). Coverage on those is roughly 35%; covered rows reach
   about 64%. Task 8 targets these.

## Task 8: reviewed synthetic examples

`synthetic_examples.jsonl`: 695 phrasings for the 29 canonical intents with no live examples.
Qwen3-14B-AWQ on the GPU host generated 713; the maintainer pruned them in `synthetic_review.txt`. The model never
saw the eval set, and 7 exact eval matches were dropped automatically. Train with `bench.py --synthetic`.

| system | headline % | unseen % | false accept % | high-risk fires | first thr with FA <= 20% | McNemar p vs baseline |
|---|---|---|---|---|---|---|
| char + logreg | 56.2 | 40.7 | 62.2 | 6 | 0.55: 53% cov / 89% prec | 7e-7 |
| char + logreg + synthetic | 61.8 | 48.1 | 82.2 | 10 | 0.70: 51% / 94% | 9e-11 |
| bge-small + logreg | 56.6 | 44.4 | 64.4 | 4 | 0.45: 67% / 77% | 2e-6 |
| **bge-small + logreg + synthetic** | **65.5** | **56.2** | 73.3 | 7 | 0.65: 67% / 90% | 1e-11 |

- **The synthetic examples help, and the gain is real.** For bge-small, with vs without:
  p = 0.0016 (34 vs 12). Rows that were zero-shot now train on examples.
- **High-risk fires get worse (4 → 7, 6 → 10).** Positive examples give the high-risk intents
  more surface area, so near-misses land on them.
- The bge+synthetic fires fall into three classes:
  - negation/near-miss: "do not unlock the door", "he's in shutdown mode again",
    "door's open, come on in"
  - mention, not command: "how do i unlock my phone", "tell home assistant to unlock the front
    door" (gold ha.assist)
  - high-risk vs high-risk confusion: "let me in the side door" → cover.open
- **A threshold cannot fix this.** Confidences on wrong high-risk fires (bge: up to 0.994; char: up
  to 0.997) overlap the correct hits. Risk-scaled thresholds alone cannot reach zero wrong fires
  while keeping correct ones.

**Implication:** reaching "zero high-risk wrong fires" through the classifier needs explicit
negatives (a decline/none class trained on "don't X" and "how do I X" phrasings), and probably
still won't be enough. The robust fix is policy: high-risk intents always CONFIRM and never
auto-execute (architecture step 5). Then a wrong high-risk prediction costs one "did you mean
unlock the front door?" prompt, not an unlocked door. That would mean re-scoping the guardrail
from "wrong fire" to "wrong execution", which is the owner's call.

**Decision (2026-10-11): always-confirm for high-risk intents.** `score.py` now reports high-risk
wrong EXECUTIONS (the blocker; 0 for every system under this policy) separately from wrong
CONFIRM PROMPTS (the annoyance to minimize: bge+synthetic 7, char+synthetic 10). The leading
candidate, bge-small + logreg + synthetic, is now unblocked. Next lever for the confirm-prompt
count: explicit negatives ("don't X", "how do I X") as a decline class.

## Real household utterances (128 rows, category `real`) — the reality check

The rows came from the OVOS logs and the maintainer labeled them (Claude drafted). They were added AFTER every
model and data choice above, so they work as a fresh test set: nothing was tuned on them.

| system (387-row dataset) | headline % | unseen % | **real %** | wrong exec | wrong confirms |
|---|---|---|---|---|---|
| live baseline | 50.1 | 30.0 | **74.2** | 0 | 0 |
| bge-small + logreg + synthetic | 69.5 | 60.0 | **77.3** | 0 | 7 |
| char + logreg + synthetic | 67.1 | 51.4 | **77.3** | 0 | 10 |

- On real rows, bge+synthetic vs baseline: **p = 0.62** (20 vs 16). **No measurable gain on real use.**
- 80/128 (62%) of real rows already match a live template, versus ~34% of the hand-authored rows. The
  household mostly says what the skills expect, and the live stack handles that. The large
  headline gains come from the hand-authored paraphrase, implicit, negation, and near-miss rows.
- Caveat: the real logs are biased toward phrasing that already works. People adapt to the
  assistant, and many rows are the maintainer testing his own easter-egg skill.
- **Methodology caveat:** encoder, head, and synthetic data were all chosen by scoring on the
  hand-authored eval set itself, so the winner's headline is optimistically biased. Future
  experiments should select on a dev split and report on a held-out test split once.

## Neon Hub rows, stock comparator, and exact-first (2026-10-11, 522-row dataset)

135 Neon Hub rows added (`source: neon_hub`, category `real`). Comparator is now STOCK OVOS
(`run_ovos.py --stock` -> `baseline_stock.jsonl`: Padatious + m2v-high via a named session).

**The "seen" metric was misleading.** Of 30 stock failures on template-matching rows, 25 matched
a template belonging to a DIFFERENT intent (one-label mapping undercount, or real skill template
collisions such as alerts matching "set the thermostat to 72"). The real "magic phrase" metric
counts only rows where the gold intent's OWN template matches. On that metric stock Padatious
gets ~97%. Classifiers alone are worse (char 92%, bge 76%) because they blur exact phrases.

**Exact-first design (simulated):** if the utterance matches templates of exactly one canonical
intent, take it; otherwise fall back to the classifier. This is padacioso-exact first with a
better fallback and without persona-high / m2v hijackers in front.

Go/no-go bar, TEST split only (276 rows). The design was chosen after seeing all-row numbers,
so test is the honest number:

| criterion | stock | exact-first + char+syn | bar | status |
|---|---|---|---|---|
| 1 magic phrases (n=93) | 97.8% | 98.9% | >= 98% | pass (1 miss) |
| 2 determinism across restarts | — | — | 0 flips | NOT RUN |
| 3 real Hub (n=64) | 42.2% | 65.6% | p < 0.05 | pass, p = 0.0007 (17+/2-) |
| 3 real Mac (n=63) | 69.8% | 81.0% | p < 0.05 | fail, p = 0.09 (underpowered) |
| 4 false accept on negatives (n=42) | 100% | 71.4% | <= 25% | FAIL, biggest gap |
| 5 high-risk wrong executions | 0 | 0 | 0 | pass (policy) |

Caveat: Hub utterances were probed against the Mac's OVOS skill set, not Neon's skills, for both
systems. Neon-only capabilities (Node app launches, briefing) count against stock and are
learnable by the candidate only through synthetic examples.

Next lever: criterion 4 (decline). Candidates always pick an intent. A decline class trained on
negatives, a threshold, or both, measured on dev.

**Update:** the exact layer is now real (`bench.py --templates introspection.json`, class
`ExactLayer`). It routes a unique template owner, including unmapped skills, and hands collisions
to the classifier. On test it reproduces the simulation exactly: magic 98.9%, Hub 65.6%
(p = 0.0007), Mac 81.0%, false accept 71.4%. The go/no-go bar is now a script:
`bar.py dataset.jsonl intents.yaml baseline_stock.jsonl <candidates...> [--split dev|test]`.

## Criterion 4 (decline) and OCP templates — 2026-10-11 late

- **INTENT-4 introspection omits pipeline-plugin intents.** OCP's play/pause/next/prev/stop
  templates never appear in `ovos.intent.list`/`describe` (0 OCP entries). `dump_introspection.py`
  now adds them from `ocp_pipeline/locale` (`OCP_INTENTS`). This is worth reporting upstream: the
  manifest is incomplete for anyone building on it. `ocp:resume -> media.resume` is proposed, not yet mapped.
- **Decline lever 1 (threshold on classifier confidence, tuned on dev).** Exact-layer routes carry
  confidence 1.0 and are never declined. Dev choice: 0.6 (0.5 gave FA 26.5%).
- A "min 2 literal words" rule for the exact layer (keeping `play {query}` out) was tested on dev
  and REJECTED: magic phrases fell to ~92% with no FA gain. All OCP templates stay in.

TEST, locked config = exact layer (+OCP templates) -> char+logreg+synthetic, threshold 0.6:

| criterion | stock | candidate | status |
|---|---|---|---|
| 1 magic phrases (n=103) | 96.1% | 99.0% | PASS |
| 3 real Hub (n=64) | 42.2% | 73.4%, p = 9e-5 (23+/3-) | PASS |
| 3 real Mac (n=63) | 69.8% | 76.2%, p = 0.39 (8+/4-) | FAIL (direction right, underpowered) |
| 4 false accept (n=42) | 100% | 11.9% | PASS |
| 5 high-risk wrong exec | 0 | 0 | PASS |

Known cost: declines of valid requests (dev in-domain 73% -> 67%). In the full design a decline
goes to the LLM fallback (Phase 2), not a flat refusal. Test has now been consulted 3 times this
session; consult it only at milestones from here.

## Decline lever 2: dialog-derived negatives (2026-10-11 late)

No open "device-directed speech" dataset or model was found (one search; Amazon/Apple corpora are
proprietary). Out-of-scope corpora such as CLINC150 cover addressed-but-unsupported requests,
which belong to the LLM fallback, NOT the decline class. A text-only decline class has a ceiling:
"turn off the lights" said to a person reads the same as to the device.

`harvest_dialogs.py` builds `dialog_negatives.jsonl` from skills' own `.dialog` files (what the
assistant says aloud = self-echo): 2,599 lines from 926 files. Eval matches (1) and exact-template
matches (33) were dropped. Trained as class `__decline__` via `bench.py --synthetic a,b`.
LLM-generated negatives (`negative_review.txt`) were NOT used.

Dev: at no threshold, FA 79% -> 47% with no in-domain cost. Operating point moves from 0.6 to
0.5 (FA 14.7%, in-domain 69.5% vs 23.5% / 67.0% threshold-only). Small samples: 3 and 5 rows.

TEST (milestone check), exact+OCP -> char+logreg+synthetic+dialog-decline, threshold 0.5:
magic 99.0% PASS | Hub 75.0% p = 5e-5 PASS | Mac 74.6% p = 0.55 FAIL | FA 7.1% PASS | high-risk 0 PASS.
**Adopted as the current candidate.**

## Determinism (criterion 2): instrument bug found mid-run

`run_ovos.py` used `wait_for_response` with no request correlation. After a core restart, the
probe at row 1 timed out, and its late reply was then recorded as a later row's answer, so A1/A2
showed 27 spurious "flips". FIXED: each probe carries a unique context id (`intent_eval_req`)
and only its own reply is accepted. Timeouts are now counted and warned. The clean
`baseline_stock.jsonl` (all bar numbers) had zero slow rows, so it was unaffected.
B1 (forced retrain, no timeouts) differs from the clean stock run on 182/522 rows. Pending a
clean rerun with `--settle` to separate a background-compile transient from a different
retrained model.
- CORRECTION: B1's 561 s was NOT ~9 min of Padatious training. 512 s of it was the probe:
  after a forced retrain every query ran ~10x slower (~1 s/row vs 0.09 s in A1), consistent with
  padaos recompiling in the background and serving stale state. Restart-to-first-answer was
  ~50 s. The maintainer flagged "9 minutes seems too long", correctly.

## Criterion 2 (determinism) — RESULT (clean rerun, fixed runner, 2026-10-11 ~23:30)

`determinism.py --settle 60`, stock OVOS, 522 rows, zero probe timeouts in all runs.
After a forced retrain, Padatious serves ~2.4 s/row for about the first 130 rows (~5 min),
then normal speed. Rows split at that point:

| comparison | slow window (130 rows) | settled (392 rows) |
|---|---|---|
| restart vs restart (cached model) | 1 | **0** |
| retrain vs retrain | 39 | **155 (40%)** |
| original model vs retrain 1 | 48 | 152 |
| original model vs retrain 2 | 31 | 39 |

- **Restarts are deterministic; retraining is a lottery.** Two retrains of identical skills disagree
  on 40% of settled rows. Retrain 1 drew a degenerate model where alerts `reschedule_alert`
  swallowed unrelated requests ("unlock the back door for the dog walker", "put eggs and coffee
  on the grocery list"). Retrain 2 landed near the original. Every skill install or update
  retrains, which explains the maintainer's "even the magic phrases don't always work". n = 2 retrains,
  but the effect is large.
- For ~5 min after a retrain, stock is also slow (~2.4 s per query).
- **Candidate (exact+OCP -> char+logreg+synthetic+dialog-decline, t = 0.5): 0/522 flips across
  independent retrains from scratch. Criterion 2: stock FAIL, candidate PASS.**

## Phase 2 — LLM tool definitions from the manifest (2026-10-11)

`tools.py introspection.json` -> `tools.json` + `tools_review.md`. **Manifest data only** (skill_id,
intent_name, templates, slot names, slot_blacklist); intents.yaml is not read, so this works
upstream with no skill changes and no hand-written descriptions.
- 145 registrations + 4 meta tools (`ask_knowledge`, `stop`, `unsupported`, `ignore`) = 149.
- Slots become string params; required iff every template guarantees the slot. Blacklists become
  "Never one of: ...". Examples: greedy diverse pick over template expansions.
- **All 149 tools = ~12.1k prompt tokens (~81/tool), over the GPU host's 12,288 context.** A shortlist
  is mandatory. `shortlist.py`: top-12 by char-ngram cosine to each tool's expansions + 4 meta
  -> ~1.5-1.9k tokens per call.

Smoke test (6 hand-picked utterances, Qwen3-14B-AWQ, temp 0, thinking off; anecdote, not a result):

| utterance | tool_choice=auto | tool_choice=required |
|---|---|---|
| turn on the desk lamp | homeassistant__turn_on {entity: desk lamp} | same |
| what's the weather like in denver | weather__weather {location: denver} | same |
| skip this song | ocp__next | same |
| honey where are my keys | **no tool, chatted back in prose** | **unsupported** (should be ignore) |
| unlock the front door | unsupported | unsupported |
| who was the first president | ask_knowledge | ask_knowledge |

- `auto` lets the model answer outside the tools: unacceptable (it would speak to overheard talk).
  `required` forces a schema-constrained call (vLLM guided decoding).
- Latency 0.3-1.4 s per call. The required run came second, so prefix caching confounds the
  auto-vs-required speed comparison.
- Open: telling `ignore` apart from `unsupported` (both no action, but `unsupported` SPEAKS — a
  false trigger for the family). Catch-all tools (`homeassistant__assist`, `ddg__search_duck`)
  rarely make a similarity shortlist; "unlock the front door" never saw the HA assist tool.

## Phase 2 -- first LLM fallback results, DEV only (2026-10-11)

`llm_fallback.py --preds preds/exact_ocp__char__logreg__syn_dlg__t0.5.jsonl [--system-file F] [--think]`
Routes classifier-declined dev rows (79/246 = 32%) to Qwen3-14B-AWQ with shortlist top-12 + 4 meta
tools, tool_choice=required, temp 0. Responses cached in `llm_cache.jsonl`. Test split untouched.

| system (dev) | magic | Hub (n=71) | Mac (n=65) | false accept (n=34) | LLM p50/p95 |
|---|---|---|---|---|---|
| stock OVOS | 93.9 | 49.3 | 69.2 | 91.2 | -- |
| candidate (no LLM) | 99.0 | 69.0 | 76.9 | 14.7 | -- |
| + LLM, base prompt | 100.0 | 73.2 | 83.1 | **41.2 FAIL** | 958 / 1148 ms |
| + LLM, base prompt, thinking | 99.0 | 74.6 | 81.5 | 32.4 FAIL | 4305 / 10400 ms |
| + LLM, strict prompt (`prompts/fallback_strict.txt`) | 100.0 | **78.9** (p=6e-6) | **83.1** (p=0.012) | **26.5 FAIL by 1 row** | 962 / 1185 ms |

- The base prompt acts on overheard speech: "yeah" -> greetings, "one" -> count, "bacon and eggs" -> knowledge.
- Classifier confidence does NOT separate LLM fixes from LLM false accepts (both span 0.13-0.63),
  so "route only low-confidence declines" isn't a lever.
- Thinking mode: worse and 4-10x slower. Dropped.
- Leakage caught: the first strict prompt quoted three dev utterances as examples and scored 23.5%.
  Replaced with non-dataset examples ("sure", "three", "the blue one") -> 26.5%. Only the clean number counts.
- Remaining 9 false accepts: 5 from the LLM (self-echo "...hey mycroft, play", "goethe", "bacon and eggs",
  "turn up for the weekend"), 4 pre-existing from exact/classifier ("thanks", "roll with it",
  "he's in shutdown mode again", "...the garage door was open").
- n=34 negatives: one row = 2.9 points. The bar can't resolve the difference between 23.5 and 26.5.
- Scoring gaps (proposals, not committed): wikipedia:wiki -> knowledge.ask; hello_world greetings /
  thank_you_intent and parrot did_you_hear_me are unmapped, so reasonable LLM picks score as wrong.
- No tool exists for thermostat, locks, or the app launcher (fallback skill, absent from the
  manifest), so `unsupported` is the right answer there and those gold rows can't be recovered.

## Phase 2 — addressee gate (logprob P(yes)), DEV only (2026-10-12)

Local analogue of Jev-style calibrated decisions (the maintainer has no Jev access; it's a proprietary cloud
API anyway). `llm_fallback.py ... --gate T`: one extra call per routed row, `prompts/addressee_gate.txt`,
max_tokens=1, P(yes) renormalized over yes/no top-logprob tokens. Below T -> declined.
Gate call latency p50 32 ms, p95 117 ms (79 calls).

| system (dev) | magic | Hub (n=71) | Mac (n=65) | false accept (n=34) |
|---|---|---|---|---|
| + LLM strict prompt | 100.0 | 78.9 | 83.1 | 26.5 FAIL |
| + LLM strict prompt + gate 0.9 | 100.0 | **81.7** (p=1.6e-6) | **83.1** (p=0.012) | **20.6 PASS** |

- Calibration is poor: a large share of rows sit at P = 1.000, negatives included ("call the plumber",
  "please", self-echo, "testing testing"). It acts as a second binary vote, not a usable probability.
- But it is safe on dev: every LLM-fixed row has P >= 0.97, so any T in [0.66, 0.97] gives the
  identical result. T = 0.9 is not fit to individual rows. Removes "goethe" (0.17) and "bacon and
  eggs" (0.65) at no cost.
- Still accepted: "turn up for the weekend", self-echo "...hey mycroft, play", "testing testing can
  you hear me" (all P >= 0.999), plus 4 pre-LLM accepts from exact/classifier.
- n=34 negatives: 20.6% is 7 rows; the margin under 25% is one row. Needs more real negatives
  before trusting it, and a single test-split confirmation at the milestone.

## Phase 2 MILESTONE — TEST split, frozen config (2026-10-12)

Config frozen on dev before this run: candidate -> LLM fallback (strict prompt, shortlist k=12,
tool_choice=required) -> addressee gate T=0.9. One test consultation; nothing tuned afterwards.
98/276 test rows (36%) reached the LLM; LLM p50 970 ms, p95 1162 ms (+ gate ~32 ms).

| criterion (test) | stock | candidate (Phase 1) | **Phase 2** | bar |
|---|---|---|---|---|
| 1 magic phrases (n=103) | 96.1 | 99.0 | **100.0** | PASS |
| 3 real Hub (n=64, gating) | 42.2 | 75.0 | **82.8** (p=3e-8, 26+/0-) | PASS |
| 3 real Mac (n=63, reported) | 69.8 | 74.6 | **84.1** (p=0.023) | pass, not gating |
| 4 false accept (n=42) | 100.0 | 7.1 | **21.4** (9 rows) | PASS |
| 5 high-risk wrong executions | 0 | 0 | **0** | PASS |

Criterion 2 (determinism): the LLM path is temp 0 and cached in eval; live repeatability of vLLM
under batching is not yet measured. Criterion 6 (mini PC latency) still deferred.

The LLM costs false accepts (3 -> 9 of 42). The 6 it added: "restart the movie from the beginning"
-> common_play stop, "i need a caffeine break" -> create_timer, "that's an open question" -> knowledge,
self-echo "and i am an intelligent personal assistant" -> who_are_you, and two background
"make me laugh ..." -> dad joke. The gate passed all six (P >= 0.966). Label questions for the maintainer
(not changed; test rows): "restart the movie from the beginning" and the "make me laugh ..." rows
may be real requests.

### Post-milestone label correction (2026-10-12, owner ruling — not a tuned result)

The maintainer, after seeing the test false accepts: the two "make me laugh ..." Hub rows are his real requests,
with someone else in the house speaking loudly before VAD cut off. Relabelled `joke.tell`, new tag
`trailing_speech` (build_dataset.py). "restart the movie from the beginning" stays None (background).
Same predictions, re-scored, no re-run: Phase 2 test Hub 82.8 -> **85.9%** (p=7e-9, 28+/0-),
false accept 21.4 -> **17.5%** (7/40). Phase 1 candidate Hub 75.0 -> 71.9 (it declined those rows).
Both rows are test-split, so dev numbers are unchanged. Upstream-relevant finding: trailing speech
before VAD endpointing is a real household failure mode; the LLM recovers it, the classifier doesn't.

## Phase 2 — LLM repeatability (criterion 2 for the LLM path, 2026-10-12)

`llm_determinism.py`: all 177 routed rows (dev+test), uncached, frozen config, 3 runs.

| run | tool flips | arg flips | gate flips | final flips | max abs dP |
|---|---|---|---|---|---|
| seq1 (1 at a time) | 0 | 0 | 0 | **0/177** | 0.0000 |
| seq2 (1 at a time) | 0 | 0 | 0 | **0/177** | 0.0000 |
| par8 (8 concurrent) | 0 | 0 | 0 | **0/177** | 0.0063 |

vLLM batch composition moves gate probabilities by <= 0.006; nothing near T = 0.9 flips. n = 3 runs on
one model/server config; re-check after any vLLM or model upgrade.

### Stale-input bug found by this measurement (mine), and its correction

The re-runs agreed with each other but differed from the milestone on 1 row (ask_knowledge -> ocp__next),
and 16 prompts missed the cache. Cause: the milestone used an `examples_live.jsonl` built at 22:31, BEFORE
OCP templates were added to introspection.json (22:35); I regenerated it during the label correction.
The shortlist and the Phase 1 classifier read that file; the exact layer reads introspection.json directly.
Re-ran the frozen config end to end on the regenerated (correct) file, no selection:
Phase 1 changed 5/522 predictions; **bar results identical on dev and test** (test: magic 100, Hub 85.9,
Mac 84.1, FA 17.5, high-risk 0). Committed results are now `preds/v2__*`, reproducible from committed
introspection.json via harvest_live.py.

## External negatives stress slice (2026-10-12)

`build_ext_negatives.py` -> `negatives_ext.jsonl`: 600 rows, gold None, 200 each from AMI meetings (CC BY 4.0),
MAGPIE spoken-BNC idioms whose first word starts a live template (CC BY 4.0), DailyDialog turns
(CC BY-NC-SA: private eval only). Neon Hub STT form. Separate from criterion 4. `score_ext.py` scores it.
Worst-case framing: it assumes every one of these sentences produced a false wake.

| system | FA all | ami | magpie | dailydialog | rows sent to LLM |
|---|---|---|---|---|---|
| Phase 1 (exact + classifier + decline) | **1.3%** (8) | 1.5 | 0.0 | 2.5 | — |
| Phase 2 frozen (LLM strict + gate 0.9) | **14.3%** (86) | 9.5 | 8.0 | 25.5 | 592/600, p50 945 ms |

- The LLM adds 78 false accepts; 56 are `ask_knowledge` on questions people ask EACH OTHER ("do you think
  you studied enough for the exam", "could you do me a favour"). The addressee gate passes them: they are
  phrased to "you". Text alone can't tell who "you" is.
- Policy lever, measured on dev + this slice (not test): of the 78 LLM false accepts, 71 come from rows the
  decline CLASS won; only 7 from below-threshold rows. But 13 of the 21 dev household LLM fixes also come
  from decline-class rows (mostly knowledge questions). Routing only below-threshold rows: external FA
  ~2.5%, LLM calls 592 -> 35, but loses 13/21 dev fixes. Same shape on both sides: questions.
- Implication: the remaining separation needs non-text signal (wake-word confidence, whether the assistant
  just spoke, ASR confidence, speaker). Decision for the maintainer: which side of the trade to take.

## Clarify tool experiment (2026-10-12)

Single variable: `tools.py --clarify` adds `clarify(question)` to the meta tools (`tools_clarify.json`);
prompt, shortlist, gate 0.9 unchanged. A clarification SPEAKS: it counts as a false accept on negatives
and never as correct on requests. `clarify_compare.py` classifies each question against the frozen run.

| | frozen | + clarify |
|---|---|---|
| dev Hub (full routing) | 81.7 | 77.5 |
| dev household FA (n=34) | 20.6 | 32.4 FAIL |
| external FA, full routing | 14.3 | 29.5 |
| external FA, uncertain-only routing | 2.5 | 4.5 |

- Dev: 7 questions asked = 3 recoveries ("use military time", "lunch mail", "make me love"),
  3 interruptions of overheard Hub speech ("please", "where you want", self-echo), 1 softened wrong action.
  0 lost fixes. 5 more clarify calls were suppressed by the addressee gate.
- External: 111 questions, 0 recoveries possible, 86 interruptions where the frozen run stayed silent,
  25 softened (replaced an ask_knowledge answer with a question).
- Why: the property that makes speech "ambiguous" to the LLM (fragments, garble, mid-conversation turns)
  is the same property that marks it as overheard. Without an addressee signal, clarify mostly fires on
  speech nobody aimed at the assistant.
- Not adopted. Revisit once Phase 3 provides an addressee signal (active conversation / explicit wake
  word): clarify only when there is evidence the user is talking to the assistant.

## Phase 3 start — addressee signal from real Hub logs (2026-10-12)

`hub_context.py` joins all 343 Hub transcripts (voice.log) with 720 assistant utterances (audio.log
"Speak:", since 2026-07-15); 152 transcripts match labelled dataset rows. Raw logs stay in data/hub/
(gitignored, household data). Not available: wake-word confidence (detected on satellites, not logged
centrally); STT confidence is always 1.0.

**Recency of assistant speech is a real addressee signal:**

| assistant last spoke | n | addressed |
|---|---|---|
| < 30 s ago | 61 | **87%** |
| 30 s - 2 min | 18 | 72% |
| > 2 min / never | 73 | **49%** |

(addressed = request, trailing_speech, or a reply within 30 s of the assistant asking a question)

- **Echo:** longest contiguous run of words shared with assistant speech in the last 60 s. Requests reuse the
  assistant's words, not its phrases: 0/100 requests reach run >= 4; true self-echo does (run 9). Word-overlap
  fraction did NOT separate (11 requests >= 0.5). n of true echoes is tiny (2-3).
- **"Context replies" are mostly human-human:** 26/37 came > 2 min after the assistant spoke. Only 2 followed
  an assistant question ("Yes." after "Shall I read them all?"); those are addressed and belong to converse.
- Label questions for the maintainer: "ai communities the mccroft ai community team" (11 s after the assistant said
  "...OpenVoice OS and Neon dot AI community") and "initiate key aspects of all time" (21 s into the assistant
  reading news, run 4) are labelled background_speech but look like self-echo.
- Correction: raw Hub STT output has casing and punctuation ("Launch the Reminders app."); the lowercase form
  is my harvest normalization. Dataset and external slice share it, so results are consistent; a live
  deployment must apply the same normalization.

## Phase 3 addressee rules — overfitting check and whole-pipeline evaluation (2026-10-12)

The maintainer confirmed the two "background_speech" rows were the Hub hearing itself -> retagged self_echo.
`context_policy.py` recombines existing predictions under rules using `hub_row_context.jsonl` (first
occurrence per Hub row; 111/135 Hub rows have log context; rows without it, plus hand/Mac/external rows,
are COLD). No model calls, so sweeps are deterministic.

**Contamination:** the 30 s window and echo>=4 were first chosen looking at ALL Hub rows (test included).
Mitigations: (1) dev-only tables put the addressed-rate cliff at 2 min, not 30 s; (2) threshold sweep on dev;
(3) config declared before the test run; (4) the clean test is PROSPECTIVE: Hub traffic after 2026-10-12.

Dev sweep (route = decline-class rows go to the LLM only while the assistant spoke within the window):
- Window is smooth, not knife-edge: Hub 73.2 (15 s), 74.6 (30 s), 77.5 (60 s, 120 s), 78.9 (300 s);
  household FA 17.6 at every window.
- Echo rule: no effect on dev at >=4/5/6 (dev's one self-echo was already declined); >=3 costs a magic phrase.
  Untested guard; n=1 on dev, and the test self-echo rows were also already declined.
- Clarify gated to active conversation still raises FA 17.6 -> 23.5 (overheard "please", "where you want"
  landed within 20 s of the assistant speaking). Stays off.

Declared config: echo>=4, active 120 s, route=active, clarify=never.

| | route=never (upstream default) | **context policy** | route=always (opt-in) |
|---|---|---|---|
| dev Hub / household FA | 71.8 / 17.6 | **77.5 / 17.6** | 81.7 / 20.6 |
| test Hub / household FA | 75.0 / 10.0 | **78.1 / 10.0** (p=6e-6 vs stock) | 85.9 / 17.5 |
| external overheard FA | 2.5 | **2.5** (all cold) | 14.3 |

The context policy dominates route=never on every slice (same FA, higher Hub). It recovers part of full
routing's Hub gain (dev 5.7 of 9.9 points, test 3.1 of 10.9) at zero household or external FA cost.
Mac rows have no timing context, so they behave as cold; Mac OVOS would need equivalent speak/transcribe logs.

## Mac timing context (2026-10-12)

The maintainer pointed out the Mac logs ARE timestamped (I had asserted otherwise without checking):
~/.local/state/mycroft/voice.log "Raw transcription:" + audio.log "Speak:", since 2026-02. `hub_context.py --device mac`
-> mac_row_context.jsonl (83/128 Mac rows). Mac use is bursty testing: 87/101 Mac requests and 16/25 Mac
negatives arrive within 2 min of assistant speech. Declared context policy with Mac context: dev Mac 80.0 -> 83.1
(= route=always), household FA unchanged 17.6; test Mac 84.1 under all policies. No regressions.

## Prefix rule for trailing speech (2026-10-12)

`bench.py --prefix-min-literal K`: for rows the classifier declines, slot-free templates with >= K literal words
also match as a PREFIX followed by more speech (unique owner). Runs before the LLM. Constraints fixed before results.

| K | dataset rows routed (all TEST) | correct | external false accepts |
|---|---|---|---|
| 2 | 4 | 3 | 4 (greetings, "thank you for all the help") |
| **3** | 4 | 3 | **0** |
| 4 | 0 | 0 | 0 |

K=3 fixes both trailing-speech rows ("make me laugh hey dad ...") and "turn it up on the kitchen speaker";
one wrong ("what's the temperature of the living room" -> weather.current, gold device.status).
**Unvalidated:** zero dev rows affected, and the idea came from test rows. Candidate for the prospective holdout,
not yet part of the declared config.

## Hub label mappings applied; schema v3 (2026-10-12)

The maintainer reviewed hub_mapping_proposals.yaml. Applied: 91 Hub labels to existing canonical intents; new canonical
`camera`, `demo`, `email.draft` (DraftEmailIntent stays in scope, unlike other comms); `out_of_scope` list
(email_chat_history, contact_support = comms; audio recording and speed test dropped) = excluded from scoring.
Verified from source: device_controls `exit` asks numeric confirmation then emits neon.shutdown -> system.shutdown.
Pending the maintainer: skill-homeassistant enable / disable / get.all.devices (not settings.change; target TBD).

Side effect: the classifier trains on every canonical intent's description, so 3 new classes entered the MAC
model too (no Mac skill backs them). 3/522 Phase 1 predictions changed (2 correct -> declined, 1 new knowledge
accept). Re-ran the frozen config end to end (preds/v3__*, no selection): **bar numbers identical to v2 on dev and
test under both routing policies**. check_router.py now verifies against v3.
Design flaw surfaced (predates today; lock.*/cover.* too): the classifier's class set is the whole schema, not
what the DEVICE can execute. Candidate fix: per-device class set (handled intents + __decline__), the rest to the
LLM. Needs a dev measurement before adoption.

## Hub router: Neon's own skills (2026-10-12)

`DeviceConfig.hub()`: introspection_hub.json (83 padatious + 6 OCP + 87 adapt parsed from source), examples_hub.jsonl,
synthetic + Hub dialog negatives (1,732), tools_hub.json (180 tools; Adapt tools described by keyword combos).
`ha.admin` added (the maintainer). `eval_device_router.py` scores a device router on the bar slices. ("magic phrases" is
defined by MAC templates: not meaningful for the Hub router. Mac rows are not Hub traffic.)

First measurement: Hub router Phase 1 on dev Hub rows 53.5% vs Mac router 70.4%. Not an Adapt-starvation effect:
10/14 losses were app launches routed to web.browse because `skill-launcher.neongeckocom:launch_program` was
pre-mapped (Phase 0, from the repo) to web.browse, contradicting the maintainer's app.launch labels. Remapped -> app.launch.

| dev, Hub rows (n=71) | Hub | household FA |
|---|---|---|
| Mac router, Phase 1 | 70.4 | 14.7 |
| **Hub router, Phase 1** | **74.6** | 14.7 |
| Mac router, full routing | 81.7 | 20.6 |
| **Hub router, full routing** | **81.7** (p=2e-6) | 23.5 |
| Mac router, context rule | 77.5 | 17.6 |
| **Hub router, context rule** | **78.9** | 17.6 |

The approach carries over to Neon's own skill set. Mac baseline unaffected (check_router 522/522).
LLM is punctuation-sensitive: "whats shaking" (dataset form) routes to knowledge, "what's shaking" (Neon's logged,
post-transformer form) is declined -> apply our own normalization before routing.

## Hub history replay (2026-10-12)

`replay_hub.py --policy always`: all 689 utterances Neon's intent service handled since 2026-07-02, routed by the
Hub router (Neon's own skills) with exactly the text Neon saw, next to Neon's ACTUAL logged decision.
7 converse rows excluded (converse runs before any intent pipeline). Since voice.log began (2026-09-17), 323/330
inputs are voice; earlier rows can't be split voice vs typed.

| | ours | Neon (actual) |
|---|---|---|
| labeled occurrences (n=240, 126 distinct) | **85.8%** | 69.2% |
| discordant | **60** ours-only right | 20 Neon-only right (McNemar p = 9e-6) |
| labeled overheard (gold None, n=51) | **acted 4** | acted 51 |
| agreement on all 682 scored | 69% | |

Neon-only-right: 12x "who's your daddy (and what does he do)" (our LLM -> assistant.about; labelled knowledge.ask),
4x "what is the weather right now" -> forecast, 3x "what's shaking" (punctuation sensitivity).
Caveats: labels are occurrence-weighted (repeats count); designs were chosen with these rows in view (dev/test
contamination as before); Neon's history spans skill-set changes, ours uses today's skills (e.g. the old HA skill).
Instrument check: an apparent parser bug ("play steve steinberg" -> knowledge.ask) was real Neon behaviour: 13
occurrences, OCP on most, LLM fallback on 3 within 4 minutes on 2026-08-02.
Hub-router weaknesses surfaced in unlabeled rows: "what is the weather" -> weather.forecast (x20: Neon weather is
Adapt-only, 2 keyword examples for current weather), "thank you" -> translate.phrase (x15).
`replay_candidates.txt`: 81 unlabeled disagreement utterances with draft labels for the maintainer (no system outputs shown).

## Experiment: device class sets and shared canonical phrases (2026-10-12, dev only, not adopted)

Diagnosis of the two Hub misroutes the maintainer flagged:
- "what is the weather" -> forecast: Hub weather.current trains on 201 KEYWORD fragments ("temperature", "high",
  even "forecast": HandleCurrentWeather is one_of(weather, forecast)) and no full sentence. Adapt gets it right by
  structure (no day word = current), which a classifier can't learn from fragments.
- "thank you" -> translate.phrase (0.71): the only training text containing it is synthetic translation examples
  ("how do you say thank you in arabic"); synthetic rows carry no slot spans, so slot VALUES are learned as
  intent evidence. The Hub has no thank-you skill. Fix B (slot-annotate synthetic examples) pending the maintainer's label.
The maintainer: "who's your daddy" -> assistant.about is reasonable, not a bug.

`DeviceConfig(device_classes_only=True)`: train only canonical intents with a handler on this device
(+ knowledge.ask, stop); `shared_examples`: also borrow other devices' declared phrases for those intents.
Phase 1 (dev): Hub router on Hub rows 74.6 -> 76.1 (devcls) -> 78.9 (+Mac phrases); Mac router on Mac rows 76.9
throughout. Full pipeline, Hub router, dev Hub rows:

| | always: Hub / FA | active: Hub / FA |
|---|---|---|
| baseline | 81.7 / 23.5 | 78.9 / 17.6 |
| device classes only | **84.5 / 20.6** | **83.1 / 11.8** |
| + Mac phrases | 85.9 / 23.5 | 84.5 / 17.6 |

Device classes only dominates the baseline on both axes. Shared phrases fix "what is the weather" (forecast 0.58
-> current 0.998) but add near-miss false accepts via Mac templates ("the kids are monsters today" -> monster.scan);
dev has no "what is the weather" rows, so judge it on the replay after the maintainer labels replay_candidates.txt.

### Slot-masked synthetic examples (Fix B) and the bar's criterion 3 (2026-10-12)

The maintainer: "thank you" with no thank-you skill is small talk -> knowledge.ask via the fallback.
`annotate_synthetic.py`: the GPU host marks slot values in the 460 synthetic rows of slot-bearing intents (JSON schema;
every value verified as an exact substring: 0 rejected) -> synthetic_examples_slotted.jsonl ("masked" text) +
synthetic_slots_review.txt (the maintainer reviews). bench.load_synthetic trains on "masked" when present.
"thank you": translate.phrase 0.74 -> declined (falls to the LLM); "translate thank you into spanish" still 1.0.

Full pipeline, dev, each device on its OWN traffic:

| | always: own rows / FA | active: own rows / FA |
|---|---|---|
| Hub router baseline (Hub rows) | 81.7 / 23.5 | 78.9 / 17.6 |
| Hub router devcls + masked | **84.5 / 20.6** | **83.1 / 11.8** |
| Hub router devcls + masked + Mac phrases | 84.5 / 23.5 | 83.1 / 17.6 |
| Mac router baseline (Mac rows) | 83.1 / 20.6 | 83.1 / 17.6 |
| Mac router devcls + masked | **83.1 / 17.6** | **83.1 / 14.7** |

Device classes + masking dominates the baseline on both devices. Shared phrases: no dev gain, more FA; their target
("what is the weather", x20 in history) isn't in dev -> decide on replay labels.
**Criterion 3 re-scope needed:** it was measured with the MAC router on Hub rows (cross-device; the only option before
a Hub router existed). With device class sets the Mac router no longer claims Hub-only intents (its Hub-row score
falls 81.7 -> 67.6), which is correct behaviour. Proposed: score each device's router on its own traffic.

### Replay with the maintainer's labels: class-set variants on the full history (2026-10-12)

`replay_labels.jsonl` (the maintainer-reviewed replay_candidates.txt; "thank you" -> knowledge.ask by ruling; 13 SMS rows
PENDING the text-message category) overlays at scoring time. Kept out of dataset.jsonl: rows were selected by
disagreement, which suits a paired comparison but would bias the bar's slices.

| Hub router, full routing | labelled history (n=360) | ours-only / Neon-only right | McNemar | acted on overheard (n=71) |
|---|---|---|---|---|
| current | 72.8% | 103 / 76 | p=0.05 | 7 |
| devcls + masked | 73.3% | 105 / 76 | p=0.04 | 6 |
| **devcls + masked + Mac phrases** | **79.4%** | 104 / **53** | **p=5e-5** | 7 |
| Neon (actual) | 65.3% | | | 71 |

- Mac phrases fix all 17 "what is the weather" (forecast -> current) with no change in overheard accepts.
- "thank you" (x15): masking stops the translate.phrase misroute, but the LLM then calls `ignore` (strict prompt:
  "a remark or reaction ... ignore"). The maintainer's ruling is small talk -> fallback. Root cause: the LLM sees no
  context; this "thank you" came 11 s after the assistant spoke. Candidate: pass gap + last assistant line to the
  LLM prompt (requires re-measuring dev, history, external).
- Dev showed Mac phrases adding near-miss FAs (34-row mixed slice); the history's 71 REAL overheard rows show none.

## Adopted config and v5 re-baseline (2026-10-12)

The maintainer's decisions: calls out of scope (skill-communication:call -> out_of_scope); texts -> new canonical
`message.send` (skill-communication:SendMessageIntent; the maintainer removed the blacklist entry that blocked texting over
the Node loopback; verified live: call.intent + SendMessageIntent now register); "thanks"/"thank you" = small talk
-> knowledge.ask (dataset "thanks" relabelled; hello-world thank_you_intent + greetings -> knowledge.ask); criterion 3
= each device's router on its own traffic (CLAUDE.md); adopted router defaults: device class sets + slot-masked
synthetic for both devices, Mac phrases shared into the Hub router.

| | dev: own rows / FA | test: own rows / FA |
|---|---|---|
| Hub router, always | 84.5 / 27.3 (FAIL, 9/33) | 90.6 (p=9e-10) / 17.5 |
| Hub router, active | 81.7 / 24.2 | 84.4 / 12.5 |
| Mac router, always | 83.1 / 18.2 | 84.1 / 20.0 |
| Mac router, active | 83.1 / 15.2 | 84.1 / 12.5 |

New Hub dev false accepts, both consequences of today's decisions: "call the plumber" -> skill-communication call
(exact layer; the skill is loaded, so a real Hub would place the call) and "thank advice" -> knowledge.ask (thanks
phrases now train small talk). References: preds/v5__<device>__<route>__<SPLIT>.jsonl; check_router.py now
regression-tests the router against them. Hub log archive runs daily via launchd (ops/install_archive_job.sh;
archive moved to ~/.local/share/intent-eval/hub-archive because launchd can't read ~/Documents).

## Slots, first pass: template extraction (2026-10-12)

`slots.py` SlotExtractor: the matched template's {slot} wildcards become named groups. Exact-layer picks use a full
match; classifier/prefix picks try the predicted intent's own templates as a LOOSE matcher (extra words allowed
before the template, end anchored, trailing please/thanks trimmed). Template slot names -> canonical slots via
intents.yaml renames, else same-name, else the single-slot correspondence; unmappable slots are dropped.
LLM rows keep tool arguments. Hub: `shared_manifests="introspection.json"` lends the Mac's templates for SLOTS
ONLY (Neon alerts are Adapt-only, so alarm/timer/reminder had nothing to extract from); the exact layer still uses
only the device's own templates. Intent results unchanged.

| slot exact match (rows with correct intent and gold slots) | dev | test (once) |
|---|---|---|
| Mac router: before -> after | 4.2 -> **79.2** (n=48) | 8.2 -> **76.7** (n=73) |
| Hub router: before -> own templates -> + Mac templates | 3.8 -> 40.4 -> **73.1** (n=52) | 3.8 -> **72.2** (n=79) |

Remaining misses (dev): values with determiners ("play some jazz" -> query "some jazz"), semantic slots templates
can't express (settings: "use military time" -> setting=time format), LLM rows whose tools lack those params.
Next: deterministic typed parsers (duration/time/date/number), then Home Assistant entity resolution.

## LLM conversation context: rejected (2026-10-12)

The maintainer approved trying it to fix "thank you". `DeviceConfig(llm_context=True)`: the LLM and the gate get "the assistant
last spoke N s ago and said: ..." (prompts/fallback_context.txt, addressee_gate_context.txt). First run also changed
the prompt for NO-context rows (two variables at once): 3 of 4 new false accepts were context-free hand-written rows.
Fixed: context prompts only when a note exists. Then:
- dev Hub: 84.5/27.3 -> 83.1/30.3 (always), 81.7/24.2 -> 80.3/27.3 (active); dev Mac FA 18.2 -> 24.2, 15.2 -> 21.2.
- History (n=374 labelled): 83.7% -> 83.4%; acted on overheard 7/71 -> **14/71**.
Same lesson as clarify: "the assistant just spoke" is too weak an addressee signal to hand the LLM. Not adopted.
"thank you" is fixed WITHOUT it: hello-world thank-you phrases (now small talk) shared into the Hub classifier
route all 15 to knowledge.ask at the classifier.

Adopted Hub router on the full archived history (693 utterances, 374 labelled): **83.7% vs Neon 62.8%**,
114 vs 36 discordant (McNemar p ~ 1e-10); acted on overheard 7/71 vs Neon 71/71.
Calls ("call the plumber" -> call intent): accepted by the maintainer as the cost of enabling texting.

## Entity resolution, first pass (2026-10-12)

`resolver.py`: intent -> eligible HA domains; device-type words narrow them ("lamp" -> light/switch, "garage" -> cover);
char n-gram cosine vs friendly name (authoritative) and entity_id slug (x0.9: slugs go stale after renames);
accept if >= threshold and beats the runner-up by margin, else UNRESOLVED (first-class: an upstream misroute
handing "me laugh" to the HA skill should not act). Registry via the home-assistant skill (`ha-client states`,
read-only; 1,438 entities; snapshot gitignored). No per-home aliases.
Eval (`resolver_eval.py`): The maintainer's house split by hash into tune/test (resolver_gold_home.tsv, DRAFT gold for
the maintainer's review, drafted from names not resolver output); a second house = HA's demo integration
(ha_demo_registry.json from HA source; omitted test-only players Browse/Group/Search; gold by Claude's judgment).

| (threshold 0.7, margin 0.1, selected on home TUNE) | correct | wrong entity | acted on nothing | missed |
|---|---|---|---|---|
| home TUNE (n=16) | 11 | 0 | 1 | 4 |
| home TEST (n=26) | 15 | 0 | 0 | 11 |
| demo house (n=34) | 30 | 0 | 1 | 3 |

Precision is the strength (0 wrong entities); recall on ASR noise ("desk liam", "gas clamp") is weak at 0.7.
Caveats: type words + slug weight were added after seeing ALL errors (test included), so home TEST and demo are no
longer clean; tune n=16 and the threshold hinges on one uncertain gold ("own bedroom light"). Clean check =
device names in future Hub traffic (daily archive).

### Entity resolution: not the router's job (the maintainer, 2026-10-12)

The skills resolve entities themselves (skill-homeassistant fuzzy-matches the raw entity string, and handled almost
all the garbled examples). The router passes the right intent + raw slot string + original utterance; matching stays
in the skill (no duplicate matcher, no skill changes). resolver.py stays as an evaluation tool, NOT in the pipeline.
Typed slot parsers dropped for the same reason (skills parse their own values). Slot work = raw-string extraction
(slots.py, 73-79% exact match). Gold entity answers withheld here: they name the home's devices.

## Stage B: skill label selection + skill-native match_data (gap 1 + gap 2, 2026-10-12)

Live dispatch needs `skill_id:intent_name` and the data the skill reads; canonical intents are for sharing and
scoring. Hub: 21 canonical intents have >1 label (settings.change 29, weather.current 12); Mac: 15.
`labels.py` LabelSelector, from the device's own declarations only: Adapt rules evaluated like Adapt (require /
one_of / optionally over the skill's .voc phrases and .rx regex entities, now carried in introspection_hub.json),
yielding Adapt-style match_data ({"location": "seattle"}); Padatious templates (full, then loose) yielding slot
values under the template's own names; nearest label examples as the last resort; ties go to the more specific
match. Unresolvable: DuckMore requires DuckKnows, a runtime Adapt context keyword (only fires mid-conversation).

Proxy check (`label_agreement.py`, no label gold): replayed history rows where Neon and our router agree on the
canonical intent and Neon dispatched to a skill intent -> **multi-label 162/165, single-label 248/248** agree with
Neon's label. The 3 misses: "pause/unpause on office speaker", where OCP and Music Assistant BOTH declare pause.
router.route() now returns `label` + `match_data` (LLM: the tool's label + arguments). Intents unchanged.
Name decided (the maintainer): the plugin is **Switchboard**; silent decline is its own component.

## Experiment: classifier dispatches unmapped intents (2026-10-12, not adopted)

Trigger: live Mac shadow run. "where were you born" topped the classifier at 0.959, but
`ovos-skill-personal.openvoiceos:where_were_you_born` is not in intents.yaml, so the classifier passed it to the
LLM (right answer, +1 s). On the Mac, 62 of 146 registered intents (42%) are unmapped.
Single variable: Switchboard `ClassifierOptions.dispatch_unmapped` (branch exp/dispatch-unmapped). `exp_unmapped.py`
routes every row through the packaged router with the flag off and on; cached LLM answers only.

| | magic | own traffic | FA | hx |
|---|---|---|---|---|
| dev, all 4 device/route configs | unchanged | unchanged | unchanged | 0 |
| test mac (always, active) | 100 = | 84.1 -> 82.5 | = | 0 |
| test hub (always, active) | 97.1 = | = | = | 0 |

4 changed decisions = 2 utterances (`preds/exp_unmapped_flips.jsonl`), both wrong:
- "how far is the moon from earth" (Mac, real): knowledge.ask via LLM -> date-time `time_until` (0.578).
- "how do i unlock my phone" (Hub): LLM declined -> Neon communication **`call`** (0.676). Would place a call.
Blind spot: gold is canonical, so a CORRECT unmapped dispatch can never score; this harness can only show harm.
The replay labels are canonical too: no label-level gold exists anywhere. Lead, not a setting: both losses sat at
0.58-0.68, the live win at 0.96, so a stricter bar for unmapped picks may separate them (n=3, two from test).
Not adopted. Needs label-level evaluation data first (docs/dynamic-schema.md in switchboard).
Draft mappings for the Mac's 62 unmapped labels: `mac_mapping_proposals.yaml` (awaiting the maintainer).

Wake phrases in transcripts (same night): Hub 1/347 (0.3%), Mac 9/598 (1.5%); 2 of the Mac's are the assistant's
own alarm speech read back. Leading wake phrase cost one exact match live ("Hey Mycroft, where were you born").

## Experiment: wake-phrase stripping and the prefix stage (2026-10-13)

The maintainer, after the live "Hey Mycroft, where were you born" miss: regex matching does reach unmapped intents (the exact
stage dispatches any registered template); it missed only because the anchored match can't skip a leading wake
phrase. Two variables, alone and together (`exp_wake_prefix.py`, packaged router, cached LLM answers only):
- wake: strip a leading wake phrase before the exact/prefix match (Mac "hey mycroft", Hub "hey neon").
- prefix: PrefixStage after the classifier, slot-free templates of >= 3 literal words matched as a prefix.

Bar (magic/own/FA/hx) is unchanged in every device/route/split except **test hub active: own 84.4 -> 87.5** with
prefix (alone or with wake). 8 changed decisions = the same 2 real rows on both routers' active route, both fixed
(gold joke.tell): "make me laugh hey dad uhoh ..." and "make me laugh let us see i mean ...". 0 losses, 0 FA change.
- wake changes nothing: dataset rows carry no wake phrases, so the harness can't test it (live: 1 miss in 9).
- prefix: still not independently validated. These 2 rows motivated the rule; dev has none. Both features go to
  the Mac shadow run for prospective evidence (`preds/exp_wake_prefix_flips.jsonl`).

## Measurement: does the right canonical reach the right handler? (2026-10-13)

Prompted by the grouping rule (agreed with the maintainer: a canonical groups labels that do the same thing, never sibling
actions of one skill). 29 of 70 canonicals in intents.yaml group siblings (weather.current: 17 handlers;
settings.change: 31). Gold is canonical, so a wrong handler inside the right canonical scores as correct.
`exp_handler_choice.py` uses `seen_in_templates` as partial label gold on rows the v5 router gets canonically right
(active route, dev+test):

| device | judged rows | right handler | sibling (wrong handler) | unjudgeable |
|---|---|---|---|---|
| mac | 204 | 204 | 0 | 0 |
| hub | 104 + 89 | 87 | 17: 16 aliases, **1 real** | 89 |

- Hub sibling aliases: the Hub's randomness build registers `roll-multiple-dice`, the harvest has
  `roll_multiple_dice`: the same intent. Real: "enable spicy mode" -> easter_egg via the classifier, then
  `LabelSelector.select` fell back to nearest examples and picked `malibu_stacey_intent`, not `adult_mode_intent`.
- Hub unjudgeable: gold labels come from OVOS skill templates, the Hub dispatched Neon's equivalent skill
  (`CreateAlarm` for `create_alarm`). Instrument blind spot, not errors.
- Read: 197 of the Mac's 204 are exact-stage dispatches, where the template match picks the handler, so grouping
  can't hurt there. The risk sits in classifier dispatches into multi-handler canonicals (Mac 6, Hub 19 judged
  rows): 1 failure, too few rows to size. No evidence of a widespread problem today; also no evidence against the
  rule for paraphrases, which is where it matters. Retrofitting the existing 29 waits for label-level gold
  (dynamic-schema phase 1). The Mac draft follows the rule now.

## Experiment: the Mac mapping draft on real traffic and on the bar (2026-10-13)

The maintainer: "I want to see evidence and data before I approve anything as final." `exp_mac_draft.py` merges
`mac_mapping_proposals.yaml` (rule-compliant revision) into intents.yaml as the only variable.

Bar (magic/own/FA/hx, active route; cache misses asked of the GPU host and cached):

| split device | base | draft |
|---|---|---|
| dev mac | 99.0 / 83.1 / 15.2 / 0 | 99.0 / 83.1 / 18.2 / 0 |
| dev hub | 90.1 / 81.7 / 24.2 / 0 | 90.1 / 83.1 / 21.2 / 0 |
| test mac | 100.0 / 84.1 / 12.5 / 0 | 100.0 / 82.5 / 10.0 / 0 |
| test hub | 97.1 / 84.4 / 12.5 / 0 | 97.1 / 84.4 / 12.5 / 0 |

Within noise everywhere (about +/-6 points). The Hub moves because it shares the Mac's examples and manifest.

Real Mac traffic (508 archived transcripts, no gold, LLM off so "pass" = would ask the LLM): 15 changed decisions.
- Win: "What does Arnold say?" x5, LLM -> classifier easter_egg (arnold_intent): right (the maintainer, 2026-10-12).
- Wrong new dispatches: "shift" -> alert.reschedule (0.94); "How far is the moon from Earth?" -> date.time_until
  (0.62, the same pick exp_unmapped found); "You started" -> assistant.ready (0.66, likely overheard).
- Neutral (dispatch -> LLM, none of the old dispatches were right): "Who is your daddy and what does he do?" x3
  (was who_made_you), "No, open the curtains... I'm good, thanks" (was thank_you), "Hey, Minecraft" (was greetings).
- Renames only: two exact matches below.

Finding independent of the draft: open-slot templates make the exact layer wrong at confidence 1.0. Alerts'
`(change|set|adjust|make) (my|the|...) {schedkind} to {mediaform}` takes "set the thermostat to 72", "set the music
volume to 30", "set my location to dallas, Texas"; date-time's `how far away is {date}` / `when is {date}`
take "how far away is the moon" and "when is sunset"; reschedule_alert took an overheard sentence about screen time.
OVOS constrains such slots with `.entity` files: alerts' schedkind.entity is a closed list (alarm, timer, reminder,
event, ...); date-time's date.entity is an example list for an open slot. The harvested manifest carries no entity
values. Breaks design goal 1 (exact phrases always right). Not fixed; see the decisions for the maintainer.

Read: mapping an intent is what lets the classifier dispatch it, so the draft's costs are exp_unmapped's again
(confident wrong picks of weakly-trained classes), and its one real-traffic win is a pre-existing class shifting.
Not applied.

## Exact-layer audit: every template, and speech nobody meant as a command (2026-10-13)

The maintainer: "We should be able to test everything, right?" `exact_audit.py` checks design goal 1 (exact phrases always
right) deterministically, no review needed, both devices:
- templates: every expansion of every registered template, slots filled with "the blue box", through the exact layer.
- negatives: assistant dialog files, speak.log, AMI meeting speech (negatives_ext), dataset gold-None rows. Any
  exact dispatch is a finding.

v5 matcher, then with the word-boundary fix (switchboard `fix/exact-guard`):

| | Mac v5 | Mac fixed | Hub v5 | Hub fixed |
|---|---|---|---|---|
| expansions reaching their own intent | 26163/31380 | 26256 | 4476/4554 | 4477 |
| collisions (to the classifier, by design) | 5217 | 5124 | 78 | 77 |
| non-command speech dispatched | 23 | 23 | 14 | 5 |

No expansion is ever stolen by another intent (0 "other"): collisions are the only failure mode, and the largest is
`count to {n}` vs OCP `play {query}`-shaped templates (2625).

Every negative hit is one template shape: ONE literal word plus a slot. Mac: `timer [for] {duration}` takes the
assistant's own "timer for ten minutes, starting now" (x12), `{schedkind} (earlier|later|longer)` takes "give me a
little while longer" and the weather forecast "later it will be overcast", `play {query}` "play it by ear",
`spell ... {word}` "spell it out for me, are you serious". Hub: `Call {{contact}}` takes Neon's own self-introduction
"call me Neon. I am an intelligent personal assistant..." (would place a call; the communication skill is out of scope
but registered), `stop (the|my|) {entity}` "stop taking the mickey". The rest are slot-free greetings ("thank you",
"hi there"), which are real commands when a person says them.

Matcher bug, fixed: a template space matched zero characters, so `play {query}` matched "playing by" and
`(play|start) {query}` matched the assistant's "starting a speed test". Fix: a space needs whitespace or a word edge.
Harness gate 8/8 identical; bar unchanged on every cell. Real Mac transcripts: "Roll 2D20." and "Roll a dime." lose
the exact match (glued tokens) and still route right via the classifier (0.99, 0.94).

`exp_exact_guard.py`, min_literal 2 (one-literal templates stop claiming the exact layer): a wash, not adopted.
| split device | v5 magic/own/FA | ml2 |
|---|---|---|
| dev mac | 99.0 / 83.1 / 15.2 | 96.0 / 83.1 / 15.2 |
| dev hub | 90.1 / 81.7 / 24.2 | 92.1 / 78.9 / 24.2 |
| test mac | 100.0 / 84.1 / 12.5 | 98.1 / 85.7 / 12.5 |
| test hub | 97.1 / 84.4 / 12.5 | 96.1 / 84.4 / 10.0 |
About 11 rows won (catch-alls stealing phrases: "play npr news" -> media.play, "open the pod bay doors" -> app.launch,
"play something like glados" -> media.play) and about 9 lost (real "play X" commands to a weaker classifier: "play
moonlight sonata" -> media.resume). It also drops slot-free one-word commands ("Stop.", "Pause.") from the exact
layer, which is wrong by design: the risky shape is one literal PLUS a slot, not a short template.
Lead: the wins are all cases where the classifier was confident (0.95-0.99) in a different intent than the catch-all,
and the losses are cases where it wasn't. Candidate: a one-literal-plus-slot match yields to a confident classifier
disagreement. Threshold must be chosen on dev only.

## Experiment: guard rules for catch-all templates (2026-10-13)

`exp_guard_rules.py`, on the word-boundary matcher. A catch-all = a slot behind < 2 literal words. Variables: capK
(a catch-all slot takes at most K words; Daniel McKnight: Padatious, undocumented, takes only one word into
`{contact}`, which is why Neon's `Call {contact}` mostly works there) and yieldT (a catch-all match passes when the
classifier ranks a different intent or the decline class first at >= T). Selection rule fixed before results: best
dev own accuracy (mac+hub mean), dev magic and FA no worse than base on both devices; ties -> fewer non-command hits.

Dev (magic/own/FA/hx; non-command exact dispatches mac/hub):
| variant | mac | hub | non-command |
|---|---|---|---|
| base | 99.0/83.1/15.2/0 | 90.1/81.7/24.2/0 | 22/5 |
| yield0.8 | 99.0/84.6/15.2/0 | 92.1/81.7/24.2/0 | 19/3 |
| yield0.9 | 99.0/84.6/15.2/0 | 92.1/81.7/24.2/0 | 19/5 |
| cap1 | 96.0/83.1/15.2/0 | 92.1/78.9/24.2/0 | 4/0 |
| cap2+yield0.8 | 96.0/83.1/15.2/0 | 92.1/81.7/24.2/0 | 4/1 |
(all 16 cells: `preds/exp_guard_rules.txt`). Chosen: **yield0.8**.

Test: Mac 100.0/84.1/12.5/0 -> 100.0/**87.3**/12.5/0; Hub 97.1/84.4/12.5/0 -> **98.1**/84.4/12.5/0 (magic bar >= 98
now met on Hub test). 5 changed test rows, all fixed: "play/say something like glados/lados" (x4 across devices) ->
easter_egg, "open the pod bay doors" -> easter_egg. Real Mac transcripts: 3 changes, all better (the overheard
screen-time sentence no longer reschedules an alert). Non-command speech no longer dispatched: Neon's "call me Neon,
I am an intelligent personal assistant..." (x2) and the forecast "later it will be overcast..." (x3).
Caps buy the most non-command protection but break real catch-all commands ("play the new radiohead album"):
Padatious's one-word slot is a real cost there, not just a quirk. Still dispatched under yield0.8: the assistant's
own "timer for N, starting now" (echo guard territory: 6 words, 60 s window), "call the plumber", "stop taking the
mickey", "play it by ear", greetings.

Implemented as `ExactOptions.catch_all_yield` (switchboard feat/catch-all-yield). `refs_v6.py` writes reference
predictions from THIS experiment's implementation; the packaged gate reproduces them 8/8 identical.

Determinism and latency (bar items 2 and 6, `det_latency.py`, fix/exact-guard): 3 fresh builds in separate
processes, 1044 rows: **0 flips** (lbfgs logistic regression and TF-IDF have no random step). Non-LLM path on an
Apple M3 Max, warm, 1030 utterances: p50 1.09 ms, **p95 1.88 ms**, p99 11.86 ms, max 13.31 ms (classifier tail).
Not a mini-PC measurement.

## Live active-mode trials on the Mac (2026-10-13, owner consent; `live_trial.py`, `trials/`)

`live_trial.py` EXECUTES handlers (recognizer_loop:utterance, default session); the maintainer consented for these runs
(lamps, timers, alarms allowed if reverted; every state change was reverted and verified).

1. **Active, full stock pipeline after Switchboard** (19:59). Declared phrases, wake and prefix right in ms. Leaks: "play
   something like glados" yielded at exact (overruled_by easter_egg 0.968) but stock OCP-high, placed between exact and
   classifier, played music; warm-context non-command speech (utterances ~7 s apart, so "assistant active"): "i
   turned the lamp on for you" -> LLM turn_on (gate p 1.0, a lamp really turned on, reverted), "can you pass me the
   salt" -> classifier decline class passed on -> LLM unsupported -> stock m2v missed_alerts. "play groove salad" was
   never dispatched by Switchboard here (OCP-high first).
2. **Pure: Switchboard + stop, converse, OCP-high after the classifier, common-query, fallback-low; padacioso, padatious,
   m2v, persona, adapt blacklisted** (20:10). Cold non-command speech: 3/4 declined silently by the classifier
   (4th: `{schedkind} later` catch-all, harmless). GLaDOS -> easter egg via classifier ("Still Alive" played). Knowledge
   question -> LLM -> wikipedia (right, but the skill reads the whole intro: needs a length setting). "play groove salad":
   Switchboard dispatched ocp:play and OCP's handler crashed (KeyError media_type). "what is your place of birth" ->
   when_were_you_born (wrong sibling; stock m2v made the same mistake). "turn off the desk lamp": exact collision ->
   LLM, right but ~1.2 s. Restart to fully ready: ~51 s (morning, full stock) -> ~30 s (blacklist).
3. **Fixes deployed (switchboard PRs #9-#11)** (20:26). OCP verdicts delegated to OCP's step: "play groove salad" ->
   OCP matched it (then the YouTube Music skill failed, as it did under stock). Warm "can you pass me the salt" ->
   declined (decline_class_unsupported). Readiness: "I am ready" now 0.2 s AFTER Switchboard is ready; restart to fully
   ready 22.8 s.

Open: the LLM addressee gate scored "i turned the lamp on for you" 1.0 (warm context); `{schedkind} later`; the
"turn off X" exact collision; when/where sibling paraphrase; wikipedia answer length (skill setting).

## Live speed benchmark, pure Switchboard pipeline (2026-10-13, Mac M3 Max; `live_trial.py --repeat 3`)

`trials/2026-10-13-speed.txt`, 14 utterances x 3 rounds, every state change reverted (all 42 routed and verified
right). Times from sending `recognizer_loop:utterance`: decision = ovos.intent.matched, text = first `speak`,
audio = audio_output_start. Raw rows: `preds/live_speed_2026-10-13.jsonl`.

| path | decision (median) | text | audio |
|---|---|---|---|
| exact (time, joke, timers, alarms, lamp on, wake phrase) | 580-710 ms | +20-160 ms | 690-1360 ms (TTS length) |
| prefix ("where were you born again") | 649 ms | 677 | 700 |
| LLM, fresh call (round 1: "place of birth", "turn off the desk lamp") | ~1020-1030 ms | | |
| declines (silent) | 650-684 ms | - | - |
LLM rounds 2-3 hit Switchboard's in-memory response cache (LLM step 17 ms vs 384 ms fresh): only round 1 is a real
LLM time.

Where the decision time goes ("what time is it", bus trace + decision log): Switchboard echo at +19 ms, exact decides
in 4-8 ms, but ~550 ms passes between them in stock **converse**: it pings the 5 active skills (date-time, hangup,
homeassistant, personal, jokes) on the broadcast and per-skill topics, gets NO pong on any topic, and waits out its
fixed 0.5 s window. Stock OVOS pays the same on every utterance (converse runs before every matcher). Lead, unverified:
alpha ovos-core 3.7.4a1 vs ovos-workshop 9.8.16a1 converse protocol mismatch. Switchboard's own share: <10 ms on the
deterministic paths, ~400 ms on a fresh LLM call (the GPU host, Qwen3-14B-AWQ).

## Converse regression and the full live speed matrix (2026-10-13, Mac M3 Max)

**Regression found:** ovos-core marks every dispatched skill a converse candidate and waits for ALL candidates to
answer, up to 0.5 s (unchanged 2023 -> now: checked 2023-05, 2025-06, 2026-01, 2026-08, now). Until ovos-workshop
**#339** ("refactor!: split converse_skill from ovos_skill", JarbasAl, self-merged 2025-06-07, only an automated
CodeRabbit review, released as workshop 5.0.0 the same day), every OVOSSkill answered the per-skill ping with
can_handle False; #339 moved the ping handlers to ConversationalSkill only, so plain skills went silent and every
utterance after any plain skill ran waits the full 0.5 s. The 2026 OVOS-CONVERSE-1 work (core #863, #933) carried it
forward, didn't cause it. Not reported upstream (ovos-core #1027 is a different converse timeout).
Fix: plain skills decline the poll again (`ops/patches/ovos-workshop-plain-skills-decline-converse.patch`, 4 new tests,
fail on upstream, workshop suite otherwise unchanged: the 4 other failures also fail on upstream main). Applied to the
Mac's installed workshop (original kept as ovos.py.orig-2026-10-13). Live: converse round 0.5 s -> 4 ms; get_response
(alerts asking for an alarm time) still works.

Decision time, median of 3 rounds, `trials/2026-10-13-speed.txt` (raw: `preds/live_speed_*_2026-10-13.jsonl`):

| | stock Padacioso, unpatched | Switchboard, unpatched | stock Padacioso, patched | **shipped default Padatious, patched** | **Switchboard, patched** |
|---|---|---|---|---|---|
| declared phrases | 640-710 | 580-710 | 78-137 | 57-114 | 46-108 |
| wake / prefix | 623-636 | 612-649 | 102-135 | 68-100 (wrong) | 58-79 |
| wrong routes per round (of 14) | 5 | 1 | 5 | 5 | 1 |

Wrong routes: Padacioso "turn off the desk lamp" -> cancel_alert (lamp stays on), overheard speech -> GLaDOS easter
egg and missed_alerts, "place of birth" -> fallback 6.3 s then wrong sibling. Padatious: "where were you born again"
and "he said he'd be home around six" -> DuckDuckGo search, "place of birth" -> list_alerts, salt -> missed_alerts,
"hey mycroft, where were you born" -> DDG in round 1 but right in rounds 2-3. Switchboard: only "place of birth" ->
when_were_you_born (wrong sibling, also LLM). Fresh LLM decisions ~420-470 ms patched (rounds 2-3 hit the cache).
Read: with the converse fix, Switchboard ties Padatious on speed (single M3 Max, not a mini PC) and routes this set
correctly 13/14 vs 9/14; the live set is small, so the harness bar carries the accuracy claim.

## Languages and scale, deterministic layers only (2026-10-14, `exp_lang_scale.py`)

The maintainer: OVOS adoption needs multilingual proof and scale. Manifests built straight from skill template files
(`build_lang_manifest.py`, frozen in `manifests/`): the same 31 installed skills for every language; scale tiers add
cloned repos read statically (no skill code imported or run): T1 +OpenVoiceOS org, T2 +OVOSHatchery, T3
+OscillateLabsLLC (duplicates and cookiecutter/template repos skipped). No LLM, no synthetic rows, no dialog negatives
(English-only, so left out for every language). Hold-out: every 5th template line per intent (intents with >= 5 lines)
removed from training; up to 3 expansions each ("lugo" in slots) through the shipped exact step (catch-all yield
0.8) then the classifier (0.5 threshold). Skill-intent level. Template volume per language (installed skills): en
58 lines/intent, gl 125, ca 116, fr 14 (gl has the most template lines, not the most intents).

| manifest | intents | template lines | train rows | classifier train | peak RSS | colliding expansions | hold-out right / wrong / pass | p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| en-us | 146 | 8,459 | 31,028 | 10.2 s | 0.94 GB | 5,328 | 83.0 / 8.6 / 8.4 | 1.2 / 2.3 ms |
| gl-es | 132 | 16,557 | 20,571 | 7.9 s | 0.80 GB | 1,048 | 93.9 / 2.5 / 3.6 | 1.8 / 2.3 ms |
| ca-es | 126 | 14,648 | 31,385 | 10.0 s | 0.99 GB | 5,230 | 93.9 / 2.3 / 3.8 | 1.5 / 2.1 ms |
| fr-fr | 146 | 2,048 | 2,948 | 3.1 s | 0.63 GB | 285 | 69.4 / 7.7 / 23.0 | 0.8 / 1.0 ms |
| en-us T1 | 222 | 8,919 | 37,619 | 20.7 s | 1.59 GB | 5,937 | 80.7 / 8.8 / 10.5 | 1.2 / 1.9 ms |
| en-us T2 | 459 | 10,081 | 42,996 | 59.1 s | 3.94 GB | 8,431 | 72.9 / 13.4 / 13.7 | 1.4 / 2.2 ms |
| en-us T3 | 465 | 10,115 | 43,030 | 56.6 s | 4.16 GB | 8,505 | 69.0 / 17.5 / 13.5 | 1.2 / 1.9 ms |

Peak RSS is the experiment process (two stacks built, plus training): an upper bound, not a device's steady state.
Read: languages with rich templates (gl, ca) do BETTER than English here (fewer cross-skill collisions); thin
locales (fr) lose coverage (pass), not precision. Scale is the weak spot: accuracy falls and training time and memory
grow with classes; most of the accuracy drop is one catch-all template (application launcher, below), not
general degradation. (The ~3,500 extra collisions first blamed on OscillateLabsLLC skills mostly went away with
the maintainer's exclusions: 11,963 -> 8,505 colliding expansions; which excluded skill caused them wasn't measured.)
Latency stays ~1-2 ms at every scale.

T3 rerun (2026-10-14) on the corrected manifest: The maintainer's exclusions applied (openhab, musicassistant,
skill-fallback_llm, podcasts-for-kids, the Neon launcher), 483 -> 465 intents; the old row (66.6 / 18.8 / 14.6,
11,963 colliding) is superseded. T3 adds only 6 intents over T2 (Mac application launcher, meal-plan).

Where the scale errors come from (`exp_scale_errors.py`): ONE template. Of T3's 394 wrong hold-out answers, 304 are
the exact layer routing to `ovos-skill-application-launcher:launch` (T2: 210 of 301; every other error source is
identical between the tiers), and 301 of those are `ovos-skill-count` lines ("start counting forever") caught by
`[please] (launch|start|open|run|fire up) {application}`. The hold-out removes the count skill's own line, so only
the launcher matches; with every template installed the two would collide and the classifier would decide. So the
hold-out overstates this, but it is also the real paraphrase risk: any unlisted "start/open/run ..." phrasing goes
to the launcher. The skill knows: it drops slot values in its `application.blacklist` at runtime, a guard the
exact layer can't see. T2 -> T3 (+94 launcher errors from 6 new intents) is probably the catch-all yield getting
rarer as classes dilute classifier confidence (yield needs >= 0.8 on another intent): likely, not measured.
All tiers (`exp_scale_errors.py`; % of hold-out utterances):
| tier | intents | wrong: launcher catch-all | wrong: everything else | pass |
|---|---|---|---|---|
| en-us | 146 | 113 (5.7%) | 57 (2.9%) | 8.4% |
| T1 | 222 | 116 (5.6%) | 66 (3.2%) | 10.5% |
| T2 | 459 | 210 (9.3%) | 91 (4.0%) | 13.7% |
| T3 | 465 | 304 (13.5%) | 90 (4.0%) | 13.5% |
The launcher/count collision is the largest wrong-answer source at EVERY size, and scale amplifies it (113 -> 304);
other wrong answers creep 2.9 -> 4.0%; passes grow 8.4 -> 13.7% (confidence dilution: more caution, not more
mistakes). The maintainer, 2026-10-14: "so it's not scale per se, it's particular intents?" Mostly yes.
Read: the scale drop in the hold-out is mostly one catch-all, not general degradation; excluding launcher
errors, wrong is ~4% at T2 and T3 alike. But see the yield sweep below: on real traffic this catch-all
causes no errors, so the hold-out overstates it.

Found on the way (fixed): gl-ES `ovos-skill-news` ships 20 templates with an unclosed "(" (e.g. "quero escoitar as
(noticias|novas"); one such line made Switchboard's ExactLayer raise, so no exact/classifier stage built at all for
the language (switchboard `fix/malformed-templates`: malformed lines skipped; typed slots like {number:priority}
kept; the first version of that fix wrongly dropped typed slots and was corrected after en-us training rows fell
31,028 -> 27,641). The Mac's installed news skill has the 20 lines patched (original kept as .orig-2026-10-13);
upstream issue + PR pending.
Update 2026-10-14: no upstream fix possible or needed. OpenVoiceOS/ovos-skill-news is ARCHIVED (deprecated
2026-09-02, replaced by the OCP media provider `ovos-media-provider-news`, which ships no templates at all; the
issue attempt was refused as read-only). The bug lives only in the archived skill the Mac still runs (0.4.10a4).
The fix stays in ops/patches/ for the record; switchboard #13's guard covers any skill with the same mistake.

## No-LLM configuration (rung 0 of the small-model ladder, 2026-10-14, `exp_no_llm.py`)

Test split, each device's own router; v6 = with Qwen3-14B-AWQ:
| | stock | no LLM | v6 (14B) |
|---|---|---|---|
| magic (Hub router / Mac router) | 96.1 | 95.1 FAIL / 97.1 FAIL | 98.1 / 100 |
| real Hub (Hub router) | 42.2 | 76.6, p=2.7e-5 (25+/3-) | 84.4, p=4.6e-7 |
| real Mac (Mac router) | 69.8 | 77.8, p=0.27 (9+/4-) FAIL | 87.3, p=0.0074 |
| false accept | 100 | 7.5 / 10.0 | 12.5 |
Dev agrees. The LLM is worth ~8-10 points on real traffic and the last points of magic (collisions and yielded
catch-alls the classifier passes); without it Switchboard is more conservative (fewer false accepts) and still beats
stock decisively on Hub traffic, not significantly on the Mac's.

## Classifier memory (2026-10-14, `exp_classifier_memory.py`, `exp_memory_phases.py`, `exp_memory_train.py`)

Hypothesis "the dense weight matrix is the problem" was wrong: at 146 intents the weights are 14.3 MB (17,670
features x 106 classes; 7.1 MB as float32). Vectorizer tightening barely matters there (min_df 2: 15,309 features,
hold-out 83.1 vs 83.0; cap 100k: no change). Memory by build phase, one device build:
| phase | 146 intents | 465 intents (T3) |
|---|---|---|
| imports | 36 MB | 36 MB |
| device + label resolver + exact layer | 151 MB | 162 MB |
| classifier trained (in process) | 921 MB | 3,454 MB (peak 3,517) |
| after training data released | 921 MB | 3,454 MB |
T3 training split: TF-IDF features +296 MB (25,247 features, 4.3M non-zeros); logistic-regression fit +2.76 GB (425
classes; peak 3.7 GB). The SAME T3 model loaded from its saved state in a fresh process: **207 MB** total (82 MB
weights on disk). So the resident gigabytes are the fit's scratch memory, kept by the allocator, not the model.
Decision (the maintainer, 2026-10-14): train in a child process, and try float32, but only together with a full benchmark
rerun; memory optimisation is required before telling OVOS users to install Switchboard, not now. The training PEAK
(3.7 GB at 465 intents) is a separate problem for Pi-class devices even in a child process (solver choice, float32
features: unmeasured).
py-spy (the maintainer, 2026-10-14; `preds/pyspy_train_T3.txt`, Python frames only, 6,346 samples ~63 s, T3 training):
logistic-regression fit (L-BFGS) 93.7% of the time: sparse x dense products in loss/gradient 55.7%, the L-BFGS-B
routine itself 28.7%; TF-IDF ~2%, device/training data ~2%. One root cause for time AND memory: 425 classes x 25,247
features = 10.7M parameters; L-BFGS keeps 10 correction pairs of parameter-sized vectors (86 MB each, ~1.7 GB) plus
gradients and copies, matching the ~2.8 GB the fit adds. Levers for the memory work: fewer features (min_df), a
history-free solver (saga/SGD) or one-vs-rest, float32. All need the full benchmark rerun.

## Small-model ladder: the LLM step on local models (2026-10-14, `exp_small_llm.py`, LM Studio on the Mac M3 Max)

The maintainer: "if we can have an option that is quick and accurate and smaller that would be better." Same frozen v6 config,
prompts, shortlist (12 skill tools + 4 meta) and routers; only the LLM changes. LM Studio differs from vLLM in three
ways the client works around (see the script docstring): no logprobs (the addressee gate reads a text yes/no, so
P(yes) is 1.0 or 0.0, uncalibrated), `chat_template_kwargs` ignored (Qwen3: " /no_think" appended; Qwen3.6:
`reasoning_effort: "none"`), tool_choice=required not enforced (a text answer that is exactly a tool name is taken
as that call: 51 salvaged for 4B, 0 for 1.7B; accuracy unchanged either way). Each device's own router; magic and
false accept shown as Hub router / Mac router.

| | magic test | magic dev | real Hub test / dev | real Mac test / dev | FA test | FA dev | route p50 | gate p50 |
|---|---|---|---|---|---|---|---|---|
| stock OVOS | 96.1 | 92.1 | 42.2 / 49.3 | 69.8 / 69.2 | 100 | 90.9 | | |
| no LLM | 95.1 / 97.1 FAIL | 89.1 / 97.0 FAIL | 76.6 / 76.1 | 77.8 / 78.5 (n.s.) | 7.5 / 10 | 24.2 / 15.2 | | |
| Qwen3-1.7B (local) | 98.1 / 100 | 92.1 / 99.0 | 81.2 / 77.5 | 84.1 / 83.1 | 12.5 / 10 | 24.2 / 15.2 | 956 ms | 116 ms |
| Qwen3-4B (local) | 98.1 / 99.0 | 92.1 / 99.0 | 81.2 / 81.7 | 84.1 / 81.5 | 15 / 17.5 | **33.3** / 21.2 | 2.2 s | 226 ms |
| Qwen3.6-35B-A3B (local, MoE) | 98.1 / 99.0 | 90.1 / 98.0 | 85.9 / 78.9 | 87.3 / 83.1 | 10 / 12.5 | 24.2 / 18.2 | (8.4 s, swapping) | (4.1 s, swapping) |
| Qwen3-14B-AWQ (v6, an RTX 3090, vLLM) | 98.1 / 100 | 92.1 / 99.0 | 84.4 / 81.7 | 87.3 / 84.6 | 12.5 | 24.2 / 15.2 | ~400 ms | |

Dev magic on the Hub router is 92.1 for every LLM config: magic rows are defined by the Mac's installed templates,
and the Hub runs different skills; the same 8 rows miss each time (volume up/down/louder/quieter/turn it up ->
volume.set or none, "turn on grandma mode" -> device.on, an easter egg, the shopping list). Stock's 92.1 is a
coincidence. The Mac router's 99.0 is the magic number that matters on dev.
Read:
- 1.7B passes every test criterion and every dev criterion the 14B passes; it gives up ~3 points of real traffic
  (Hub test 81.2 vs 84.4; Mac 84.1 vs 87.3; within noise at n~65-70) for a 1.84 GB model file (4B: 2.28 GB).
- 4B is not better than 1.7B: same test accuracy, slower, and it accepts more non-commands (dev Hub router 33.3%,
  FAILS criterion 4; test 15-17.5%). One-directional: 7 non-command rows (dev+test) 4B accepted and 1.7B declined,
  0 the other way ("time flies" -> knowledge.ask, "set the table" -> alerts, "honey where are my keys" ->
  settings.query). Bigger is not monotonically better at declining with these prompts.
- 35B-A3B is level with the 14B, not better: best test Hub score of any rung (85.9 vs 84.4) but lower on dev
  Hub (78.9 vs 81.7) and dev magic (98.0 vs 99.0); within noise everywhere. 1 unusable answer. Its 20.5 GB
  doesn't fit beside the OVOS stack on the 36 GB Mac: the run swapped (16 GB swap in use), so its timings are
  void, and a first attempt run alongside the Galician experiment ran at ~15 s per call. Not a co-hosting option.
- 6 of 1.7B's answers were unusable (neither a tool call nor a tool name): the plugin declined them.
- Latency: 1.7B re-timed uncached with nothing else running but the OVOS stack (2026-10-14, `--cache` on a fresh
  file): route p50 965 / p95 1,153 ms, gate 114 ms; matches the first run, so that wasn't load-distorted. The
  4B re-timed the same way (below). Apple-silicon Metal timings say nothing about a mini PC CPU.
- Repeatability: the uncached 1.7B rerun reproduced all four prediction files byte for byte (temperature 0).
  The 4B did NOT: 9 of ~178 LLM-routed decisions changed (2 gate answers, 2 now unusable, 5 different tools,
  e.g. device.off -> lights.color; `preds/llm-qwen3-4b__*__rerun.jsonl`). Conclusions hold: the rerun
  reproduces the dev FA failure (33.3%); test moves <= 1 point (Mac-router magic 99.0 -> 98.1), dev real -1.4/-1.5.
  Cause unmeasured (LM Studio / llama.cpp Metal kernels, or the model): a property of the serving stack, not of
  Switchboard's deterministic stages. 4B uncached timing: route p50 1,978 / p95 2,411 ms, gate 204 ms.

## Galician end to end, natural paraphrases (2026-10-14, `exp_galician.py`, `data/gl/dataset_gl.jsonl`)

62 rows (48 requests over 26 intents, 14 non-commands) drafted by Claude: The maintainer doesn't speak Galician and asked me to
use my own judgement. NOT native-speaker reviewed; 6 rows I'm unsure of are flagged `uncertain` (5 requests, 1
non-command) and scored both ways. Natural phrasings, not template text, so this tests generalisation, unlike the
template hold-out above (93.9 right). Device: the installed skills' gl-ES templates; decline class from 1,909 gl-ES
.dialog lines; English prompts; cold context for every row (upstream "active" policy). Needs switchboard #13
(malformed templates) and #14 (unknown tool).

| | right | wrong | pass | false accept | right, excluding uncertain |
|---|---|---|---|---|---|
| no LLM | 68.8 | 8.3 | 22.9 | 14.3 (2/14) | 69.8 |
| Qwen3-1.7B (local) | 75.0 | 8.3 | 16.7 | 14.3 (2/14) | 72.1 |

For scale, English real traffic on the Mac router is 77.8 (no LLM) / 84.1 (1.7B), on a different and larger set: a
rough comparison only. Read:
- Galician fails like English: mostly passes (coverage), few wrong answers, and false accepts well under the 25% bar.
- The LLM fixed 3 rows ("sabes que hora temos?", "a cantos estamos hoxe?", "que día da semana cae o vinte e cinco?")
  and broke none. It only sees rows the classifier didn't keep for itself.
- Most remaining passes are the decline class winning: 5 of the 7 passed requests checked have `__decline__` on top,
  3 with the right intent second (humidity -> weather.current, "tira un dado" -> random.dice). Under the upstream
  policy a cold decline-class win never reaches the LLM. The maintainer's opt-in "always" policy would route them: untested
  on Galician. Not a data gap: decline rows come from each language's own .dialog files, which ship wherever a
  skill is translated. Why the class over-wins is unmeasured; one hypothesis is balance (English intents get
  reviewed synthetic rows, Galician ones don't). The maintainer, 2026-10-14: call out the concern for the first iteration,
  don't fix it yet. What IS missing per language is native-reviewed evaluation data.
- "dime algo sobre a catedral de Santiago" -> alerts.list via the EXACT layer: the alerts skill's gl-ES template
  `(lista|amosa|dime|le) {alertkind}` is a verb + slot catch-all (like English `open {x}`). The classifier wasn't
  confident about knowledge.ask, so the template didn't yield. A skill-template issue, and a reason to keep the
  catch-all yield rule.
- Found on the way (switchboard #14): Qwen3-1.7B answered with a tool that doesn't exist, `easter_eggs_joke` (a blend
  of `easter_eggs__*` and `icanhazdadjokes__joke`), and the LLM stage crashed with KeyError. Servers without enforced
  tool choice (LM Studio, Ollama) allow this; it now declines as `unknown_tool`.

## Experiment: a lower catch-all yield against the launcher catch-all (2026-10-14, `exp_yield_sweep.py`, not adopted)

The maintainer asked for the launcher fix first. Shipped `catch_all_yield` 0.8 (exp_guard_rules tried 0.8-0.95, nothing
lower). Swept down to 0.01 (= yield whenever another intent ranks first), NO LLM. Selection rule fixed before
results: no worse than 0.8 on dev magic, own and FA on both devices, then lowest T3 hold-out wrong.

| yield | dev Mac magic/own/FA | dev Hub magic/own/FA | non-command exact mac/hub | hold-out 146 r/w/p | hold-out 465 r/w/p |
|---|---|---|---|---|---|
| 0.8 (shipped) | 97.0/78.5/15.2 | 89.1/76.1/24.2 | 19/3 | 83.0/8.6/8.4 | 69.0/17.5/13.5 |
| 0.6 | 96.0/76.9/15.2 | 89.1/76.1/24.2 | 18/3 | 84.1/7.5/8.4 | 77.5/9.0/13.5 |
| 0.5 | 96.0/76.9/15.2 | 89.1/76.1/24.2 | 18/3 | 88.5/3.1/8.4 | 77.7/8.7/13.5 |
| 0.4 | 96.0/76.9/15.2 | 89.1/76.1/24.2 | 17/3 | 88.5/3.0/8.5 | 77.7/8.2/14.1 |
| 0.3 | 94.1/75.4/15.2 | 88.1/74.6/24.2 | 17/2 | 88.5/2.9/8.5 | 77.7/4.1/18.2 |
| 0.2 | 94.1/75.4/15.2 | 88.1/74.6/24.2 | 17/2 | 88.5/2.8/8.6 | 77.7/4.0/18.3 |
| 0.01 | 93.1/75.4/15.2 | 88.1/74.6/24.2 | 17/2 | 88.5/2.8/8.6 | 77.7/3.9/18.4 |
Chosen by the rule: **0.8, unchanged**. Every lower value loses real dev rows; no high-risk executions anywhere.

The rows that flip (dev + test, both devices) are the answer: they are legitimate verb + slot commands a lower yield
breaks: "say ram a limiting dong", "say thank you", "say hi to john" (speak.repeat -> pass or knowledge.ask),
"play moonlight sonata", "play samoa fm" (media.play -> media.resume), "play the album random access memories...",
"open my reminders" (app.launch -> pass). One gain: "start a pasta timer for 11 minutes" (media.play -> timer.set,
at 0.6 on the Hub only). NO bar row anywhere is a launcher misroute: the 113-304 launcher errors exist only in the
hold-out, where the victim's own template has been removed, turning a real-world collision (classifier decides)
into a free win for the catch-all.
Read: the confidence threshold cannot tell "say/play {x}" (the slot IS the request, should win) from "start {x}"
taking another skill's phrase. The hold-out overstates catch-all harm; the real residual risk is UNLISTED
paraphrases that share a verb with a catch-all ("start counting to ten" if the count skill doesn't list it).
Not tried: per-template signals (e.g. does the slot value look like the catch-all's vocabulary). The launcher's
runtime `application.blacklist` (pronouns, doors, garage, gate, blinds, news, weather, computer) doesn't cover
"counting", so honouring it wouldn't fix this case. It IS a real gap elsewhere (measured, no LLM): on the Mac,
"open the garage / garage door / blinds / gate" collide (HA vs launcher), go to the classifier and pass: safe.
On the Hub, Neon's launcher (`skill-launcher.neongeckocom:launch_program`) takes all four plus "open the
weather" through the EXACT layer: no garage opens (wrong intent, not a high-risk execution), but the user gets
a launcher reply. The OVOS launcher's runtime blacklist exists to stop exactly this; Switchboard can't see it.
Lever (untested): read `*.blacklist` files and refuse a catch-all match whose slot value is blacklisted.

## Experiment: 6 vs 12 skill tools for the small model (2026-10-14, `exp_small_llm.py --k 6`, Qwen3-1.7B)

Prompt size is the small model's latency lever (12 tools ~1,740 prompt tokens). Same everything else.
| | magic test (Hub/Mac) | real Hub test / dev | real Mac test / dev | FA test | FA dev | route p50 / p95 |
|---|---|---|---|---|---|---|
| 12 tools (shipped) | 98.1 / 100 | 81.2 / 77.5 | 84.1 / 83.1 | 12.5 / 10 | 24.2 / 15.2 | 965 / 1,153 ms |
| 6 tools | 97.1 FAIL / 100 | 79.7 / 77.5 | 84.1 / 81.5 | 15 / 12.5 | 24.2 / 15.2 | 623 / 776 ms |
Read: 6 tools save ~340 ms per LLM route (-35%) and cost ~1.5 points on real traffic plus one magic row on the Hub
router (below the 98% bar). Never better. Keep 12; `shortlist_k` 6 is a documented option where latency matters
more than the last points (e.g. a CPU-only LLM host).

## Measurement: a small LLM on CPU only (2026-10-14, `llama-bench`, Qwen3-1.7B Q8_0 GGUF, M3 Max with no GPU)

The maintainer: people without an LLM host may still want one; is a ~1.7B model usable on CPU? The ladder's 1.7B/4B were MLX
builds (Apple GPU); this is the Q8_0 GGUF of the same model (answers not checked for equality with MLX).
Prompt size from the 1.7B cache (`prompt_tokens`): route with 12 skill tools median 1,751 (max 2,262), with 6
tools 1,155; gate 107. Each skill tool ~100 tokens; the fixed part (system prompt + 4 meta tools + utterance) ~550.
| llama-bench, `-ngl 0 -dev none` | 4 threads | 8 threads | 12 threads |
|---|---|---|---|
| prompt reading, 1,800 tokens (Accelerate BLAS = Apple AMX coprocessor) | 160 t/s | 281 t/s | 245 t/s |
| prompt reading, 1,800 tokens, micro-batch 16 (plain CPU kernels, no BLAS) | 137 t/s | 224 t/s | |
| generation, 16 tokens | 61 t/s | 60 t/s | 35 t/s |
Read: prompt reading is the whole cost. Plain CPU kernels at 8 threads: a 12-tool route ~7.8 s, 6 tools ~5.2 s,
gate ~0.5 s, generation ~0.3 s. M3 Max performance cores are far faster than a mini PC's (an N100 has 4 efficiency
cores): there it would be tens of seconds (estimate, not measured). Prompt caching could reuse only the fixed ~550
tokens. Not viable for voice on typical home CPUs with this prompt design. The maintainer then raised decision models
(discriminative scorers with low latency) as the direction to look at instead.

## Measurement: the shipped build on the external overheard slice (2026-10-14, `exp_ext_active.py`)

Closes the gap the evaluation doc listed: the `active` policy had only an estimate (~2.5%). Shipped Hub router,
every sentence cold (a false wake with the assistant silent), Qwen3-14B-AWQ on the GPU host. `score_ext.py`:
| variant | false accept (600) | ami | dailydialog | magpie | accepted by stage |
|---|---|---|---|---|---|
| no LLM | 1.8% (11) | 2.0 | 2.5 | 1.0 | classifier 10, exact 1 |
| **active (default)** | **2.7% (16)** | 3.0 | 4.0 | 1.0 | classifier 10, llm 5, exact 1 |
| always (opt-in) | 13.8% (83) | 9.5 | 25.0 | 7.0 | llm 72 (ask_knowledge 59), classifier 10, exact 1 |
`always` reproduces the 2026-10-12 14.3% (instrument check). The default policy keeps the LLM away from cold
overheard speech: 5 LLM accepts instead of 72.

## Experiment: Jev-style decision models in place of the LLM (2026-10-14, `exp_decider.py`, CPU only)

The maintainer: homes without an LLM host; decision models (Jev and the open reproductions on the Jev Decision Index,
huggingface.co/spaces/multimodalart/jev-decision-index) return probabilities, no text, at low latency. The index has
benchmarks shaped like our task: CLINC150+OOS (assistant intent or out of scope), When2Call (answer / call / ask /
decline), BANKING77. Small open-weight entries with good scores there, checked for license and code:
| model | size | CLINC | When2Call | license | used |
|---|---|---|---|---|---|
| tasksource-decider-nano (v4) | 150M encoder | 0.867 | 0.786 | Apache-2.0 | yes |
| DecisionTune 1.0 | 395M ModernBERT-large | 0.919 | 0.690 | Apache-2.0, permissive data, signed manifest, ONNX | smoke only |
| Crivo-150M | 150M | 0.918 | 0.719 | CC-BY-NC-4.0 | no (non-commercial) |
| Dinah-0 | 150M | 0.888 | 0.784 | CC-BY-NC-4.0 | no |
| vLLM-SR d3-lite | 0.85B decoder | 0.940 | 0.796 | Apache-2.0, much custom code | not yet |
| OneJev1.5-0.8B | 0.8B decoder | 0.911 | 0.875 | Apache-2.0 | not yet |
(Jev itself: 0.893 / 0.810, hosted only.) Inference code reviewed before running and vendored in `third_party/`;
DecisionTune files verified against its manifest SHA-256s.

Setup: the model replaces only the LLM CLIENT in Switchboard's LLM stage (shortlist, policy, dispatch unchanged):
route = one choice over the offered tools (tool name + the LLM's tool description), empty tool arguments; gate = one
yes/no. Framing swept on dev (with/without the `ignore` option, long/short gate question, gate 0.3-0.9), rule fixed
before results.

Result, tasksource-decider-nano: **adds nothing with this framing.** Every dev cell is within a row of no LLM (dev
Mac 97.0/78.5/15.2, Hub 89.1/76.1/24.2); the chosen cell's TEST numbers equal no LLM exactly (Mac 97.1/77.8/10.0,
Hub 95.1/76.6/7.5): it declines what reaches it. Smoke test: "kill the porch light" -> ignore; the gate gives real
commands 0.22-0.31. **Latency is the good part: route p50 140 ms, gate 33 ms on CPU (10 threads); 175 / 28 ms at 4
threads** (M3 Max cores), against ~8 s for the 1.7B LLM on CPU.
DecisionTune smoke test (5 utterances, CPU shared with the sweep): gate near 0 for everything (0.02-0.26), routes
mostly wrong; ~450 ms per route, ~85 ms per gate.
Read: this tests our FRAMING as much as the models. Both were trained on short plain option descriptions
(DecisionTune's card: short descriptions "give much better results"); ours are the LLM tool descriptions
('name: Example requests: "..."; "..."'), and "is this addressed to an assistant" is unlike anything in their
training mix. Next (not run): a small pre-registered set of framings (plain intent descriptions; no gate, a "none of
these" option plus a probability threshold), selected on dev. The maintainer: if a decider is adopted it should be its own
pipeline stage, so users choose LLM or decider (or both in order).

### Round 2: plain-language options (pre-registered, `exp_decider.py --framing plain`)

Options in plain words (a tool's first description sentence + up to 2 examples; "a general question or chit-chat for
the assistant"; "stop what the assistant is doing"; no `unsupported`/`ignore`), framings A (NONE option, no gate,
act if top probability >= tau), B (no NONE, short gate), C (NONE + short gate). Same rule, dev then test once.
| test | real Hub (Hub router) | real Mac (Mac router) | magic Hub / Mac router | FA Hub / Mac | route p50 (CPU) |
|---|---|---|---|---|---|
| no LLM | 76.6 (p=2.7e-5 vs stock) | 77.8 (p=0.27) | 95.1 / 97.1 | 7.5 / 10.0 | |
| **decider-nano, plain, A tau 0.7** | **79.7 (p=8.4e-6)** | **81.0 (p=0.065)** | 96.1 / **98.1** | 7.5 / 10.0 | **82 ms, no gate call** |
| DecisionTune 1.0, plain, chosen B 0.7 | 76.6 | 77.8 | 95.1 / 97.1 | 7.5 / 10.0 | 130 ms + gate 50 ms |
| Qwen3-1.7B LLM (Apple GPU, MLX) | 81.2 | 84.1 | 98.1 / 100 | 12.5 / 10 | 965 ms + gate 114 ms |
Read:
- The nano, asked in plain words, recovers about half the gap between no LLM and the 1.7B LLM with NO increase in
  false accepts and no gate call, at 82 ms on CPU (vs ~8 s for the 1.7B on CPU). But the gain over no LLM is ~2
  rows per device on test: within noise, not demonstrated. It does not pass the full bar (Hub-router magic 96.1;
  real Mac not significant vs stock); neither does no LLM.
- DecisionTune (395M) accepts far too much when it acts (dev FA 30-51% in framing A); the rule's pick declines
  everything (= no LLM). The 150M nano beats it here.
- Framing mattered more than model choice: the nano went from "adds nothing" (round 1) to the best non-LLM result.
  Two rounds of framing selection on dev is still tuning; a second household or fresh rows would be the real test.
- Side measurement (shortlist recall, v6 rows that reach the LLM, gold a skill intent): 12-tool shortlist 66.7%
  Hub / 53.0% Mac, classifier top-12 71.8 / 51.5. But most misses are intents the device has NO tool for (Mac
  25/30, mostly app.launch; Hub 5/9). Where a tool exists, shortlist + classifier together reach 88% on both
  devices; the rest are slang ("kill the lights", "light up the porch") and nonsense speak.repeat rows. Open
  question: the Mac router has no app.launch tool although the Mac runs a launcher skill.
Design (the maintainer): a decider would be its own pipeline step (`ovos-switchboard-decider-pipeline-plugin`), so users pick
LLM or decider, or both in order (decider for confident cases, LLM for what it passes). Not built yet.

## How each configuration fails on real traffic (2026-10-14, test split)

The maintainer: "most of the failures without LLM or Decider come down to false action, which is actually an improvement over
stock". Measured, the improvement holds but the failure mode is the other way round:
| | right | wrong action | did nothing |
|---|---|---|---|
| stock, Hub / Mac | 42.2 / 69.8 | **56.2** / 28.6 | 1.6 / 1.6 |
| no LLM, Hub / Mac | 76.6 / 77.8 | **1.6** / 9.5 | 21.9 / 12.7 |
| decider nano, Hub / Mac | 79.7 / 81.0 | 1.6 / 9.5 | 18.8 / 9.5 |
| 14B LLM, Hub / Mac | 84.4 / 87.3 | 1.6 / 11.1 | 14.1 / 1.6 |
Stock fails by acting wrongly; Switchboard fails by doing nothing. Mac wrong actions stay ~10% at every tier (open).
Decision (the maintainer): the decider ships as an opt-in experimental step (switchboard feat/decider-step); the
recommendation is an LLM on a small GPU or an M-series Mac. "Progress not perfection."

## Classifier memory, part 2: child-process training and float32 (2026-10-14)

`exp_memory_isolated.py` (switchboard feat/isolated-training: a spawned child trains and writes the no-pickle .npz;
the parent loads it), each mode in a fresh process, M3 Max:
| intents | resident after training, in process | resident after training, child process (parent) | child's peak |
|---|---|---|---|
| 146 | 899 MB | **162 MB** | 882 MB |
| 465 | 4,027 MB | **243 MB** | 4,009 MB |
ovos-core keeps 82-94% less; the model is bit-identical to in-process training (switchboard test compares every
array), so no benchmark rerun is needed for this part. The PEAK is unchanged: it moved to a short-lived child.

`exp_memory_float32.py`, 465 intents, in process: float64 peak 4,027 MB, 58.5 s, 39 L-BFGS iterations; float32
peak **3,388 MB (-16%)**, **39.4 s (-33%)**, 36 iterations, float32 coefficients. Prediction was "little": the
L-BFGS correction history stays float64 inside SciPy's L-BFGS-B, so only the feature matrix and temporaries shrink.
A different model (fewer iterations): adopting float32 needs the full benchmark rerun (the maintainer's rule). Typical
install (146 intents) peaks at 882 MB, transient in the child; the ~4 GB peak is the whole-org + Hatchery case.
Remaining peak levers, unmeasured: fewer features (min_df 2 cut features 13% at 146 intents with hold-out
unchanged), a history-free solver (saga/sag), float32 (above).

## Full benchmark: float32 classifier features (2026-10-14, `exp_float32.py` + the usual scripts on a float32 build)

The maintainer: "do the full float32 benchmark". Rule fixed before results: adopt only if (1) no bar criterion flips PASS ->
FAIL on dev or test, (2) no significant loss vs float64 (McNemar) on any real-traffic slice, (3) determinism 0 flips,
(4) non-LLM p95 latency not worse beyond noise. **All four hold.**
- Bar with the 14B LLM, every device x policy x split (8 runs): magic / own / FA / high-risk identical to v6; McNemar
  0 vs 0 discordant on every real slice. 4 rows changed, all outside the bar's slices: "that's enough" stop -> none
  (Hub dev, a loss); "give me a random number from one to ten" count skill (wrong) -> none (Mac dev); "list all my
  meals" wrong -> a different wrong (Mac test, both policies).
- No-LLM bar: identical, dev and test, both devices. Galician set: identical (68.8 / 8.3 / 22.9).
- External overheard slice: no LLM 1.8% and always 13.8% identical; active 2.7% -> 2.5% (16 -> 15).
- Determinism: 3 fresh builds, 1,044 rows, 0 flips.
- Latency (Mac router, LLM off, warm, idle machine): whole path p50 1.10 / p95 2.01 ms (float64) vs 1.10 / 1.91 (float32).
- Hold-outs: within 0.3 points everywhere except T2 (459 intents): 72.9/13.4/13.7 -> 68.8/17.5/13.7. Entirely the
  launcher catch-all: launcher misroutes 210 -> 303, the other 91 errors identical (`exp_scale_errors.py`). The known
  hold-out artifact (the victim's own template held out; no real-traffic row shows it), but the clearest sign yet
  that the 0.8 catch-all yield sits on a knife edge at that scale.
- Cost side: training peak (experiment process, two stacks) T3 4.22 -> 3.32 GB, en-us 0.94 -> 0.78 GB; training
  time T3 56.6 -> 39.8 s, en-us 10.2 -> 8.1 s.
Adopting it changes 4 gate reference rows, so switchboard's references would be regenerated (v7) with it.

## Why the Mac router's wrong actions stay ~10% (Hub 1.6%) (2026-10-14, v7 + no-LLM preds, real Mac rows)

15 real Mac rows acted on wrongly (dev + test), almost all identical with and without the LLM, mostly EXACT layer:
- Right handler, scored wrong (6): the OVOS weather skill's single `weather` intent DECLARES "what is the weather
  tomorrow / tonight / on monday ..." among its own templates (it serves current and forecast), and the OVOS alerts
  skill's single `cancel_alert` serves timers, alarms and reminders. intents.yaml maps a label to ONE canonical
  intent: `ovos-skill-weather:weather` -> weather.current only; `cancel_alert` -> timer.cancel only (alarm.cancel
  has no mapping). Neon's skills split these (HandleOneDayForecast, ...), which is why the Hub doesn't show it.
- Probably right handler (2): "what's the day today" -> date-time what_weekday_is_it (LLM); "what was the weather
  last night" -> weather skill.
- Label question (1): "thank you" (dev) gold None, but the maintainer's 2026-10-12 ruling: thanks = small talk -> knowledge.ask.
- Real wrong actions (6): "dismiss alarm" -> alarm.set and "dismiss timer" -> timer.set (classifier; "dismiss" is in
  no template), "when is sunset" -> date-time's `when is {x}` catch-all (exact), "who is albert einstein" ->
  assistant.about, "does barbie say" -> speak.repeat, an overheard Bill & Ted quote -> easter egg.
Test only: 3 of the 7 Mac wrong actions are mapping artifacts; real rate ~6% (4/63), not 11.1%.
Pending the maintainer (not changed, per the no-auto-mapping rule): (1) let a label map to several canonical intents
(weather -> current + forecast; cancel_alert -> timer/alarm/reminder cancel), a schema + scorer change; (2) relabel
the dev "thank you" row.

## Scoring: one skill intent serving several canonical intents; "thank you" relabelled (2026-10-14)

The maintainer approved both after the Mac wrong-action analysis.
- dataset: dev u0285 "thank you" None -> knowledge.ask (risk low), as "thanks" (ruling 2026-10-12).
- intents.yaml `serves:` (scoring only): ovos-skill-weather:weather also serves weather.forecast;
  ovos-skill-alerts:cancel_alert and skill-alerts.neongeckocom:CancelAlert also serve alarm.cancel (each verified
  against the skill's templates / Adapt keywords). `serves.py` turns that into (predicted, gold) pairs per device:
  a pair counts only when EVERY label the predicting device has for the predicted intent serves the gold one, so no
  dispatched label is needed and the rule applies to stock (no labels recorded; ran on the Mac's skills) exactly as
  to Switchboard. Result: mac {timer.cancel -> alarm.cancel}, hub {timer.cancel -> alarm.cancel}. The weather pair
  does NOT fire: on the Mac, weather.current also maps humidity / sunrise / sunset / temperature, so a weather.current
  prediction doesn't prove the forecast-serving handler ran. Stock predicted weather.current on all 4 of those rows
  too, so paired comparisons are unaffected; crediting them would need a stock re-run with labels recorded.
- score.correct / mcnemar take each system's pairs; bar.py infers the device from the file name (stock = mac).
  context_policy.evaluate unchanged (past selection runs stay reproducible).
Test, real Mac: stock 69.8 -> 71.4, v7 Mac router 87.3 -> 88.9 (p=0.0074 unchanged), no LLM 77.8 -> 79.4; Hub
unchanged. Credited both ways: stock's "dismiss alarm" -> timer.cancel now counts right (Switchboard: alarm.set,
wrong).
Re-scored, float32 (2026-10-14, after switchboard #18): the decider and Qwen3-1.7B (k12, k6) prediction files
were regenerated on the float32 model (2 new LLM calls per rung; the float64 versions are in git history). Real
desktop test with the serves rule: stock 71.4, no LLM 79.4, decider 82.5, 1.7B 85.7 (p=0.022), 14B 88.9; speaker
unchanged. docs/evaluation.md updated to these (switchboard docs/rescored-numbers).

## Experiment: exact-layer collision priority (Daniel McKnight's idea, 2026-10-14, `exp_collision_priority.py`)

switchboard feat/collision-priority: ExactOptions.collision_priority "none" (default, classifier decides) / "core" /
"user" (the one colliding intent whose skills are all core / all not core wins; core = .openvoiceos, .neongeckocom
suffixes + "ocp", whose id has no suffix). Collisions on the dataset: speaker 0; desktop 13 rows, 5 core-vs-user.
LLM off and 14B, dev + test, serves-aware scoring:
- speaker: no change anywhere.
- core: only harm: "turn off the (living room) lights ..." -> timer.cancel (alerts' `turn off {alertkind}`); with
  the 14B, test magic 100 -> 99.0 ("turn off the living room lights", which the LLM had right).
- user: without an LLM fixes "turn off the living room lights" (test magic 97.1 -> 98.1), "turn off the lights but
  not in the nursery" (dev) and a compound; breaks "turn on grandma mode" (easter egg -> Home Assistant device.on).
  With the 14B only the compound and the nursery row gain; grandma mode still breaks.
- own-device real accuracy and false accepts unchanged in every cell.
Read: default stays "none"; "user" is a reasonable opt-in for LLM-less installs whose own skills overlap core
catch-alls; "core" not recommended.

Follow-up (the maintainer on #20: Neon and OVOS installs each have their own core org; the speaker's 5 .openvoiceos skills
are ones the maintainer added, but NeonCore's default and extended skill sets include OVOS skills (wikipedia, dadjokes; 2024
checkout), so both suffixes stay; and "list specific skills ... if on this list, route there"):
ExactOptions.skill_priority, an ordered list checked before collision_priority; the colliding intent of the
earliest listed skill wins. Same runs:
- [Home Assistant]: identical to "user" (HA is the only user skill in these collisions).
- [easter eggs, Home Assistant]: the lights rows fixed AND grandma mode kept, the only setting with both. Desktop
  magic, no LLM: dev 97.1 -> 98.0, test 97.1 -> 98.1; 14B: dev 99.0 -> 100, test 100. Nothing else changes.
Caveat: 3 rows, and the list was written knowing them; it shows the mechanism works, not a default to ship. Default
stays empty.

## Mini PC latency, bar criterion 6 (2026-10-14, `det_latency.py latency --rows-only`)

The maintainer approved Pan for this: Beelink mini PC, 4-core x86_64, 7.5 GB, Ubuntu, Python 3.12, also a k3s agent with
its normal workloads running (load 1.0-2.3 during the runs; not drained). Switchboard main a45857a, sklearn 1.9.1,
numpy 2.5.4 (Mac 2.4.6). Desktop router, LLM off, warm, the 522 dev+test rows (no archived transcripts on the
host). Two runs; the M3 Max on the same rows for comparison:

| host | build (train) | route p50 | route p95 | p99 | max | peak RSS |
|---|---|---|---|---|---|---|
| Pan run 1 | 76.8 s | 5.70 ms | **9.18 ms** | 39.7 ms | 43.8 ms | 1.92 GB |
| Pan run 2 | 76.8 s | 6.07 ms | **9.47 ms** | 42.3 ms | 49.1 ms | 1.95 GB |
| M3 Max | 21.4 s | 1.15 ms | 2.25 ms | 12.0 ms | 12.4 ms | |

Criterion 6 (non-LLM p95 < 300 ms on a mini PC): **PASS**, ~30x margin. Pan is ~4x slower than the M3 Max per
utterance; exact and classifier split the time evenly; the p99 tail is the classifier. Build and peak RSS are the
harness's in-process build (the OVOS runtime trains in a child process, once, then loads the cached model).
Raspberry Pi not measured.

## Per matcher: Padatious, Adapt and Model2Vec on their own (2026-10-14)

Asked by an OVOS maintainer: the headline compares Switchboard with OVOS's default pipeline as a whole, not with
each matcher. Method: ovos-config's shipped pipeline with the other matchers removed (`run_ovos.py --without`, new),
so stop, converse, OCP and the fallbacks are identical in every variant; the live desktop's intent query path; the
three matchers un-blacklisted for the run and blacklisted again afterwards. Instrument check: the fresh full stock
run matched the published baseline on 516/522 rows; the 6 that differ are hello-world thank_you/greetings, which
the schema has since mapped to knowledge.ask (same routing, newer labels). The fresh run scores test 42.2 / 71.4,
as published. Files: data/baselines/matchers/ (scored with the stock device's serves rule: names start "stock").

Test (dev in brackets), real requests right, declared phrases, non-commands acted on:
| pipeline | speaker | desktop | declared | non-commands |
|---|---|---|---|---|
| default (Adapt + Padatious + Model2Vec) | 42.2 (49.3) | 71.4 (75.4) | 96.1 (93.1) | 100 (90.9) |
| Padatious alone | 42.2 (52.1) | 65.1 (69.2) | 96.1 (92.2) | 100 (90.9) |
| Padatious + Adapt | 42.2 (50.7) | 65.1 (69.2) | 96.1 (92.2) | 100 (90.9) |
| Model2Vec alone (high tier, as shipped) | 18.8 (23.9) | 46.0 (43.1) | 32.0 (32.4) | 100 (84.8) |
| Adapt alone | 18.8 (25.4) | 14.3 (16.9) | 7.8 (9.8) | 100 (84.8) |
| Switchboard, no LLM | 76.6 (76.1) | 79.4 (81.5) | 95.1-97.1 | 7.5-10.0 |
| Switchboard, 14B | 84.4 (81.7) | 88.9 (87.7) | 98.1-100 | 12.5 |
Against Padatious alone (test): no LLM p = 2.7e-5 speaker, 0.012 desktop; 14B p = 4.6e-7, 2.8e-4.
Padatious carries the default pipeline; Adapt adds nothing on top of it here; Model2Vec adds desktop rows.

Latency, live desktop query path (median / p95 ms): default 16.3 / 37.0, Padatious alone 19.3 / 35.9, Padatious +
Adapt 19.6 / 35.5, Model2Vec alone 11.3 / 15.6, Adapt alone 11.8 / 18.7; the desktop's live Switchboard pipeline:
rows decided by exact or classifier (406 of 522) 6.7 / 21.3, rows decided by the 14B 1,043 / 1,240.
The live Switchboard run's accuracy isn't comparable (every query is cold, so "not for me" rows the harness sends
to the LLM with real warm context are declined; and the query path scores a Hangup decline as an action unless
mapped to none: then non-commands 12.5%, matching the harness).

## Collision resolution "specific": the template that matched more words literally wins (2026-10-14)

Found by reading the 44 OVOS golden utterances (en-US) Switchboard left unanswered without an LLM: sibling intents
of one skill whose templates both match ("change my {kind} to repeat {when}" vs "change my {kind} to {time}"). New
ExactOptions.collision_resolution "specific" (default stays "classifier"): after skill_priority and
collision_priority, the one owner whose matching template covers the most utterance words literally (words minus
those slots captured) wins; a tie goes to the classifier. Experiment: exp_specific.py.
Pre-registered: no bar criterion flips PASS -> FAIL; no significant loss on real rows; clean gold not worse;
default only if clean gold also gains.

OVOS golden utterances, skill-intent level, no LLM (right / wrong):
| set | rows | classifier | specific |
|---|---|---|---|
| en-US (motivating), desktop router | 683 | 93.0 / 0.6 | 98.1 / 1.0 |
| ca-ES (clean), ca-es manifest | 59 | 67.8 / 15.3 | 79.7 / 15.3 |
| gl-ES (clean) | 60 | 68.3 / 16.7 | 80.0 / 16.7 |
| fr-FR (clean) | 59 | 74.6 / 15.3 | 81.4 / 15.3 |
Harness, both devices, dev and test, LLM off and on: zero changed predictions. check_harness 8/8 with the default.

The first version counted the TEMPLATE's literal words (optional words counted zero) and sent "set the volume to
five/30" (desktop test, magic rows) to alerts' change_media_properties: magic 100.0 -> 98.1 with the 14B. Counting
the words each MATCH covers fixed it (volume covers "set the volume to", alerts "set the ... to"). That fix was
shaped by a test row, so the harness test split isn't a clean confirmation of the final rule; ca/gl/fr are
(results identical under both counts). All criteria pass, including the bar for default; made the default
in switchboard #29 (check_harness 8/8 identical with it on). Released in 1.1.0. check_harness compares
intents: with specific on, 5 desktop rows (dev 3, test 2) keep their intent but are now decided by the exact step
instead of the classifier (stage and confidence change in the v7/nollm mac preds).

## All 1,007 OVOS golden utterances, live, with the set's skills installed (2026-10-14)

exp_ovos_gold scored the 683 en-US rows whose skill the desktop had. Here the 18 missing skills were installed on
the desktop at the ovos-intents v6.1 pins (ovos-m2v-pipeline train/sources.yaml at e0b2f4e; installed with the
existing environment as constraints: 26 packages added, none changed; snapshot data/baselines/ovos_gold/
mac_env_before.txt). The application launcher was already installed. Skills the desktop had kept their versions.
Every system ran live through the query path; Switchboard 1.1.1, the desktop's pipeline with its 14B, and the same
without the LLM step (route_decline_class never). Runs and the live manifest: data/baselines/ovos_gold_full/.
Experiment: exp_ovos_gold_full.py. Afterwards: conf restored, the 26 packages removed (355 again), audio back.

108 rows no system on this desktop can answer, scored nowhere: mark1-ctrl refuses to load off Mark 1 hardware (47),
the application launcher registers nothing on macOS (16), and the desktop's alerts / volume / date-time / naptime
predate the intent renames the labels use (45). Of the 899 answerable rows, 885 repeat one of their intent's
templates; 14 don't.

| system | answerable (899) | template (885) | unseen (14) |
|---|---|---|---|
| OVOS default | 96.4 | 97.5 | 28.6 |
| Padatious alone | 96.2 | 97.5 | 14.3 |
| Padatious + Adapt | 96.2 | 97.5 | 14.3 |
| Model2Vec alone | 50.5 | 50.5 | 50.0 |
| Adapt alone | 0.0 | 0.0 | 0.0 |
| Switchboard, no LLM | 97.8 | 98.9 | 28.6 |
| Switchboard + 14B | 98.8 | 99.1 | 78.6 |
Paired against OVOS default: no LLM 23 vs 11 rows, p = 0.058; 14B 26 vs 5, p = 1.9e-4.

Found on the way (switchboard #31, released in 1.1.1): installing ovos-skill-days-in-history (136,639 dialog lines)
crashed classifier training. Its lines became decline rows (175k rows, 821k features), and 2 x 10 x features x
classes = 2.7e9 overflowed SciPy L-BFGS-B's 32-bit work-array index (SIGBUS at the same address both times;
reproduced outside OVOS on SciPy 1.17.1 and 1.18.1). Fix: dialog_negatives_exclude (default days-in-history) and at
most 2,000 dialog lines per package. Also: Movie Master announces a missing TMDB key out loud on every load, so OVOS
audio was off during the run.

### Rerun with alerts, volume and naptime at the dataset's pins (2026-10-15)

Upgraded on the desktop (kept): alerts, volume, naptime to the v6.1 pins (constraints: nothing else changed; alerts'
settings and alarm data backed up locally first, alarms unchanged after). date-time not upgraded: its pin needs
timezonefinder 9 -> NumPy 2, and the desktop's OVOS runs on NumPy 1.26. Then the same procedure as above, every
system rerun (data/baselines/ovos_gold_full/ replaced). 927 answerable rows (913 template, 14 unseen); 80 not
(63 skill not loaded, 17 date-time renames).

| system | answerable (927) | template (913) | unseen (14) |
|---|---|---|---|
| OVOS default | 94.2 | 95.2 | 28.6 |
| Padatious alone | 93.6 | 94.9 | 14.3 |
| Padatious + Adapt | 93.6 | 94.9 | 14.3 |
| Model2Vec alone | 49.5 | 49.5 | 50.0 |
| Adapt alone | 0.0 | 0.0 | 0.0 |
| Switchboard, no LLM | 97.8 | 98.9 | 28.6 |
| Switchboard + 14B | 99.1 | 99.5 | 78.6 |
Paired against OVOS default: no LLM 45 vs 11, p = 5.4e-6; 14B 48 vs 2, p = 2.3e-12.
OVOS default lost 22 rows it had right before: timer_status (new in the upgraded alerts) now gets weather (8) and
wallpaper (5) questions; weekday_matches_date -> weekday_for_date (3); create_alarm -> play (2). Not attributable to
the upgrade alone: both runs retrained Padatious, and retraining identical skills already flips 40% of rows
(criterion 2 above). One run per install can't separate the new intent from a different draw.
