# Next hardware session: what to run, and why each step earns its place

Written 2026-08-28. Supersedes the "run PILOT02" plan, because two findings changed what
the session needs to accomplish.

**Budget ~2 hours.** Three recordings, in this order. Steps 1 and 2 are gates: if one
fails, stop and fix rather than continuing, because everything after inherits the problem.

---

## Why the plan changed

| finding | consequence |
|---|---|
| The alpha validation was recorded on **TP9/TP10**, the index runs on **AF7/AF8**, and the effect does not transfer (`finding_channel_validation.md`) | The sensing path is unvalidated on the channels the study uses. This gates participant data. |
| The analysis latency is a **dominated configuration**, and retuning may cut required n from 25 to ~7 per arm (`finding_analysis_latency.md`) | Worth measuring on the right channels — but **not worth running a session at**, because the controller chatters under the retuned estimator (`finding_ladder_hysteresis.md`). |

Both are answered by the same session if it is run in the right order.

---

## 0. Before the headset (10 min)

```bash
python scripts/run_tests.py
```

98 checks. If anything fails, fix it before recording.

Charge the Muse. Run the laptop **on battery** — mains through the charger was the largest
single source of 60 Hz contamination during rig validation.

---

## 1. GATE — contact, on the frontal pair specifically

```bash
python scripts/contact_check.py --seconds 30
```

**AF7 and AF8 must both read `GOOD`.** This matters more than it did before: the alpha
test failed on frontal channels because AF8 exceeded 100 µV on 49% of samples and AF7 on
17.8%. PILOT01 managed 3.3% and 7.1%, so it is achievable — it needs work.

Wipe the forehead, push hair clear, dampen the sensors slightly. Run this **with the
headphones on**, since over-ear cups sit where TP9/TP10 do.

Do not proceed on `FAIR`. The whole point of this session is to answer a question that a
marginal recording cannot answer.

---

## 2. GATE — alpha validation on AF7/AF8

**This is the session's primary purpose.**

```bash
python scripts/alpha_test.py --channels AF7,AF8
```

Six 60 s blocks alternating eyes open and closed, with audible cues. ~7 minutes.

| result | meaning | what to do |
|---|---|---|
| ratio ≥ 1.5×, rejection < 15% | the gap closes | proceed; Figure 0 is replaced with this |
| effect present but weak | proceed, reporting frontal SNR as a measured limitation | continue to step 3 |
| **no effect, contact good** | the index is not validated on its own channels | **stop.** Switch to TP9/TP10 or justify β/α without an alpha manipulation. Do not collect participant data. |

Then, immediately:

```bash
python scripts/signal_quality.py --session sessions/<the alphatest you just recorded>
```

**As of 9/5 this now prints an EYE-CLOSURE CHECK below the main table**, and that is the
row to read. The table above it reports prominence over the whole recording, which
averages eyes-open and eyes-closed together — on the August session that scores AF7 at
1.78, "clear alpha peak", a **pass on the channel trap 2 disqualified**. The eye-closure
ratio scores the same channel at 0.75 and flags it. Since this session moves the index to
AF7/AF8, those become the starred channels, so read the ratio and not the prominence.

Anything below **1.2×** is suspect even if its prominence looks healthy.

Read the verdict on the starred channels. `contact_check` answers "is it attached";
this answers "is it cortex". AF7 previously passed the first and failed the second.

---

## 3. The estimator comparison, on the right channels at last

Same recording — no extra session needed:

```bash
python scripts/estimator_sweep.py --session sessions/<the alphatest you just recorded>
```

The sweep has only ever run on TP9/TP10, because that was the only labelled recording.
Running it on AF7/AF8 answers whether the dominated-configuration result holds on the
channels the study uses.

~~Then record what the retuned settings would need…~~ **Removed 2026-09-07. That step
could never have worked.** It named `calibrate_hysteresis.py --session <the alphatest>`,
and that script refuses any session without a completed baseline — an alpha test has no
baseline phase, so it would have printed a refusal and nothing else. Dry-run before you
were standing there with the headset on.

It is also obsolete twice over: the trend thresholds it derived belonged to the trend
suffix, which was deleted on 9/5 because the slope it gated on is smaller than the noise
of the estimator measuring it. The script now reports *whether the trend is measurable at
all*, which is worth knowing on the new channels — but it needs a baseline, so run it in
step 5 on PILOT02 instead:

```bash
python scripts/calibrate_hysteresis.py --session sessions/<PILOT02> --window 2 --hop 0.5 --tau 0.5
```

**Do not change `music_engine.py` during the session, and do not run a session at the
retuned settings** — not because they chatter (the dwell added on 9/5 fixes that) but
because nobody has listened to them yet.

---

## 4. PILOT02, at the DEPLOYED settings

```bash
python src/live_music.py --participant PILOT02 --condition pilot --duration 20
```

**Do not pass `--window 2 --hop 0.5 --tau 0.5` on its own.** An earlier version of this
document recommended it bare, and that was wrong: with no dwell it produces 382 prompt
changes with 136 arriving faster than a crossfade — worse than the defect that made
PILOT01's audio unusable. `live_music.py` now **refuses to start** in that configuration
rather than letting you find out afterwards.

The cause is not the estimator and not the trend thresholds. The rung flips whenever z
crosses a half-integer boundary, and a less-smoothed z crosses them constantly.

As of 9/5 the controller work is done and those settings are survivable **with
`--min-dwell 1.0`**: zero sub-crossfade switches, and 3.68 s end-to-end against the
deployed 6.67 s. **Do not use them in this session anyway.** Its job is a clean yoke
source at known-good settings, and nobody has yet listened to what the retuned controller
sounds like — 299 changes in twenty minutes is click-free but unjudged. See
[finding_ladder_hysteresis.md](finding_ladder_hysteresis.md).

So this session has one purpose:

**A clean yoke source.** Every session on disk is disqualified — the pre-fix ones for
chatter, all of them for the 7 s replay-origin bias. Without one, the sham arm cannot run
at all, which blocks the study regardless of anything else.

Watch for: rejection under ~20%, zero underruns, and no chatter. At the deployed settings
PILOT01 replays to 36 prompt changes with a 3 s median gap and zero sub-crossfade
switches, so anything much faster means something else has changed.

---

## 5. Immediately after, before you put the headset away

```bash
python src/analyze_session.py sessions/<PILOT02>
python scripts/signal_quality.py --all
python scripts/verify_claims.py
```

**Name the session explicitly.** Bare `analyze_session.py` takes the most recently
modified directory. That happens to be PILOT02 if you run these steps in order — but if
you re-record the alpha test after PILOT02, or the session aborts and you retry, it
silently analyses the wrong one and every check below becomes meaningless.

Check, in order:

- `session complete` present, not `session FAILED`
- underruns 0, `missing_prompts` empty
- rejection under ~40%
- `audio_chattering` **false** — if true the session is disqualified as a yoke source.
  At the deployed settings this should not happen; if it does, something has changed and
  the session must be repeated before it can be used
- the starred channels' verdicts

### What `verify_claims` should and should not say afterwards

**Expected to move — update these deliberately, once.** The alpha-validation claims track
the *newest* recording on purpose, because replacing Figure 0 on AF7/AF8 is the point of
this session:

- `eyes-closed alpha ratio`, `same effect on AF7/AF8`
- `alpha ratio at deployed 350 uV`, `effect survives every threshold`
- `deployed detection latency`, `deployed info per minute`, `retuned info per minute`
- `alternatives dominating deployed`, `info rate at the floor / deployed`

**Expected NOT to move.** Every pilot claim is pinned to `PILOT01_20260822_153652` by
name (`PINNED_PILOT` in `verify_claims.py`), so recording PILOT02 must not touch them. If
one of these moves, something is wrong — do **not** update the asserted value:

- `prompt changes before the fix` (491) — a fact about PILOT01's *pre-fix* audio, and the
  evidence behind §6.3. PILOT02 will show ~24 because the defect was fixed; re-basing
  this claim would delete the finding
- `PILOT01 lag-1 autocorrelation`, `effective sample size`, `intervention rejection`
- `power at the registered n = 10`, `participants per arm for 0.15 z` — both in the
  **frozen** plan
- the replay, ladder-margin and trend-noise claims

A test asserts the pin, so a future edit that re-points the pilot claims fails the suite
rather than passing quietly.

---

## What this session cannot do

It cannot produce efficacy data, and it is not a participant session. It answers whether
the instrument measures what it claims on the channels it uses, and produces the clean
yoke source without which the sham arm cannot run. Both currently block the study, and
neither needs ethics approval to answer.

The SRC submission does not depend on this session and should be in flight already.
