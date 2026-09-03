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

30 checks. If anything fails, fix it before recording.

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

Then record what the retuned settings would need, for later analysis rather than for use
today:

```bash
python scripts/calibrate_hysteresis.py --session sessions/<the alphatest> --window 2 --hop 0.5 --tau 0.5
```

**Do not change `music_engine.py` during the session, and do not run a session at the
retuned settings.** Recalibrating the trend thresholds does not fix the chatter — it is
the ladder, not the suffix. Record the numbers; the controller work comes later.

---

## 4. PILOT02, at the DEPLOYED settings

```bash
python src/live_music.py --participant PILOT02 --condition pilot --duration 20
```

**Do not pass `--window 2 --hop 0.5 --tau 0.5`.** An earlier version of this document
recommended it. That recommendation was tested against PILOT01's raw recording and is
wrong: it produces 459 prompt changes with 168 arriving faster than a crossfade — worse
than the defect that made PILOT01's audio unusable. See
[finding_ladder_hysteresis.md](finding_ladder_hysteresis.md).

The cause is not the estimator and not the trend thresholds. `state_rung` has **no
hysteresis**, so the rung flips whenever z crosses a half-integer boundary; a less-smoothed
z crosses them constantly. Fixing it needs controller work, not a flag.

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
python src/analyze_session.py
python scripts/signal_quality.py --all
python scripts/verify_claims.py
```

Check, in order:

- `session complete` present, not `session FAILED`
- underruns 0, `missing_prompts` empty
- rejection under ~40%
- `audio_chattering` **false** — if true the session is disqualified as a yoke source.
  At the deployed settings this should not happen; if it does, something has changed and
  the session must be repeated before it can be used
- the starred channels' verdicts

`verify_claims` will now disagree with several stored numbers, because the alpha
validation and estimator sweep will have new values. **That is expected and correct** —
update the asserted values in `scripts/verify_claims.py` deliberately, once, rather than
loosening the tolerances.

---

## What this session cannot do

It cannot produce efficacy data, and it is not a participant session. It answers whether
the instrument measures what it claims on the channels it uses, and produces the clean
yoke source without which the sham arm cannot run. Both currently block the study, and
neither needs ethics approval to answer.

The SRC submission does not depend on this session and should be in flight already.
