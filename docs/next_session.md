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
| The analysis latency is a **dominated configuration**, and retuning may cut required n from 25 to ~7 per arm (`finding_analysis_latency.md`) | The retuned settings are worth measuring now, because they may move the study from feasibility-only back to powered. |

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

Then derive the thresholds the retuned settings would need:

```bash
python scripts/calibrate_hysteresis.py --session sessions/<the alphatest> --window 2 --hop 0.5 --tau 0.5
```

**Do not change `music_engine.py` during the session.** Record the numbers; decide later.

---

## 4. PILOT02, at the retuned settings

```bash
python src/live_music.py --participant PILOT02 --condition pilot --duration 20 --window 2 --hop 0.5 --tau 0.5
```

Two purposes, and the second is new:

1. **A clean yoke source.** Every session on disk is disqualified — pre-fix ones for
   chatter, all of them for the 7 s replay-origin bias. Without one, the sham arm cannot
   run at all.
2. **Information rate on the real contrast.** The n = 7 projection assumes the
   discriminability measured on an eyes-open/closed contrast carries to the subtler
   adaptive-versus-sham one. This session measures it directly.

**If step 3 shows the retuned settings behave badly on AF7/AF8, run PILOT02 at the
deployed settings instead** and record why. A clean yoke source at known settings is worth
more than an unvalidated speed-up.

Watch for: rejection under ~20%, zero underruns, and **no chatter** — the hysteresis
thresholds were calibrated for τ = 3, and at τ = 0.5 they may be wrong. If prompt changes
come faster than a few seconds, stop and revert to the deployed settings.

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
- `audio_chattering` **false** — if true the session is disqualified as a yoke source and
  the retuning is at fault
- the starred channels' verdicts

`verify_claims` will now disagree with several stored numbers, because the alpha
validation and estimator sweep will have new values. **That is expected and correct** —
update the asserted values in `scripts/verify_claims.py` deliberately, once, rather than
loosening the tolerances.

---

## What this session cannot do

It cannot produce efficacy data, and it is not a participant session. It answers whether
the instrument measures what it claims on the channels it uses, and whether the
retuning is safe. Both currently block the study, and neither needs ethics approval to
answer.

The SRC submission does not depend on this session and should be in flight already.
