# The ladder has no hysteresis, and that blocks the retuning

**Found 2026-08-28, testing a recommendation I had already written into
`next_session.md`. It was wrong, and this is the correction.**

## What I recommended, and why it fails

`finding_analysis_latency.md` showed the deployed estimator is a dominated configuration
and recommended `--window 2 --hop 0.5 --tau 0.5` as "a two-parameter change, no new code".

Replaying PILOT01's raw recording through those settings:

| configuration | prompt changes | median gap | switches under 1 s |
|---|---|---|---|
| deployed (4 s / 1 s / tau 3) | 36 | 3.0 s | **0** |
| retuned (2 s / 0.5 s / tau 0.5) | **459** | 1.0 s | **168** |

37% of switches arrive faster than the crossfade can resolve — worse than the 30% that
made PILOT01's audio unusable. **The recommendation as written would have reproduced the
defect it was meant to follow.**

## It is not the trend suffix, and thresholds cannot fix it

The obvious suspect was the trend hysteresis, calibrated for tau = 3. It is not:

| thresholds | changes | under 1 s |
|---|---|---|
| current (0.35 / 0.175) | 459 | 168 |
| recalibrated 5 x sd (0.191) | 459 | 168 |
| above the observed maximum (0.170) | 459 | 168 |

Identical. Suffix-only changes are **zero** in both configurations — the suffix is inert,
exactly as designed after the 8/16 fix.

What chatters is the **rung**:

| configuration | z sd | rung flips | prompt changes |
|---|---|---|---|
| deployed | 0.92 | 217 | 36 |
| retuned | 1.37 | **1122** | 459 |

`state_rung(z)` is `round(2 + z)` clipped to the ladder. It has **no hysteresis at all**.
The rung flips whenever z crosses a half-integer boundary, and `_DEADBAND_Z` only applies
near the *target*, not at rung edges. A noisier z crosses those boundaries constantly.

The deployed configuration survives this only by accident: `build_prompt` maps several
`here` values onto the same output level, so 217 rung flips collapse into 36 prompt
changes. The mapping absorbs the chatter rather than the controller preventing it. Reduce
the smoothing and the absorption stops being sufficient.

## Adding ladder hysteresis: helps, and is not sufficient

A Schmitt trigger on the rung — leave a rung only once z is `margin` past the boundary:

| configuration | changes | median gap | under 1 s |
|---|---|---|---|
| retuned, no hysteresis | 459 | 1.0 s | 168 |
| retuned + margin 0.25 | 296 | 1.5 s | 68 |
| retuned + margin 0.50 | 208 | 2.0 s | **28** |
| **deployed + margin 0.25** | **14** | **16.0 s** | **0** |

Two conclusions, and they point in different directions.

**The retuning is not ready.** Even at margin 0.5 — which is large, half a rung — 28
switches still arrive inside a crossfade. Making those settings safe needs more than a
margin: a minimum dwell time, or an acceptance that this estimator is too noisy for a
five-rung quantiser. That is real controller work, not a flag change.

**Ladder hysteresis is worth having anyway.** On the *deployed* configuration it cuts
prompt changes from 36 to 14 and raises the median gap from 3 s to 16 s, with zero
sub-crossfade switches. It removes chatter the current system is absorbing by luck rather
than by design.

## Consequences

**`next_session.md` is corrected.** PILOT02 runs at the **deployed** settings. A clean
yoke source at known-good settings is the thing that unblocks the study; an unvalidated
speed-up is not.

**The latency finding still stands.** The deployed estimator remains dominated on
information rate, and the analysis path remains 85% of the budget. What changed is the
cost of acting on it: adopting a faster estimator requires giving the ladder hysteresis
and probably a dwell limit, so it is a controller change rather than a configuration
change.

**The n = 7 projection is unaffected in principle and further away in practice.** It
assumed the information rate transfers; that assumption is untested and now sits behind
controller work as well.

## What I got wrong, and why the check caught it

I wrote "a two-parameter change, no new code" from a benchmark that measured the
*estimator* in isolation. The estimator was fine. The controller downstream of it was
tuned — implicitly, by nobody — against the noise level the estimator happened to produce,
and I did not test the pair together until after recommending the change.

The general lesson is in the numbers above: the deployed system absorbs 217 rung flips
into 36 prompt changes. That absorption is not a designed property, it is a coincidence of
the mapping, and any change to the noise level spends it.
