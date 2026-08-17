# Closed-loop latency: what was measured and what it licenses

Every number here comes from `benchmarks/latency_probe.py`. Raw JSON for each run is
in `benchmarks/`. Reproduce with the command shown under each table.

The claim this document supports is **not** "we made generation fast". It is: *here
is the irreducible latency floor for consumer-EEG closed-loop audio, here is which
term dominates, and here is why the system is architected the way it is.*

---

## 1. The headline

Live MusicGen generation is slower than realtime on every machine and every
configuration tested — including a datacenter GPU.

| machine | backend | best case measured | realtime factor |
|---|---|---|---|
| GTX 1650 Ti (laptop, 4 GB) | transformers | fp16-half, 4 s | **2.08×** |
| Tesla T4 (Colab, 15.6 GB) | transformers | fp32, 4 s | **1.14×** |

Viable streaming needs **< 1.0×**. Nothing measured comes under it. The T4 gets
close, which is the interesting part — but "close" still means the queue starves,
permanently, and audio drops out.

This is why the system uses a precomputed segment library
(`scripts/build_library.py`, `src/library_engine.py`) rather than live generation.
The conclusion does not depend on hardware tier, which is a far stronger position
than "our GPU was too small".

---

## 2. The latency budget, by term

From `--skip-musicgen`, so no GPU is involved. These are structural: they come from
the DSP window geometry, not from anything that can be optimized away.

| term | old (5 s window, 2 s hop, 5-tap boxcar) | current (4 s window, 1 s hop, τ=3 s) |
|---|---|---|
| window centroid delay | 2.50 s | 2.00 s |
| hop quantization | 1.00 s | 0.50 s |
| smoother group delay | 4.00 s | 3.00 s |
| **total analysis lag** | **7.50 s** | **5.50 s** |

The smoother is the largest single term in both configurations — larger than the
window itself. Anyone trying to cut analysis lag further should start there, not at
the window length, which is where the intuition usually goes.

DSP compute per window is **1.3–5.5 ms** — three orders of magnitude below the 1 s
hop. Analysis compute is not a bottleneck and never was; analysis *geometry* is.

The 2.00 s saved here cost no GPU work at all. That is worth stating plainly in the
paper: the cheapest latency win in the whole system came from window design, not
from the model.

```bash
python benchmarks/latency_probe.py --skip-musicgen
```

---

## 3. Between-run variance dominates — read this before trusting any single number

Three runs of the *same configuration* on the *same machine*, audiocraft backend:

| config | min | median | max | max/min |
|---|---|---|---|---|
| fp32 4 s | 12.80 s | 18.05 s | 25.08 s | **1.96×** |
| fp32 8 s | 25.32 s | 27.66 s | 42.00 s | 1.66× |
| fp16 4 s | 12.08 s | 12.56 s | 18.03 s | 1.49× |
| fp16 8 s | 26.27 s | 26.74 s | 47.95 s | 1.82× |

Within any single run, p95/median is 1.01–1.11 — tight. Between runs, the same
configuration varies by up to **1.96×**. Between-run variance beats within-run
variance by an order of magnitude.

**Consequences that shaped everything after:**

- A single run of this probe is not a reproducible measurement on this hardware.
  `--trials` does not fix it, because all trials in a run share one thermal state.
- Report the **range**, not a point estimate.
- Cross-run comparisons on the laptop are unreliable. The same fp32 8 s
  configuration measured 15.47 s and 18.38 s in two runs hours apart — a 1.19× swing
  from thermal state alone.
- Adding an untimed warm-up made fp32 4 s **slower** (12.80 s → 18.05 s). On a
  thermally limited part, warm-up mostly adds heat before the timed trials.
  Warm-up practice borrowed from server-GPU benchmarking is counterproductive here.

The likely mechanism is thermal: a laptop GPU also driving the display throttles
under sustained load, and its clocks depend on thermal history rather than on
anything the benchmark controls.

**What survives all of it:** every measurement in every run lands between 3.0× and
6.3× realtime on the laptop. None is near 1.0×. The architectural conclusion is
robust precisely because it does not depend on which run you believe.

---

## 4. Precision: autocast is the wrong tool, and fp16 is not the win it looks like

All three modes in **one process, one thermal history** — the only way this
comparison is trustworthy on this hardware (see §3).

GTX 1650 Ti, transformers, 3 trials:

| mode | 4 s | 8 s | vs fp32 (4 s / 8 s) |
|---|---|---|---|
| fp32 | 8.80 s | 18.38 s | — |
| fp16 (autocast) | 9.95 s | 19.94 s | **0.88× / 0.92×** — slower |
| fp16-half (`.half()`) | 8.32 s | 16.96 s | **1.06× / 1.08×** — faster |

The two fp16 modes land on **opposite sides of fp32**. That is the whole finding.

- `torch.amp.autocast` inserts a cast at every op it covers. It is built for
  *training*, where those casts amortize over large batched matmuls. Batch-1
  autoregressive decoding has no such matmuls to amortize against, so the casts are
  pure overhead — **16–20% of runtime**.
- Converting weights once with `.half()` pays no per-op cast and is genuinely
  faster than fp32.
- But the format's own win is **6–8%, not the 2× fp16 is supposed to buy**. This
  card has no tensor cores, so the gain is memory bandwidth (half the bytes moved),
  not matmul throughput.

Residual thermal drift works *against* this result rather than for it: fp16-half ran
last and hottest, and still came out fastest.

```bash
python benchmarks/latency_probe.py --backend transformers \
  --precisions fp32 fp16 fp16-half --durations 4 8 --trials 3
```

### The hypothesis this refuted

The original prediction was that fp16 showed no benefit on the 1650 Ti *because the
GTX 16-series has no tensor cores*, and that a tensor-core GPU would show a clear
fp16 win. The T4 tested it directly:

| machine | tensor cores | fp16 (autocast) speedup, 4 s | 8 s |
|---|---|---|---|
| GTX 1650 Ti | no | 1.06× | 0.93× |
| Tesla T4 | **yes** | **0.75×** | **0.80×** |

fp16 autocast is *more* consistently slower on the card that has tensor cores. The
tensor-core explanation is dead. The replacement is about the workload, not the
hardware: batch-1 autoregressive generation is bound by memory bandwidth and kernel
launch overhead, and autocast's per-op casts make both worse — worse on the faster
card, where launch overhead is a larger share of the total.

**Open, and it needs a T4:** `fp16-half` has only been measured on the laptop. On a
T4 the tensor cores could turn a 6–8% bandwidth win into a real one. Rerun the
notebook with `--precisions fp32 fp16 fp16-half`.

---

## 5. Hardware comparison

Same backend, same code, both machines. This is the only valid cross-machine
comparison in this document — see the backend warning in §7.

| config | GTX 1650 Ti | Tesla T4 | T4 advantage |
|---|---|---|---|
| fp32 4 s | 8.89 s | 4.54 s | 1.96× |
| fp32 8 s | 15.47 s | 9.78 s | 1.58× |
| fp16 4 s | 8.37 s | 6.08 s | 1.38× |
| fp16 8 s | 16.60 s | 12.26 s | 1.35× |

The 1.58–1.96× fp32 gap is the expected generational difference, which is a useful
sanity check that both runs measured the same thing.

**Caveat: the T4 column is a single run.** Given §3, one run is not a measurement.
Whether a datacenter GPU has tighter between-run variance than a thermally
throttled laptop is itself an open question — and a result worth having, since it
determines whether cloud benchmarking needs the same range-reporting discipline.
Within-run spread on the T4 was tight (p95/median 1.01–1.09) versus 1.00–1.23 on the
laptop, which is suggestive but not the same measurement.

---

## 6. What the library engine changes

`src/library_engine.py` replaces generation with selection from a prebuilt library.
Verified by `scripts/verify_library.py` (14 checks).

| | streaming | library |
|---|---|---|
| time to produce 8 s of audio | 15.5–27.7 s | **2.4 µs** (selection) |
| worst-case prompt→audible (design bound) | 8.0 s (queue_depth × segment) | **1.0 s** (one crossfade) |
| **measured** prompt→audible, median | up to 8.0 s | **11–13 ms** |
| **measured** prompt→audible, p95 | up to 8.0 s | **967 ms** |
| underruns | permanent starvation | 0 |
| clipped samples | n/a | 0 |
| coverage of controller output | n/a | 20/20 prompts, exact |

Measured through the full closed loop (`live_music.py --engine library`) against
synthetic EEG, 21 switches over a 36 s intervention.

**The median and the p95 mean different things and both matter.** A prompt change
normally takes effect on the next audio block — hence 11–13 ms. The 967 ms p95 is a
change that arrives *while a crossfade is already running* and has to wait for it to
finish. So 1.0 s is a real bound that is genuinely hit, not a theoretical worst case,
and lowering `--crossfade` trades transition smoothness against that tail directly.

Streaming has no equivalent distinction: a prompt change there was inaudible for up
to a full segment with nothing happening in between.

The 8× latency improvement is **not** because generation was slow. It is because
streaming *committed*: once a segment entered the queue it played to completion, so
a prompt change waited up to a full segment no matter how fast the GPU was. A
resident library can abandon the current segment mid-playback and crossfade into the
new prompt from any offset. Removing the commitment is the contribution; removing
the wait only fixes starvation.

**What it gives up:** streaming used `generate_continuation`, conditioning each
segment on the tail of the last, so the music developed as one piece. Library
transitions are equal-power crossfades between independent renders — a blend, not a
development. That is the honest cost of the 8×.

**Why this is a complete solution rather than a fallback:** `build_prompt()` is a
pure function of `(z, target_z, trend)` with a finite range — 5 rungs × 4 trend
variants = 20 strings, and it cannot emit anything else. A library covering those 20
covers the controller's entire output space exactly. Only novelty *within* a state
is lost, and `--variants` buys that back.

---

## 7. Reading rules

**Never compare across backends.** audiocraft and transformers have different
sampling loops and defaults. transformers looks faster than audiocraft on the same
card (8.89 s vs 12.80 s, fp32 4 s) and that difference is not a real speed
difference. Every JSON records `backend` at top level and per row for this reason.
Compare transformers-to-transformers across machines, audiocraft-to-audiocraft
across runs.

**Never quote a single laptop run.** See §3. Report the range.

**Precision comparisons must come from one process.** Cross-run thermal drift on the
laptop (1.19×) is comparable to the effects being measured (6–20%).

---

## 8. Open questions

| question | what it needs | why it matters |
|---|---|---|
| T4 between-run variance | 2–3 more Colab runs | §5 rests on n=1; determines whether cloud benchmarking needs range reporting |
| `fp16-half` on tensor cores | one Colab run with `--precisions fp32 fp16 fp16-half` | the only configuration that could plausibly approach 1.0× realtime |
| Are ladder rungs 0 and 4 dead by design? | a therapeutic decision, not a measurement | `build_prompt` can never emit them under either arm — see `enumerate_prompts.__doc__` |
| Does the crossfade sound acceptable? | listening to `library_engine.py --wav demo.wav` | the one cost of the 8× that no metric captures |
