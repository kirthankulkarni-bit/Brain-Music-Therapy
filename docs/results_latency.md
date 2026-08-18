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
| Tesla T4 (Colab, 15.6 GB) | transformers | fp32, 4 s | **1.05×** |

Viable streaming needs **< 1.0×**. Nothing measured comes under it, across 3
independent T4 runs and every precision mode. The T4 gets to 1.05× — close enough to
be tantalizing, and still on the wrong side of the line, which means the queue
starves permanently and audio drops out.

The closeness is worth stating precisely, because it is the obvious objection: a
GPU roughly 5% faster would cross 1.0× at 4 s. But §4 shows the workload is bound by
a sequential decode loop rather than by arithmetic, and 8 s segments sit at 1.13×
rather than 1.05×, so the margin shrinks as segments get long enough to be musically
useful. Crossing the line at one duration is not the same as viable streaming.

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
tensor-core explanation is dead.

### The decisive test: fp16-half on tensor cores

If the workload were matmul-bound, this is where it would show. A T4 does fp16 on
tensor cores at roughly **8× its fp32 throughput** (~65 vs ~8.1 TFLOPS), and
`fp16-half` pays no per-op cast. A matmul-bound workload would be dramatically
faster. Tesla T4, transformers, **3 independent runs**, medians:

| mode | 4 s | 8 s | vs fp32 (4 s / 8 s) |
|---|---|---|---|
| fp32 | **4.21 s** | **9.00 s** | — |
| fp16-half | 4.51 s | 9.45 s | 0.94× / 0.95× — slower |
| fp16 (autocast) | 5.61 s | 11.47 s | 0.75× / 0.79× — slower |

**On the T4, fp32 is the fastest mode. Both fp16 modes lose.** The ordering
`fp32 < fp16-half < fp16` held in **6 of 6** run × duration cells — as consistent as
n=3 can be.

This settles the mechanism. Giving the workload 8× the arithmetic throughput makes
it *slower*, so it was never arithmetic-bound. MusicGen at batch 1 is bound by the
**sequential decode loop**: 50 tokens per second of audio, each step depending on the
last, with per-step kernel launch overhead that no numeric format touches. fp16 only
adds conversion cost and different kernel selection.

Note the two machines disagree in *direction* — `fp16-half` is 6–8% faster on the
laptop and 5–6% slower on the T4. That is consistent with the mechanism: the 1650 Ti
is bandwidth-starved enough that halving bytes moved recovers a little, while the T4
has bandwidth to spare and only pays the conversion.

**What this predicts:** a faster GPU will not rescue streaming. The fix would have to
change the decode loop — batching, speculative decoding, or a non-autoregressive
model — not the hardware or the precision. That is a claim about the workload class,
which is worth considerably more than a claim about one laptop.

---

## 5. Hardware comparison

Same backend, same code, both machines. This is the only valid cross-machine
comparison in this document — see the backend warning in §7.

T4 column is the median of 3 independent runs; laptop column is a single run of the
same configuration.

| config | GTX 1650 Ti | Tesla T4 | T4 advantage |
|---|---|---|---|
| fp32 4 s | 8.89 s | 4.21 s | 2.11× |
| fp32 8 s | 15.47 s | 9.00 s | 1.72× |
| fp16 4 s | 8.37 s | 5.61 s | 1.49× |
| fp16 8 s | 16.60 s | 11.47 s | 1.45× |

The 1.7–2.1× fp32 gap is the expected generational difference, which is a useful
sanity check that both runs measured the same thing.

### Between-run variance is a property of the *machine*, not the benchmark

This was an open question and the three T4 runs answered it.

| machine | between-run max/min | within-run p95/median |
|---|---|---|
| GTX 1650 Ti (laptop) | **up to 1.96×** | 1.01–1.11 |
| Tesla T4 (Colab) | **1.02–1.18×** | 1.01–1.09 |

On the T4, between-run and within-run spread are the *same order*. On the laptop they
differ by an order of magnitude. That is direct support for the thermal explanation
in §3 — a datacenter part with fixed cooling and no display to drive has no thermal
history for the benchmark to inherit.

**Methodological consequence:** the range-reporting discipline §3 demands is required
on thermally limited consumer hardware and largely unnecessary in the cloud. A single
Colab run is close to a measurement; a single laptop run is not. Anyone benchmarking
generative audio on a laptop needs this discipline, which is a transferable finding
rather than a quirk of this project.

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

### Closed

| question | answer |
|---|---|
| T4 between-run variance | **Tight** — 1.02–1.18× across 3 runs, versus up to 1.96× on the laptop. Range reporting is a consumer-hardware requirement, not a universal one. See §5. |
| `fp16-half` on tensor cores | **Slower than fp32** (0.94×/0.95×), in 6 of 6 cells. Giving the workload 8× the arithmetic throughput made it slower, so it is not arithmetic-bound. See §4. |
| Can any tested configuration stream? | **No.** Best across all machines, modes and runs is 1.05× realtime. |

### Still open

| question | what it needs | why it matters |
|---|---|---|
| Does a non-autoregressive or batched decode reach 1.0×? | a different model or a speculative-decoding path | §4 says the decode loop is the bottleneck, so this is the only remaining lever on streaming — and it is a separate project |
| Are ladder rungs 0 and 4 dead by design? | a therapeutic decision, not a measurement | `build_prompt` can never emit them under either arm — see `enumerate_prompts.__doc__` |
| Does the crossfade sound acceptable? | listening to `library_engine.py --wav demo.wav` | the one cost of the 8× that no metric captures |
| Can analysis lag be cut below 5.5 s? | shorter window, faster smoother, and a re-validation of alpha SNR | with the library engine it is now **85%** of the end-to-end budget — the GPU is no longer on the critical path |
