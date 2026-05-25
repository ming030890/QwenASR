# Android Improvement Experiments (Speed & Memory)

Protocol, scripts, thermal control, and acceptance rules:
[android-autoresearch-program.md](android-autoresearch-program.md).

## Current State

- Android benchmark infrastructure exists and defaults to the corrected Auto
  language prompt policy.
- Dotprod remains the chosen Android build flag for non-BLAS runs, but OpenBLAS
  is the clear performance path.
- Best accepted Android baseline, rebuild steps, and latest rerun comparison
  live in [android-best-baseline.md](android-best-baseline.md).
- Latest rejected OpenBLAS variants are summarized in the completed result
  index below; use linked result logs for detailed numbers.
- The active work is now bottleneck isolation. Do not resume broad OpenBLAS
  rebuild guessing until phase-split profiling and shape-level SGEMM data point
  to a specific target.

## Completed Result Index

Use progressive disclosure: keep this section as the decision summary, and put
full commands, caveats, transcripts, and per-clip numbers in the linked result
file for each completed experiment.

| ID | Summary | Decision | Detail |
| --- | --- | --- | --- |
| A0 | Added Android benchmark binary, fixed assets, and `bench/android_run.py`. | Keep as infrastructure. | [A0 details](android-speed-results/a0-android-benchmark-harness-and-assets.md) |
| H1 | macOS host reference: original median `196.125 ms`; prompt-correct median `262.911 ms`, encode `39.634 ms`, decode `224.362 ms`. | Use as host reference only, not Android acceptance. | [H1 details](android-speed-results/h1-macos-host-baseline-reference.md) |
| A1 | Legal no-dotprod Android smoke build: median `25937.755 ms`, guardrail `46855.435 ms`. | Keep build correctness fix; not a speed comparator. | [A1 details](android-speed-results/a1-legal-no-dotprod-android-baseline-build.md) |
| A2 | Dotprod acceptance: `26761.235 ms` vs no-dotprod `29299.505 ms`, an `8.66%` median speedup with exact guardrail. | Keep `-C target-feature=+dotprod` for Android experiments. | [A2 details](android-speed-results/a2-dotprod-build-feature-controlled-check.md) |
| A3 | No-dotprod controlled rerun reached `31510.236 ms` but thermal status hit warning. | Reject as contaminated; keep as thermal caution. | [A3 details](android-speed-results/a3-baseline-controlled-rerun-thermal-contaminated.md) |
| B1 | Fast gate found dotprod neutral/slightly slower on one clip but faster on guardrail, with exact transcript. | Promote to B2 instead of rejecting. | [B1 details](android-speed-results/b1-clean-fast-gate-baseline-vs-dotprod.md) |
| B2 | Candidate gate found dotprod `7.8%` faster; acceptance follow-up confirmed `8.66%` speedup. | Dotprod passes acceptance. | [B2 details](android-speed-results/b2-candidate-gate-baseline-vs-dotprod.md) |
| B11 | Android OpenBLAS static reduced prompt-correct median to `1789.114 ms`, about `14.96x` faster than old Android dotprod/no-BLAS and `6.81x` slower than macOS. | OpenBLAS is the Android performance path; continue with post-OpenBLAS plan. | [B11 details](android-speed-results/b11-android-openblas-static-benchmark-arm.md) |
| C1 | Profiled corrected Android/OpenBLAS static build: `sgemm` was about `62-66%` of profiled time, followed by `bf16_matvec` at about `20-24%`. | Keep as coarse profile evidence only; current work needs phase-split attribution before choosing kernels. | [C1 details](android-speed-results/c1-profile-corrected-openblas-build.md) |
| C2 | Fixed-performance baseline reached `1677.491 ms`; engine `threads=2` regressed to `92916.558 ms` and empty Common Voice transcripts. | Reject engine thread counts above `1`; keep engine `threads=1`. | [C2 details](android-speed-results/c2-openblas-engine-thread-sweep-fixedperf.md) |
| C3 | NDK 29 generic `ARMV8` static OpenBLAS reached `1602.726 ms` Common Voice; `DYNAMIC_ARCH=1`, `USE_THREAD=1`, `NUM_THREADS=8`, API 33 reached `2466.708 ms`. Both were slower than the prompt-correct fast baseline. | Reject NDK-version-only and threaded dynamic-arch rebuilds at Stage 1. Stop broad OpenBLAS rebuild guessing until phase-split and shape-level measurements isolate the failure mode. | [C3 details](android-speed-results/c3-openblas-ndk29-armv8-rebuild.md) |
| Q2 | Pre-quantize decoder weights to INT8 on host; replace BF16 tensors in model file. Model file `419 MB` smaller (`1370 MB` vs `1789 MB`). Fast gate: neutral — `model_load_ms` unchanged at `1361 ms`, inference within cross-session noise, RSS `~0.9%` lower. | Reject (neutral). Smaller file does not reduce load or inference time; on-device BF16→INT8 quantization rides free on already-paged mmap data; load is CPU/alloc bound, not I/O bound. | [Q2 details](android-speed-results/q2-decoder-int8-weights.md) |
| Q2B | Zero-copy mmap INT8 decoder weights: read INT8 model weights directly via zero-copy mmap pointers rather than allocating and copying to the heap. Fuses gate+up projections in model file. | Accept. Achieves ~31.4% faster model load (935 ms vs 1364 ms) and ~24.5% lower memory footprint (3.71 GB vs 4.91 GB RSS) with neutral/slightly faster inference. | [Q2B details](android-speed-results/q2b-int8-mmap-zero-copy.md) |
| M1 | Transient prefill dequantization: INT8 decoder prefill weights dequantize into a shared F32 scratch buffer instead of permanent per-layer F32 heap copies. Stage 3 reached `404.7 ms` model load, `1567828 KB` load RSS, and `2136024 KB` peak RSS. | Accept. Model load improved `56.7%`, load RSS improved `57.7%`, peak RSS improved `44.3%`, and Common Voice median was `1.1%` faster versus Q2B. | [M1 details](android-speed-results/m1-transient-prefill-dequant.md) |

## Current Android Experiment Plan

Use this section as the active backlog, sorted by expected ROI. The current
fixed-performance Android/OpenBLAS median is `1677.491 ms` versus macOS
`262.911 ms`, so the remaining gap is about `6.38x`, split roughly as Android
encode `383.011 ms` versus macOS `39.634 ms`, and Android decode `1305.652 ms`
versus macOS `224.362 ms`.

C1 says the corrected static OpenBLAS profile is dominated by `sgemm`
(`62-66%` of profiled time), followed by `bf16_matvec` (`20-24%`),
`conv2d_op` (`7-10%`), and causal attention (`3-5%`). That aggregate is not
phase-split: `sgemm` mixes encoder, decoder prefill, and any sequence-length
greater-than-one paths. Do not treat the current profile as proof that the
whole pipeline is memory-bound.

Working hypothesis:
- The latest C3 regression is OpenBLAS configuration overhead, especially
  internal threading and/or dynamic-arch dispatch, because encode regressed far
  more than decode on the same fast-gate clip.
- Encoder and prefill SGEMM are probably kernel-shape/threading limited on
  Android's small and skinny matrices.
- Single-token decode matvecs are more likely memory-bandwidth limited, but the
  current timing combines prefill and token loop decode, so this must be
  measured before optimizing decoder kernels.

ROI scoring:
- `5`: likely large user-visible win, low to moderate implementation risk.
- `3`: useful diagnostic or medium-size win, moderate risk.
- `1`: uncertain payoff or high product/format risk.

Gate scripts: `bench/android_run.py` (collect numbers, waits for thermal internally) and `bench/android_check_gate.py --result LABEL --baseline LABEL --stage 1|2|3` (accept/reject with detailed output).

| Rank | ID | ROI | Experiment | Main metrics |
| --- | --- | ---: | --- | --- |
| 1 | M2 | 5 | Lazy/Dynamic KV Cache scaling starting at 128 tokens. | `memory_kb`, `peak_rss_kb`, `inference_ms` |
| 2 | M3 | 4 | Transient layer-by-layer conversion of BF16 encoder weights. | `memory_kb`, `peak_rss_kb`, `encoder_ms` |
| 3 | M4 | 4 | Host-side pre-quantization of the tied LM head. | `memory_kb`, `peak_rss_kb`, `model_load_ms` |
| 4 | P0 | 5 | Phase-split benchmark timing. | `mel_ms`, `encoder_ms`, `prefill_ms`, `decode_loop_ms/token` |
| 5 | P1 | 4 | Isolate threaded OpenBLAS runtime settings. | encode/decode split and fast-gate latency |
| 6 | P3 | 4 | Android SGEMM shape microbench. | per-shape Android/macOS gap |
| 7 | Q3 | 3 | Encoder weight quantization feasibility probe. | encoder timing, RSS, CER |
| 8 | P2 | 3 | Dynamic-arch single-thread OpenBLAS build if P1 requires it. | fast-gate and acceptance latency |
| 9 | P4 | 3 | Decoder token-loop matvec work after phase split. | `decode_loop_ms/token` |
| 10 | Q4 | 1 | Whole-model quantized file format. | package size, startup, fallback complexity |

### M1: Transient layer-by-layer dequantization of F32 prefill weights

Status:
- Done. Accepted.

ROI:
- 5/5.

Purpose:
- Refactor the decoder initialization (`decoder.rs`) to dequantize the INT8 decoder weights (`wq`, `wk`, `wv`, `wo`, `gate_up_fused`, `down`) into a single, shared F32 prefill scratch buffer during the prefill phase, instead of pre-converting all layers to F32 permanently on the heap at startup.

Why:
- The current implementation consumes **1.68 GB of permanent heap** solely to store pre-converted F32 prefill weights. Since prefill is compute-bound and transient, we can share a single layer-sized F32 scratch buffer (~60 MB) across all layers, dequantizing on the fly in <2 ms per layer, saving **1.62 GB of permanent heap** with 0% latency regression.

### M2: Lazy/dynamic KV Cache allocation

Status:
- Proposed. High ROI memory-saver.

ROI:
- 5/5.

Purpose:
- Initialize the `KvCache` with a small default sequence capacity (e.g. `128` or `256` tokens) instead of pre-allocating for `2048` tokens. Rely on the existing dynamic `.grow()` method to automatically resize the cache on demand for longer audio inputs.

Why:
- The current pre-allocation at `max_seq = 2048` consumes **448 MB of heap** immediately upon loading, even though ASR transcripts are typically short (often <100 tokens per chunk). Starting at `128` tokens consumes only **~28 MB** (saving **420 MB / 93%**) and grows gracefully only when needed.

### M3: Transient layer-by-layer conversion of BF16 encoder weights

Status:
- Proposed. High ROI memory-saver.

ROI:
- 4/5.

Purpose:
- Keep the encoder weights in their native BF16 format in the `mmap` region, and perform on-the-fly SIMD conversion to a small single-layer F32 scratch buffer during the encoder forward pass immediately before running each layer's SGEMM.

Why:
- The encoder currently converts all layer weights from BF16 in the safetensors file to F32 on startup, consuming **714 MB of permanent heap**. Keeping them as BF16 in the mmap file and converting them layer-by-layer on-the-fly into a shared ~12.8 MB scratch buffer saves **~700 MB of permanent heap** while still utilizing optimized F32 BLAS `sgemm` kernels.

### M4: Tied LM Head host-side pre-quantization

Status:
- Proposed. High ROI memory-saver.

ROI:
- 4/5.

Purpose:
- Update `prepare_int8_model.py` to pre-quantize the tied `lm_head` weights on the host, saving them directly as INT8 in the model file. Refactor `decoder.rs` to load the tied LM head via zero-copy mmap pointers rather than allocating and quantizing on-device at startup.

Why:
- The engine currently quantizes the massive `lm_head` (vocabulary size `151,936` x `1024` hidden) to INT8 on the heap at startup, consuming **149 MB of heap**. Pre-quantizing it on the host lets the client read it directly from `mmap` zero-copy, saving startup CPU overhead and 149 MB of heap.

### Q2: decoder INT8 weight replacement

Status:
- Done. Rejected.

ROI:
- 5/5.

Purpose:
- Pre-quantize all decoder attention/MLP/lm_head weights to INT8 on the host
  and ship them as part of the model assets pushed to the device.
- On load, read the INT8 weights directly instead of loading BF16 and
  quantizing at runtime. No sidecar file; no separate read path.

Why:
- Q1 showed that a sidecar avoids on-device quantization but still costs a
  `570M` extra file read from device storage, which is slower than the mmap'd
  BF16 path it was meant to replace.
- Embedding INT8 weights in the packaged assets eliminates the sidecar file
  read entirely. The device reads exactly what it needs and no BF16-to-INT8
  conversion happens at load time.
- The decoder single-token path already uses INT8 on `aarch64`, so the
  inference math is unchanged.

Run:

```bash
LABEL=android-fast-q2-decoder-int8-weights-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 1
```

### Q3: encoder quantization feasibility

Status:
- Run after Q1/Q2 and P0 phase timing.

ROI:
- 3/5.

Purpose:
- Quantize a single high-cost encoder projection family and compare it against
  the current F32/OpenBLAS path.

Why:
- Encoder weights are currently loaded as F32 and call BLAS-like kernels.
  Quantization may reduce memory, but could lose optimized SGEMM throughput or
  introduce dequantization overhead.

Run (after Q2 and P0 phase timing):

```bash
LABEL=android-fast-q3-encoder-quant-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 1
```

### Q4: whole-model quantized file format

Status:
- Deferred.

ROI:
- 1/5 until Q1-Q3 show measurable Android value.

Purpose:
- Design a native quantized model file or GGUF-like export only after sidecar
  and encoder data justify the format cost.

Why:
- A replacement model file requires schema/versioning, conversion tooling,
  architecture fallback, and accuracy-debug paths. Current x86 and prefill
  paths still expect BF16/F32-compatible data.

Dropped from the active plan:
- Engine thread sweeps above `threads=1`: C2 showed `threads=2` can produce
  pathological decode latency and empty transcripts; keep engine `threads=1`.
- More NDK-version-only OpenBLAS rebuilds: C3 showed this does not move the
  needle.
- OpenBLAS internal threading as a default path: the threaded dynamic-arch
  archive regressed badly. Revisit only if a phase/shape microbench shows a
  large single BLAS call that can amortize thread overhead.
- Prompt-length tuning: useful for quality policy, but unlikely to close a large
  runtime gap now that the intended prompt is fixed.
- CPU affinity/core-placement policy: can improve a benchmark, but is hard to
  ship reliably. Revisit only after algorithmic and kernel bottlenecks are
  understood.
- Shared versus static OpenBLAS comparison: static is already the benchmarked
  path and is closer to the intended single native Android artifact. Revisit
  only for binary-size or packaging reasons.

### P0: phase-split benchmark timing

Status:
- Next (promoted to rank 1 after Q2 rejection).

Purpose:
- Split the current coarse `encode_ms` and `decode_ms` into actionable buckets:
  `mel_ms`, `encoder_ms`, `prefill_ms`, `decode_loop_ms`,
  generated token count, and `decode_loop_ms/token`.

Why:
- The current `decode_ms` includes decoder prefill plus autoregressive token
  generation. Optimizing decoder matvecs before separating those phases risks
  chasing the wrong part of the runtime.
- The current profile counters are operation-level, not phase-level. `sgemm`
  needs to be attributed to mel, encoder, or prefill before choosing a kernel
  experiment.

Decision:
- Do not start another OpenBLAS build variant until this timing split exists and
  a profile run records the phase-level breakdown on the fast assets.

### P1: isolate threaded OpenBLAS runtime settings

Status:
- Run after P0, using the existing rejected threaded dynamic-arch archive.

Purpose:
- Determine whether the latest C3 regression is caused by OpenBLAS internal
  threading or by dynamic architecture/kernel selection itself.

Command shape:

First add or use a benchmark harness passthrough that prefixes the device-side
command with runtime environment variables. Host-side exports alone do not
reach `adb shell`.

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS-v0.3.33-dynarch-thread8 \
OPENBLAS_ANDROID_STATIC=1 \
BENCH_REMOTE_ENV="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1" \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.py \
  --label android-fast-openblas-v0333-dynarch-thread8-forced1-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1 \
  --profile
```

After the run:

```bash
bench/android_check_gate.py \
  --result android-fast-openblas-v0333-dynarch-thread8-forced1-<timestamp> \
  --baseline BASELINE_LABEL \
  --stage 1
```

Interpret the output: if `cv_yue_003` encode recovers toward `292.986 ms`
baseline, the regression was internal OpenBLAS threading. If it stays slow,
dynamic-arch dispatch or kernel selection is suspect.

### P2: dynamic-arch single-thread build only if needed

Status:
- Conditional.

Purpose:
- Isolate `DYNAMIC_ARCH=1` from OpenBLAS internal threading only if P1 does not
  already prove threading is the failure mode.

Build shape:
- `TARGET=ARMV8`
- `DYNAMIC_ARCH=1`
- `USE_THREAD=0`
- `NUM_THREADS=1`
- `ONLY_CBLAS=1`
- Android API `33` or the current benchmark default API

After the run:

```bash
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 1
```

If stage 1 fails, stop OpenBLAS build-variant work and move to P3.

### P3: Android SGEMM shape microbench

Status:
- Next after P1/P2, or immediately after P0 if P1 confirms threading overhead.

Purpose:
- Benchmark the actual SGEMM shapes used by this pipeline on Android for each
  candidate OpenBLAS archive.

Shapes to include:
- mel DFT and mel filter matrix multiplies
- encoder convolution projection
- encoder attention Q/K/V/O and FFN projections
- decoder prefill Q/K/V/O and FFN projections

Decision:
- If one or two skinny shapes dominate the gap, build a targeted custom kernel
  or packing path for those shapes.
- If OpenBLAS is broadly slow across all important shapes, keep the best known
  static archive and move optimization effort to reducing calls, fusing adjacent
  work, or changing data layout.

### P4: decoder token-loop matvec work

Status:
- Blocked on P0 phase timing.

Purpose:
- Optimize only if `decode_loop_ms/token` is a large share after prefill is
  separated.

Likely targets:
- INT8 QKV, O, SwiGLU, down-projection, and lm_head argmax throughput.
- Weight layout and packing for single-token matvec locality.
- Reducing generated tokens or prompt/prefix work only if quality remains
  stable.

Run only after P0 confirms token loop dominates decode cost:

```bash
LABEL=android-fast-p4-decode-matvec-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 1
```

### P5: guardrail normalization for Auto language

Status:
- Done. `bench/android_run.py` and `bench/android_check_gate.py` both use
  keyword matching (`你好吗`, `坐低咗`) instead of an exact string, so
  script/punctuation variants under Auto language no longer cause false
  rejections.
