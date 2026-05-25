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
- M3 A is currently implemented as a memory-prioritized tradeoff despite
  missing the speed-first Stage 3 acceptance rule: it cuts load RSS to about
  `508 MB` and peak RSS to about `1.45 GB`, while full Common Voice median
  regressed `+3.5%` in the controlled benchmark.
- M4 is accepted on top of M3 A: tied `lm_head` host pre-quantization cuts
  model load to about `27.5 ms`, load RSS to about `52.7 MB`, and peak RSS to
  about `1.15 GB`, with full Common Voice median neutral/slightly faster.
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
| M2 | Lazy/dynamic KV cache allocation: start `KvCache` at `128` tokens and grow on demand instead of constructing `2048` tokens up front. Stage 3 was neutral: Common Voice median `1718.5 ms` vs M1 `1722.5 ms`, load RSS `1567760 KB` vs `1567828 KB`, peak RSS `2135716 KB` vs `2136024 KB`. | Reject and revert. The untouched zeroed KV pages do not appear to commit into Android RSS, so the smaller logical capacity does not improve measured memory metrics and adds a future long-input growth/copy event. | [M2 details](android-speed-results/m2-lazy-kv-cache.md) |
| M3 | Transient BF16 encoder weight conversion: tested all encoder projection weights transient, then transformer-layer-only transient with stem/final projections kept F32. Stage 3 reached `507756-558156 KB` load RSS and `149.5-185.6 ms` model load, but full Common Voice median regressed `+3.5-3.7%`. | Implement M3 A as a memory-prioritized tradeoff. It misses the speed-first `+2%` inference regression limit, but saves the most RSS and may improve low-memory robustness. | [M3 details](android-speed-results/m3-transient-bf16-encoder.md) |
| M4 | Tied `lm_head` host-side pre-quantization: emits an I8 mmap-backed `thinker.lm_head.weight` while keeping BF16 token embeddings for lookup. Stage 3 reached `27.5 ms` model load, `52700 KB` load RSS, `1147644 KB` peak RSS, and `1771.6 ms` full Common Voice median. | Accept. Meets model-load, load-RSS, and peak-RSS thresholds with neutral/slightly faster full Common Voice median versus M3 A. | [M4 details](android-speed-results/m4-tied-lm-head-int8.md) |
| P0 | Phase-split benchmark timing: bench_android JSON now emits `mel_ms`, `encoder_ms`, `prefill_ms`, `decode_loop_ms`, `decode_loop_ms_per_token`, and `median_tokens` per file/run. Fast-gate cv_yue_003 split: mel `3.3` / encoder `403.9` / prefill `893.2` / decode_loop `371.9` ms over `7` tokens (`53.1` ms/token), total `1672.5` ms — neutral vs `1677.5` baseline. | Accept as infrastructure. Confirms prefill (`70%` of decode_ms) is the largest decode-side slice, not the autoregressive token loop. | [P0 details](android-speed-results/p0-phase-split-benchmark-timing.md) |

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

P0 phase-split findings (2026-05-25):
- Mel is `<1%` of total inference (~3-5 ms). Encoder forward is `~25%`.
  Prefill is `~50-53%` of total inference (`70%` of legacy `decode_ms`).
  The autoregressive token loop is `~22%` of total inference at `46-53`
  ms/token over typically `7-13` tokens for these clips.
- Kernel work that touches the BF16 GEMM call surface (encoder + prefill)
  attacks `~75%` of total inference; pure token-loop work attacks `~22%`.
- These shares come from cv_yue_003 and ptt_guardrail fast-gate runs; rerun
  the phase split on a longer clip if a future experiment is specifically
  about long-utterance behavior.

Reranking notes (after M4 acceptance, 2026-05-25):
- Memory is largely solved: peak RSS `1.15 GB`, load RSS `52.7 MB`, model load
  `27.5 ms`. Pure memory-only ideas now have low marginal ROI; speed dominates.
- Encode gap (`9.67x` macOS) is larger than decode (`5.84x`), so kernel work that
  touches encoder/prefill shapes pays back the most. Decode still has a `5.84x`
  gap so it remains relevant, but mostly via the prefill portion until P0
  splits prefill from the token loop.
- The current `bf16_matvec` NEON kernel (`crates/qwen-asr/src/kernels/neon.rs`)
  unpacks BF16 with `vshll_n_u16`+`vfmaq_f32` — it does **not** use the Armv8.6
  `BFDOT`/`BFMMLA` instructions. Pixel 9 (Tensor G4, Armv9-A) supports both, so
  a native BF16 dot-product kernel is a concrete, well-scoped, high-ROI target
  attacking the `20-24%` bf16_matvec slice and any matmat caller of it.
- `bf16_matvec` is also called in a row loop to implement matmat in encoder and
  prefill. A true packed GEMM path (BFMMLA-based, or routed through a BF16-aware
  BLAS) would be a second multiplier on the same code surface.
- C3 already proved broad OpenBLAS rebuild guessing is dead. New BLAS work must
  be either (a) phase/shape-targeted after P0/P3, or (b) a different library
  (KleidiAI), not another OpenBLAS configuration sweep.
- Q3 (encoder weight quantization probe) is reformulated as E1 (encoder INT8
  weights with a real native INT8 GEMM kernel). The original probe was too
  loose given M3's signal that touching encoder math is sensitive.
- Q4 (whole-model quantized format) is dropped. The current pre-quantization
  scheme via `prepare_int8_model.py` already covers the wins a new format would
  buy, with no schema/versioning cost.
- P2 (dynamic-arch single-thread OpenBLAS rebuild) stays conditional behind P1
  and is demoted; if P1 shows threading is the regression cause, P2 is moot.

Gate scripts: `bench/android_run.py` (collect numbers, waits for thermal internally) and `bench/android_check_gate.py --result LABEL --baseline LABEL --stage 1|2|3` (accept/reject with detailed output).

| Rank | ID | ROI | Experiment | Main metrics |
| --- | --- | ---: | --- | --- |
| 1 | K1 | 5 | NEON `BFDOT`/`BFMMLA` BF16 matvec/matmul kernel for the bf16_matvec hotpath. | `bf16_matvec` profile share, `encoder_ms`/`prefill_ms` |
| 2 | K3 | 4 | True packed GEMM path for prefill/encoder (replace matvec-in-a-loop matmat). | `encoder_ms`/`prefill_ms`, sgemm share |
| 3 | P3 | 4 | Android SGEMM shape microbench across candidate kernels (K1/K3 vs OpenBLAS vs KleidiAI). | per-shape Android/macOS gap |
| 4 | P1 | 3 | Isolate threaded OpenBLAS runtime settings. | `encoder_ms`/`prefill_ms` split, fast-gate latency |
| 5 | K2 | 3 | KleidiAI (Arm) GEMM library probe for top shapes from P3. | per-shape latency vs OpenBLAS/K1 |
| 6 | E1 | 3 | Encoder INT8 weights + native INT8 GEMM kernel (replaces Q3). | `encoder_ms`, RSS, CER |
| 7 | P4 | 2 | Decoder token-loop matvec work — demoted by P0: token loop is only ~22% of total inference. | `decode_loop_ms_per_token` |
| 8 | C1K | 2 | Replace `conv2d_op` with im2col+GEMM through the K1/K3 path. | `conv2d_op` profile share, `encoder_ms` |
| 9 | P2 | 2 | Dynamic-arch single-thread OpenBLAS build if P1 requires it. | fast-gate and acceptance latency |

Removed from backlog:
- **Q3** — reformulated as E1 (the probe form was too loose; we need a real INT8
  kernel, not just a feasibility BF16-swap test that risks repeating M3's
  encoder-math sensitivity).
- **Q4** — whole-model quantized file format. The host pre-quantization scheme
  already in `prepare_int8_model.py` (M4, Q2B) delivers the savings without a
  schema/versioning/fallback cost. Revisit only if a kernel needs a layout that
  the safetensors path cannot express.

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
- Done. M3 A implemented as memory-prioritized tradeoff.

ROI:
- 5/5.

Purpose:
- Initialize the `KvCache` with a small default sequence capacity (e.g. `128` or `256` tokens) instead of pre-allocating for `2048` tokens. Rely on the existing dynamic `.grow()` method to automatically resize the cache on demand for longer audio inputs.

Why:
- The current pre-allocation at `max_seq = 2048` consumes **448 MB of heap** immediately upon loading, even though ASR transcripts are typically short (often <100 tokens per chunk). Starting at `128` tokens consumes only **~28 MB** (saving **420 MB / 93%**) and grows gracefully only when needed.

Result:
- Stage 3 showed no meaningful RSS, peak RSS, model-load, or inference
  improvement versus M1. The untouched zeroed cache pages appear not to commit
  into Android RSS, so the smaller initial logical capacity does not improve
  the measured app memory metrics.
- Reverted to the fixed `2048` token cache.

### M3: Transient layer-by-layer conversion of BF16 encoder weights

Status:
- Done. Rejected and reverted.

ROI:
- 4/5.

Purpose:
- Keep the encoder weights in their native BF16 format in the `mmap` region, and perform on-the-fly SIMD conversion to a small single-layer F32 scratch buffer during the encoder forward pass immediately before running each layer's SGEMM.

Why:
- The encoder currently converts all layer weights from BF16 in the safetensors file to F32 on startup, consuming **714 MB of permanent heap**. Keeping them as BF16 in the mmap file and converting them layer-by-layer on-the-fly into a shared ~12.8 MB scratch buffer saves **~700 MB of permanent heap** while still utilizing optimized F32 BLAS `sgemm` kernels.

Result:
- Stage 3 showed the expected memory/startup improvement but failed acceptance
  because full Common Voice median inference regressed beyond the memory-win
  limit: `+3.5%` for fully transient encoder projections and `+3.7%` for the
  transformer-layer-only variant.
- M3 A was intentionally restored after review because it saves more RSS than
  M3 B with no meaningful controlled-benchmark speed disadvantage. See
  [M3 details](android-speed-results/m3-transient-bf16-encoder.md).

### M4: Tied LM Head host-side pre-quantization

Status:
- Done. Accepted.

ROI:
- 4/5.

Purpose:
- Update `prepare_int8_model.py` to pre-quantize the tied `lm_head` weights on the host, saving them directly as INT8 in the model file. Refactor `decoder.rs` to load the tied LM head via zero-copy mmap pointers rather than allocating and quantizing on-device at startup.

Why:
- The engine currently quantizes the massive `lm_head` (vocabulary size `151,936` x `1024` hidden) to INT8 on the heap at startup, consuming **149 MB of heap**. Pre-quantizing it on the host lets the client read it directly from `mmap` zero-copy, saving startup CPU overhead and 149 MB of heap.

Result:
- Stage 3 passed versus M3 A with neutral/slightly faster full Common Voice
  median (`1771.6 ms` vs `1782.3 ms`) and large memory/startup wins:
  `27.5 ms` model load, `52700 KB` load RSS, and `1147644 KB` peak RSS.
- See [M4 details](android-speed-results/m4-tied-lm-head-int8.md).

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

### K1: NEON BFDOT/BFMMLA BF16 matvec/matmul kernel

Status:
- Next after P0 (or in parallel — P0 doesn't block the kernel work, it just
  scopes which call sites matter most).

ROI:
- 5/5.

Purpose:
- Add an aarch64 NEON kernel path that consumes BF16 weights natively using the
  Armv8.6 BF16 extension instructions `BFDOT` (4-lane BF16 dot, accumulating to
  F32) and `BFMMLA` (2x2 BF16 matrix-multiply accumulate to F32) instead of the
  current shift-left → `vfmaq_f32` path.

Why:
- The existing `bf16_matvec_fused` at
  `crates/qwen-asr/src/kernels/neon.rs:30` unpacks BF16 into F32 with
  `vshll_n_u16(.., 16)` and then accumulates with `vfmaq_f32`. That is 1 BF16
  FMA per F32 FMA throughput — half (or worse) the Armv8.6 BF16 path.
- Pixel 9 / Tensor G4 (Cortex-X4 / A720 / A520) is Armv9-A with mandatory BF16
  support, so `BFDOT`/`BFMMLA` are available at runtime. Detect at startup, fall
  back to the current kernel otherwise (older devices, non-aarch64).
- C1 attributed `20-24%` of profiled time to `bf16_matvec`. The function is also
  called as a matvec-per-row loop to implement matmat in encoder and prefill,
  so a faster kernel also reduces the apparent `sgemm` share (which currently
  includes BF16-weighted GEMM through this path).

Plan:
- Add a `neon_bf16` cargo feature and runtime CPU detection (via `is_aarch64_feature_detected!("bf16")`).
- Implement `bf16_matvec_fused` using `vbfdotq_f32` (4-lane dot, four BF16 pairs into 4 F32 accumulators).
- Implement a packed `bf16_matmat_fused` using `vbfmmlaq_f32` (2x4 BF16 panels, 2x2 F32 output tiles).
- Add a unit test that diffs the new kernel vs the existing generic kernel on the same inputs (tolerance ~`1e-3` after BF16 rounding).
- Microbench at the call shapes that show up in the encoder/prefill before running the full Android gate.

Acceptance:
- Stage 1 fast-gate: total median speedup, no transcript regression.
- Stage 3 full Common Voice: median improvement, CER neutral.

Run:

```bash
LABEL=android-fast-k1-bfdot-bfmmla-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1 \
  --profile
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 1
```

### K3: packed GEMM path for prefill/encoder (eliminate matvec-loop matmat)

Status:
- Run after K1 lands. Uses the same kernel surface.

ROI:
- 4/5.

Purpose:
- Replace call sites in encoder and decoder prefill that currently call
  `bf16_matvec_fused` per row (an implicit matmat with poor weight reuse) with
  a true packed GEMM that streams the weight tile across multiple input rows.

Why:
- A `BFMMLA` GEMM that holds a weight tile in registers across an x-panel of
  rows hits much higher GFLOPS than calling matvec N times — the weight loads
  amortize across rows.
- Encoder forward and prefill are the highest-shape callers; K3 specifically
  attacks the encode `9.67x` gap.
- Independent of OpenBLAS, so does not interact with C3's threading regressions.

Acceptance:
- Stage 3 full Common Voice: encode_ms improvement of `>= 15%`, CER neutral.

### K2: KleidiAI (Arm) GEMM library probe

Status:
- Conditional on K1/K3 results and the P3 shape microbench.

ROI:
- 3/5.

Purpose:
- Evaluate Arm's KleidiAI micro-kernels (the same library backing XNNPACK/
  llama.cpp/MediaPipe Arm backends) for the top shapes identified by P3.

Why:
- KleidiAI ships hand-tuned BF16/F32/INT8 GEMM micro-kernels for Armv8.6+ and
  is specifically targeted at LLM/transformer shapes, including the skinny
  encoder/prefill matrices where general-purpose OpenBLAS underperformed in C3.
- Lower risk than writing a full GEMM macro-kernel from scratch; higher cost
  than K1/K3 because of the library integration surface.
- Only worth pulling in if K1/K3 leave a clear residual gap on specific shapes.

Decision rule:
- If P3 shows K1/K3 within ~10% of KleidiAI on the top shapes, do not adopt the
  dependency. If KleidiAI is meaningfully faster on shapes that dominate
  runtime, ship it as the Android BLAS path.

### E1: encoder INT8 weights + native INT8 GEMM kernel (replaces Q3)

Status:
- Run after K1/K3, and only if encoder_ms is still the dominant phase after
  those kernels land.

ROI:
- 3/5.

Purpose:
- Host-pre-quantize encoder projection weights to INT8 with per-row F32 scales
  (like the decoder path), and add an aarch64 INT8 GEMM kernel using `SDOT`
  (`vdotq_s32`) that dequantizes by F32 scale on the accumulator.

Why:
- M3 showed that swapping encoder math from F32 to "transient BF16 → F32 GEMM"
  regresses speed `+3.5-3.7%` because we lose F32 SGEMM throughput without a
  faster replacement. INT8 with `SDOT` is meaningfully higher throughput than
  F32 SGEMM on Arm (4 INT8 MACs per dot, 16 lanes per `vdotq_s32`), so the
  encoder math has a real upside path, not just a memory tradeoff.
- The decoder already validates the host-pre-quant + mmap + on-device INT8
  matmul flow end-to-end (Q2B/M1), so the safetensors and loader surface is
  reusable.

Acceptance:
- Stage 3 full Common Voice: encode_ms improvement of `>= 10%`, CER neutral.

### C1K: replace conv2d_op via im2col + K1/K3 GEMM

Status:
- After K3 lands.

ROI:
- 2/5.

Purpose:
- Route the encoder convolution projection through im2col packing into the K3
  GEMM path instead of the current direct `conv2d_op`.

Why:
- C1 attributed `7-10%` of profiled time to `conv2d_op`. Once K1/K3 give us a
  fast BF16 GEMM, the cheapest way to make `conv2d_op` faster is to convert it
  to a GEMM call — at the cost of an im2col copy that is typically cheap
  relative to the matmul itself for these shapes.
- Smaller and more uncertain than K1/K3 (the conv shape is small and im2col
  overhead could eat the win), but a natural follow-on once the GEMM kernel
  exists.

### Q3 (retired): encoder quantization feasibility probe

Status:
- Retired. Reformulated as E1.

Why retired:
- The original Q3 was a loose "swap weights, see what happens" probe. M3 showed
  the encoder is sensitive to changes in the math path: a memory-saving swap
  without a faster math path regressed Common Voice `+3.5-3.7%`. E1 replaces
  Q3 with a sharper proposal — host pre-quantization plus a real native INT8
  GEMM that has a credible speed upside (`SDOT`), not just RSS savings.

### Q4 (dropped): whole-model quantized file format

Status:
- Dropped from the active plan.

Why dropped:
- The host pre-quantization scheme in `bench/prepare_int8_model.py` (Q2B INT8
  decoder projections, M4 tied lm_head) already delivers the memory, load, and
  zero-copy wins a new file format would buy, with no schema/versioning or
  fallback cost.
- Revisit only if a future kernel (K1/K3/E1/KleidiAI) needs a tensor layout the
  current safetensors path cannot express. That is not the case for any
  currently planned experiment.

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
- Done. Accepted as infrastructure. See
  [P0 details](android-speed-results/p0-phase-split-benchmark-timing.md).

Key finding:
- Prefill is `~70%` of `decode_ms`; the autoregressive token loop is only
  `~22%` of total inference at `46-53` ms/token. Encoder forward is `~25%`
  of total inference; mel is `<1%`. K1/K3 (which touch both encoder and
  prefill BF16 GEMM) remain the highest-leverage next item; P4 stays demoted.

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
