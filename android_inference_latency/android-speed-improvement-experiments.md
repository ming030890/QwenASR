# Android Speed Improvement Experiments

Goal: improve Android arm64 inference speed on the connected Pixel 9 while
keeping transcript quality stable.

Current benchmark protocol:
- Model: `/Volumes/SSD/qwen3-asr-0.6b`
- Device workspace: `/data/local/tmp/qwen-asr-bench`
- Assets: `bench/android-assets`
- NDK: `bench/android_run.sh` defaults to
  `/Users/tony/Library/Android/sdk/ndk/29.0.14206865`; override with
  `NDK_HOME`, `ANDROID_NDK_HOME`, `ANDROID_NDK_ROOT`, or `NDK_BIN` if needed.
- Fast-gate asset subset: `/Volumes/SSD/qwen-asr-android-fast-assets`
  contains `cv_yue_003.wav`, `cv_yue_003.txt`, `ptt_guardrail.wav`, and
  `ptt_guardrail.txt`.
- Candidate-gate asset subset:
  `/Volumes/SSD/qwen-asr-android-candidate-assets` contains
  `cv_yue_003.wav`, `cv_yue_003.txt`, `cv_yue_001.wav`, `cv_yue_001.txt`,
  `ptt_guardrail.wav`, and `ptt_guardrail.txt`.
- Common Voice metric: median `inference_ms` across five fixed Yue clips.
- Guardrail: `ptt_guardrail.wav` should remain semantically and phonetically
  stable for `Hello,你好吗？已经坐低咗啦。`; exact punctuation/script can vary
  under Auto language.
- Prompt/language policy: upstream-style auto language means no `language ...`
  suffix. Rust now treats unset language and `"Auto"` as auto instead of
  defaulting to English. For Yue benchmark prompts, use concise Simplified
  Chinese context: `BENCH_LANGUAGE=Auto`,
  `BENCH_PROMPT='音频是粤语，可能夹杂英语'`.
- Standard run: `bench/android_run.sh --runs 3 --threads "1 2 4 6 8"`
- Device pre-run setup is now built into `bench/android_run.sh`:
  it wakes the device, enables `svc power stayon true`, enables fixed
  performance mode, then restores fixed performance mode and the original
  `stay_on_while_plugged_in` setting on exit.
- Controlled build-feature check: compare arms with identical `--runs` and
  `--threads`, preferably after cooldown.

Resume checklist:
- Read `android-autoresearch-program.md` first; it defines fixed inputs,
  thermal control, staged gates, and acceptance rules.
- Make sure the phone is visible:
  `/Users/tony/Library/Android/sdk/platform-tools/adb devices`
- Treat this file and `android-autoresearch-program.md` as the source of truth;
  do not rely on chat history for the plan.
- Keep Android work separate from macOS autoresearch docs and results.
- Do not resume with a full five-clip thread sweep. Use Stage 1 or Stage 2
  unless the experiment is specifically about thread/core policy.
- Recreate fast/candidate asset subsets if missing:
  `/Volumes/SSD/qwen-asr-android-fast-assets` and
  `/Volumes/SSD/qwen-asr-android-candidate-assets`.
- Start normal comparable runs only when `dumpsys thermalservice` reports
  `Thermal Status: 0`.
- Do not wait for exact skin/soc matching during Stage 1/Stage 2. Snapshot the
  state with `bench/android_snapshot.sh` and record any warmer-start caveat.
- Record every run, even if it is rejected or thermally contaminated. Put the
  detailed result in `android-speed-results/` and keep only the summary here.
- Only promote an experiment to the full acceptance gate if the staged gate
  justifies it.

Current state summary:
- Android benchmark infrastructure exists and defaults to the corrected Auto
  language prompt policy.
- Dotprod remains the chosen Android build flag for non-BLAS runs, but OpenBLAS
  is the clear performance path.
- Fixed-performance Android OpenBLAS median is `1677.491 ms`; prompt-correct macOS
  host median is `262.911 ms`, leaving about a `6.38x` Android/macOS gap.
- The latest OpenBLAS `DYNAMIC_ARCH=1`, `USE_THREAD=1`, `NUM_THREADS=8` run
  regressed badly. On `cv_yue_003`, inference moved from `1481.111 ms`
  baseline to `2466.708 ms`; encode moved from `292.986 ms` to `881.855 ms`,
  while decode moved from `1188.245 ms` to `1393.140 ms`.
- The latest regression is mostly encoder/BLAS-side. Treat it as an OpenBLAS
  configuration failure, not evidence that the model has hit a global memory
  bandwidth limit.
- The active work is now bottleneck isolation. Do not resume broad OpenBLAS
  rebuild guessing until phase-split profiling and shape-level SGEMM data point
  to a specific target.

## Completed Result Index

Use progressive disclosure: keep this section as the decision summary, and put
full commands, caveats, transcripts, and per-clip numbers in the linked result
file for each completed experiment.

| ID | Summary | Decision | Detail |
| --- | --- | --- | --- |
| A0 | Added Android benchmark binary, fixed assets, and `bench/android_run.sh`. | Keep as infrastructure. | [A0 details](android-speed-results/a0-android-benchmark-harness-and-assets.md) |
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

## Current Post-OpenBLAS Performance Plan

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
- Next.

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
bench/android_run.sh \
  --label android-fast-openblas-v0333-dynarch-thread8-forced1-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1 \
  --profile
```

Decision:
- If forced single-thread runtime mostly recovers `cv_yue_003` encode from
  `881.855 ms` toward the `292.986 ms` baseline, the failure mode is internal
  OpenBLAS threading. Drop threaded OpenBLAS from speed work.
- If it remains slow, dynamic-arch dispatch or selected kernels are suspect.

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

Decision:
- Keep only if it improves the prompt-correct fast gate and then the full
  acceptance gate. Otherwise stop OpenBLAS build-variant work and move to P3.

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

Decision:
- Promote only if phase timing shows the token loop, not prefill, dominates the
  remaining decode cost and the candidate gate improves without transcript
  drift.

### P5: guardrail normalization for Auto language

Status:
- Low speed ROI, useful for experiment velocity.

Purpose:
- Add a normalized guardrail comparison for benchmark reporting while keeping
  raw transcripts in JSON. Current Auto-language outputs can be semantically and
  phonetically stable while failing the old exact string check (`Hello` versus
  `哈喽`, punctuation variants).

Decision:
- Do this when quality triage starts slowing speed experiments, or before
  changing any default benchmark acceptance rule.
