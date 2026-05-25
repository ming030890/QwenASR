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
- The active work is now post-OpenBLAS profiling and tuning, not no-BLAS scalar
  fallback work.

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
| C1 | Profiled corrected Android/OpenBLAS static build: `sgemm` was about `62-66%` of profiled time, followed by `bf16_matvec` at about `20-24%`. | Run C2 OpenBLAS/thread sweep next; keep static OpenBLAS as primary Android packaging path. | [C1 details](android-speed-results/c1-profile-corrected-openblas-build.md) |
| C2 | Fixed-performance baseline reached `1677.491 ms`; engine `threads=2` regressed to `92916.558 ms` and empty Common Voice transcripts. | Reject engine thread counts above `1`; keep `threads=1` and move to C3 OpenBLAS rebuild tuning. | [C2 details](android-speed-results/c2-openblas-engine-thread-sweep-fixedperf.md) |
| C3 | NDK 29 generic `ARMV8` static OpenBLAS reached `1602.726 ms` Common Voice; `DYNAMIC_ARCH=1`, `USE_THREAD=1`, `NUM_THREADS=8`, API 33 reached `2466.708 ms`. Both were slower than the prompt-correct fast baseline. | Reject NDK-version-only and threaded dynamic-arch rebuilds at Stage 1; next isolate `DYNAMIC_ARCH=1` with single-threaded OpenBLAS. | [C3 details](android-speed-results/c3-openblas-ndk29-armv8-rebuild.md) |

## Current Post-OpenBLAS Performance Plan

Use this section as the active backlog, sorted by expected ROI. The current
fixed-performance Android/OpenBLAS median is `1677.491 ms` versus macOS
`262.911 ms`, so the remaining gap is about `6.38x`, split roughly as Android
encode `383.011 ms` versus macOS `39.634 ms`, and Android decode `1305.652 ms`
versus macOS `224.362 ms`.

C1 changed the shape of the plan: the corrected static OpenBLAS profile is
dominated by `sgemm` (`62-66%` of profiled time), followed by `bf16_matvec`
(`20-24%`), then `conv2d_op` (`7-10%`) and causal attention (`3-5%`). Treat
static OpenBLAS as the primary Android packaging path unless a later packaging
constraint forces a shared-library comparison.

Dropped from the active plan:
- Prompt-length tuning: useful for quality policy, but unlikely to close a large
  runtime gap now that the intended prompt is fixed.
- CPU affinity/core-placement policy: can improve a benchmark, but is hard to
  ship reliably and should only be revisited if normal thread tuning leaves an
  unexplained gap.
- Shared versus static OpenBLAS comparison: static is already the benchmarked
  path and is closer to the intended single native Android artifact. Revisit
  only for binary-size or packaging reasons, not as a near-term speed bet.
- Decoder INT8-specific kernels such as `linear_nobias_int8_qkv`,
  `linear_nobias_int8_swiglu`, `linear_nobias_int8_addto`, and
  `argmax_int8_range`: they did not appear as top C1 counters. Revisit only if
  more detailed profiling shows one of them hidden inside the large decode
  bucket after BLAS and BF16 matvec work.

### C2: current static OpenBLAS thread sweep

Status:
- Rejected. Engine-level `threads=2` caused a massive decode regression and
  empty transcripts on the candidate gate. Keep `threads=1`.

ROI:
- Highest active item. It is the cheapest way to learn whether the current
  static, single-threaded OpenBLAS build has a better production setting through
  the engine's existing thread policy. Previous accepted OpenBLAS data is only
  `threads=1`, while C1 shows most measured time is in `sgemm`.

Purpose:
- Find the best `kernels::set_threads(...)` value for the current static
  OpenBLAS build before spending time rebuilding BLAS.
- Use staged gates: candidate subset first, then full five-clip acceptance only
  if a non-1 thread count looks meaningfully better and transcript quality is
  stable.

Stage 1 command:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-candidate-thread-sweep-openblas-auto-zhprompt-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1 2 4 6 8" \
  --runs 1
```

Acceptance command, only if Stage 1 justifies it:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-thread-sweep-openblas-auto-zhprompt-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir bench/android-assets \
  --threads "1 2 4 6 8" \
  --runs 3
```

Decision:
- Keep the fastest stable thread count that improves full five-clip median
  inference and does not regress transcript quality.
- If more threads only improve guardrail or one clip, keep `threads=1` and move
  to C3.
- If thermal status changes or later thread counts start warmer, rerun only the
  promising counts after cooldown before changing the default.

### C3: OpenBLAS static rebuild tuning

Status:
- In progress. The NDK 29 generic `ARMV8`, `USE_THREAD=0`, `NUM_THREADS=1`
  rebuild was rejected at Stage 1 because it was slower than the comparable
  prompt-correct fast-gate baseline. The `DYNAMIC_ARCH=1`, `USE_THREAD=1`,
  `NUM_THREADS=8`, API 33 follow-up was also rejected because it was much
  slower on the fast Common Voice clip. Next C3 step is `TARGET=ARMV8`,
  `DYNAMIC_ARCH=1`, `USE_THREAD=0`, `NUM_THREADS=1`, so dynamic arch can be
  separated from OpenBLAS internal threading.

ROI:
- High. C1 says `sgemm` is the largest measured bucket, and the current static
  library was a first working generic `ARMV8`, `USE_THREAD=0`, `NUM_THREADS=1`
  build. A better static OpenBLAS build can attack the largest bucket while
  preserving the intended Android packaging shape.

Purpose:
- Improve the static OpenBLAS archive used by Android without changing model
  behavior or introducing a runtime `.so` dependency.

Experiments, in this order:
- Done/rejected: rebuild static OpenBLAS with NDK 29 and confirm the selected
  AArch64 kernels in the build log. Keep `ONLY_CBLAS=1`, `NOFORTRAN=1`, and
  static output.
- Done/rejected: rebuild OpenBLAS `v0.3.33` with `DYNAMIC_ARCH=1`,
  `USE_THREAD=1`, `NUM_THREADS=8`, `ONLY_CBLAS=1`, and Android API 33. It was
  `60.67%` slower than the prompt-correct fast-gate baseline on Common Voice.
- Next: rebuild OpenBLAS `v0.3.33` with `TARGET=ARMV8`, `DYNAMIC_ARCH=1`,
  `USE_THREAD=0`, `NUM_THREADS=1`, `ONLY_CBLAS=1`, and Android API 33. Benchmark
  it with the same fast gate to isolate dynamic-arch dispatch from OpenBLAS
  internal threading.
- Check whether OpenBLAS has a better ARMv8/AArch64 target for Pixel 9 than
  generic `TARGET=ARMV8`; benchmark any candidate with the same C2 Stage 1 gate.
- Do not continue OpenBLAS internal threading unless the single-threaded
  dynamic-arch result is promising or a later profile shows a clear single large
  BLAS call that can amortize its overhead.

Decision:
- Keep a rebuilt library only if it improves prompt-correct median inference on
  the acceptance gate and still links statically into the Android benchmark.
- If OpenBLAS internal threading is tested, compare it against the best C2
  external-thread setting, not only against `threads=1`.

### C4: BF16 matvec microkernel

ROI:
- Medium-high after C2/C3. C1 consistently shows `bf16_matvec` as the second
  largest measured bucket (`20-24%`), so it is the most relevant non-BLAS
  kernel target.

Purpose:
- Optimize the custom BF16 matvec paths that BLAS does not cover, especially in
  decode.

Likely work:
- Inspect `bf16_matvec` implementation and add a focused host microbenchmark if
  one does not exist.
- Tune NEON/dotprod-friendly BF16 conversion and accumulation paths one narrow
  change at a time.
- Gate every change through candidate assets before acceptance because this
  path touches decode-heavy user-visible transcription.

Decision:
- Promote only if the C1-style profile shows `bf16_matvec` time falls and the
  full benchmark median improves without quality drift.

### C5: conv2d encoder kernel

ROI:
- Medium. C1 shows `conv2d_op` at `7-10%` of profiled counter time. It is
  smaller than `sgemm` and `bf16_matvec`, but it is still a visible encoder
  bucket and may compound with OpenBLAS gains.

Purpose:
- Reduce Android encoder overhead that remains outside `sgemm`.

Decision:
- Start only after C2/C3, or if a rebuilt OpenBLAS lowers `sgemm` enough that
  `conv2d_op` becomes a larger share of the profile.

### C6: guardrail normalization for Auto language

ROI:
- Low for speed, medium for experiment velocity. Current Auto-language outputs
  can be semantically and phonetically stable while failing the old exact string
  check (`Hello` versus `哈喽`, punctuation variants). This can obscure whether a
  speed experiment truly regressed quality.

Purpose:
- Add a normalized guardrail comparison for benchmark reporting while keeping
  raw transcripts in JSON.

Decision:
- Do this when quality triage starts slowing speed experiments, or before
  changing any default benchmark acceptance rule.
