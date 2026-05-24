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
- Prompt-correct Android OpenBLAS median is `1789.114 ms`; prompt-correct macOS
  host median is `262.911 ms`, leaving about a `6.81x` Android/macOS gap.
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

## Current Post-OpenBLAS Performance Plan

Use this section as the active backlog, sorted by expected ROI. The current
prompt-correct Android/OpenBLAS median is `1789.114 ms` versus macOS
`262.911 ms`, so the remaining gap is about `6.81x`, split roughly as Android
encode `398.294 ms` versus macOS `39.634 ms`, and Android decode `1397.684 ms`
versus macOS `224.362 ms`.

Dropped from the active plan:
- Prompt-length tuning: useful for quality policy, but unlikely to close a large
  runtime gap now that the intended prompt is fixed.
- CPU affinity/core-placement policy: can improve a benchmark, but is hard to
  ship reliably and should only be revisited if normal thread tuning leaves an
  unexplained gap.

### C1: profile corrected OpenBLAS build

ROI:
- Highest. One run tells us whether the next real work should be BLAS tuning or
  decoder kernel work.

Purpose:
- Identify whether the remaining gap is still BLAS/encoder-heavy, or whether
  decoder microkernels dominate after BLAS.

Command shape:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-profile-openblas-auto-zhprompt-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1" \
  --runs 1
```

If `bench/android_run.sh` still does not expose `--profile`, run the pushed
binary directly with `--profile` and the same model/input/language/prompt.
Do not use profiling timing as final performance timing.

Decision:
- If profile time is mostly `sgemm`/encoder, prioritize C2/C4/OpenBLAS build
  tuning.
- If profile time is mostly decoder single-token kernels, prioritize C3.

### C2: BLAS/thread sweep under corrected prompt

ROI:
- High. Cheap to run and can reveal an immediate production setting. Previous
  accepted OpenBLAS data is only `threads=1`.

Purpose:
- Find the best thread count now that BLAS is enabled.

Command:

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
- Keep the fastest thread count that does not regress transcript quality.
- If OpenBLAS is rebuilt with internal threading, do not also oversubscribe the
  Rust thread pool without a specific comparison.

### C3: decoder single-token microkernels

ROI:
- High if C1 shows decode-heavy time, because decode is still the majority of
  corrected Android runtime (`1397.684 ms` median decode). More work than C2,
  but it targets the largest measured bucket.

Purpose:
- Optimize the custom BF16/INT8 single-token decode kernels that BLAS does not
  cover.

Likely targets after C1 profile confirms:
- `linear_nobias_int8_qkv`
- `linear_nobias_int8_swiglu`
- `linear_nobias_int8_addto`
- `argmax_int8_range`
- BF16 matvec paths if profile shows they are still active

Decision:
- Make one narrow kernel change at a time and gate through Stage 1/Stage 2
  before acceptance.

### C4: OpenBLAS rebuild tuning

ROI:
- Medium. Worth doing only if C1/C2 show the encoder or BLAS calls are still a
  major bucket. It costs more build/debug time than a thread sweep and may
  complicate packaging.

Purpose:
- The current library was a first working static build: generic `TARGET=ARMV8`,
  `USE_THREAD=0`, `NUM_THREADS=1`. It is not tuned for Pixel 9.

Experiments:
- Rebuild with OpenBLAS threading enabled: `USE_THREAD=1`, compare
  `NUM_THREADS=2` and `NUM_THREADS=4`.
- Check whether OpenBLAS has a more specific ARMv8/AArch64 core target that
  maps better to Pixel 9 than generic `ARMV8`.
- Confirm `-O3`/NEON kernels are used and inspect the OpenBLAS build log for
  selected kernels.
- Compare static versus shared only after performance is understood; static is
  currently simpler and already works.

Decision:
- Keep only if acceptance median improves under the corrected prompt policy and
  packaging remains realistic.
