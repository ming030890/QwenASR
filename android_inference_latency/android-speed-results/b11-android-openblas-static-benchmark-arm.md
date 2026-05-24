# B11: Android OpenBLAS static benchmark arm

Purpose:
- Directly test the biggest suspected Android/macOS gap: Android was using
  scalar fallback loops for BLAS-shaped f32 work while macOS used
  Accelerate/vDSP.

Build/setup notes:
- Added benchmark-only `qwen-asr-cli` feature `android-blas`, which forwards
  `qwen-asr/blas` for Android without changing the Flutter/JNI dependency.
- Added Android OpenBLAS link env support:
  `OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS` and
  `OPENBLAS_ANDROID_STATIC=1`.
- Built OpenBLAS from `/private/tmp/OpenBLAS` for Android arm64 using the
  OpenBLAS Android wiki guidance. The NDK 30 wrapper in this environment drops
  arguments through its `clang` shim, so the working target compiler was:
  `CC='/Users/tony/Library/Android/sdk/ndk/30.0.14904198/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang-21 --target=aarch64-linux-android35'`
  with `HOSTCC=/usr/bin/clang`, `TARGET=ARMV8`, `ONLY_CBLAS=1`,
  `NOFORTRAN=1`, `USE_THREAD=0`, and `NUM_THREADS=1`.
- Follow-up wrapper check: NDK `28.2.13676358` and `30.0.14904198` have a text
  `clang` shim (`clang-19` / `clang-21`) that fails the stdin compile test with
  `error: no input files`; NDK `29.0.14206865` has a real Mach-O `clang` binary
  and both `aarch64-linux-android30-clang` and `aarch64-linux-android35-clang`
  pass the same test. `bench/android_run.sh` now defaults to NDK 29. Prefer NDK
  29 for future OpenBLAS rebuilds, or keep using the direct
  `clang-21 --target=...` workaround with NDK 30.
- The shared-library build step failed, but the static archive
  `/private/tmp/OpenBLAS/libopenblas.a` was created and linked successfully.

Fast gate:
- Label: `android-fast-openblas-static-dotprod-t1-r1-20260524-1544`
  (`bench/android-results/android-fast-openblas-static-dotprod-t1-r1-20260524-1544`).
- Flags: `["BLAS", "NEON", "DotProd"]`; threads/runs: `1` / `1`.
- `cv_yue_003.wav`: `1222.881 ms` inference, `293.981 ms` encode,
  `928.755 ms` decode.
- Guardrail: `2356.684 ms`, exact transcript.
- Compared with prior fast dotprod `cv_yue_003.wav` at `20627.996 ms`, this is
  about `16.87x` faster.

Candidate gate:
- Label: `android-candidate-openblas-static-dotprod-t1-r1-20260524-1546`
  (`bench/android-results/android-candidate-openblas-static-dotprod-t1-r1-20260524-1546`).
- Median Common Voice inference over `cv_yue_001.wav` and `cv_yue_003.wav`:
  `1480.571 ms`.
- Per-clip inference: `cv_yue_001.wav` `1751.388 ms`, `cv_yue_003.wav`
  `1209.754 ms`.
- Guardrail: `2317.934 ms`, exact transcript.
- Compared with prior candidate dotprod median `23635.609 ms`, this is about
  `15.96x` faster.

Acceptance gate:
- Label: `android-accept-openblas-static-dotprod-t1-r3-20260524-1551`
  (`bench/android-results/android-accept-openblas-static-dotprod-t1-r3-20260524-1551`).
- Thermal status after run: `0`.
- Median Common Voice inference: `1342.469 ms`.
- Per-clip inference: `cv_yue_000.wav` `1281.776 ms`, `cv_yue_001.wav`
  `1758.571 ms`, `cv_yue_002.wav` `1671.465 ms`, `cv_yue_003.wav`
  `1276.666 ms`, `cv_yue_004.wav` `1342.469 ms`.
- Guardrail: `2434.938 ms`, exact transcript.
- Compared with accepted dotprod median `26761.235 ms`, OpenBLAS static is
  about `19.93x` faster (`94.98%` lower median inference). Guardrail is about
  `20.90x` faster than dotprod.
- Per-clip speedups versus accepted dotprod: `17.26x`, `16.35x`, `16.01x`,
  `15.30x`, and `20.09x`.

Quality caveat:
- The exact guardrail passed.
- Common Voice transcripts were unchanged for the first four clips under the
  current CER script, but `cv_yue_004.wav` changed from
  `Need to provide what information.` to
  `What information should I provide?`.
- With the current strict reference text, average Common Voice CER changes from
  `0.9500` for accepted dotprod to `0.9722` for OpenBLAS static. Therefore this
  is the clear performance winner but should be marked "performance-promising,
  quality-caveated" rather than fully accepted until the `cv_yue_004.wav`
  transcript drift is resolved or the quality policy is clarified.

Follow-up:
- Re-ran the OpenBLAS acceptance arm with Auto language and a Simplified Chinese
  Yue prompt. That prompt-correct result is recorded below.
- Current default prompt policy is now shorter:
  `BENCH_LANGUAGE=Auto`, `BENCH_PROMPT='音频是粤语，可能夹杂英语'`.
- Future guardrail checks should tolerate punctuation/script variants such as
  `Hello` versus `哈喽` when the output is semantically and phonetically stable.

Prompt-correct acceptance rerun:
- Android label:
  `android-accept-openblas-static-dotprod-auto-zhprompt-t1-r3-20260524-1909`
  (`bench/android-results/android-accept-openblas-static-dotprod-auto-zhprompt-t1-r3-20260524-1909`).
- Prompt/language: `BENCH_LANGUAGE=Auto`,
  `BENCH_PROMPT='音频是粤语，可能夹杂英语词或短语'`.
- Thermal status after run: `0`.
- Flags: `["BLAS", "NEON", "DotProd"]`; threads/runs: `1` / `3`.
- Common Voice median inference: `1789.114 ms`; median encode `398.294 ms`;
  median decode `1397.684 ms`.
- Per-clip inference: `cv_yue_000.wav` `1622.743 ms`, `cv_yue_001.wav`
  `2016.026 ms`, `cv_yue_002.wav` `2032.570 ms`, `cv_yue_003.wav`
  `1611.078 ms`, `cv_yue_004.wav` `1789.114 ms`.
- Transcripts: `我急求方法啊！`; `唔做嘢嘅佔多數咯。`; `證據嘅話，真係冇啊！`;
  `你睇下呢部劇。`; `仲要提供乜嘢資料？`.
- Guardrail: `2724.594 ms`, transcript `Hello，你好吗？已经坐低咗啦。`.
- Common Voice average CER under the current strict script/punctuation metric:
  `0.19168`.
- Compared with the prompt-correct macOS host reference
  `host-osx-auto-zhprompt-t1-r3-20260524-1910`, Android OpenBLAS is about
  `6.81x` slower on Common Voice median (`1789.114 ms` vs `262.911 ms`) and
  about `7.20x` slower on the guardrail (`2724.594 ms` vs `378.397 ms`).
- Compared with the old Android dotprod/no-BLAS acceptance median
  `26761.235 ms`, prompt-correct OpenBLAS is about `14.96x` faster. This ratio
  is not a perfectly isolated comparison because the prompt policy changed, but
  it reflects the current intended benchmark behavior.

Decision:
- OpenBLAS remains the clear Android performance path.
- The old exact guardrail string is too strict for prompt-policy comparisons
  because punctuation and `Hello`/`哈喽` rendering vary while the utterance is
  semantically and phonetically acceptable. Add normalized guardrail comparison
  before treating future Auto-language runs as failing on punctuation alone.
