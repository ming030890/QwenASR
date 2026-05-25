# C3: OpenBLAS NDK 29 generic ARMV8 static rebuild

Purpose:
- Test whether rebuilding the same generic `ARMV8`, single-threaded, CBLAS-only
  static OpenBLAS archive with NDK 29 improves Android benchmark speed.

Build command:

```bash
git clone /private/tmp/OpenBLAS /private/tmp/OpenBLAS-ndk29-armv8

make TARGET=ARMV8 \
  OSNAME=Android \
  BINARY=64 \
  CC=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android35-clang \
  HOSTCC=clang \
  AR=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ar \
  RANLIB=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ranlib \
  NOFORTRAN=1 \
  ONLY_CBLAS=1 \
  USE_THREAD=0 \
  NUM_THREADS=1 \
  -j10
```

Build result:
- Build succeeded.
- `Makefile.conf` selected `CORE=ARMV8`, `LIBCORE=armv8`,
  `SGEMM_UNROLL_M=16`, and `SGEMM_UNROLL_N=4`.
- `config.h` selected `CORE_ARMV8`, `HAVE_NEON`, and `CHAR_CORENAME "ARMV8"`.
- The rebuilt static archive is
  `/private/tmp/OpenBLAS-ndk29-armv8/libopenblas_armv8-r0.3.33.dev.a`.

Fast-gate command:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS-ndk29-armv8 \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-fast-openblas-ndk29-armv8-t1-r1-20260524-2300 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

Fast-gate result:
- Label: `android-fast-openblas-ndk29-armv8-t1-r1-20260524-2300`
  (`bench/android-results/android-fast-openblas-ndk29-armv8-t1-r1-20260524-2300`).
- Thermal status after the run was `0`.
- `cv_yue_003`: `1602.726 ms` inference, `332.286 ms` encode,
  `1270.273 ms` decode, transcript `你睇下呢部劇。`.
- Guardrail: `2810.445 ms` inference, `755.338 ms` encode,
  `2054.779 ms` decode, transcript `哈喽，你好吗？已经坐低咗啦。`.

Comparison:
- The best comparable prompt-correct fast-gate baseline was
  `android-fast-openblas-static-dotprod-auto-zhprompt-fixed-t1-r1-20260524-1653`
  with `1535.293 ms` Common Voice and `2630.653 ms` guardrail.
- The NDK 29 rebuild was `4.39%` slower on the fast Common Voice clip and
  `6.83%` slower on the guardrail.

Decision:
- Reject this rebuilt generic `ARMV8` OpenBLAS archive at Stage 1.
- Do not promote to the candidate gate.
- Continue C3 by checking a more specific AArch64 OpenBLAS target for Pixel 9,
  rather than changing only the NDK version.

## Follow-up: dynamic-arch threaded ARMV8 API 33 rebuild

Purpose:
- Test the latest OpenBLAS release tag, `v0.3.33`, with `DYNAMIC_ARCH=1` and
  OpenBLAS internal threading enabled.

Build command:

```bash
git clone https://github.com/OpenMathLib/OpenBLAS.git \
  /private/tmp/OpenBLAS-v0.3.33-dynarch-thread8
cd /private/tmp/OpenBLAS-v0.3.33-dynarch-thread8
git checkout v0.3.33

make -j10 \
  TARGET=ARMV8 \
  DYNAMIC_ARCH=1 \
  BINARY=64 \
  CC=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android33-clang \
  AR=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ar \
  RANLIB=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ranlib \
  HOSTCC=clang \
  NOFORTRAN=1 \
  ONLY_CBLAS=1 \
  USE_THREAD=1 \
  NUM_THREADS=8
```

Build result:
- Build succeeded with NDK `29.0.14206865`, Android API `33`.
- Output archive:
  `/private/tmp/OpenBLAS-v0.3.33-dynarch-thread8/libopenblasp-r0.3.33.a`.
- `libopenblas.a` is a symlink to the threaded archive.

Fast-gate command:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS-v0.3.33-dynarch-thread8 \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-fast-openblas-v0333-dynarch-thread8-api33-t1-r1-20260525-083524 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

Fast-gate result:
- Label:
  `android-fast-openblas-v0333-dynarch-thread8-api33-t1-r1-20260525-083524`
  (`bench/android-results/android-fast-openblas-v0333-dynarch-thread8-api33-t1-r1-20260525-083524`).
- Flags: `["BLAS", "NEON", "DotProd"]`.
- `cv_yue_003`: `2466.708 ms` inference, `881.855 ms` encode,
  `1393.140 ms` decode, transcript `你睇下呢部劇。`.
- Guardrail: `2856.480 ms` inference, `751.903 ms` encode,
  `2104.200 ms` decode, transcript `哈喽，你好吗？已经坐低咗啦。`.

Comparison:
- Versus the prompt-correct fast-gate baseline
  `android-fast-openblas-static-dotprod-auto-zhprompt-fixed-t1-r1-20260524-1653`,
  this dynamic-arch threaded build was `60.67%` slower on Common Voice and
  `8.58%` slower on the guardrail.
- Versus the generic NDK 29 `ARMV8`, `USE_THREAD=0`, `NUM_THREADS=1` rebuild
  above, it was `53.91%` slower on Common Voice and `1.64%` slower on the
  guardrail.

Decision:
- Reject `DYNAMIC_ARCH=1`, `USE_THREAD=1`, `NUM_THREADS=8` at Stage 1.
- Do not promote to the candidate gate.
- Do not continue broad OpenBLAS rebuild guessing from this result alone. The
  next step is phase-split timing plus an isolated runtime-thread control for
  this threaded archive. Only build `DYNAMIC_ARCH=1`, `USE_THREAD=0`,
  `NUM_THREADS=1` if that control does not already prove OpenBLAS internal
  threading is the failure mode.
