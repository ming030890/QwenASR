# C2: OpenBLAS engine thread sweep under fixed performance mode

Purpose:
- Check whether increasing the engine thread count improves the current static
  OpenBLAS Android build before rebuilding BLAS or tuning custom kernels.

Device setup:
- `bench/android_run.sh` now runs these before every benchmark invocation:

```bash
adb shell input keyevent KEYCODE_WAKEUP
adb shell cmd power set-fixed-performance-mode-enabled true
```

- Baseline and candidate snapshots both started with `Thermal Status: 0`.
- Post-run snapshots also reported `Thermal Status: 0`, though the candidate
  thread run warmed `soc_therm` to status `1`.

Baseline command:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-baseline-openblas-fixedperf-t1-r3-20260524-215914 \
  --assets-dir bench/android-assets \
  --threads "1" \
  --runs 3
```

Focused candidate command:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-candidate-thread12-openblas-fixedperf-20260524-220110 \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1 2" \
  --runs 1
```

Baseline result:
- Label: `android-baseline-openblas-fixedperf-t1-r3-20260524-215914`
  (`bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260524-215914`).
- Flags: `["BLAS", "NEON", "DotProd"]`; threads/runs: `1` / `3`.
- Full Common Voice median inference: `1677.491 ms`; median encode
  `383.011 ms`; median decode `1305.652 ms`.
- Guardrail: `2609.735 ms`, transcript `哈喽，你好吗？已经坐低咗啦。`.
  This is semantically/phonetically stable but fails the old exact string check.

Focused candidate result:
- Label: `android-candidate-thread12-openblas-fixedperf-20260524-220110`
  (`bench/android-results/android-candidate-thread12-openblas-fixedperf-20260524-220110`).
- `threads=1` candidate Common Voice median: `1715.387 ms`; guardrail
  `2588.293 ms`.
- `threads=2` candidate Common Voice median: `92916.558 ms`; median encode
  `536.370 ms`; median decode `92380.033 ms`; transcript empty on both Common
  Voice clips.
- `threads=2` guardrail: `3534.938 ms`; transcript empty.
- A stale earlier candidate run had already shown `threads=4` pathological
  (`91833.065 ms` Common Voice, `155486.073 ms` guardrail), so the fixed
  performance rerun was intentionally limited to `threads=1 2`.

Decision:
- Reject external engine thread counts above `1` for the current static
  OpenBLAS build.
- Keep `threads=1` as the production benchmark setting.
- Move next to C3: rebuild/tune static OpenBLAS itself. If OpenBLAS internal
  threading is tested later, compare against this fixed-performance
  `threads=1` baseline and avoid combining it with engine-level threading.
