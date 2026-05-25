# Android best baseline

Canonical result for quick comparison:

| Result | Label | Common Voice median | Common Voice encode | Common Voice decode | Guardrail median | Guardrail encode | Guardrail decode | Guardrail transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best accepted | `android-baseline-openblas-fixedperf-t1-r3-20260524-215914` | `1677.491 ms` | `383.011 ms` | `1305.652 ms` | `2609.735 ms` | `722.009 ms` | `1908.856 ms` | `哈喽，你好吗？已经坐低咗啦。` |
| Latest rerun | `android-baseline-openblas-fixedperf-t1-r3-20260525-rerun` | `1797.315 ms` | `378.084 ms` | `1419.142 ms` | `2798.915 ms` | `748.484 ms` | `2054.865 ms` | `哈喽，你好吗？已经坐低咗啦。` |

Latest rerun delta versus best accepted:

| Input | Delta |
| --- | ---: |
| Common Voice median inference | `+7.14%` slower |
| Guardrail median inference | `+7.25%` slower |

Best accepted config:

- Static Android arm64 OpenBLAS archive at `/private/tmp/OpenBLAS/libopenblas.a`.
- OpenBLAS build: `TARGET=ARMV8`, `OSNAME=Android`, `BINARY=64`, `ONLY_CBLAS=1`, `NOFORTRAN=1`, `USE_THREAD=0`, `NUM_THREADS=1`.
- Benchmark build: `CARGO_FEATURES_EXTRA=android-blas`, `RUSTFLAGS_EXTRA="-C target-feature=+dotprod"`, `OPENBLAS_ANDROID_STATIC=1`.
- Benchmark run: fixed-performance mode, `BENCH_LANGUAGE=Auto`, `BENCH_PROMPT="音频是粤语，可能夹杂英语"`, engine `threads=1`, `runs=3`.

Rebuild the baseline OpenBLAS archive with:

```bash
bench/build-openblas-android-baseline.sh --force
```

Rerun the full baseline with:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-baseline-openblas-fixedperf-t1-r3-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir bench/android-assets \
  --threads "1" \
  --runs 3
```

JSON references:

- `bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260524-215914/common_voice-t1.json`
- `bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260524-215914/ptt_guardrail-t1.json`
- `bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260525-rerun/common_voice-t1.json`
- `bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260525-rerun/ptt_guardrail-t1.json`
