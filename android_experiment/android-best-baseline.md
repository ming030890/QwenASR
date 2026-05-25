# Android best baseline

Canonical result for quick comparison:

| Result | Label | Common Voice median | Common Voice encode | Common Voice decode | Guardrail median | Guardrail encode | Guardrail decode | Guardrail transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best accepted (M1 transient prefill dequant) | `android-accept-m1-transient-prefill-dequant-20260525-141733` | `1722.546 ms` | `378.444 ms` | `1345.889 ms` | `2716.500 ms` | `748.495 ms` | `1967.856 ms` | `哈喽，你好吗？已经坐低咗啦。` |
| Previous accepted (INT8 mmap) | `android-accept-q2b-int8-mmap-20260525-131609` | `1742.230 ms` | `378.124 ms` | `1364.184 ms` | `2750.264 ms` | `742.325 ms` | `2003.709 ms` | `哈喽，你好吗？已经坐低咗啦。` |
| Best BF16 accepted | `android-baseline-openblas-fixedperf-t1-r3-20260524-215914` | `1677.491 ms` | `383.011 ms` | `1305.652 ms` | `2609.735 ms` | `722.009 ms` | `1908.856 ms` | `哈喽，你好吗？已经坐低咗啦。` |
| Rerun (bash script) | `android-baseline-openblas-fixedperf-t1-r3-20260525-rerun` | `1797.315 ms` | `378.084 ms` | `1419.142 ms` | `2798.915 ms` | `748.484 ms` | `2054.865 ms` | `哈喽，你好吗？已经坐低咗啦。` |
| Latest rerun (python script) | `android-baseline-py-full-20260525-103710` | `1778.161 ms` | `377.781 ms` | `1399.354 ms` | `2813.800 ms` | `751.200 ms` | `2065.700 ms` | `哈喽，你好吗？已经坐低咗啦。` |

M1 transient prefill dequant delta versus previous Q2B baseline:

| Metric | Delta | Details |
| --- | ---: | --- |
| Common Voice median inference | `-1.1%` faster | `1722.5 ms` vs `1742.2 ms` |
| Model load time | `-56.7%` faster | `404.7 ms` vs `935.2 ms` |
| Memory after load (RSS) | `-57.7%` lower | `1.57 GB` vs `3.71 GB` |
| Peak inference RSS | `-44.3%` lower | `2.14 GB` vs `3.83 GB` |

Q2B INT8 mmap delta versus latest Python rerun:

| Metric | Delta | Details |
| --- | ---: | --- |
| Common Voice median inference | `-2.0%` faster | `1742.2 ms` vs `1778.2 ms` |
| Model load time | `-31.4%` faster | `935.2 ms` vs `1363.7 ms` |
| Memory after load (RSS) | `-24.5%` lower | `3.71 GB` vs `4.91 GB` |
| Peak inference RSS | `-23.9%` lower | `3.83 GB` vs `5.04 GB` |

Best accepted config:

- Static Android arm64 OpenBLAS archive at `/private/tmp/OpenBLAS/libopenblas.a`.
- OpenBLAS build: `TARGET=ARMV8`, `OSNAME=Android`, `BINARY=64`, `ONLY_CBLAS=1`, `NOFORTRAN=1`, `USE_THREAD=0`, `NUM_THREADS=1`.
- Benchmark build: `CARGO_FEATURES_EXTRA=android-blas`, `RUSTFLAGS_EXTRA="-C target-feature=+dotprod"`, `OPENBLAS_ANDROID_STATIC=1`.
- Model build: Pre-quantized using `prepare_int8_model.py` with zero-copy INT8 decoder projections and pre-fused gate+up MLP tensors.
- Decoder prefill: INT8 projection weights dequantize transiently into a shared
  F32 scratch buffer instead of permanent per-layer F32 heap copies.
- Benchmark run: fixed-performance mode, `BENCH_LANGUAGE=Auto`, `BENCH_PROMPT="音频是粤语，可能夹杂英语"`, engine `threads=1`, `runs=3`.

Rebuild the baseline OpenBLAS archive with:

```bash
bench/build-openblas-android-baseline.sh --force
```

Rerun the full baseline with:

```bash
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.py \
  --label android-accept-m1-transient-prefill-dequant-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir bench/android-assets \
  --threads "1" \
  --runs 3
```

JSON references:

- `bench/android-results/android-accept-m1-transient-prefill-dequant-20260525-141733/common_voice-t1.json`
- `bench/android-results/android-accept-m1-transient-prefill-dequant-20260525-141733/ptt_guardrail-t1.json`
- `bench/android-results/android-accept-q2b-int8-mmap-20260525-131609/common_voice-t1.json`
- `bench/android-results/android-accept-q2b-int8-mmap-20260525-131609/ptt_guardrail-t1.json`
- `bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260524-215914/common_voice-t1.json`
- `bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260524-215914/ptt_guardrail-t1.json`
- `bench/android-results/android-baseline-py-full-20260525-103710/common_voice-t1.json`
- `bench/android-results/android-baseline-py-full-20260525-103710/ptt_guardrail-t1.json`
