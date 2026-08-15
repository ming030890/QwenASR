# Android best baseline

The current deployable baseline is K8, measured on a Pixel 10 with one engine
thread and fixed-performance mode. The standalone canonical run is retained for
reproduction; optimization deltas come from the same-session K7/K8 A/B because
Android decoder timings vary materially between sessions.

| Result | Label | Common Voice median | Encoder | First visible token | Peak RSS | Guardrail transcript |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Current K8 full linear INT8 | `android-pixel10-k8-full-encoder-int8-20260815` | `2196.233 ms` | `252.257 ms` | `1497.915 ms` | `2192632 KiB` | `哈喽，你好嘛？已经坐低咗啦。` |
| Historical K4 INT8 SDOT prefill | `android-accept-k4-int8-prefill-sdot-20260525-185000` | `1565.846 ms` | — | — | `1.42 GB` | `Hello，你好吗？已经坐低咗啦。` |
| Historical M1 transient prefill dequant | `android-accept-m1-transient-prefill-dequant-20260525-141733` | `1722.546 ms` | — | — | `2.14 GB` | `哈喽，你好吗？已经坐低咗啦。` |

Historical rows were captured on different devices/builds and are not direct
performance comparisons with K8.

## Controlled K7 to K8 result

The accepted same-session Pixel 10 A/B showed:

| Metric | K7 | K8 | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | `1574.075 ms` | `1500.350 ms` | `-4.7%` |
| Common Voice median encoder | `238.371 ms` | `193.163 ms` | `-19.0%` |
| Common Voice median first-visible token | `1088.979 ms` | `1017.995 ms` | `-6.5%` |
| Guardrail inference | `2428.365 ms` | `2314.875 ms` | `-4.7%` |
| Guardrail encoder | `457.682 ms` | `339.112 ms` | `-25.9%` |
| Guardrail first-visible token | `1536.008 ms` | `1416.941 ms` | `-7.8%` |
| Common Voice peak RSS | `2264128 KiB` | `2145332 KiB` | `-5.2%` |
| Artifact size | `2027.83 MiB` | `1957.83 MiB` | `-3.5%` |

All five Common Voice transcripts and the push-to-talk guardrail transcript
matched exactly. See
[`android-speed-results/k8-full-encoder-linear-int8-pixel10.md`](android-speed-results/k8-full-encoder-linear-int8-pixel10.md)
for the full experiment and rejected convolution extension.

## Accepted configuration

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-kleidiai-i8mm-full-encoder-packed`.
- 315 matrices are INT8 and packed offline: 147 encoder and 168 decoder.
- Every encoder linear weight uses the KleidiAI I8MM backend; runtime packing,
  experimental fallbacks, and obsolete opt-in flags are absent.
- The BF16 convolution filters remain because quantizing them caused all six
  validation transcripts to become empty.
- OpenBLAS 0.3.34 static Android arm64 archive at
  `/private/tmp/OpenBLAS-0.3.34`, single-threaded CBLAS.
- KleidiAI I8MM archive at `/private/tmp/qwen-kleidiai-review/build-android-qwen`.
- Build features: `android-blas kleidiai`; Rust target feature: `+dotprod`.
- Runtime: language `Auto`, prompt `音频是粤语，可能夹杂英语`, one engine
  thread, three repetitions, fixed-performance mode.

Rebuild OpenBLAS with:

```bash
bench/build-openblas-android-baseline.sh --force
```

Rerun the baseline with:

```bash
MODEL_DIR=/Volumes/SSD/Qwen3-ASR-1.7B-int8-kleidiai-i8mm-full-encoder-packed \
KLEIDIAI_LIB_DIR=/private/tmp/qwen-kleidiai-review/build-android-qwen \
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS-0.3.34 \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA="android-blas kleidiai" \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.py \
  --label android-pixel10-k8-full-encoder-int8-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir bench/android-assets \
  --threads "1" \
  --runs 3
```

Canonical raw JSON (gitignored local benchmark output):

- `bench/android-results/android-pixel10-k8-full-encoder-int8-20260815/common_voice-t1.json`
- `bench/android-results/android-pixel10-k8-full-encoder-int8-20260815/ptt_guardrail-t1.json`

The checked-in snapshot is `bench/latest_baseline.json`.
