# Q1: decoder INT8 sidecar cache

Purpose:
- Persist the decoder INT8 weights and per-row scales generated during
  `QwenCtx::load()`, then reuse them on later loads when the cache schema,
  model fingerprint, and detected decoder dimensions match.

Implementation tested:
- Added a versioned binary sidecar at `decoder-int8-cache.v1.bin`.
- The sidecar stores all decoder attention/MLP INT8 matrices plus the INT8
  `lm_head`.
- Cache validation checks magic, schema version, model fingerprint derived from
  safetensors metadata, and decoder dimensions before reuse.
- Normal loads are gated behind `QWEN_ASR_DECODER_INT8_CACHE=1` after this
  result, because the experiment did not clear acceptance.
- Added `BENCH_REMOTE_ENV` to `bench/android_run.sh` so on-device benchmark
  env vars can be set explicitly.

Fast-gate commands:

```bash
PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
CARGO_FEATURES_EXTRA=android-blas \
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-fast-q1-decoder-int8-sidecar-t1-r1-20260525-1058 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

```bash
PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
CARGO_FEATURES_EXTRA=android-blas \
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-fast-q1-decoder-int8-sidecar-cached-t1-r1-20260525-1101 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

Run conditions:
- Device was visible as `44241FDAQ000DX`.
- Thermal status before and after the fast gate was `0`.
- Generated sidecar size on device:
  `/data/local/tmp/qwen-asr-bench/model/decoder-int8-cache.v1.bin`, `570M`.

Results:

| Arm | Input | `model_load_ms` | `memory_after_model_load_kb` | `peak_rss_kb` | `inference_ms` | `encode_ms` | `decode_ms` | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| create sidecar | `cv_yue_003` | `5722.832` | `4891368` | `4999556` | `1688.035` | `337.313` | `1344.604` | `你睇下呢部劇。` |
| create sidecar, then cached | `ptt_guardrail` | `1822.784` | `4609156` | `4731500` | `2900.983` | `760.295` | `2140.297` | `哈喽，你好吗？已经坐低咗啦。` |
| cached sidecar | `cv_yue_003` | `1476.444` | `4609196` | `4717236` | `1618.438` | `332.805` | `1285.459` | `你睇下呢部劇。` |
| cached sidecar | `ptt_guardrail` | `1477.904` | `4609188` | `4731588` | `2818.768` | `752.919` | `2065.601` | `哈喽，你好吗？已经坐低咗啦。` |

Rerun after gating the cache behind `QWEN_ASR_DECODER_INT8_CACHE=1`:

| Arm | Input | `model_load_ms` | `memory_after_model_load_kb` | `peak_rss_kb` | `inference_ms` | `encode_ms` | `decode_ms` | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default | `cv_yue_003` | `1329.168` | `4911624` | `5019392` | `1606.159` | `332.027` | `1273.944` | `你睇下呢部劇。` |
| default | `ptt_guardrail` | `1368.550` | `4911760` | `5034252` | `2828.434` | `756.932` | `2071.180` | `哈喽，你好吗？已经坐低咗啦。` |
| cached sidecar | `cv_yue_003` | `1489.056` | `4609336` | `4717080` | `1621.330` | `332.878` | `1288.256` | `你睇下呢部劇。` |
| cached sidecar | `ptt_guardrail` | `1482.627` | `4609212` | `4731548` | `2863.303` | `763.825` | `2099.142` | `哈喽，你好吗？已经坐低咗啦。` |

Interpretation:
- A cached sidecar still has to read `570M` from device storage and allocate/copy
  the decoded buffers into Rust vectors.
- The default path uses the mmap'd BF16 safetensors pages already needed by the
  model and quantizes while streaming through memory. On this device, that
  compute is cheaper than loading the sidecar format.
- The RSS improvement is repeatable but remains below the acceptance threshold.

Comparison:
- Current full baseline rerun:
  `android-baseline-openblas-fixedperf-t1-r3-20260525-rerun`.
- Baseline `model_load_ms`: `1402.421` for Common Voice and `1366.191` for
  guardrail.
- Cached sidecar `model_load_ms` was slower than that baseline
  (`1476.444` / `1477.904`).
- Cached sidecar RSS was lower by roughly `6%`, below the `10%` memory
  acceptance threshold.
- Fast-gate inference also regressed versus the prompt-correct fast baseline
  (`1535.293 ms` Common Voice, `2630.653 ms` guardrail).

Decision:
- Reject Q1 as a generated-on-device sidecar cache.
- Do not promote to Q2 packaged sidecar as-is: a `570M` sidecar does not
  produce a startup win and misses the memory threshold.
- Keep the prototype code gated by `QWEN_ASR_DECODER_INT8_CACHE=1` only for
  future diagnostics; it must not become the default load path.
