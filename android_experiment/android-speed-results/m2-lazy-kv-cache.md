# M2: Lazy/dynamic KV cache allocation

Purpose:
- Initialize `KvCache` with a small default sequence capacity (`128` tokens)
  instead of pre-allocating the full `2048` token capacity.
- Rely on the existing `KvCache::grow()` path when longer inputs exceed the
  initial capacity.

Implementation tested:
- `crates/qwen-asr/src/decoder.rs`: added a `128` token default constant.
- `crates/qwen-asr/src/context.rs`: changed `QwenCtx::load()` to construct
  `KvCache` with the smaller default.
- Added a focused `KvCache::grow()` preservation test.

Fast gate command:

```bash
THERMAL_TIMEOUT=0 \
MODEL_DIR=./model \
PATH=$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH \
bench/android_run.py \
  --label android-fast-m2-lazy-kv-cache-20260525-1525 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads 1 \
  --runs 1

bench/android_check_gate.py \
  --result android-fast-m2-lazy-kv-cache-20260525-1525 \
  --baseline android-fast-m1-transient-prefill-dequant-20260525-141601 \
  --stage 1
```

Fast gate result:
- Stage 1 passed.
- Guardrail passed: `哈喽，你好吗？已经坐低咗啦。`
- `cv_yue_003 inference_ms`: `1651.2 ms` vs M1 `1651.7 ms`.
- `memory_after_model_load_kb`: `1567624 KB`.
- `peak_inference_rss`: `2119368 KB`.

Acceptance command:

```bash
THERMAL_TIMEOUT=0 \
MODEL_DIR=./model \
PATH=$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH \
bench/android_run.py \
  --label android-accept-m2-lazy-kv-cache-20260525-1530 \
  --assets-dir bench/android-assets \
  --threads 1 \
  --runs 3

bench/android_check_gate.py \
  --result android-accept-m2-lazy-kv-cache-20260525-1530 \
  --baseline android-accept-m1-transient-prefill-dequant-20260525-141733 \
  --stage 3
```

Acceptance results:

| Metric | M1 baseline | M2 | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | `1722.5 ms` | `1718.5 ms` | `-0.2%` faster |
| `model_load_ms` | `404.7` | `405.4` | `+0.2%` slower |
| `memory_after_model_load_kb` | `1567828` | `1567760` | `-0.0%` lower |
| `peak_rss_kb` | `2136024` | `2135716` | `-0.0%` lower |

Stage 3 gate:
- **FAIL / REJECT**: no metric met the acceptance threshold.
- Per-clip inference stayed within the regression limit.
- Guardrail passed.

Conclusion:
- Reject and revert M2.
- The original `vec![0.0; total]` allocation for the `2048` token KV cache
  appears not to commit the untouched zero pages into Android RSS, so reducing
  the logical initial capacity does not improve the measured app memory metrics
  for short voice-input clips.
- Keep the current fixed `2048` token cache until a future experiment can show
  a concrete RSS, peak RSS, startup, or long-input stability benefit.
