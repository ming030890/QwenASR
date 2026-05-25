# M3: Transient BF16 encoder weight conversion

Purpose:
- Keep large encoder weights in native BF16 mmap storage and convert them into a
  shared F32 scratch buffer during encoder forward instead of permanently
  storing all encoder projection weights as F32 on the heap.

Implementation tested:
- Variant A made `conv_out`, all transformer layer projections, `proj1`, and
  `proj2` transient BF16-to-F32 conversions.
- Variant B narrowed the change to transformer layer projections only, keeping
  `conv_out`, `proj1`, and `proj2` as permanent F32 weights to avoid repeated
  conversion of non-layer projection weights.
- Variant A is now implemented intentionally as a memory-prioritized tradeoff.
  It saves more RSS than Variant B and did not show a meaningful controlled
  benchmark speed disadvantage versus Variant B.

Run conditions:
- Device: Pixel 9, fixed-performance mode.
- Model: `/Volumes/SSD/qwen-asr-int8dec`.
- Config inherited from M1: static OpenBLAS, `android-blas`,
  `-C target-feature=+dotprod`, `threads=1`, Auto language with Yue prompt.
- Baseline: `android-accept-m1-transient-prefill-dequant-20260525-141733`.

Caveat:
- Ignore `android-build-smoke-m3-bf16-encoder`; it accidentally used the
  original BF16 model directory instead of `/Volumes/SSD/qwen-asr-int8dec`.
  The valid smoke label is `android-build-smoke-m3-bf16-encoder-int8dec`.

## Variant A: all encoder BF16 projection weights transient

Commands:

```bash
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-fast-m3-bf16-encoder-20260525-1534 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads 1 \
  --runs 1

bench/android_check_gate.py \
  --result android-fast-m3-bf16-encoder-20260525-1534 \
  --baseline android-fast-m1-transient-prefill-dequant-20260525-141601 \
  --stage 1

MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-candidate-m3-bf16-encoder-20260525-1535 \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads 1 \
  --runs 1

bench/android_check_gate.py \
  --result android-candidate-m3-bf16-encoder-20260525-1535 \
  --baseline android-fast-m1-transient-prefill-dequant-20260525-141601 \
  --stage 2

MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-accept-m3-bf16-encoder-20260525-1536 \
  --assets-dir bench/android-assets \
  --threads 1 \
  --runs 3

bench/android_check_gate.py \
  --result android-accept-m3-bf16-encoder-20260525-1536 \
  --baseline android-accept-m1-transient-prefill-dequant-20260525-141733 \
  --stage 3
```

Stage 3:
- Common Voice median `1782.3 ms` vs M1 `1722.5 ms`: `+3.5%` slower.
- `model_load_ms` `149.5` vs `404.7`: `63.0%` faster.
- `memory_after_model_load_kb` `507756` vs `1567828`: `67.6%` lower.
- `peak_rss_kb` `1451268` vs `2136024`: `32.1%` lower.
- Guardrail passed: `哈喽，你好吗？已经坐低咗啦。`

Decision:
- Implemented by product decision as the preferred memory-saving variant.
  Memory/startup thresholds were met, but median inference regressed `+3.5%`,
  above the speed-first Stage 3 `+2%` regression limit for memory/load wins.
- Verification rerun after restoring M3 A:
  `android-fast-m3-bf16-encoder-rerun-20260525-160331` passed Stage 1 with
  load RSS `507904 KB`, peak RSS `1435216 KB`, guardrail OK, and `cv_yue_003`
  `+3.3%` slower than M1 fast baseline.

## Variant B: transformer layer weights transient only

Commands:

```bash
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-fast-m3-layer-bf16-encoder-20260525-1540 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads 1 \
  --runs 1

bench/android_check_gate.py \
  --result android-fast-m3-layer-bf16-encoder-20260525-1540 \
  --baseline android-fast-m1-transient-prefill-dequant-20260525-141601 \
  --stage 1

MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-candidate-m3-layer-bf16-encoder-20260525-1541 \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads 1 \
  --runs 1

bench/android_check_gate.py \
  --result android-candidate-m3-layer-bf16-encoder-20260525-1541 \
  --baseline android-fast-m1-transient-prefill-dequant-20260525-141601 \
  --stage 2

MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-accept-m3-layer-bf16-encoder-20260525-1542 \
  --assets-dir bench/android-assets \
  --threads 1 \
  --runs 3

bench/android_check_gate.py \
  --result android-accept-m3-layer-bf16-encoder-20260525-1542 \
  --baseline android-accept-m1-transient-prefill-dequant-20260525-141733 \
  --stage 3
```

Stage 3:
- Common Voice median `1787.0 ms` vs M1 `1722.5 ms`: `+3.7%` slower.
- `model_load_ms` `185.6` vs `404.7`: `54.1%` faster.
- `memory_after_model_load_kb` `558156` vs `1567828`: `64.4%` lower.
- `peak_rss_kb` `1484924` vs `2136024`: `30.5%` lower.
- Guardrail passed: `哈喽，你好吗？已经坐低咗啦。`

Decision:
- Not implemented. Keeping non-layer projections as F32 did not reduce the
  full-run inference regression and saved less RSS than Variant A; median
  inference still exceeded the `+2%` acceptance limit.

Takeaway:
- Encoder BF16 conversion saves roughly `1.0 GB` of load RSS and another
  `220-255 ms` of model load, but the repeated per-forward conversion cost is
  too high for the current acceptance rule.
- Do not retry this exact transient conversion path unless it includes caching,
  shape-specific conversion avoidance, or another way to keep full Common Voice
  median regression under `+2%`.
