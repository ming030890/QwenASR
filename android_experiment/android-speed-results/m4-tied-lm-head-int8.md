# M4: Tied LM head host-side pre-quantization

Purpose:
- Pre-quantize the tied ASR `lm_head` from `thinker.model.embed_tokens.weight`
  on the host and store it as an INT8 safetensors tensor, while keeping the
  BF16 token embedding tensor for embedding lookup.
- Load the INT8 `lm_head` via zero-copy mmap in `decoder.rs` instead of
  allocating and quantizing the full vocabulary projection at startup.

Implementation:
- `bench/prepare_int8_model.py` now emits:
  - `thinker.lm_head.weight` as `I8 [151936, 1024]`
  - `thinker.lm_head.weight_int8_scale` as `F32 [151936]`
- `Decoder::load()` now prefers an mmap-backed I8 `thinker.lm_head.weight`
  when it matches `lm_head_dim * dec_hidden`; otherwise it falls back to the
  original on-device BF16-to-INT8 quantization path.
- Generated model:
  `/Volumes/SSD/qwen-asr-m4-lmhead-int8/model.safetensors`
  - Original: `1789 MB`
  - M4: `1222 MB`
  - Saved: `567 MB`

Run conditions:
- Device: Pixel 9, fixed-performance mode.
- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8`.
- Config inherited from M3 A: static OpenBLAS, `android-blas`,
  `-C target-feature=+dotprod`, `threads=1`, Auto language with Yue prompt.
- Baseline: `android-accept-m3-bf16-encoder-20260525-1536`.

Commands:

```bash
python3 \
  bench/prepare_int8_model.py \
  --model-dir ./model \
  --output-dir ./qwen-asr-m4-lmhead-int8

THERMAL_TIMEOUT=0 \
bench/android_run.py \
  --label android-fast-m4-lmhead-int8-20260525-try2 \
  --model-dir /Volumes/SSD/qwen-asr-m4-lmhead-int8 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads 1 \
  --runs 1

bench/android_check_gate.py \
  --result android-fast-m4-lmhead-int8-20260525-try2 \
  --baseline android-build-smoke-m3-bf16-encoder-int8dec \
  --stage 1

THERMAL_TIMEOUT=0 \
bench/android_run.py \
  --label android-accept-m4-lmhead-int8-20260525-try2 \
  --model-dir /Volumes/SSD/qwen-asr-m4-lmhead-int8 \
  --threads 1 \
  --runs 3

bench/android_check_gate.py \
  --result android-accept-m4-lmhead-int8-20260525-try2 \
  --baseline android-accept-m3-bf16-encoder-20260525-1536 \
  --stage 3
```

Fast gate:
- `cv_yue_003` `1715.9 ms` vs M3 smoke `1712.0 ms`: `+0.2%` slower.
- `model_load_ms` `18.9 ms`.
- `memory_after_model_load_kb` `52520 KB`.
- `peak_rss_kb` `1131808 KB`.
- Guardrail passed: `哈喽，你好吗？已经坐低咗啦。`

Stage 3:
- Common Voice median `1771.6 ms` vs M3 A `1782.3 ms`: `0.6%` faster.
- `model_load_ms` `27.5` vs `149.5`: `81.6%` faster.
- `memory_after_model_load_kb` `52700` vs `507756`: `89.6%` lower.
- `peak_rss_kb` `1147644` vs `1451268`: `20.9%` lower.
- Guardrail passed: `哈喽，你好吗？已经坐低咗啦。`

Decision:
- Accept. M4 clears Stage 3 on model load, load RSS, and peak RSS while keeping
  full Common Voice median inference neutral/slightly faster versus M3 A.
- Keep this as the preferred memory/startup follow-up to M3 A.

