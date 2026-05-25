# K4: INT8 SDOT packed matmat for decoder prefill

Purpose:
- Replace the decoder prefill F32 matmul path (which dequantized INT8 weights to F32 scratch on-the-fly and routed through OpenBLAS F32 `sgemm`) with a direct, packed multi-row INT8 matmatmul using the `vdotq_s32` (SDOT) instructions.
- Eliminate the startup and runtime CPU/memory overhead of dequantizing weights, saving significant peak memory and drastically speeding up model loading and inference.

Implementation:
- Modified `DecoderBuffers` to include zero-allocation `pref_x_int8: Vec<i8>` and `pref_x_scales: Vec<f32>` activation quantization scratch buffers.
- Added `quantize_f32_to_int8_row` and `int8_matmat_threaded` in `mod.rs` to dynamically quantize activations row-by-row and execute parallelized matrix-matrix multiplication over the `seq_len` dimension using the lock-free `parallel_for` thread pool and optimized `neon::matvec_int8` assembly.
- Updated `decoder_prefill` in `decoder.rs` to route projections (`wq`, `wk`, `wv`, `wo`, `gate_up`, `down`) to the new `kernels::linear_nobias_int8_matmat` kernel on `aarch64` when `use_int8_weights` is active, falling back gracefully to the F32 dequantization path on other targets.
- Added a host-side unit test `test_linear_nobias_int8_matmat` in `tests/kernels.rs` to verify mathematical correctness.

Run conditions:
- Device: Pixel 9, fixed-performance mode.
- Model: `/Volumes/SSD/qwen-asr-int8dec` (INT8 quantized model).
- Config: static OpenBLAS, `android-blas`, `-C target-feature=+dotprod`, `threads=1`, Auto language with Yue prompt.
- Baseline: `android-accept-m1-transient-prefill-dequant-20260525-141733` (M1 transient dequant baseline).

Commands:

```bash
# Host unit test
cargo test --release --test kernels

# Stage 1 Fast Performance Gate
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-fast-k4-int8-prefill-sdot-20260525-183000 \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1 \
  --profile

bench/android_check_gate.py \
  --result android-fast-k4-int8-prefill-sdot-20260525-183000 \
  --baseline android-fast-p0-phase-split-20260525-165425 \
  --stage 1

# Stage 3 Acceptance Gate
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label android-accept-k4-int8-prefill-sdot-20260525-185000 \
  --assets-dir bench/android-assets \
  --threads "1" \
  --runs 3

bench/android_check_gate.py \
  --result android-accept-k4-int8-prefill-sdot-20260525-185000 \
  --baseline android-accept-m1-transient-prefill-dequant-20260525-141733 \
  --stage 3
```

Stage 1 Fast Gate:
- `cv_yue_003` total inference: `1500.0 ms` vs `1672.5 ms` (`-10.3%` faster ✓).
- `cv_yue_003` `prefill_ms` phase: `713.7 ms` vs `893.2 ms` (`-20.1%` faster ✓).
- `ptt_guardrail` total inference: `2535.6 ms` vs `2852.2 ms` (`-11.1%` faster ✓).
- `ptt_guardrail` `prefill_ms` phase: `1244.1 ms` vs `1459.5 ms` (`-14.8%` faster ✓).
- Guardrail transcript matched perfectly: `'Hello，你好吗？已经坐低咗啦。'`

Stage 3 Acceptance Gate:
- **Common Voice median inference**: **`1565.8 ms` vs `1722.5 ms`** (**`9.1%` faster** ✓ - exceeds 5% gate).
- **Common Voice decode**: `1131.8 ms` vs `1345.9 ms` (**`15.9%` faster** ✓).
- **Guardrail median**: `2481.9 ms` vs `2716.5 ms` (**`8.6%` faster** ✓).
- **Model load time**: **`128.1 ms` vs `404.7 ms`** (**`68.3%` faster** ✓ - exceeds 5% gate).
- **Initial app RSS (load)**: **`508,024 KB` vs `1,567,828 KB`** (**`67.6%` lower** ✓ - exceeds 10% gate).
- **Peak inference RSS**: **`1,426,984 KB` vs `2,136,024 KB`** (**`33.2%` lower** ✓ - exceeds 10% gate).
- Guardrail transcript matched perfectly: `'Hello，你好吗？已经坐低咗啦。'`

Decision:
- **Accept**. K4 is an exceptional, sweep-level win on Pixel 9. It sets a new standard for both performance and efficiency, achieving a `9.1%` speedup in full-utterance inference, a `20.1%` speedup in prefill phase, a `68.3%` startup speedup, and massive memory savings (`67.6%` lower startup RSS and `33.2%` lower peak RSS).
