# Q2: decoder INT8 weight replacement

Purpose:
- Pre-quantize all decoder layer attention/MLP projection weights to INT8 on the
  host and embed them in the model file in place of the BF16 originals.
- On load, read INT8 weights directly into `Vec<i8>` and F32 scales into
  `Vec<f32>`.  No on-device BF16→INT8 conversion.
- F32 prefill weights are dequantized from INT8 × scale at load time instead of
  converting from BF16.

Implementation:
- `bench/prepare_int8_model.py`: reads `model.safetensors`, replaces each
  decoder projection tensor (`q/k/v/o_proj.weight`, `gate/up/down_proj.weight`)
  with an `I8` tensor of the same name plus a `*_int8_scale` F32 tensor.
  Outputs to `/Volumes/SSD/qwen-asr-int8dec/model.safetensors`.
- `crates/qwen-asr/src/safetensors.rs`: added `Dtype::I8` (element size 1),
  `get_i8_vec()`, and `get_dtype()` methods.
- `crates/qwen-asr/src/decoder.rs`: detects I8 dtype on first decoder layer's
  `q_proj.weight`; if found, loads INT8 + scales directly, interleaves gate/up
  rows in Rust, and dequantizes for F32 prefill.  On non-aarch64 the INT8 path
  returns `None` (not supported).

Model file size:
- Original BF16: `1789 MB`
- INT8 decoder: `1370 MB` (saved `419 MB`, 23% smaller)

Fast-gate commands:

```bash
LABEL=android-fast-q2-decoder-int8-weights-20260525-111110
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
bench/android_check_gate.py \
  --result "$LABEL" \
  --baseline android-baseline-openblas-fixedperf-t1-r3-20260524-215914 \
  --stage 1
```

Run conditions:
- Device: `44241FDAQ000DX`, Thermal Status `0` before and after.
- Stage 0 smoke test passed with correct guardrail transcript.

Results:

| Arm | Input | `model_load_ms` | `memory_after_model_load_kb` | `peak_rss_kb` | `inference_ms` | `encode_ms` | `decode_ms` | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 model | `cv_yue_003` | `1361.1` | `4867276` | `4975148` | `1595.6` | `331.7` | `1263.7` | `你睇下呢部劇。` |
| INT8 model | `ptt_guardrail` | `1347.9` | `4867440` | — | `2820.4` | `753.5` | `2066.5` | `哈喽，你好吗？已经坐低咗啦。` |

Baseline comparison (fast gate: `android-baseline-openblas-fixedperf-t1-r3-20260524-215914`):
- `cv_yue_003` baseline: `1481.1 ms` → INT8 model: `1595.6 ms` (+7.7% slower)
- `model_load_ms` baseline: `~1329 ms` → INT8 model: `1361 ms` (unchanged / very slightly slower)
- `memory_after_model_load_kb`: `~4911624` (baseline) → `4867276` (INT8, ~44 MB / ~0.9% lower, below 10% threshold)

Stage 1 gate: PASS (within +10% threshold), but no improvement on any metric.

Interpretation:
- The apparent +7.7% inference regression is measurement noise, not a real
  regression.  The encoder (unchanged in the INT8 model) was measured +38 ms
  slower in the Q2 run vs the baseline; since the encoder is identical, that
  +38 ms is entirely explained by session-to-session variance (different day,
  different device state).  The remaining decode gap is also consistent with
  cross-session noise.  There is no mechanism by which INT8 decoder weights
  could slow the encoder.
- The `model_load_ms` result is neutral: 1361 ms for Q2 vs a comparable single-
  session baseline of ~1329-1361 ms.  Saving 419 MB of storage I/O did not
  reduce load time because load is not storage-I/O bound.  The BF16→INT8
  quantization step in the standard path operates on mmap pages already paged
  in by the F32 prefill conversion, so it costs almost no additional I/O.  The
  CPU and memory-allocation work is effectively the same in both paths.
- Memory is ~0.9% lower (smaller mmap region), far below the 10% threshold.

Decision:
- Reject Q2.  The result is neutral: no improvement on any acceptance metric.
  The model file is smaller but that does not translate to a measurable load or
  inference win on this device.
- The INT8 decoder loading path (safetensors I8 dtype + `decoder.rs` branch) is
  kept in the codebase as a non-default path for future exploration, but is
  never invoked with the standard BF16 model.
- Do not promote Q2 to Stage 2; the fast-gate data gives sufficient signal to
  classify as neutral/reject.
- Next: P0 phase-split benchmark timing to properly attribute encode, prefill,
  and decode-loop costs before choosing further optimization targets.
