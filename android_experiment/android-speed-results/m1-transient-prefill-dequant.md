# M1: Transient prefill dequantization

Purpose:
- Stop storing permanent F32 prefill copies of pre-quantized INT8 decoder
  weights.
- During decoder prefill, dequantize each layer projection into the shared F32
  scratch buffer immediately before the SGEMM call.

Implementation:
- `crates/qwen-asr/src/decoder.rs`:
  - INT8 model path now leaves `wq_weight_f32_prefill`,
    `wk_weight_f32_prefill`, `wv_weight_f32_prefill`,
    `wo_weight_f32_prefill`, `gate_up_fused_f32_prefill`, and
    `down_weight_f32_prefill` empty instead of allocating per-layer F32 copies.
  - Added `dequantize_int8_to_f32_scratch()` and uses `DecoderBuffers`
    `bf16_scratch` as the reusable F32 prefill matrix buffer.
  - BF16 model path keeps its previous permanent F32 prefill behavior.

Run command:

```bash
LABEL=android-fast-m1-transient-prefill-dequant-20260525-141601
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads 1 \
  --runs 1

bench/android_check_gate.py \
  --result "$LABEL" \
  --baseline android-fast-q2b-int8-mmap-20260525-124250 \
  --stage 1
```

Run conditions:
- Device: `44241FDAQ000DX`.
- Thermal Status `0` before run.
- Model: `/Volumes/SSD/qwen-asr-int8dec`.
- Config inherited from accepted Q2B: static OpenBLAS, `android-blas`,
  `-C target-feature=+dotprod`, `threads=1`, Auto language with Yue prompt.

Results:

| Arm | Input | `model_load_ms` | `memory_after_model_load_kb` | `peak_rss_kb` | `inference_ms` | `encode_ms` | `decode_ms` | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M1 | `cv_yue_003` | `400.1` | `1567604` | `2119344` | `1651.7` | `330.7` | `1320.8` | `你睇下呢部劇。` |
| M1 | `ptt_guardrail` | `408.8` | `1567620` | `2133792` | `2801.2` | `751.4` | `2049.5` | `哈喽，你好吗？已经坐低咗啦。` |

Comparison vs Q2B fast gate (`android-fast-q2b-int8-mmap-20260525-124250`):

| Metric | Q2B | M1 | Delta |
| --- | ---: | ---: | ---: |
| `cv_yue_003 inference_ms` | `1571.9` | `1651.7` | `+5.1%` slower |
| `model_load_ms` | `919.1` | `400.1` | `-56.5%` faster |
| `memory_after_model_load_kb` | `3708184` | `1567604` | `-57.7%` lower |
| `peak_rss_kb` | `3816340` | `2119344` | `-44.5%` lower |

Stage 1 gate:
- **PASS**: `cv_yue_003` was `+5.1%` slower, within the Stage 1 `+10%`
  regression limit.
- Guardrail passed with expected Cantonese/code-switch transcript.

Caveat:
- An earlier invalid run used `/Volumes/SSD/qwen3-asr-0.6b`, the original BF16
  model, and failed the gate with no memory improvement. Ignore that label:
  `android-fast-m1-transient-prefill-dequant-20260525-141323`.

## Stage 3: Acceptance Gate Run

Run command:

```bash
LABEL=android-accept-m1-transient-prefill-dequant-20260525-141733
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir bench/android-assets \
  --threads 1 \
  --runs 3

bench/android_check_gate.py \
  --result "$LABEL" \
  --baseline android-accept-q2b-int8-mmap-20260525-131609 \
  --stage 3
```

Stage 3 results:

| Metric | Q2B baseline | M1 | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | `1742.2 ms` | `1722.5 ms` | `-1.1%` faster |
| `cv_yue_000 inference_ms` | `1660.8` | `1671.5` | `+0.6%` slower |
| `cv_yue_001 inference_ms` | `2054.4` | `2051.1` | `-0.2%` faster |
| `cv_yue_002 inference_ms` | `1970.0` | `1981.8` | `+0.6%` slower |
| `cv_yue_003 inference_ms` | `1552.6` | `1577.5` | `+1.6%` slower |
| `cv_yue_004 inference_ms` | `1742.2` | `1722.5` | `-1.1%` faster |
| `model_load_ms` | `935.2` | `404.7` | `-56.7%` faster |
| `memory_after_model_load_kb` | `3707968` | `1567828` | `-57.7%` lower |
| `peak_rss_kb` | `3832060` | `2136024` | `-44.3%` lower |

Stage 3 gate:
- **PASS / ACCEPT**: model load, load RSS, and peak RSS thresholds all met.
- Per-clip inference regressions stayed within the Stage 3 limit.
- Guardrail passed: `哈喽，你好吗？已经坐低咗啦。`

Decision:
- **Accept M1.** It removes roughly another `2.14 GB` of load-time RSS versus
  Q2B and cuts model load to about `405 ms`, while preserving full Common Voice
  inference behavior.
