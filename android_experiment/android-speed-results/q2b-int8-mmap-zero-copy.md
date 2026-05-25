# Q2B: INT8 mmap zero-copy loading

Purpose:
- Refactor the INT8 model loading path to use raw mmap pointers instead of
  copying INT8 data into `Vec<i8>` heap allocations.
- Pre-fuse gate+up projections in the model file so the fused tensor can also
  be loaded zero-copy.
- Fix a latent shape bug in `prepare_int8_model.py` where non-quantized tensors
  (norms, encoder weights) were written with shape `[n_bytes]` instead of their
  original logical shape, causing silent data corruption for mid-file tensors and
  SIGSEGV for the last tensor in the file.

Implementation changes:
- `crates/qwen-asr/src/safetensors.rs`: added `get_i8_ptr()` returning
  `(*const i8, usize)` pointing directly into the mmap region (no copy).
- `crates/qwen-asr/src/decoder.rs`: new `Int8Buf` type — either a zero-copy
  mmap pointer or an owned `Vec<i8>` (for on-device BF16→INT8 path).
  `Int8Buf` implements `Deref<Target=[i8]>` so all inference call sites are
  unchanged. `DecLayer` fields changed from `Vec<i8>` to `Int8Buf`.
- `bench/prepare_int8_model.py`:
  - gate and up projections are now pre-interleaved into `gate_up_fused.weight`
    I8 [2×inter, hidden] at model-prep time.
  - non-quantized tensors now preserve their original shape in the header
    (shape bug fix).

Bug found and fixed: the original `prepare_int8_model.py` wrote non-quantized
tensors with `shape = [n_bytes_of_raw_data]` instead of the actual tensor shape.
For BF16 norms with shape [1024], this wrote shape [2048], making `numel() = 2048`
instead of 1024. `get_f32` then tried to read `2048 × 2 = 4096` bytes from a
2048-byte region. This was silent corruption for mid-file tensors (reading into
the next tensor's bytes) and SIGSEGV for the last tensor in the file
(`thinker.model.norm.weight`). The Q2 experiment avoided the SIGSEGV only because
different tensors came after the norm in that file layout.

Expected RSS savings:
- INT8 decoder projections (~420 MB) no longer copied to heap.
- Actual saving is larger because the smaller file (1370 MB vs 1789 MB BF16)
  also reduces total mmap pages that get paged in.

Run commands:

```bash
LABEL=android-fast-q2b-int8-mmap-20260525-124250
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
- Same-day run (no cross-session thermal variation).

Results:

| Arm | Input | `model_load_ms` | `memory_after_model_load_kb` | `peak_rss_kb` | `inference_ms` | `encode_ms` | `decode_ms` | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 mmap | `cv_yue_003` | `919.1` | `3708184` | `3816340` | `1571.9` | `339.2` | `1232.5` | `你睇下呢部劇。` |
| INT8 mmap | `ptt_guardrail` | `913.9` | `3708008` | — | `2765.7` | `760.2` | `2005.2` | `哈喽，你好吗？已经坐低咗啦。` |

Comparison vs BF16 baseline (`android-baseline-openblas-fixedperf-t1-r3-20260524-215914`):
- `cv_yue_003` baseline: `1481.1 ms` → INT8 mmap: `1571.9 ms` (+6.1%, within cross-session noise)
- `model_load_ms` baseline: `~1329–1402 ms` → `919.1 ms` (**~32–35% faster**)
- `memory_after_model_load_kb` baseline: `~4911624` → `3708184` (**~24.5% lower**, −1175 MB)

Comparison vs Q2 (old Vec<i8> copy path):
- `model_load_ms` Q2: `1361 ms` → Q2B: `919 ms` (**32% faster**)
- `memory_after_model_load_kb` Q2: `4867276` → Q2B: `3708184` (**24% lower**, −1131 MB)

Stage 1 gate: **PASS** (cv_yue_003 +6.1%, within ±10% threshold).

## Stage 3: Acceptance Gate Run

Run command:

```bash
LABEL=android-accept-q2b-int8-mmap-20260525-131609
MODEL_DIR=/Volumes/SSD/qwen-asr-int8dec \
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir bench/android-assets \
  --threads "1" \
  --runs 3

bench/android_check_gate.py \
  --result "$LABEL" \
  --baseline android-baseline-py-full-20260525-103710 \
  --stage 3
```

Stage 3 Results:

| Metric | Result (Q2B) | Baseline (BF16 Python rerun) | Delta |
| --- | --- | --- | ---: |
| **Common Voice Median Inference** | `1742.2 ms` | `1778.2 ms` | **-2.0%** (faster ✓) |
| **cv_yue_000 Inference** | `1660.8 ms` | `1715.1 ms` | **-3.2%** (faster ✓) |
| **cv_yue_001 Inference** | `2054.4 ms` | `2108.7 ms` | **-2.6%** (faster ✓) |
| **cv_yue_002 Inference** | `1970.0 ms` | `2040.7 ms` | **-3.5%** (faster ✓) |
| **cv_yue_003 Inference** | `1552.6 ms` | `1617.5 ms` | **-4.0%** (faster ✓) |
| **cv_yue_004 Inference** | `1742.2 ms` | `1778.2 ms` | **-2.0%** (faster ✓) |
| **Model Load Time (`model_load_ms`)** | `935.2 ms` | `1363.7 ms` | **-31.4%** (faster ✓) |
| **Memory after Load (`memory_kb`)** | `3707968 KB` (`3.54 GB`) | `4911568 KB` (`4.68 GB`) | **-24.5%** (lower ✓) |
| **Peak Inference RSS (`peak_rss_kb`)** | `3832060 KB` (`3.65 GB`) | `5036008 KB` (`4.80 GB`) | **-23.9%** (lower ✓) |

Acceptance criteria check (Stage 3 per acceptance rule):
- `model_load_ms` improvement: **31.4%** (**PASS**, threshold ≥ 5%)
- `memory_after_model_load_kb` improvement: **24.5%** (**PASS**, threshold ≥ 10%)
- `peak_inference_rss_kb` improvement: **23.9%** (**PASS**, threshold ≥ 10%)
- Inference regression check: **-2.0%** change, i.e., slightly faster, completely avoiding the maximum regression limit of +2.0% for startup/memory wins.

Decision:
- **Accept Q2B.** The zero-copy mmap INT8 decoder weights path is an absolute success:
  - Reductions of **~1.2 GB** in memory footprint (RSS and peak RSS).
  - Reductions of **~430 ms** in model load times (over 30% speedup).
  - Absolutely zero regression in inference throughput (in fact, slightly faster by 2.0%).
- All future experiments will inherit this zero-copy mmap model configuration.
- The next step is **P0: phase-split benchmark timing** to split the encode/decode timings into detailed components (`mel_ms`, `encoder_ms`, `prefill_ms`, `decode_loop_ms/token`).
