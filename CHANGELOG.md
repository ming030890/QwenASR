# Changelog

## [0.3.0] - 2026-07-12

### Performance & Quantization
- **Memory-Mapped Weight Loading (mmap):** Added support for zero-copy memory-mapped loading of pre-quantized INT8 weight files (`M4` format / `Q2B` optimization) to avoid memory allocations and speed up model load times.
- **Host-Side Pre-Quantized Tied `lm_head`:** Moved the tied vocabulary classification projection (`lm_head`) quantization to the host, storing it pre-quantized in Safetensors, completely avoiding on-device startup quantization overhead.
- **Dynamic Prefill Activation Quantization (`K4`):** Implemented a dynamic quantization prefill kernel on `aarch64` that quantizes activations on-the-fly and utilizes NEON `SDOT` (`vdotq_s32`) instruction sets, lowering RSS memory usage and improving prefill execution times.
- **Transient Prefill Dequantization (`M1`):** Optimizes memory bandwidth by keeping projection weights in INT8 and dequantizing them transiently into a shared thread-local F32 scratch buffer during compute.
- **Static OpenBLAS on Android:** Standardized compilation and static linking configurations for single-threaded CBLAS static OpenBLAS libraries on Android `arm64-v8a` target builds.

---

## [0.2.3] - 2026-02-23

### Features
- **Live audio capture** (macOS) — `--live` flag captures from audio input devices in real time with segmented, streaming, or VAD modes.
- **VAD live mode** — `--vad` flag uses energy-based Voice Activity Detection to detect speech segments and transcribe each independently.
- **Model download subcommand** — `qwen-asr download --list` and `qwen-asr download <model>` for built-in model management.
- **Forced alignment** — `--align` flag produces word-level timestamps for a known transcript using the ForcedAligner model variant.

### Performance
- **Lazy encoder re-encoding** — Partial encoder tail is only re-encoded every other chunk in streaming mode, cutting decoder prefill cost by ~50%.
- **Streaming robustness** — Degeneracy detection resets decoder state when repetitive output is detected; periodic re-anchoring prevents sequence growth.

---

## [0.2.0] - 2026-02-15

### Performance
- **Reusable BF16→F32 scratch buffer** — Pre-allocated scratch in `DecoderBuffers` eliminates ~140 heap allocations per prefill pass.
- **SIMD BF16→F32 bulk conversion** — NEON and AVX2 paths for 4-8x faster weight conversion.
- **Threaded encoder attention** — `bidirectional_attention` parallelized across heads via thread pool for near-linear scaling.
- **Cached mel filter bank** — `OnceLock`-based lazy initialization avoids redundant computation in streaming mode.
- **SIMD activation functions** — Vectorized `rms_norm`, `gelu`, and `swiglu` with fast polynomial exp approximation.
- **Encoder buffer reuse** — Persistent scratch avoids per-call allocations in encoder forward pass.
- **vDSP integration** (macOS) — leverages Apple AMX coprocessor for gemm and vector math operations.

---

## [0.1.0] - 2026-02-15

Initial release — pure Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- CPU-only Qwen3-ASR inference (0.6B and 1.7B models)
- Three runtime modes: offline, segmented, streaming
- NEON SIMD (aarch64) and AVX2+FMA (x86_64) acceleration
- BLAS via Accelerate (macOS) / OpenBLAS (Linux)
- Zero runtime Rust crate dependencies (only `libc`)
