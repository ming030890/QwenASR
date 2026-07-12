# Changelog

## [0.3.0] - 2026-07-12

### Performance & Quantization
- **Memory-Mapped Weight Loading (mmap):** Added support for zero-copy memory-mapped loading of pre-quantized INT8 weight files (`M4` format / `Q2B` optimization) to avoid memory allocations and drastically speed up model load times (31% faster load, 25% lower memory footprint).
- **Host-Side Pre-Quantized Tied `lm_head`:** Moved the tied vocabulary classification projection (`lm_head`) quantization to the host, storing it pre-quantized in Safetensors, completely avoiding on-device startup quantization overhead.
- **Dynamic Prefill Activation Quantization (`K4`):** Implemented a high-performance dynamic quantization prefill kernel on `aarch64` that quantizes activations on-the-fly and utilizes NEON `SDOT` (`vdotq_s32`) instruction sets, lowering RSS memory usage by 33% and improving prefill execution times.
- **Transient Prefill Dequantization (`M1`):** Optimizes memory bandwidth on mobile devices by keeping projection weights in INT8 and dequantizing them transiently into a shared thread-local F32 scratch buffer during compute.
- **Static OpenBLAS on Android:** Standardized compilation and static linking configurations for single-threaded CBLAS static OpenBLAS libraries on Android `arm64-v8a` target builds.

### Mobile & Testing Infrastructure
- **Same-Session Mobile Benchmark Comparison:** Developed a back-to-back testing and comparison harness (`bench/android_run.py` and `bench/android_compare.py`) to run baseline and candidate builds in immediate succession, eliminating thermal and CPU throttling noise on Android devices.
- **Phonetic & Semantic Guardrails:** Integrated transcript correctness tests with keyword checks (`你好吗`, `坐低咗`) into the mobile benchmark verification loop to prevent regressions on Cantonese live caption flows.
- **Multi-Stage Mobile Gate Checks:** Built `bench/android_check_gate.py` to evaluate candidates against a control baseline across all performance dimensions: end-to-end latency, first-token latency, memory usage (RSS/peak), and target binary size.

### Project & Documentation Cleanups
- **Sanitized Paths:** Removed all hardcoded personal paths (e.g. `/Users/tony` and `/Volumes/SSD/`) from scripts, result tables, and configurations to make execution user-agnostic.
- **Refined Autoresearch Ledger:** Merged mobile experiment instructions and history tables into unified `android-autoresearch-program.md` and `android-improvement-experiments.md` references.
- **Pruned Redundancies:** Cleaned up unused Flutter directories, old experimental notes, and duplicate document files to reduce repository weight.

---

## [0.2.3] - 2026-02-23

### Features

- **Live audio capture** (macOS) — `--live` flag captures from audio input devices (microphone, BlackHole) in real time with segmented, streaming, or VAD modes
- **VAD live mode** — `--vad` flag uses energy-based Voice Activity Detection to detect speech segments and transcribe each independently, with cross-segment prompt conditioning for improved accuracy
- **Model download subcommand** — `qwen-asr download --list` and `qwen-asr download <model>` for built-in model management
- **Forced alignment** — `--align` flag produces word-level timestamps for a known transcript using the ForcedAligner model variant

### Performance

- **Lazy encoder re-encoding** — Partial encoder tail is only re-encoded every other chunk in streaming mode, giving near-perfect LCP (Longest Common Prefix) reuse and cutting decoder prefill cost by ~50% on skip chunks
- **Streaming robustness** — Degeneracy detection resets decoder state when stale or repetitive output is detected; periodic re-anchoring prevents unbounded sequence growth

### Changed

- Debug messages (`[stream degen]`, `[stream reanchor]`) now only appear in `--debug` mode
- Project restructured into workspace: `crates/qwen-asr` (library), `crates/qwen-asr-cli` (CLI binary)
- Removed WIP banner from all README files

---

## [0.2.0] - 2026-02-15

### Performance

- **Reusable BF16→F32 scratch buffer** — Pre-allocated scratch in `DecoderBuffers` eliminates ~140 heap allocations per prefill pass
- **SIMD BF16→F32 bulk conversion** — NEON (`vshll_n_u16`) and AVX2 (`_mm256_cvtepu16_epi32`) paths for 4-8x faster weight conversion
- **Threaded encoder attention** — `bidirectional_attention` parallelized across heads via thread pool for near-linear scaling
- **Cached mel filter bank** — `OnceLock`-based lazy initialization avoids redundant computation in streaming mode
- **SIMD activation functions** — Vectorized `rms_norm`, `gelu`, and `swiglu` with fast polynomial exp approximation (NEON + AVX2)
- **Encoder buffer reuse** — New `EncoderBuffers` struct with persistent scratch avoids per-call allocations in encoder forward pass
- **vDSP integration** (macOS, `--features vdsp`) — `vDSP_dotpr`, `vDSP_vsmul`, `vDSP_vsma`, `vvexpf` leverage Apple AMX coprocessor

### Features

- **Built-in profiling** — `--profile` flag prints per-operation timing breakdown (call count, total/avg time)
- **iOS support** — Static library target with C-FFI API (`src/c_api.rs`): `qwen_asr_load_model`, `qwen_asr_transcribe_file`, `qwen_asr_transcribe_pcm`, `qwen_asr_free`
- **Android support** — Shared library target with JNI API (`src/jni_api.rs`) for `com.qwenasr.QAsrEngine` Java class
- **Feature flags** — `blas` (default), `vdsp`, `ios`, `android` for platform-specific builds
- **Cross-compilation config** — `.cargo/config.toml` with tuned CPU targets for iOS (`apple-a14`) and Android (`cortex-a76`)

### Changed

- Library crate renamed to `qwen_asr` (was `q-asr`) for valid Rust identifier in imports
- Library target now produces `lib`, `staticlib`, and `cdylib` outputs
- Thread pool workers recover from poisoned mutex instead of panicking
- Regression tests serialized via `Mutex` to prevent thread pool race conditions
- README updated with per-platform build instructions (macOS, Linux, iOS, Android)

---

## [0.1.0] - 2026-02-15

Initial release — pure Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- CPU-only Qwen3-ASR inference (0.6B and 1.7B models)
- Three runtime modes: offline, segmented (`-S`), streaming (`--stream`)
- NEON SIMD (aarch64) and AVX2+FMA (x86_64) acceleration
- BLAS via Accelerate (macOS) / OpenBLAS (Linux)
- Zero runtime Rust crate dependencies (only `libc`)
- 22 tests (kernels, audio, tokenizer, regression)
