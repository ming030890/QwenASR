# qwen-asr

[![OctoCounts](https://api.octocounts.com/badge/huanglizhuo/QwenASR/branch/main)](https://octocounts.com/?q=https%3A%2F%2Fgithub.com%2Fhuanglizhuo%2FQwenASR&ref=main)

A **blazing fast**, pure Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. It features zero heavy runtime dependencies (only `libc`).

This repository is a highly-optimized fork of [antirez/qwen-asr](https://github.com/antirez/qwen-asr) with significant enhancements for performance, memory efficiency, and mobile (Android) compatibility.

### Major Changes Since Fork

* **Zero-Copy Memory-Mapped Loading (`mmap`):** Implemented zero-copy memory-mapped loading for INT8 weights (`M4`/`Q2B`) to drastically reduce startup latency and initialization memory overhead.
* **Host-Side Pre-Quantized Tied `lm_head`:** Added host-side pre-quantization of the tied `lm_head` vocabulary projection into INT8 format, avoiding on-device quantization overhead at startup.
* **Dynamic Prefill Activation Quantization (`K4`):** Developed a custom dynamic quantization prefill kernel utilizing NEON `SDOT` (`vdotq_s32`) on `aarch64`, significantly accelerating prompt ingestion.
* **Transient Prefill Dequantization (`M1`):** Optimized memory bandwidth by transiently dequantizing INT8 weights into a shared thread-local F32 scratch buffer during compute.
* **Static OpenBLAS on Android:** Ported and optimized the engine for Android `arm64-v8a` environments, linking with a customized CBLAS-only static OpenBLAS target configuration.
* **Offline I8MM-Packed Mobile Model:** Quantized every encoder linear matrix and packed the accepted INT8 decoder/encoder weights into their final KleidiAI I8MM layout in the model artifact, eliminating runtime packing and duplicate row-major copies.
* **Same-Session Mobile Benchmark & Gate Checks:** Created an automated on-device testing and gate-check harness (`bench/android_run.py`, `bench/android_compare.py`, `bench/android_check_gate.py`) that monitors device thermals, enforces phonetic guardrails, and compares baseline vs. candidate builds on E2E latency, first-token latency, RSS/peak memory, and binary footprint.

Supports 0.6B and 1.7B models with multiple modes: offline, segmented, streaming, live capture, VAD live, and forced alignment.

**🚀 Extreme Performance:** On Apple Silicon (M5), the highly optimized CPU implementation transcribes a 28.2s audio sample in just **676ms** (**41.69x realtime**), outperforming both the upstream pure C implementation and measured MLX GPU baselines.



## Auto Research

Performance optimizations were discovered autonomously using the [autoresearch](https://github.com/karpathy/autoresearch) pattern: an AI agent loops over hypothesize-implement-benchmark-keep/revert cycles on the inference code. The experiment protocol is defined in [`android_experiment/android-autoresearch-program.md`](android_experiment/android-autoresearch-program.md).

## Benchmark

> [!NOTE]
> **TODO:** Provide the latest macOS (Apple Silicon M-series) and Android (Pixel 9 / Tensor G4) CPU benchmarks.


## Quick Start

```bash
# Install
cargo install qwen-asr-cli

# Download model
qwen-asr download qwen3-asr-0.6b

# Transcribe
qwen-asr -d qwen3-asr-0.6b -i audio.wav
```

Or download a pre-built binary from [GitHub Releases](https://github.com/huanglizhuo/QwenASR/releases).

## Usage

```
qwen-asr -d <model_dir> (-i <file> | --stdin | --live) [options]
```

```bash
qwen-asr -d qwen3-asr-0.6b -i audio.wav              # basic
qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent      # transcript only
cat audio.wav | qwen-asr -d qwen3-asr-0.6b --stdin     # pipe from stdin
qwen-asr -d qwen3-asr-0.6b -i long.wav -S 30           # segmented (long audio)
qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream       # streaming
qwen-asr -d qwen3-asr-0.6b --live --device "BlackHole 2ch"         # live capture (macOS)
qwen-asr -d qwen3-asr-0.6b --live --vad --device "BlackHole 2ch"   # VAD live
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English  # alignment
ffmpeg -i video.mp4 -f s16le -ar 16000 -ac 1 - | qwen-asr -d qwen3-asr-0.6b --stdin        # raw PCM
```

<details>
<summary>All options</summary>

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input WAV file | — |
| `--stdin` | Read audio from stdin (WAV or raw s16le 16kHz) | off |
| `--live` | Live capture from audio device (macOS) | off |
| `--device <name>` | Input device for live capture | system default |
| `--list-devices` | List audio input devices | — |
| `--vad` | VAD live mode | off |
| `-t <n>` | Thread count | all CPUs |
| `-S <secs>` | Segment target seconds | 0 (full) |
| `--stream` | Streaming mode | off |
| `--stream-chunk-sec <s>` | Chunk size for streaming | 2.0 |
| `--language <lang>` | Force output language (`en`, `zh`, `ja`, ...) | auto |
| `--silent` | Transcript only, no status output | off |
| `--profile` | Print timing breakdown | off |

</details>

## Build

**Always use release mode.** Debug builds are 10-50x slower.

```bash
# macOS
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Linux
sudo apt install libopenblas-dev   # Debian/Ubuntu
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Without BLAS
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features

# iOS (static library + C-FFI)
cargo build --release --target aarch64-apple-ios --features ios

# Android (shared library + JNI)
cargo ndk -t arm64-v8a build --release --features android
```

| Feature | Description |
|---------|-------------|
| `blas` (default) | BLAS linking (Accelerate on macOS, OpenBLAS on Linux) |
| `vdsp` | Accelerate vDSP/vForce for AMX (macOS) |
| `ios` | C-FFI API (`src/c_api.rs`) |
| `android` | JNI API (`src/jni_api.rs`) |

## OpenClaw Skill

One-command install for [OpenClaw](https://github.com/anthropics/openclaw) users:

```bash
bash skills/qwen-asr/scripts/install.sh
bash skills/qwen-asr/scripts/transcribe.sh audio.wav
```

## Acknowledgments

Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr), a pure C implementation of Qwen3-ASR inference by antirez.

## License

Same license as [antirez/qwen-asr](https://github.com/antirez/qwen-asr).
