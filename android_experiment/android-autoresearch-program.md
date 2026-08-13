# QwenASR Android Autoresearch Program

> Android-on-device optimization loop for the QwenASR Rust inference engine.
> This unified document defines the protocol, benchmark rules, experiment history, and active backlog for Android optimization.

## Goal

Improve Android arm64 inference speed, startup time, binary footprint, and first token latency on the connected device while preserving transcript quality.

**Latest Production Model Baseline**: The current production configuration uses the **`qwen-asr-m4-lmhead-int8-v16-manual-code-switching-cp4000`** model. This integrates:
- Host-side pre-quantized tied `lm_head` (M4 format) for zero-copy startup loading.
- On-device zero-copy memory-mapped INT8 weight loading.
- Manual language code-switching thresholds.

Primary metric:
- Median `inference_ms` over the fixed Common Voice Yue clips.

Secondary metrics:
- `first_token_ms`: decoder prefill latency.
- `model_load_ms`: elapsed time inside `QwenCtx::load()`.
- `memory_after_model_load_kb`: process RSS immediately after model load.
- `peak_rss_kb`: sampled peak process RSS during each inference run.
- `binary_size_bytes`: compiled size of the candidate binary.

Guardrails:
- The benchmark binary must report `OpenBLAS`; builds using another BLAS backend
  are not comparable or eligible for acceptance.
- Every candidate transcript must exactly equal the controlled baseline transcript
  for the corresponding audio file, including Unicode characters, capitalization,
  whitespace, and punctuation.
- Every repeated run within both baseline and candidate must produce the same transcript.
- The candidate and baseline audio-file sets must be identical.
- Common Voice average CER may still be reported diagnostically, but it cannot
  override an exact-transcript failure.
- Host Rust tests must continue to pass (`cargo test --release --test kernels --test audio`).
- Android JNI production builds must not be broken.

---

## Fixed Inputs & Assets

Model:
- Host source: Path to model directory (e.g. `qwen3-asr-0.6b`)
- Device path: `/data/local/tmp/qwen-asr-bench/model`

Assets:
- Host directory: `bench/android-assets`
- Device directory: `/data/local/tmp/qwen-asr-bench/assets`
- Common Voice subset: five deterministic clips from Common Voice Cantonese (`yue`)
  - `cv_yue_000.wav`
  - `cv_yue_001.wav`
  - `cv_yue_002.wav`
  - `cv_yue_003.wav`
  - `cv_yue_004.wav`

Create target gate asset directories on the host:
```bash
# Fast-gate (Stage 1)
FAST_ASSETS=./bench/android-fast-assets
mkdir -p "$FAST_ASSETS"
cp -f bench/android-assets/cv_yue_003.wav "$FAST_ASSETS/"
cp -f bench/android-assets/cv_yue_003.txt "$FAST_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.wav "$FAST_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.txt "$FAST_ASSETS/"

# Candidate-gate (Stage 2)
CANDIDATE_ASSETS=./bench/android-candidate-assets
mkdir -p "$CANDIDATE_ASSETS"
cp -f bench/android-assets/cv_yue_003.wav "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/cv_yue_003.txt "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/cv_yue_001.wav "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/cv_yue_001.txt "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.wav "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.txt "$CANDIDATE_ASSETS/"
```

---

## Benchmark Harness Setup

Standard environment variables:
- `ADB`: adb binary path
- `MODEL_DIR`: host model directory
- `NDK_HOME`: Android NDK path (prefers NDK 29)
- `CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER`: target linker wrapper path (e.g., `./android-aarch64-clang`)

OpenBLAS static baseline:
- Expected at `/private/tmp/OpenBLAS/libopenblas.a`. Build: static Android arm64, `TARGET=ARMV8`, `ONLY_CBLAS=1`, `USE_THREAD=0`.

Prompt/Language Policy:
- `BENCH_LANGUAGE=Auto` and `BENCH_PROMPT='音频是粤语，可能夹杂英语'` (matches keyboard code-switching).

---

## Thermal Control & Discipline

`bench/android_run.py` polls `dumpsys thermalservice` and blocks execution until `Thermal Status: 0` is reached to prevent thermal throttling.
- Run device plugged in using the same charger.
- Prevent background app activity.
- Perform tests in alternating baseline-candidate runs to cancel out thermal bias.

---

## Acceptance Rules

Stage 1 and Stage 2 serve as filters. Stage 3 defines the final acceptance gate.

Stage 3 accepts a change only if the exact-transcript gate passes and one of the
target metrics improves by at least the specified threshold, provided there is
no major regression in other metrics (with inference latency regression capped at `2%`):
- **Latency (Inference):** Full Common Voice median `inference_ms` improves by at least `5%`.
- **First Token Latency:** Median `first_token_ms` (decoder prefill) improves by at least `5%`.
- **Memory Usage:** `memory_after_model_load_kb` or `peak_inference_rss_kb` improves by at least `10%`.
- **Binary Size:** Candidate binary size (`binary_size_bytes`) improves by at least `5%`.
- **Model Load:** `model_load_ms` improves by at least `5%`.

All measurements are performed back-to-back using the comparison harness script:
```bash
bench/android_compare.py --baseline-binary bench/android-baselines/BASELINE --label LABEL --stage 3
```
This triggers `bench/android_check_gate.py` to evaluate the thresholds and report the PASS/FAIL verdict.

Startup, memory, or binary size wins must not regress full Common Voice median
inference latency by more than `2%`. Any transcript mismatch automatically
rejects the experiment regardless of the performance win.

---

## Experiment Loop

### Stage 0: Host and Android Build Gate
Run after every code change to catch compile/runtime crashes:
```bash
cargo test --release --test kernels --test audio
bench/android_run.py --label android-build-smoke --assets-dir ./bench/android-fast-assets --threads "1" --runs 1
```

### Stage 1: Fast Performance Gate
Runs on single-clip assets:
```bash
LABEL=android-fast-<idea>-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py --label "$LABEL" --assets-dir ./bench/android-fast-assets --threads "1" --runs 1
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 1
```

### Stage 2: Candidate Gate
Runs on two-clip assets:
```bash
LABEL=android-candidate-<idea>-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py --label "$LABEL" --assets-dir ./bench/android-candidate-assets --threads "1" --runs 1
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 2
```

### Stage 3: Acceptance Gate
Full evaluation sweep:
```bash
LABEL=android-accept-<idea>-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py --label "$LABEL" --assets-dir bench/android-assets --threads "1" --runs 3
bench/android_check_gate.py --result "$LABEL" --baseline BASELINE_LABEL --stage 3
```

---
