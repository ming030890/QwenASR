# QwenASR Android Autoresearch Program

> Android-on-device optimization loop for the QwenASR Rust inference engine.
> Keep this separate from the macOS autoresearch notes because the build,
> benchmark, and performance bottlenecks are different.

## Goal

Improve Android arm64 inference speed on the connected Pixel 9 while preserving
transcript quality.

Primary metric:
- Median `inference_ms` over the five fixed Common Voice Yue clips.

Secondary metrics:
- `model_load_ms`: elapsed time inside `QwenCtx::load()`.
- `memory_after_model_load_kb`: process RSS immediately after model load and
  prompt/language setup.
- `peak_rss_kb`: sampled peak process RSS during each inference run.

Guardrails:
- `ptt_guardrail.wav` transcript must remain semantically and phonetically
  stable for `Hello,你好吗？已经坐低咗啦。`; exact punctuation/script can vary
  under Auto language.
- Common Voice average CER must not regress versus the current best baseline.
- Host Rust tests must continue to pass:
  `cargo test --release --test kernels --test audio`
- Android production/JNI builds must not be broken by benchmark-only changes.

## Fixed Inputs

Model:
- Host source: `/Volumes/SSD/qwen3-asr-0.6b`
- Device path: `/data/local/tmp/qwen-asr-bench/model`

Assets:
- Host directory: `bench/android-assets`
- Device directory: `/data/local/tmp/qwen-asr-bench/assets`
- Common Voice subset: five deterministic clips from
  `ming030890/common_voice_21_0_yue`
- Guardrail source: `/Users/tony/Downloads/PTT-20241208-WA0011.opus`

Fixed Common Voice files:
- `cv_yue_000.wav`: reference `我急求方法啊`
- `cv_yue_001.wav`: reference `唔做嘢嘅佔多數囉`
- `cv_yue_002.wav`: reference `證據嘅話真係冇呀`
- `cv_yue_003.wav`: reference `你睇下呢部劇`
- `cv_yue_004.wav`: reference `仲要提供乜嘢資料？`

Prepare or refresh assets with:

```bash
bench/android_prepare_assets.py
```

Create the fixed fast-gate asset subset once:

```bash
FAST_ASSETS=/Volumes/SSD/qwen-asr-android-fast-assets
mkdir -p "$FAST_ASSETS"
cp -f bench/android-assets/cv_yue_003.wav "$FAST_ASSETS/"
cp -f bench/android-assets/cv_yue_003.txt "$FAST_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.wav "$FAST_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.txt "$FAST_ASSETS/"
```

Create the fixed two-clip candidate subset once:

```bash
CANDIDATE_ASSETS=/Volumes/SSD/qwen-asr-android-candidate-assets
mkdir -p "$CANDIDATE_ASSETS"
cp -f bench/android-assets/cv_yue_003.wav "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/cv_yue_003.txt "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/cv_yue_001.wav "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/cv_yue_001.txt "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.wav "$CANDIDATE_ASSETS/"
cp -f bench/android-assets/ptt_guardrail.txt "$CANDIDATE_ASSETS/"
```

## Benchmark Harness

Android benchmark binary:

```bash
PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
bench/android_run.py --label android-baseline --runs 3 --threads "1"
```

Standard environment for every run:

```bash
export ADB=/Users/tony/Library/Android/sdk/platform-tools/adb
export MODEL_DIR=/Volumes/SSD/qwen3-asr-0.6b
export NDK_HOME=/Users/tony/Library/Android/sdk/ndk/29.0.14206865
export PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$PWD/android-aarch64-clang"
```

NDK note:
- Prefer NDK `29.0.14206865` for Android benchmark builds and OpenBLAS
  rebuilds on this machine. The installed NDK `30.0.14904198` has a text
  `clang` shim that drops compiler arguments when OpenBLAS invokes the standard
  `aarch64-linux-android*-clang` wrapper. `bench/android_run.py` defaults to
  NDK 29 but still honors `NDK_HOME`, `ANDROID_NDK_HOME`, `ANDROID_NDK_ROOT`, or
  `NDK_BIN` overrides.

The harness is separate from production JNI code:
- Source: `crates/qwen-asr-cli/src/bin/bench_android.rs`
- Target: `aarch64-linux-android`
- Output JSON fields include setup/model/audio timing, encode/decode timing,
  model-load RSS, sampled inference peak RSS, median inference timing, RTF,
  transcript, optimization flags, target arch, per-run details, and optional
  profiling counters.

Prompt/language policy:
- Match upstream Qwen3-ASR semantics: only append `language {Language}` when a
  language is explicitly forced.
- Unset language and `Auto` mean auto language detection; they must not default
  to English.
- For fixed Yue Android assets, use concise Simplified Chinese context when
  testing code-switching behavior, for example:
  `音频是粤语，可能夹杂英语`.
- `bench/android_run.py` defaults to `BENCH_LANGUAGE=Auto` and
  `BENCH_PROMPT='音频是粤语，可能夹杂英语'`.

Canonical results files:
- Summary TSV: `results-android.tsv`
- Full JSON: `bench/android-results/<label>/*.json`
- New TSV rows include `model_load_ms`, `memory_after_model_load_kb`, and
  `peak_inference_rss_kb`. Older rows may not have these fields.

Best accepted Android baseline so far:
- `android-baseline-openblas-fixedperf-t1-r3-20260524-215914`
- Config: OpenBLAS static archive, `android-blas`, `-C target-feature=+dotprod`,
  engine `threads=1`, fixed-performance mode.
- `bench/android_run.py` defaults to this baseline config. Override defaults
  only for a named experiment.
- Common Voice median: `1677.491 ms`, encode `383.011 ms`, decode
  `1305.652 ms`.
- Result JSON:
  `bench/android-results/android-baseline-openblas-fixedperf-t1-r3-20260524-215914/common_voice-t1.json`.

Accepted OpenBLAS baseline archive:
- Expected local path: `/private/tmp/OpenBLAS/libopenblas.a`.
- Build kind: static Android arm64 OpenBLAS, `TARGET=ARMV8`,
  `ONLY_CBLAS=1`, `NOFORTRAN=1`, `USE_THREAD=0`, `NUM_THREADS=1`.
- Rebuild from scratch if the archive is missing:

```bash
rm -rf /private/tmp/OpenBLAS
git clone https://github.com/OpenMathLib/OpenBLAS.git /private/tmp/OpenBLAS
cd /private/tmp/OpenBLAS

make -j10 \
  TARGET=ARMV8 \
  OSNAME=Android \
  BINARY=64 \
  CC=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android35-clang \
  HOSTCC=clang \
  AR=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ar \
  RANLIB=/Users/tony/Library/Android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-ranlib \
  NOFORTRAN=1 \
  ONLY_CBLAS=1 \
  USE_THREAD=0 \
  NUM_THREADS=1
```

Rejected configs that must not be defaults:
- Engine thread counts above `1`.
- OpenBLAS `DYNAMIC_ARCH=1`, `USE_THREAD=1`, `NUM_THREADS=8` dynamic-arch
  threaded build.
- NDK-version-only OpenBLAS rebuilds.

## Thermal Control

`bench/android_run.py` waits for `Thermal Status: 0` automatically before
running. It prints the current status every 15 seconds and proceeds once the
device is cool. Set `THERMAL_TIMEOUT` (default 600 s) to control the maximum
wait.

Physical discipline (cannot be scripted):
- Run plugged in with the same charger/cable each time.
- Do not run unrelated apps or background workloads.
- Compare arms in alternating order when possible (baseline, candidate, baseline,
  candidate) rather than running all of one arm first.

Snapshot before and after each arm:

```bash
LABEL=android-exp-name
bench/android_snapshot.sh --label "$LABEL" --phase before
bench/android_run.py --label "$LABEL" --threads "1" --runs 3
bench/android_snapshot.sh --label "$LABEL" --phase after
```

## Acceptance Rules

Stage 1 and Stage 2 can reject or promote an idea but do not define final
acceptance thresholds.

Stage 3 accepts a change only if at least one of:
- Full Common Voice median `inference_ms` improves by at least `5%`.
- `model_load_ms` improves by at least `5%`.
- `memory_after_model_load_kb` or `peak_inference_rss_kb` improves by at least
  `10%`.

Startup or memory wins must not regress full Common Voice median inference.

All accepted changes must satisfy every guardrail listed in the Goal section.

## Experiment Loop

Use staged gates so iteration stays fast. Do not run all five clips or a full
thread sweep for every idea.

### Stage 0: Host and Android Build Gate

Run after every code change:

```bash
PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
cargo test --release --test kernels --test audio

PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
bench/android_run.py \
  --label android-build-smoke \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

Purpose:
- Catch compile/runtime failures.
- Use only `cv_yue_003.wav` plus `ptt_guardrail.wav`.
- Do not accept a performance change from this stage alone.

### Stage 1: Fast Performance Gate

Run for every experiment idea that passes Stage 0:

```bash
LABEL=android-fast-<idea>-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

Then check the result (replace `BASELINE_LABEL` with the current best fast-gate label):

```bash
bench/android_check_gate.py \
  --result "$LABEL" \
  --baseline BASELINE_LABEL \
  --stage 1
```

Exit 0 = promote to Stage 2. Exit 1 = reject; read the output for the reason.

### Stage 2: Candidate Gate

Run only for ideas that pass Stage 1:

```bash
LABEL=android-candidate-<idea>-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1" \
  --runs 1
```

```bash
bench/android_check_gate.py \
  --result "$LABEL" \
  --baseline BASELINE_LABEL \
  --stage 2
```

Exit 0 = promote to Stage 3. Exit 1 = reject.

### Stage 3: Acceptance Gate

Run only for candidates that pass Stage 2:

```bash
LABEL=android-accept-<idea>-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir bench/android-assets \
  --threads "1" \
  --runs 3
```

```bash
bench/android_check_gate.py \
  --result "$LABEL" \
  --baseline BASELINE_LABEL \
  --stage 3
```

Exit 0 = accepted; update `android-best-baseline.md` and the Completed Result
Index. Exit 1 = rejected; the output explains which threshold was missed.

### Stage 4: Full Thread Sweep

Run only when the experiment is explicitly about threading/core scheduling, or
after an accepted build-feature/kernel change needs a new default thread count:

```bash
LABEL=android-thread-sweep-<idea>-$(date -u +%Y%m%d-%H%M%S)
bench/android_run.py \
  --label "$LABEL" \
  --assets-dir bench/android-assets \
  --threads "1 2 4 6 8" \
  --runs 3
```

Do not use Stage 4 as the default loop. It is too slow and too thermally
sensitive for every experiment.

### Recording

For every stage that runs:
- Append/keep the generated row in `results-android.tsv`.
- Add a short note to `android-speed-improvement-experiments.md`.
- Link the JSON directory and record whether thermal snapshots were clean.

Resume source of truth:
- This file defines the stable protocol and acceptance rules.
- `android-speed-improvement-experiments.md` contains current state, completed
  attempts, and the ordered next-experiment backlog with concrete commands.
- A future session should read both files before running anything, then continue
  from the first unresolved item in that backlog.

## Rules

- One idea per experiment.
- Keep benchmark harness changes separate from JNI/Flutter production paths.
- Do not mix Android results into macOS `results.tsv`.
- Do not rely on single-run comparisons for keep/reject decisions.
- Use exact labels that encode the experiment arm and date.
- Preserve user files and avoid destructive cleanup unless explicitly approved.
