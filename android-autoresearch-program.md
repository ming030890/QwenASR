# QwenASR Android Autoresearch Program

> Android-on-device optimization loop for the QwenASR Rust inference engine.
> Keep this separate from the macOS autoresearch notes because the build,
> benchmark, and performance bottlenecks are different.

## Goal

Improve Android arm64 inference speed on the connected Pixel 9 while preserving
transcript quality.

Primary metric:
- Median `inference_ms` over the five fixed Common Voice Yue clips.

Guardrails:
- `ptt_guardrail.wav` transcript must remain exactly:
  `Hello,你好吗？已经坐低咗啦。`
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
bench/android_run.sh --label android-baseline --runs 3 --threads "1 2 4 6 8"
```

Standard environment for every run:

```bash
export ADB=/Users/tony/Library/Android/sdk/platform-tools/adb
export MODEL_DIR=/Volumes/SSD/qwen3-asr-0.6b
export PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$PWD/android-aarch64-clang"
```

The harness is separate from production JNI code:
- Source: `crates/qwen-asr-cli/src/bin/bench_android.rs`
- Target: `aarch64-linux-android`
- Output JSON fields include setup/model/audio timing, encode/decode timing,
  median inference timing, RTF, transcript, optimization flags, target arch,
  per-run details, and optional profiling counters.

Canonical results files:
- Summary TSV: `results-android.tsv`
- Full JSON: `bench/android-results/<label>/*.json`

## Thermal Control

Android phone benchmarks are highly sensitive to temperature, battery state, and
CPU governor decisions. Treat thermal control as part of the protocol, not as
post-hoc explanation.

Default practice:
- Run plugged in with the same charger/cable each time.
- Keep the screen state consistent.
- Do not run unrelated apps or background workloads.
- Run a short warmup only if every arm gets the same warmup.
- Do not block routine Stage 1/Stage 2 iteration on tightly matched
  temperatures. `Thermal Status: 0` is still required for normal comparable
  runs; relax only the skin/soc matching and cooldown strictness.
- Cool down before long acceptance/thread-sweep arms, or when `Thermal Status`
  is nonzero.
- Record device temperature and battery/thermal state before each arm.
- Compare arms in alternating order when possible, for example baseline,
  dotprod, baseline, dotprod, instead of running all of one arm first.

Use the same labels and snapshots for every arm. Prefer the helper script so
snapshot capture is one command:

```bash
LABEL=android-exp-name

bench/android_snapshot.sh --label "$LABEL" --phase before

PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
RUSTFLAGS_EXTRA="" \
bench/android_run.sh --label "$LABEL" --threads "1" --runs 3

bench/android_snapshot.sh --label "$LABEL" --phase after
```

Fast readiness check before starting a routine Stage 1/Stage 2 arm:

```bash
adb shell dumpsys thermalservice | grep 'Thermal Status'
```

For Stage 1/Stage 2:
- Start only when `Thermal Status: 0`.
- If status is nonzero, wait or run a diagnostic only and mark it as thermally
  contaminated.
- Do not wait for exact skin/soc matching. Record the snapshot and interpret
  close results conservatively.

Stricter cooldown check before acceptance runs, thread sweeps, or when a result
will decide the default build/thread policy:

```bash
while true; do
  adb shell dumpsys thermalservice | grep 'Thermal Status'
  adb shell dumpsys thermalservice | grep -E 'VIRTUAL-SKIN|soc_therm|BIG|MID|LITTLE|battery' | head -20
  adb shell 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do echo "$f=$(cat "$f" 2>/dev/null)"; done'
  sleep 60
done
```

Start long decision-making arms when:
- `Thermal Status` is `0`.
- Skin/soc readings are not obviously hotter than the previous arm. They do not
  need to be exact.
- CPU frequency readings are not obviously clamped lower than the previous arm.

Controlled build-feature comparison:

```bash
BASE_LABEL=android-control-baseline-t1-r3
DOT_LABEL=android-control-dotprod-t1-r3

# Arm 1: baseline
RUSTFLAGS_EXTRA="" \
bench/android_run.sh --label "$BASE_LABEL" --threads "1" --runs 3

# Cool down and snapshot before continuing.

# Arm 2: dotprod
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
bench/android_run.sh --label "$DOT_LABEL" --threads "1" --runs 3
```

Frequency locking:
- Only lock CPU frequency/governor if the device exposes the controls and the
  benchmark user has permission, usually root.
- Prefer a documented helper script over one-off shell commands.
- Always save the original governor/frequency settings before changing them.
- Always restore the original settings after the benchmark, even if a run fails.
- Record whether frequency locking was used in the result label and notes.

Suggested locked-frequency flow:

```bash
# Inspect first. This usually requires root for writes.
adb shell 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$f=$(cat "$f" 2>/dev/null)"; done'
adb shell 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_available_frequencies; do echo "$f=$(cat "$f" 2>/dev/null)"; done'

# Save originals on the device before changing anything.
adb shell su -c 'for d in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do cpu=${d%/cpufreq}; cpu=${cpu##*/cpu}; printf "%s\t%s\t%s\t%s\n" "$cpu" "$(cat "$d/scaling_governor")" "$(cat "$d/scaling_min_freq")" "$(cat "$d/scaling_max_freq")"; done > /data/local/tmp/qwen-asr-cpufreq-original.tsv'
adb pull /data/local/tmp/qwen-asr-cpufreq-original.tsv bench/android-cpufreq-original.tsv

# Root-only example: pin every CPU to its current max frequency.
# Do not run this unless the device allows adb root or su.
adb shell su -c 'for d in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do max=$(cat "$d/cpuinfo_max_freq"); echo performance > "$d/scaling_governor"; echo "$max" > "$d/scaling_min_freq"; echo "$max" > "$d/scaling_max_freq"; done'

# Restore immediately after the benchmark from the saved values.
adb shell su -c 'while IFS="$(printf "\t")" read -r cpu gov min max; do d="/sys/devices/system/cpu/cpu${cpu}/cpufreq"; [ -d "$d" ] || continue; echo "$max" > "$d/scaling_max_freq"; echo "$min" > "$d/scaling_min_freq"; echo "$gov" > "$d/scaling_governor"; done < /data/local/tmp/qwen-asr-cpufreq-original.tsv'

# Verify restore.
adb shell 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do echo "$f=$(cat "$f" 2>/dev/null)"; done'
```

If frequency locking is not available, use cooldown plus temperature gates:
- Start each arm only when battery/skin/CPU thermal readings are within a narrow
  range of the previous arm.
- Reject or rerun comparisons where one arm starts materially hotter.
- Avoid drawing conclusions from a single long run after sustained throttling.

## Experiment Loop

Use staged gates so iteration stays fast. Do not run all five clips or a full
thread sweep for every idea.

### Stage 0: Host and Android Build Gate

Run after every code change:

```bash
PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
cargo test --release --test kernels --test audio

PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
RUSTFLAGS_EXTRA="${RUSTFLAGS_EXTRA:-}" \
bench/android_run.sh \
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
RUSTFLAGS_EXTRA="${RUSTFLAGS_EXTRA:-}" \
bench/android_run.sh \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

Inputs:
- `cv_yue_003.wav`, the shortest fixed Common Voice clip.
- `ptt_guardrail.wav`, because it catches the most important transcript drift.

Decision:
- Reject immediately if the guardrail transcript changes.
- Reject immediately if `cv_yue_003.wav` inference regresses by more than 10%
  versus the current best fast-gate result under comparable thermal conditions.
- Promote only if speed is neutral or better, or if the experiment is diagnostic.

### Stage 2: Candidate Gate

Run only for ideas that pass Stage 1:

```bash
LABEL=android-candidate-<idea>-$(date -u +%Y%m%d-%H%M%S)
RUSTFLAGS_EXTRA="${RUSTFLAGS_EXTRA:-}" \
bench/android_run.sh \
  --label "$LABEL" \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1" \
  --runs 1
```

Inputs:
- `cv_yue_003.wav`: short clip.
- `cv_yue_001.wav`: longer clip with more generated tokens.
- `ptt_guardrail.wav`: exact transcript guardrail.

Decision:
- Reject if guardrail changes.
- Reject if either Common Voice clip regresses materially.
- Promote to acceptance only if median inference is at least 3% faster, or if
  the change is required infrastructure.

### Stage 3: Acceptance Gate

Run only for candidates worth keeping:

```bash
LABEL=android-accept-<idea>-$(date -u +%Y%m%d-%H%M%S)
RUSTFLAGS_EXTRA="${RUSTFLAGS_EXTRA:-}" \
bench/android_run.sh \
  --label "$LABEL" \
  --assets-dir bench/android-assets \
  --threads "<current-best-thread-count>" \
  --runs 3
```

Inputs:
- All five fixed Common Voice clips.
- `ptt_guardrail.wav`.

Decision:
- Keep only if Common Voice median inference improves by at least 5%, or
  startup/model load improves by at least 5% without inference regression.
- Reject if build/tests fail, guardrail changes, Common Voice CER worsens, or
  median inference regresses.

### Stage 4: Full Thread Sweep

Run only when the experiment is explicitly about threading/core scheduling, or
after an accepted build-feature/kernel change needs a new default thread count:

```bash
LABEL=android-thread-sweep-<idea>-$(date -u +%Y%m%d-%H%M%S)
RUSTFLAGS_EXTRA="${RUSTFLAGS_EXTRA:-}" \
bench/android_run.sh \
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
- `android-speed-improvement-experiments.md` contains the current state,
  completed attempts, and the ordered next-experiment backlog with concrete
  commands.
- A future session should read both files before running anything, then continue
  from the first unresolved item in that backlog.

## Initial Priorities

### A1: Android Build-Feature Tuning

Compare:
- baseline Android arm64 build
- `-C target-cpu=cortex-a76`
- `-C target-feature=+dotprod`
- combined CPU/feature variants if the individual arms justify it

Reason:
- Pixel 9 supports dot-product instructions, but enabling a feature globally
  can change LLVM code generation beyond the intended hot path. Treat this as
  data-driven, not automatically beneficial.

### A2: Android Thread/Core Scheduling

Sweep:
- `--threads "1 2 4 6 8"`
- optional `taskset` masks for big/mid/all cores after the plain thread sweep

Reason:
- Parallel phases may wait on slower cores. The best Android policy may differ
  from macOS and from host defaults.

### A3: Android No-BLAS Fallback Kernels

Target current fallback hot paths one at a time:
- `linear`
- `linear_accumulate`
- `matmul_nn`
- `matmul_t`
- `conv2d`

Reason:
- Android currently cannot rely on Accelerate and may not link BLAS. Large
  dense operations falling back to scalar Rust loops are the likely biggest
  code-level opportunity.

### A4: Profiling Counters

Run the benchmark with `--profile` only for diagnosis, not final timing.
Use per-file counters to choose the next kernel target.

### Deferred: Android OpenBLAS

Try only if focused NEON fallback work stalls. Treat as a separate experiment
because it changes runtime dependencies and build complexity.

## Rules

- One idea per experiment.
- Keep benchmark harness changes separate from JNI/Flutter production paths.
- Do not mix Android results into macOS `results.tsv`.
- Do not rely on single-run comparisons for keep/reject decisions.
- Use exact labels that encode the experiment arm and date.
- Preserve user files and avoid destructive cleanup unless explicitly approved.
