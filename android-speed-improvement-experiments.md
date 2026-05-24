# Android Speed Improvement Experiments

Goal: improve Android arm64 inference speed on the connected Pixel 9 while
keeping transcript quality stable.

Current benchmark protocol:
- Model: `/Volumes/SSD/qwen3-asr-0.6b`
- Device workspace: `/data/local/tmp/qwen-asr-bench`
- Assets: `bench/android-assets`
- Fast-gate asset subset: `/Volumes/SSD/qwen-asr-android-fast-assets`
  contains `cv_yue_003.wav`, `cv_yue_003.txt`, `ptt_guardrail.wav`, and
  `ptt_guardrail.txt`.
- Candidate-gate asset subset:
  `/Volumes/SSD/qwen-asr-android-candidate-assets` contains
  `cv_yue_003.wav`, `cv_yue_003.txt`, `cv_yue_001.wav`, `cv_yue_001.txt`,
  `ptt_guardrail.wav`, and `ptt_guardrail.txt`.
- Common Voice metric: median `inference_ms` across five fixed Yue clips.
- Guardrail: `ptt_guardrail.wav` must transcribe exactly
  `Hello,你好吗？已经坐低咗啦。`
- Standard run: `bench/android_run.sh --runs 3 --threads "1 2 4 6 8"`
- Controlled build-feature check: compare arms with identical `--runs` and
  `--threads`, preferably after cooldown.

Resume checklist:
- Read `android-autoresearch-program.md` first; it defines fixed inputs,
  thermal control, staged gates, and acceptance rules.
- Make sure the phone is visible:
  `/Users/tony/Library/Android/sdk/platform-tools/adb devices`
- Treat this file and `android-autoresearch-program.md` as the source of truth;
  do not rely on chat history for the plan.
- Keep Android work separate from macOS autoresearch docs and results.
- Do not resume with a full five-clip thread sweep. Use Stage 1 or Stage 2
  unless the experiment is specifically about thread/core policy.
- Recreate fast/candidate asset subsets if missing:
  `/Volumes/SSD/qwen-asr-android-fast-assets` and
  `/Volumes/SSD/qwen-asr-android-candidate-assets`.
- Start only when `dumpsys thermalservice` reports `Thermal Status: 0` or a
  comparable clean starting state.
- Record every run here, even if it is rejected or thermally contaminated.
- Only promote an experiment to the full acceptance gate if the staged gate
  justifies it.

Current state summary:
- Android benchmark infrastructure exists.
- Default Android no-dotprod build now compiles because `sdot` asm is gated
  behind `target_feature = "dotprod"` and a fallback exists.
- Existing baseline/dotprod timing rows are not enough to decide whether
  dotprod helps; prior runs were not clean/comparable.
- Next work should use the staged protocol from `android-autoresearch-program.md`
  and avoid long one-thread/full-sweep runs unless a fast gate justifies them.

## A0: Android benchmark harness and assets

Change:
- Added/reused Android benchmark binary `qwen-asr-bench-android`.
- Added deterministic Android assets under `bench/android-assets`.
- Added `bench/android_run.sh` to build, push, run thread sweeps, and append
  rows to `results-android.tsv`.

Status:
- Kept as infrastructure.

## A1: legal no-dotprod Android baseline build

Change:
- Gated the inline `sdot` asm behind `target_feature = "dotprod"`.
- Added a portable aarch64 fallback for default Android builds without
  `+dotprod`.

Reason:
- The Android default build previously failed because the compiler rejected
  `sdot` instructions unless dot-product support was explicitly enabled.

Smoke result:
- Label: `android-baseline-buildcheck`
- Threads/runs: `1` thread, `1` run.
- Common Voice median: `25937.755 ms`
- Guardrail: `46855.435 ms`
- Guardrail exact: yes.
- Flags: `["NEON"]`

Decision:
- Kept as a build correctness fix. This result is only a smoke baseline and is
  not a valid speed comparison against multi-run experiment arms.

## A2: dotprod build-feature controlled check

Hypothesis:
- `RUSTFLAGS_EXTRA="-C target-feature=+dotprod"` may improve Android decoder
  INT8 matvec and argmax paths on Pixel 9.

Final controlled result:
- Accepted for now as the best Android build-feature arm.
- Matching full acceptance pair, both `--threads "1" --runs 3`, both thermal
  status `0` before/after:
  - Baseline: `android-accept-baseline-t1-r3-20260524-1406`, Common Voice
    median `29299.505 ms`, guardrail `56240.017 ms`, exact transcript.
  - Dotprod: `android-accept-dotprod-t1-r3-20260524-1348`, Common Voice median
    `26761.235 ms`, guardrail `50890.267 ms`, exact transcript.
- Acceptance speedup: `8.66%` on five-clip Common Voice median, exceeding the
  `5%` acceptance threshold.
- Per-clip inference speedups: `cv_yue_000.wav` `5.55%`, `cv_yue_001.wav`
  `5.61%`, `cv_yue_002.wav` `8.66%`, `cv_yue_003.wav` `7.94%`,
  `cv_yue_004.wav` `11.93%`; guardrail `9.51%`.
- Thermal caveat: the matching baseline started after a long cooldown with
  overall thermal status `0` and similar soc temperature to the dotprod
  acceptance start, but one charge-wired skin sensor still reported warning
  status. The direction of the result agrees with B2, so keep dotprod while
  avoiding over-interpretation of sub-percent differences.

Partial prior result:
- Label: `android-exp1-dotprod-20260524-1130`
- Threads/runs: `1` thread, `3` runs.
- Common Voice median: `29628.330 ms`
- Guardrail: `53402.477 ms`
- Guardrail exact: yes.
- Flags: `["NEON", "DotProd"]`

Interpretation:
- This is not yet a valid keep/reject comparison because the available baseline
  was only `1` run while this arm used `3` runs and ran later in the session.
- Possible confounders: thermal throttling, CPU governor state, and LLVM codegen
  shifts from global `+dotprod`.

Next action:
- Use `RUSTFLAGS_EXTRA="-C target-feature=+dotprod"` as the chosen build flag
  for the next Android experiments until better evidence replaces it.
- Continue with B3 profiling under dotprod or B5 target-cpu checks if the goal
  is to test build policy before kernel work.

## A3: baseline controlled rerun, thermal contaminated

Hypothesis:
- Rerun the no-dotprod baseline with the same settings planned for the dotprod
  arm so the comparison is valid.

Result:
- Label: `android-control-baseline-t1-r3-20260524-1205`
- Threads/runs: `1` thread, `3` runs.
- Common Voice median: `31510.236 ms`
- Guardrail: `59343.367 ms`
- Guardrail exact: yes.
- Flags: `["NEON"]`

Thermal notes:
- During the run, `dumpsys thermalservice` reported `Thermal Status: 1`.
- CPU cooling devices were active.
- CPU frequencies included LITTLE cores at `820000`, mid cores at `1418000`,
  and CPU7 at `700000`.
- Several skin/soc sensors were in warning status.

Decision:
- Do not use this as a clean baseline for dotprod acceptance. Keep it in the
  log as evidence that long one-thread Android runs can become thermally
  dominated.

Next action:
- Cool down until thermal status and skin/soc readings are comparable before
  rerunning build-feature arms.
- Prefer shorter staged checks or alternating order for build-feature
  comparisons.

## Next Experiment Backlog

Run these in order unless new profile data says otherwise.

The backlog is intentionally front-loaded with measurement and build-policy
experiments. Do not start kernel work until B1/B2/B3 identify whether dotprod
or no-BLAS fallbacks are the better target.

### B1: clean fast-gate baseline vs dotprod

Purpose:
- Resolve whether `+dotprod` is promising without paying for all five clips.

Run notes:
- Recreated the fixed asset subsets on 2026-05-24:
  `/Volumes/SSD/qwen-asr-android-fast-assets` and
  `/Volumes/SSD/qwen-asr-android-candidate-assets`.
- First attempted baseline label
  `android-fast-baseline-t1-r1-20260524-1315` was invalid for B1 because
  `bench/android_run.sh` did not clear stale device-side assets before pushing
  the fast subset; the JSON included all five `cv_yue_*.wav` files.
- Fixed `bench/android_run.sh` to clear remote `assets` and `common_voice`
  before pushing each asset set.
- Corrected baseline label:
  `android-fast-baseline-t1-r1-20260524-1322`
  (`bench/android-results/android-fast-baseline-t1-r1-20260524-1322`).
  Thermal status before/after: `0` / `0`. Common Voice input contained only
  `cv_yue_003.wav`. Result: `20242.988 ms` inference, `7570.442 ms` encode,
  `12672.336 ms` decode. Guardrail: `44908.348 ms`, exact transcript.
- Dotprod label:
  `android-fast-dotprod-t1-r1-20260524-1325`
  (`bench/android-results/android-fast-dotprod-t1-r1-20260524-1325`).
  Thermal status before/after: `0` / `0`; dotprod started slightly warmer than
  the corrected baseline by skin/soc readings. Result: `20627.996 ms`
  inference, `9127.805 ms` encode, `11499.999 ms` decode. Guardrail:
  `42478.744 ms`, exact transcript.
- Interpretation: dotprod was `1.9%` slower on `cv_yue_003.wav`, but not over
  the `10%` fast-gate rejection threshold, and the guardrail was faster with
  exact transcript.

Commands:

```bash
export ADB=/Users/tony/Library/Android/sdk/platform-tools/adb
export PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"

# Wait until this is clean before each arm.
$ADB shell dumpsys thermalservice | grep 'Thermal Status'

RUSTFLAGS_EXTRA="" \
bench/android_run.sh \
  --label android-fast-baseline-t1-r1-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1

# Cool down to comparable state.

RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
bench/android_run.sh \
  --label android-fast-dotprod-t1-r1-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-fast-assets \
  --threads "1" \
  --runs 1
```

Decision:
- If dotprod is slower by more than 10% on `cv_yue_003.wav`, reject dotprod for
  now and move to profiling.
- If dotprod is neutral or faster, promote to B2.

### B2: candidate-gate baseline vs dotprod

Purpose:
- Check that any B1 dotprod win holds on both a short and longer clip.

Run notes:
- Baseline label:
  `android-candidate-baseline-t1-r1-20260524-1331`
  (`bench/android-results/android-candidate-baseline-t1-r1-20260524-1331`).
  Thermal status before/after: `0` / `0`. Common Voice inputs were exactly
  `cv_yue_001.wav` and `cv_yue_003.wav`. Median Common Voice inference:
  `25623.693 ms`. Per-clip inference: `cv_yue_001.wav` `30464.421 ms`,
  `cv_yue_003.wav` `20782.965 ms`. Guardrail: `45329.255 ms`, exact
  transcript.
- Dotprod label:
  `android-candidate-dotprod-t1-r1-20260524-1335`
  (`bench/android-results/android-candidate-dotprod-t1-r1-20260524-1335`).
  Thermal status before/after: `0` / `0`; dotprod started warmer than the
  baseline candidate arm by current skin/soc readings. Median Common Voice
  inference: `23635.609 ms`. Per-clip inference: `cv_yue_001.wav`
  `27857.561 ms`, `cv_yue_003.wav` `19413.656 ms`. Guardrail:
  `42438.685 ms`, exact transcript.
- Interpretation: dotprod improved candidate median by `7.8%`; per-clip
  improvements were `8.6%` for `cv_yue_001.wav` and `6.6%` for
  `cv_yue_003.wav`. Because it passes the `3%` candidate threshold and the
  guardrail transcript is exact, promote dotprod to an acceptance run after
  cooldown.
- Acceptance follow-up:
  - Dotprod: `android-accept-dotprod-t1-r3-20260524-1348`, Common Voice median
    `26761.235 ms`, guardrail exact.
  - Matching baseline: `android-accept-baseline-t1-r3-20260524-1406`, Common
    Voice median `29299.505 ms`, guardrail exact.
  - Decision: dotprod passes acceptance with `8.66%` median inference speedup.

Commands:

```bash
RUSTFLAGS_EXTRA="" \
bench/android_run.sh \
  --label android-candidate-baseline-t1-r1-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1" \
  --runs 1

# Cool down to comparable state.

RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
bench/android_run.sh \
  --label android-candidate-dotprod-t1-r1-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1" \
  --runs 1
```

Decision:
- Promote dotprod to a full acceptance run only if candidate median is at least
  3% faster and the guardrail transcript is exact.
- Otherwise reject for now and record the result.

### B3: profile the candidate baseline

Purpose:
- Identify the next kernel target instead of guessing.

Command:

```bash
RUSTFLAGS_EXTRA="" \
bench/android_run.sh \
  --label android-profile-baseline-candidate-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1" \
  --runs 1

# Then run the binary directly with --profile if android_run.sh does not expose
# profiling yet:
# adb shell '/data/local/tmp/qwen-asr-bench/qwen-asr-bench-android --model-dir /data/local/tmp/qwen-asr-bench/model --input /data/local/tmp/qwen-asr-bench/common_voice --runs 1 --threads 1 --profile'
```

Decision:
- Use the largest profile counters to choose B4/B5/B6.
- Do not compare final timing with profiling enabled.

### B4: no-BLAS scalar `linear`/`linear_accumulate` investigation

Purpose:
- Android has no Accelerate and may use scalar fallback loops for large dense
  encoder operations. These are likely bigger than dotprod if profile confirms
  `linear`-family time dominates.

Approach:
- Read `crates/qwen-asr/src/kernels/mod.rs` fallback blocks for `linear` and
  `linear_accumulate`.
- Add only one narrow Android/aarch64 NEON or blocked-loop change.
- Run Stage 0, then Stage 1, then Stage 2 if promising.

Decision:
- Keep only if candidate gate improves or acceptance gate improves by the
  thresholds in `android-autoresearch-program.md`.

### B5: Android target-cpu build-feature check

Purpose:
- Test whether a CPU tuning flag helps independently from dotprod.

Arms:
- Baseline: `RUSTFLAGS_EXTRA=""`
- Cortex-A76: `RUSTFLAGS_EXTRA="-C target-cpu=cortex-a76"`
- Dotprod only: already covered by B1/B2.
- Combined only if both individual arms look promising:
  `RUSTFLAGS_EXTRA="-C target-cpu=cortex-a76 -C target-feature=+dotprod"`

Gate:
- Use Stage 1 first.
- Promote to Stage 2 only if Stage 1 is neutral or faster.
- Do not run a full thread sweep for build flags until one arm passes Stage 2.

Decision:
- Keep only if the acceptance gate improves by the normal threshold.
- If target-cpu and dotprod disagree, prefer the arm with better Stage 2 median
  and exact guardrail.

### B6: no-BLAS `conv2d` fallback investigation

Purpose:
- If profiling shows encoder convolution/im2col time is large on Android,
  target `conv2d` before decoder micro-optimizations.

Approach:
- Keep the experiment focused: one packing or inner-loop improvement only.
- Do not change audio preprocessing behavior.
- Run Stage 1 before broader clips.

### B7: scalar `matmul_nn` / `matmul_t` fallback investigation

Purpose:
- If profiling points at generic matrix multiply rather than `linear` or
  `conv2d`, add one Android/aarch64-specific blocked implementation.

Approach:
- Inspect call shapes before coding. Record the dominant shapes in this file.
- Target one function and one shape class first.
- Keep the scalar fallback for unsupported dimensions.
- Run Stage 0, Stage 1, then Stage 2 if promising.

Decision:
- Keep only after acceptance-gate improvement, because these kernels can affect
  many paths and need broader coverage.

### B8: decoder INT8 dotprod microkernel shape check

Purpose:
- If dotprod is neutral or slower, determine whether the existing INT8 matvec
  layout is leaving performance on the table.

Approach:
- Do not immediately rewrite kernels.
- First add profiling or debug counters for INT8 matvec shapes/call counts if
  current profile output is insufficient.
- Compare time spent in `argmax_int8_range`, attention projections, and MLP
  projections.
- Only then try one narrow kernel layout change.

Decision:
- Keep only if Stage 2 improves and guardrail remains exact.

### B9: Android thread/core policy

Purpose:
- Find whether Pixel 9 should use fewer threads or avoid slower cores.

When to run:
- Only after one build/kernel configuration is chosen.
- Do not run this for every experiment.

Commands:

```bash
RUSTFLAGS_EXTRA="<chosen build flags>" \
bench/android_run.sh \
  --label android-thread-sweep-current-$(date -u +%Y%m%d-%H%M%S) \
  --assets-dir bench/android-assets \
  --threads "1 2 4 6 8" \
  --runs 3
```

Optional follow-up:
- Try `taskset` masks for big/mid/all cores only if the plain thread sweep shows
  thread count matters and thermal state is controlled.

### B10: benchmark harness improvements

Purpose:
- Reduce manual work and make comparisons less error-prone.

Completed:
- `bench/android_run.sh` now clears the device-side `assets` and
  `common_voice` directories before pushing an asset subset. This prevents
  Stage 1/Stage 2 runs from accidentally including stale clips from a previous
  full-assets run.

Potential focused changes:
- Add `--profile` passthrough to `bench/android_run.sh`.
- Add a `--thermal-snapshot` option or wrapper script.
- Add a `--no-push-model` option so repeated Stage 1 runs do not push the 1.8GB
  model every time.
- Add a `--common-voice-only` or `--guardrail-only` mode if needed.

Decision:
- Keep harness improvements if they do not alter benchmark semantics and make
  the staged loop faster or more reproducible.

### Deferred: Android OpenBLAS

Try only if focused no-BLAS fallback work does not produce wins. Treat it as
separate because it changes runtime dependencies and package complexity.
