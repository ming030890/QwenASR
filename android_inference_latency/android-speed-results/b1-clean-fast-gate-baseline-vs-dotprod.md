# B1: clean fast-gate baseline vs dotprod

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
