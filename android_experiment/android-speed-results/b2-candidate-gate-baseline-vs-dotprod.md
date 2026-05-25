# B2: candidate-gate baseline vs dotprod

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
