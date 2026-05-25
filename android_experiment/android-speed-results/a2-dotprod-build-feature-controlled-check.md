# A2: dotprod build-feature controlled check

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

Follow-up:
- `RUSTFLAGS_EXTRA="-C target-feature=+dotprod"` became the chosen Android
  build flag for subsequent experiments.
- Later OpenBLAS results superseded no-BLAS build-flag tuning as the main
  Android performance path.
