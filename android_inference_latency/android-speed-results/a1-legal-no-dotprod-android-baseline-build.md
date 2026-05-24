# A1: legal no-dotprod Android baseline build

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
