# A3: baseline controlled rerun, thermal contaminated

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

Follow-up:
- Later dotprod acceptance used cooler/comparable runs.
- Keep using shorter staged checks or alternating order for future
  build-feature comparisons.
