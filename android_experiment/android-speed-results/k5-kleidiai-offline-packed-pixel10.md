# K5: offline-packed KleidiAI decoder on Pixel 10

Purpose:

- Test the model-file packed KleidiAI DotProd path on a connected Android
  device after the Apple M4 result failed the first-token gate.
- Eliminate runtime RHS packing by mmap-loading a dedicated packed safetensors
  shard generated on the host.

## Fixed setup

- Device: Pixel 10 (`frankel`), Tensor G5, Android 17.
- CPU features: DotProd, I8MM, SVE, SVE2, and SVE-I8MM.
- Thermal status: 0 before both control and candidate.
- Fixed-performance mode: enabled during each run and restored afterwards.
- Threads: 1.
- Repetitions: 3 per input.
- Inputs: five Common Voice clips plus `ptt_guardrail.wav`.
- Final model: a 675.87 MiB shared/base shard plus a 1.35 GiB packed decoder
  shard. All 168 row-major decoder projections and scales are omitted, reducing
  logical deployed size from 3.3 GiB to 2.0 GiB.
- Binary: same Android arm64 binary for control and candidate, linked with
  static OpenBLAS and KleidiAI revision
  `6787251d9cc2f38a3a6024b11fd7ace10cde4cd9`.
- No backend environment flags. The Android KleidiAI build automatically uses
  the packed backend only when all 168 projections are present.

Result directories:

- `bench/android-results/android-pixel10-kleidiai-control-20260815`
- `bench/android-results/android-pixel10-kleidiai-offline-packed-20260815`
- `bench/android-results/android-pixel10-kleidiai-packed-all-20260815`
- `bench/android-results/android-pixel10-kleidiai-packed-only-20260815`

## Results

| Metric | Control | Packed-only backend | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | 4766.391 ms | 2849.027 ms | **-40.2%** |
| Common Voice median first-visible token | 3803.202 ms | 2115.961 ms | **-44.4%** |
| Common Voice median prefill | 2324.371 ms | 708.485 ms | **-69.5%** |
| Common Voice median decode loop | 1784.403 ms | 1471.657 ms | **-17.5%** |
| Guardrail inference | 7615.522 ms | 4425.004 ms | **-41.9%** |
| Guardrail first-visible token | 5867.167 ms | 3037.871 ms | **-48.2%** |
| Guardrail prefill | 3854.890 ms | 1152.733 ms | **-70.1%** |
| Guardrail decode loop | 2565.011 ms | 2090.337 ms | **-18.5%** |
| Peak RSS, Common Voice | 2268456 KiB | 2253084 KiB | **-0.7%** |
| Peak RSS, guardrail | 2282952 KiB | 2278588 KiB | **-0.2%** |
| Load-time RSS, Common Voice | 59924 KiB | 43808 KiB | **-26.9%** |

The first prefill-only candidate reduced TTFT but raised peak RSS by 60.7%:
prefill faulted in the packed model-file pages while single-token decode still
faulted in the original row-major INT8 pages. Routing decode through the same
offline-packed weights prevents the duplicate projection layouts from becoming
resident. Routing both phases through one backend removed that regression. The
final packed-only artifact also removes the unused row-major tensors from disk
and eliminates their loader representation entirely.

## Transcript gate

Five of six clips match exactly. One fails:

| Input | Control | Candidate |
| --- | --- | --- |
| `cv_yue_002.wav` | `證據嘅話，真係冇啊！` | `证据嘅话，真系冇啊！` |

The final packed-only model matches all six transcripts from the prior
dual-layout packed candidate. The guardrail transcript is restored exactly.
All three repetitions are stable, so the remaining traditional/simplified
character change versus row-major control is deterministic numerical behavior.

## Decision

**Accept the packed-only model as the Android packed candidate.** Prefill and
decode are now one atomic backend; partial packed shards fail at model load.
Independent runtime switches, transient packing, the packed-weight cache, the
rejected INT8 prefill kernel, and unused F32 projection scratch were removed.
The remaining caveat is one exact transcript mismatch out of six versus the
row-major quality reference, so the backend remains Android-specific rather
than becoming the cross-platform default.

No F32 last-layer fallback is used. The packed-only path gives the lowest
storage footprint while keeping latency and output equivalent to the prior
dual-layout packed candidate.
