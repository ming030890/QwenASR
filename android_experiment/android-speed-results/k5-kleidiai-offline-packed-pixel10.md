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
- Model: Qwen3-ASR 1.7B INT8 encoder/decoder candidate plus a 1.35 GiB
  offline-packed KleidiAI shard.
- Binary: same Android arm64 binary for control and candidate, linked with
  static OpenBLAS and KleidiAI revision
  `6787251d9cc2f38a3a6024b11fd7ace10cde4cd9`.
- Control: `QWEN_KLEIDIAI_PREFILL=0`.
- Prefill-only candidate: `QWEN_KLEIDIAI_PREFILL=1
  QWEN_KLEIDIAI_PACK_CACHE=0`.
- Fully packed candidate: `QWEN_KLEIDIAI_PREFILL=1
  QWEN_KLEIDIAI_DECODE=1 QWEN_KLEIDIAI_PACK_CACHE=0`.

Result directories:

- `bench/android-results/android-pixel10-kleidiai-control-20260815`
- `bench/android-results/android-pixel10-kleidiai-offline-packed-20260815`
- `bench/android-results/android-pixel10-kleidiai-packed-all-20260815`

## Results

| Metric | Control | Packed prefill + decode | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | 4766.391 ms | 2839.390 ms | **-40.4%** |
| Common Voice median first-visible token | 3803.202 ms | 2087.632 ms | **-45.1%** |
| Common Voice median prefill | 2324.371 ms | 699.970 ms | **-69.9%** |
| Common Voice median decode loop | 1784.403 ms | 1472.354 ms | **-17.5%** |
| Guardrail inference | 7615.522 ms | 4441.602 ms | **-41.7%** |
| Guardrail first-visible token | 5867.167 ms | 3053.721 ms | **-48.0%** |
| Guardrail prefill | 3854.890 ms | 1161.539 ms | **-69.9%** |
| Guardrail decode loop | 2565.011 ms | 2097.701 ms | **-18.2%** |
| Peak RSS, Common Voice | 2268456 KiB | 2279996 KiB | **+0.5%** |
| Peak RSS, guardrail | 2282952 KiB | 2295168 KiB | **+0.5%** |

The first prefill-only candidate reduced TTFT but raised peak RSS by 60.7%:
prefill faulted in the packed model-file pages while single-token decode still
faulted in the original row-major INT8 pages. Routing decode through the same
offline-packed weights prevents the duplicate projection layouts from becoming
resident. This removes essentially the entire memory regression and also makes
the decode loop 17.5-18.2% faster.

## Transcript gate

Five of six clips match exactly. One fails:

| Input | Control | Candidate |
| --- | --- | --- |
| `cv_yue_002.wav` | `證據嘅話，真係冇啊！` | `证据嘅话，真系冇啊！` |

The guardrail transcript is restored exactly by the fully packed decode path.
All three repetitions are stable, so the remaining traditional/simplified
character change is deterministic numerical behavior rather than run-to-run
noise.

## Decision

**Accept the memory/latency optimization as the Android packed candidate.** Use
packed prefill and packed decode together; prefill-only is rejected because it
makes both weight layouts resident. The remaining caveat is one exact transcript
mismatch out of six, so the KleidiAI path remains isolated behind its Android
feature and runtime switches rather than becoming the cross-platform default.

No F32 last-layer fallback is used: the fully packed path gives the lowest
latency and keeps peak RSS within 0.5% of control.
