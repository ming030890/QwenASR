# 002-mac-openblas-baseline: Qwen inference experiment

- Status: complete; accepted platform-parity baseline
- Date (UTC): 2026-08-12T20:42:21+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: none

## Hypothesis

OpenBLAS improves Qwen3-ASR 1.7B Mac inference over the legacy Accelerate path while preserving exact transcripts

## Change

Build the existing flag-gated macOS BLAS path with QWEN_BLAS_BACKEND=openblas

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B`
- Detected variant: 1.7B
- Model inventory fingerprint: `9fb23a12239e6a3917c36d17f0951731e5bacfdbe90bcb080fe8f421908da043`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 1437.555 ms |
| P95 inference latency | 2876.204 ms |
| Median model load time | 14871.317 ms |
| First-launch model load time | 18062.348 ms |
| RSS after model load | 7882.45 MiB |
| Peak RSS | 8182.25 MiB |
| Model files | 4485.24 MiB |
| Weight files | 4480.86 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊！`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `Hello，你好嘛？已经坐低咗啦。`

Full normalized measurements: [`metrics.json`](metrics.json).

## Acceptance gates

| Gate | Result |
| --- | --- |
| Exact transcript equality | **PASS** |
| Candidate repetitions identical | PASS |
| Baseline repetitions identical | n/a |
| Same fixed input set | n/a |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Accept this run as the controlled platform-parity baseline. OpenBLAS is required
for these experiments because Android uses it; the historical Accelerate run is
diagnostic only and is not an eligible baseline or candidate.

## Conclusion

All 54 runs were internally stable, and their canonical transcripts exactly
match the six transcripts in the historical Accelerate run. The measurements
in `001` are not used for acceptance because its backend differs from Android.

Use this OpenBLAS result as the named baseline for subsequent 1.7B candidates.
The first follow-up tests explicit single-thread OpenBLAS configuration; the
separate tied-embedding candidate must use a matched INT8 model baseline.
