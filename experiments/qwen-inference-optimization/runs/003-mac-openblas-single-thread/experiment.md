# 003-mac-openblas-single-thread: Qwen inference experiment

- Status: complete; rejected
- Date (UTC): 2026-08-12T20:45:45+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 002-mac-openblas-baseline

## Hypothesis

Pinning OpenBLAS to one worker avoids nested-thread overhead and recovers the latency regression while keeping its memory reduction

## Change

Set OPENBLAS_NUM_THREADS=1 for the flag-gated OpenBLAS backend

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
| Median inference latency | 1378.919 ms |
| P95 inference latency | 3288.374 ms |
| Median model load time | 14294.928 ms |
| First-launch model load time | 15014.638 ms |
| RSS after model load | 8495.11 MiB |
| Peak RSS | 8538.69 MiB |
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
| Baseline repetitions identical | PASS |
| Same fixed input set | PASS |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Reject. Explicitly pinning OpenBLAS to one worker improves its median latency,
but it does not meet the 5% improvement threshold and does not recover the large
regression relative to Accelerate.

## Conclusion

The exact-transcript gate passed across all six inputs and 54 candidate runs.
Relative to unpinned OpenBLAS (`002`), `OPENBLAS_NUM_THREADS=1` improved median
inference by **4.1%** and median model load by **3.9%**. Both miss the 5%
acceptance threshold; P95 inference regressed by **14.3%**, RSS after load
regressed by **7.8%**, and peak RSS regressed by **4.4%**.

The Accelerate reference is not acceptance-eligible because Android uses
OpenBLAS. Keep OpenBLAS as the required experiment backend, but do not graduate
the `OPENBLAS_NUM_THREADS=1` tuning: its gains versus the OpenBLAS baseline are
below threshold and its P95 and memory measurements regress.
