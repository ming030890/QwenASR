# 001-mac-qwen3-asr-1.7b-baseline: Qwen inference experiment

- Status: complete; legacy Accelerate baseline (not eligible for future comparisons)
- Date (UTC): 2026-08-12T07:43:24+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: none

## Hypothesis

Establish the unmodified Qwen3-ASR-1.7B Mac mini reference for subsequent optimization experiments

## Change

None; baseline at the recorded Git revision and fixed configuration

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
| Median inference latency | 1223.094 ms |
| P95 inference latency | 2210.936 ms |
| Median model load time | 11577.765 ms |
| First-launch model load time | 13337.806 ms |
| RSS after model load | 9002.64 MiB |
| Peak RSS | 9571.12 MiB |
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

Baseline transcript is stable; use this experiment as the named baseline for candidates.

## Conclusion

This historical Qwen3-ASR-1.7B Mac mini measurement produced exactly identical
transcripts across all repetitions. The first model load was slower than the
warm launches, so future comparisons must keep both first-load and median-load
values visible.

This run predates the mandatory OpenBLAS policy and reports vDSP/Accelerate. It
is not an eligible comparison baseline. Establish a new OpenBLAS baseline with
the current runner before evaluating future candidates; accepted changes still
require the Android acceptance gate.
