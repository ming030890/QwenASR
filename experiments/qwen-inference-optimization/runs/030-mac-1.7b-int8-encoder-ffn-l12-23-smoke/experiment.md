# 030-mac-1.7b-int8-encoder-ffn-l12-23-smoke: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-13T06:17:03+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Rowwise INT8 on encoder FFN layers 12 through 23 identifies whether the later half is safe and suitable for a mixed-precision hybrid.

## Change

Applied fast per-row INT8 only to encoder FC1/FC2 weights in layers 12-23; layers 0-11 remain BF16.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l12-23`
- Detected variant: 1.7B
- Model inventory fingerprint: `2e95ff1e4cd67b38babcba8a42fdc12e80e6ad30a94a2d24caf5748125f39484`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 883.942 ms |
| P95 inference latency | 7659.803 ms |
| Median model load time | 137.814 ms |
| First-launch model load time | 137.814 ms |
| RSS after model load | 42.30 MiB |
| Peak RSS | 2428.02 MiB |
| Model files | 2157.95 MiB |
| Weight files | 2153.63 MiB |
| Runtime binary | 1.69 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊！`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `哈喽，你好嘛？已经坐低咗啦。`

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

Exact-transcript gate passed. Decide keep/reject from the performance thresholds.

## Conclusion

Transcript correctness is preserved. State whether the performance hypothesis was supported, the percentage deltas from baseline, important caveats, and the next experiment.
