# 031-mac-1.7b-int8-encoder-ffn-l9-23-smoke: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-13T06:18:04+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Expanding fast rowwise INT8 to encoder FFN layers 9 through 23 preserves exact baseline transcripts while increasing the RSS reduction.

## Change

Applied per-row INT8 to encoder FC1/FC2 weights in layers 9-23; layers 0-8 remain BF16.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l9-23`
- Detected variant: 1.7B
- Model inventory fingerprint: `df6eea8f5fe44ca1e97d147f355753e06c045dd45b3e3aedd5f925c7946a1b2c`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 894.563 ms |
| P95 inference latency | 7790.732 ms |
| Median model load time | 184.010 ms |
| First-launch model load time | 184.010 ms |
| RSS after model load | 42.48 MiB |
| Peak RSS | 2399.80 MiB |
| Model files | 2134.01 MiB |
| Weight files | 2129.69 MiB |
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
