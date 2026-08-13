# 028-mac-1.7b-int8-encoder-ffn-g16-smoke: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-13T06:13:13+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Symmetric INT8 groups of 16 for encoder FC1/FC2 recover exact baseline transcripts while still reducing peak RSS and model size materially.

## Change

Used one symmetric INT8 scale per 16 encoder FFN input values and optimized grouped dequantization to remove inner-loop division.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-g16`
- Detected variant: 1.7B
- Model inventory fingerprint: `5d76ede9d12665a9280b1ec420f07f9848bbbd63a200a1f205d305fe45bfa20d`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 965.214 ms |
| P95 inference latency | 7258.823 ms |
| Median model load time | 321.195 ms |
| First-launch model load time | 321.195 ms |
| RSS after model load | 137.88 MiB |
| Peak RSS | 2432.50 MiB |
| Model files | 2109.72 MiB |
| Weight files | 2105.40 MiB |
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
