# 032-mac-1.7b-int8-encoder-ffn-l6-23-smoke: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-13T06:18:54+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Expanding fast rowwise INT8 to encoder FFN layers 6 through 23 preserves exact baseline transcripts and yields roughly the group-16 storage win without grouped-dequantization latency.

## Change

Applied per-row INT8 to encoder FC1/FC2 weights in layers 6-23; layers 0-5 remain BF16.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l6-23`
- Detected variant: 1.7B
- Model inventory fingerprint: `e44f588412a5d2416c9ce5954751b2ea8d1bc8b1b7d01bc62c6fcaac593ce484`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 847.198 ms |
| P95 inference latency | 7383.606 ms |
| Median model load time | 131.890 ms |
| First-launch model load time | 131.890 ms |
| RSS after model load | 42.55 MiB |
| Peak RSS | 2370.08 MiB |
| Model files | 2110.07 MiB |
| Weight files | 2105.75 MiB |
| Runtime binary | 1.69 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `证据嘅话真系冇啊！`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `哈喽，你好嘛？已经坐低咗啦。`

Full normalized measurements: [`metrics.json`](metrics.json).

## Acceptance gates

| Gate | Result |
| --- | --- |
| Exact transcript equality | **FAIL** |
| Candidate repetitions identical | PASS |
| Baseline repetitions identical | PASS |
| Same fixed input set | PASS |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Reject: the exact-transcript acceptance gate failed.

## Conclusion

Rejected automatically because at least one transcript differed from the baseline or was unstable across repetitions.
