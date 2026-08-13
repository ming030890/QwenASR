# 029-mac-1.7b-int8-encoder-ffn-l12-smoke: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-13T06:15:23+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Rowwise INT8 on only encoder FFN layers 0 through 11 preserves exact baseline transcripts while removing about 96 MiB of resident weights without latency regression.

## Change

Applied fast per-row INT8 only to encoder FC1/FC2 weights in layers 0-11; layers 12-23 remain BF16.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l12`
- Detected variant: 1.7B
- Model inventory fingerprint: `9bbcc2335913fcc3ac11420293b8596fea270d0ec4b30c09a98e0b2f1373571a`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 888.107 ms |
| P95 inference latency | 7215.786 ms |
| Median model load time | 138.504 ms |
| First-launch model load time | 138.504 ms |
| RSS after model load | 42.31 MiB |
| Peak RSS | 2433.23 MiB |
| Model files | 2157.95 MiB |
| Weight files | 2153.63 MiB |
| Runtime binary | 1.69 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊！`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `hello，你好嘛？已经坐低咗啦。`

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
