# 027-mac-1.7b-int8-encoder-ffn-g32-smoke: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-13T06:11:36+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Symmetric INT8 groups of 32 for encoder FC1/FC2 preserve the baseline transcripts while retaining most of the per-row INT8 RSS reduction.

## Change

Replaced per-row encoder FFN scales with one symmetric INT8 scale per 32 input values; retained 256-row dequantization tiles.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-g32`
- Detected variant: 1.7B
- Model inventory fingerprint: `ba10c3e4e281f9c387ccf3aa963007bd02968b5afbd8c7078eec510c05d887bb`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 973.376 ms |
| P95 inference latency | 7301.480 ms |
| Median model load time | 270.332 ms |
| First-launch model load time | 270.332 ms |
| RSS after model load | 89.89 MiB |
| Peak RSS | 2373.58 MiB |
| Model files | 2085.72 MiB |
| Weight files | 2081.40 MiB |
| Runtime binary | 1.69 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊！`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `哈喽，你好嘛，已经坐低咗啦。`

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
