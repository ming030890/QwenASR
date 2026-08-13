# 007-mac-openblas-v15-group8-int8-embeddings-mmap: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-12T22:16:27+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 004-mac-openblas-v15-int8-keep-embeddings-baseline

## Hypothesis

Group-8 INT8 token embeddings with zero-copy mmap scales preserve exact transcripts while reducing model size without the group-16 RSS regression

## Change

Quantize token embeddings symmetrically per 8 values and read both INT8 values and F32 scales directly from the model mmap

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000-group8-mmap`
- Detected variant: 0.6B
- Model inventory fingerprint: `04fd477da57202d3ce4dc657dc08c41d0d29611571b341729106e8a5bbaf1fc0`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 469.484 ms |
| P95 inference latency | 643.621 ms |
| Median model load time | 7.047 ms |
| First-launch model load time | 72.362 ms |
| RSS after model load | 37.88 MiB |
| Peak RSS | 1210.33 MiB |
| Model files | 1152.77 MiB |
| Weight files | 1148.53 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊`
- `唔做嘢嘅佔多數囉`
- `證據嘅話真係冇啊`
- `你睇下呢部劇`
- `重要提供乜嘢資料？`
- `hello，你好嗎？已經坐低咗啦`

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
