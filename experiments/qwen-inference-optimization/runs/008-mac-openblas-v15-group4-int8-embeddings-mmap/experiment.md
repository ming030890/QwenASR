# 008-mac-openblas-v15-group4-int8-embeddings-mmap: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-12T22:17:20+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 004-mac-openblas-v15-int8-keep-embeddings-baseline

## Hypothesis

Group-4 INT8 token embeddings establish whether sufficiently fine symmetric quantization can preserve exact transcripts

## Change

Quantize token embeddings symmetrically per 4 values and mmap both values and scales

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000-group4-mmap`
- Detected variant: 0.6B
- Model inventory fingerprint: `8a3d963f55498d1baf0a1a7a3fa98e37ffe814a33bcf73252fbedbb9587bbe66`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 471.567 ms |
| P95 inference latency | 650.805 ms |
| Median model load time | 6.386 ms |
| First-launch model load time | 126.126 ms |
| RSS after model load | 37.77 MiB |
| Peak RSS | 1205.28 MiB |
| Model files | 1226.96 MiB |
| Weight files | 1222.71 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊`
- `唔做嘢嘅佔多數囉`
- `證據嘅話真係冇啊`
- `你睇下呢部劇`
- `重要提供乜嘢資料`
- `Hello，你好嗎？已經坐低咗啦`

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
