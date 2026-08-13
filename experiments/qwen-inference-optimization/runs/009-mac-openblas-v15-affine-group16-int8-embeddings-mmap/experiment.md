# 009-mac-openblas-v15-affine-group16-int8-embeddings-mmap: Qwen inference experiment

- Status: complete; rejected
- Date (UTC): 2026-08-12T22:21:09+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 004-mac-openblas-v15-int8-keep-embeddings-baseline

## Hypothesis

Affine group-16 INT8 preserves tied-token embedding decisions better than symmetric INT8 while reducing model size without adding load-time allocations.

## Change

Quantize tied decoder embeddings to affine group-16 INT8 and mmap per-group F32 scale/min metadata; retain the INT8 LM head.

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000-affine-group16-mmap`
- Detected variant: 0.6B
- Model inventory fingerprint: `2968f74ae732a1b4a1a36aeaee63c22688fecfa541ae78b8bd7f548d514869a4`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 485.544 ms |
| P95 inference latency | 662.047 ms |
| Median model load time | 6.530 ms |
| First-launch model load time | 67.067 ms |
| RSS after model load | 37.86 MiB |
| Peak RSS | 1207.62 MiB |
| Model files | 1152.77 MiB |
| Weight files | 1148.53 MiB |
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

The mmap-backed affine format saved 74.19 MiB (6.0%), but changed three stable
outputs and was 1.1% slower at the median. A later logit diagnostic found that
`仲要` led `重要` by 2.05 logits in the baseline before this quantizer reversed
the choice. Do not graduate these post-training-quantized weights; retain the
inference format for a quantization-aware recovered checkpoint.
