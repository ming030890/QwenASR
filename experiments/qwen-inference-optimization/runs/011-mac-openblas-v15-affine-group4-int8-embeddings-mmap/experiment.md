# 011-mac-openblas-v15-affine-group4-int8-embeddings-mmap: Qwen inference experiment

- Status: complete; rejected
- Date (UTC): 2026-08-12T22:23:31+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 004-mac-openblas-v15-int8-keep-embeddings-baseline

## Hypothesis

Affine group-4 embedding quantization crosses the exact-decoding fidelity threshold after group-8 left only one traditional-character variant.

## Change

Diagnostic affine group-4 INT8 tied decoder embeddings with mmap F32 scale/min metadata.

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000-affine-group4-mmap`
- Detected variant: 0.6B
- Model inventory fingerprint: `b6309f66a34f083c556ba5b03bcf30f54e3694784efd02a05915e2ce1d0283f3`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 491.008 ms |
| P95 inference latency | 675.366 ms |
| Median model load time | 6.302 ms |
| First-launch model load time | 71.358 ms |
| RSS after model load | 37.91 MiB |
| Peak RSS | 1227.19 MiB |
| Model files | 1375.33 MiB |
| Weight files | 1371.09 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊`
- `唔做嘢嘅佔多數囉`
- `證據嘅話真係冇啊`
- `你睇下呢部劇`
- `重要提供乜嘢資料？`
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

Reducing the affine group to 4 did not monotonically recover fidelity: three
outputs differed, median inference was 2.2% slower, and the model became 148.38
MiB larger than the BF16-embedding baseline. Stop the post-training group-size
search and recover group-16 quantization through training instead.
