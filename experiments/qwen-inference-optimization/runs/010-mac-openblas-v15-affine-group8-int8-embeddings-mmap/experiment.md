# 010-mac-openblas-v15-affine-group8-int8-embeddings-mmap: Qwen inference experiment

- Status: complete; rejected
- Date (UTC): 2026-08-12T22:22:20+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 004-mac-openblas-v15-int8-keep-embeddings-baseline

## Hypothesis

Halving affine embedding groups from 16 to 8 reduces accumulated embedding error enough to recover exact baseline decoding.

## Change

Use affine group-8 INT8 tied decoder embeddings with mmap F32 scale/min metadata.

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000-affine-group8-mmap`
- Detected variant: 0.6B
- Model inventory fingerprint: `b03ba42dd364a89d296f2f452e46bbe4701883e735841d28620dc3685a5086e7`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 480.765 ms |
| P95 inference latency | 667.455 ms |
| Median model load time | 6.803 ms |
| First-launch model load time | 64.924 ms |
| RSS after model load | 37.78 MiB |
| Peak RSS | 1217.67 MiB |
| Model files | 1226.96 MiB |
| Weight files | 1222.72 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊`
- `唔做嘢嘅佔多數囉`
- `證據嘅話真係冇啊`
- `你睇下呢部劇`
- `仲要提供乜嘢資料？`
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

Group 8 recovered `仲要`, punctuation, and English capitalization relative to
group 16, but retained the `占` to `佔` difference and saved no model space.
The competing logits were only 0.263 apart in the baseline, confirming a real
near tie. This is useful diagnostic evidence, not a deployable representation.
