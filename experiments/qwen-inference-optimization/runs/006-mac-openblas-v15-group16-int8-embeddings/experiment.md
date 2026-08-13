# 006-mac-openblas-v15-group16-int8-embeddings: Qwen inference experiment

- Status: complete; rejected; prototype reverted
- Date (UTC): 2026-08-12T22:10:06+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 004-mac-openblas-v15-int8-keep-embeddings-baseline

## Hypothesis

Groupwise INT8 token embeddings recover exact transcripts while retaining a meaningful model-size reduction

## Change

Replace BF16 token embeddings with symmetric INT8 values using one F32 scale per 16 values; retain the existing row-wise INT8 LM head

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000-group16`
- Detected variant: 0.6B
- Model inventory fingerprint: `aa47a6ec5b0e9123b9caf8ff1d9b1178b27f68c655d4d359453df04ef2ca6d72`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 476.019 ms |
| P95 inference latency | 653.160 ms |
| Median model load time | 10.914 ms |
| First-launch model load time | 119.618 ms |
| RSS after model load | 111.95 MiB |
| Peak RSS | 1277.28 MiB |
| Model files | 1115.68 MiB |
| Weight files | 1111.43 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊`
- `唔做嘢嘅占多數囉`
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

Using an independent scale for every 16 embedding values improved fidelity over
the row-wise INT8 reuse experiment, but one exact output still changed: `仲要`
became `重要`. Model size improved **9.1%**, median inference improved **0.9%**,
model-load RSS increased **196.4%**, and peak RSS increased **5.5%**.

The transcript gate failed and the memory tradeoff is unfavorable. The
groupwise embedding prototype was reverted; its result and generated SSD model
remain available for reproducibility.
