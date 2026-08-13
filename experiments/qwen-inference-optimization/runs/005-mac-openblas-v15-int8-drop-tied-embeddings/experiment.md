# 005-mac-openblas-v15-int8-drop-tied-embeddings: Qwen inference experiment

- Status: complete; rejected
- Date (UTC): 2026-08-12T22:06:56+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 004-mac-openblas-v15-int8-keep-embeddings-baseline

## Hypothesis

Dropping redundant BF16 tied token embeddings cuts deployed model size and resident memory without changing transcripts or materially regressing latency

## Change

Use the slim INT8 export with thinker.model.embed_tokens.weight omitted; token lookup dequantizes from the tied INT8 lm_head

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000-slim`
- Detected variant: 0.6B
- Model inventory fingerprint: `f778ade0de619fac585ed0ba2779deb39ea4f1ea55e463531a776f48c40e0592`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 471.514 ms |
| P95 inference latency | 647.467 ms |
| Median model load time | 7.025 ms |
| First-launch model load time | 179.918 ms |
| RSS after model load | 37.75 MiB |
| Peak RSS | 1208.08 MiB |
| Model files | 930.21 MiB |
| Weight files | 925.96 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊`
- `唔做嘢嘅佔多數囉`
- `證據嘅話真係冇啊`
- `你睇下呢部劇`
- `仲要提供乜嘢資料？`
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

The candidate reduced model size from 1226.96 MiB to 930.21 MiB (**-24.2%**),
with median inference **1.8% faster** and peak RSS effectively unchanged
(`-0.2%`). All repetitions were internally stable, but two exact outputs changed:
`占` became `佔`, and `hello` became `Hello`.

The size hypothesis was supported, but the mandatory transcript gate failed.
Do not graduate the dropped-embedding export; retain BF16 token embeddings by
default.
