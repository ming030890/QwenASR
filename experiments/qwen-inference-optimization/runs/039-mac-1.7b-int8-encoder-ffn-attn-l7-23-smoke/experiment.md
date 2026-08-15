# 039-mac-1.7b-int8-encoder-ffn-attn-l7-23-smoke: Qwen inference experiment

- Status: complete; smoke gate passed and promoted to repeated run 040
- Date (UTC): 2026-08-13T06:56:40+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 035-mac-1.7b-int8-encoder-ffn-l7-23

## Hypothesis

Rowwise INT8 on encoder attention projections in the exact-safe layers 7-23 reduces another ~68 MiB of resident/model weights with stable transcripts and no material encoder latency regression.

## Change

Quantized encoder Q/K/V/output projections in layers 7-23 to mmap-backed per-row INT8 in addition to the accepted FFN suffix.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-attn-l7-23`
- Detected variant: 1.7B
- Model inventory fingerprint: `b09b0dfbbe64f1a086b626addf2d45cc5931ce0936c157134338d4e678add27a`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 872.620 ms |
| P95 inference latency | 7222.427 ms |
| Median model load time | 143.982 ms |
| First-launch model load time | 143.982 ms |
| RSS after model load | 49.06 MiB |
| Peak RSS | 2324.34 MiB |
| Model files | 2050.32 MiB |
| Weight files | 2046.00 MiB |
| Runtime binary | 1.69 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊！`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `哈喽，你好嘛？已经坐低咗啦。`

Full normalized measurements: [`metrics.json`](metrics.json).

## Acceptance gates

| Gate | Result |
| --- | --- |
| Exact transcript equality | **PASS** |
| Candidate repetitions identical | PASS |
| Baseline repetitions identical | PASS |
| Same fixed input set | PASS |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Proceed to the repeated benchmark in run `040`.

## Conclusion

The one-launch smoke gate passed all six exact transcripts. Peak RSS was
2324.34 MiB and model files were 2050.32 MiB, confirming that the additional
attention weights were actually removed from the resident/model footprint.
Run `040` repeats the candidate across three launches and 54 samples.
