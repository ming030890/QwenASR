# 026-mac-1.7b-int8-encoder-ffn-smoke: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-13T06:07:41+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Per-row INT8 for all audio-encoder FC1/FC2 weights removes about 192 MiB from model and peak RSS while keeping encoder latency near baseline; transcript changes are tracked as QAT debt.

## Change

Quantized 24 encoder FC1/FC2 pairs to mmap-backed INT8 and dequantized 256 output rows at a time into reusable F32 scratch for OpenBLAS.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn`
- Detected variant: 1.7B
- Model inventory fingerprint: `24733d85c6e766059f9777b7859dfc3636d4f217beeaf1e51c2086701660839c`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 890.015 ms |
| P95 inference latency | 7356.084 ms |
| Median model load time | 269.285 ms |
| First-launch model load time | 269.285 ms |
| RSS after model load | 42.94 MiB |
| Peak RSS | 2328.02 MiB |
| Model files | 2062.19 MiB |
| Weight files | 2057.87 MiB |
| Runtime binary | 1.69 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊～`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `Hello，你好嘛？已经坐低咗啦。`

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
