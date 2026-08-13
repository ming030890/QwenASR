# 035-mac-1.7b-int8-encoder-ffn-l7-23: Qwen inference experiment

- Status: complete; accepted encoder candidate
- Date (UTC): 2026-08-13T06:21:41+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 025-mac-1.7b-full-int8-tiled256-prefill

## Hypothesis

Repeated measurements confirm that rowwise INT8 encoder FFNs in layers 7-23 reduce peak RSS and model size materially without transcript or latency regression.

## Change

Quantized encoder FC1/FC2 weights in layers 7-23 to mmap-backed per-row INT8; layers 0-6 and all encoder attention/stem projections remain BF16.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-23`
- Detected variant: 1.7B
- Model inventory fingerprint: `e8c08bf628a1dcafe2f16db54ee7a52d3f65415c3f6643b646c69b3e42c35f4a`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 863.285 ms |
| P95 inference latency | 1023.707 ms |
| Median model load time | 15.638 ms |
| First-launch model load time | 24.041 ms |
| RSS after model load | 42.30 MiB |
| Peak RSS | 2420.86 MiB |
| Model files | 2118.05 MiB |
| Weight files | 2113.73 MiB |
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

Keep. This is the largest rowwise-INT8 encoder FFN suffix that passed the exact
guardrail gate; adding layer 6 failed in run `032`.

## Conclusion

Against baseline `025`, peak RSS improved 5.4% (2560.17 to 2420.86 MiB), model
files improved 6.0% (2253.71 to 2118.05 MiB), median latency improved 0.7%
(869.26 to 863.29 ms), P95 improved 0.6%, and encoder phase latency improved
9.1% (98.65 to 89.70 ms). All 54 candidate transcripts were stable and exactly
matched baseline. Layers 0-6 remain BF16 because expanding rowwise INT8 through
layer 6 changed one Cantonese transcript; full group-16 INT8 passed the smoke
gate but regressed encoder latency substantially. Next experiment: quantize
encoder attention projections independently, starting with the same late-layer
suffix and retaining this run as the controlled baseline.
