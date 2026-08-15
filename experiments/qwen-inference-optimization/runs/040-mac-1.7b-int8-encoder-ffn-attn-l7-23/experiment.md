# 040-mac-1.7b-int8-encoder-ffn-attn-l7-23: Qwen inference experiment

- Status: complete; accepted as the encoder comparison candidate
- Date (UTC): 2026-08-13T06:57:38+00:00
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
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 855.927 ms |
| P95 inference latency | 1021.914 ms |
| Median model load time | 15.737 ms |
| First-launch model load time | 15.853 ms |
| RSS after model load | 48.92 MiB |
| Peak RSS | 2349.72 MiB |
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

Accept for encoder-specific work. Keep run `025` as the explicitly designated
overall 1.7B baseline.

## Conclusion

The hypothesis is supported. Against encoder comparison run `035`, peak RSS
fell 2.9% from 2420.86 to 2349.72 MiB, model files fell 3.2% from 2118.05 to
2050.32 MiB, and encoder phase latency improved 6.8% from 89.70 to 83.60 ms.
Median inference also improved 0.9% from 863.29 to 855.93 ms, while P95 was
effectively flat (1023.71 versus 1021.91 ms). All 54 transcripts matched
exactly. The dynamic scratch sizing from run `038` is present in this binary,
but its isolated smoke test showed no resident-memory benefit; the measured
model/RSS reduction here corresponds to the additional 67.7 MiB of attention
weight quantization. Use `040` as the next encoder comparison point, without
changing the overall `025` baseline.
