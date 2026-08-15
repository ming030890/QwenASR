# 042-mac-1.7b-int8-encoder-ffn-l7-attn-l4-smoke: Qwen inference experiment

- Status: complete; smoke gate passed, boundary search continued
- Date (UTC): 2026-08-13T07:01:18+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 040-mac-1.7b-int8-encoder-ffn-attn-l7-23

## Hypothesis

Moving the encoder-attention INT8 boundary from layer 7 to layer 4 recovers ~12 MiB while excluding the most quantization-sensitive earliest layers.

## Change

Kept FFN INT8 at layers 7-23 and quantized attention projections at layers 4-23.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-attn-l4`
- Detected variant: 1.7B
- Model inventory fingerprint: `43535a660408e65bf303f548511a714f3e7e975d0b590e645171221628cfb6a0`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 889.864 ms |
| P95 inference latency | 7111.953 ms |
| Median model load time | 147.913 ms |
| First-launch model load time | 147.913 ms |
| RSS after model load | 49.19 MiB |
| Peak RSS | 2293.75 MiB |
| Model files | 2038.37 MiB |
| Weight files | 2034.05 MiB |
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

Proceed to an earlier attention boundary in run `043`.

## Conclusion

Attention layers 4-23 preserved all six exact transcripts and reduced model
files to 2038.37 MiB. Continue the boundary search at layer 2 to capture more of
the measured all-layer memory opportunity.
