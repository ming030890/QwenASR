# 043-mac-1.7b-int8-encoder-ffn-l7-attn-l2-smoke: Qwen inference experiment

- Status: complete; smoke gate passed, boundary search continued
- Date (UTC): 2026-08-13T07:02:28+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 040-mac-1.7b-int8-encoder-ffn-attn-l7-23

## Hypothesis

Moving the encoder-attention INT8 boundary to layer 2 can retain ~20 MiB of the all-layer saving while preserving the two most sensitive early attention layers in BF16.

## Change

Kept FFN INT8 at layers 7-23 and quantized attention projections at layers 2-23.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-attn-l2`
- Detected variant: 1.7B
- Model inventory fingerprint: `84ba4c34792ad485f9b33099f76b6a7d4532e187fc7d6f713e5f646298f37f3f`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 869.517 ms |
| P95 inference latency | 7017.829 ms |
| Median model load time | 268.088 ms |
| First-launch model load time | 268.088 ms |
| RSS after model load | 49.23 MiB |
| Peak RSS | 2299.11 MiB |
| Model files | 2030.40 MiB |
| Weight files | 2026.08 MiB |
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

Proceed to the final intermediate boundary at layer 1 in run `044`.

## Conclusion

Attention layers 2-23 preserved all six exact transcripts and reduced model
files to 2030.40 MiB. Since starting at layer 0 failed, test layer 1 to isolate
whether only the first attention layer is quantization-sensitive.
