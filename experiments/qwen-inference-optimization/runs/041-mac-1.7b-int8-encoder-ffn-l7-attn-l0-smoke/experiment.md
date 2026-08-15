# 041-mac-1.7b-int8-encoder-ffn-l7-attn-l0-smoke: Qwen inference experiment

- Status: complete; rejected at the exact-transcript gate
- Date (UTC): 2026-08-13T07:00:16+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 040-mac-1.7b-int8-encoder-ffn-attn-l7-23

## Hypothesis

Encoder attention is less quantization-sensitive than FFN, so extending rowwise INT8 attention from layers 7-23 to all layers can remove another ~28 MiB without crossing the FFN transcript boundary.

## Change

Kept encoder FFN INT8 at layers 7-23 and extended Q/K/V/output projection INT8 to layers 0-23.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-attn-l0`
- Detected variant: 1.7B
- Model inventory fingerprint: `f111d051d80a4d95ae036c489f169c1a7ed86761ef120fd777a2b3d107968ee8`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 868.386 ms |
| P95 inference latency | 7055.323 ms |
| Median model load time | 138.611 ms |
| First-launch model load time | 138.611 ms |
| RSS after model load | 49.14 MiB |
| Peak RSS | 2272.94 MiB |
| Model files | 2022.43 MiB |
| Weight files | 2018.12 MiB |
| Runtime binary | 1.69 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊～`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `哈喽，你好嘛？已经坐低咗啦。`

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

Reject all-layer attention PTQ: `cv_yue_002.wav` changed from `證據嘅話，真係冇啊！`
to `證據嘅話，真係冇啊～`. The candidate did reach 2272.94 MiB peak RSS and
2022.43 MiB of model files, so runs `042`-`044` bisect the attention-only layer
boundary rather than abandoning the optimization.
