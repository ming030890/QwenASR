# 045-mac-1.7b-int8-encoder-ffn-l7-attn-l1: Qwen inference experiment

- Status: complete; accepted as the encoder comparison candidate
- Date (UTC): 2026-08-13T07:04:22+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 040-mac-1.7b-int8-encoder-ffn-attn-l7-23

## Hypothesis

Leaving only encoder attention layer 0 in BF16 isolates the observed all-layer quality regression while capturing ~24 MiB beyond run 040 with stable repeated performance.

## Change

Kept FFN INT8 at layers 7-23 and quantized attention projections at layers 1-23.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-attn-l1`
- Detected variant: 1.7B
- Model inventory fingerprint: `d86b979db69b78bf8b579ad5b80fd01b43f1d1a9ecd65fd51b84bf6f93cc6ad6`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 855.151 ms |
| P95 inference latency | 1025.163 ms |
| Median model load time | 15.423 ms |
| First-launch model load time | 21.250 ms |
| RSS after model load | 49.11 MiB |
| Peak RSS | 2327.25 MiB |
| Model files | 2026.42 MiB |
| Weight files | 2022.10 MiB |
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

The hypothesis is supported. Against run `040`, peak RSS fell 1.0% from 2349.72
to 2327.25 MiB, model files fell 1.2% from 2050.32 to 2026.42 MiB, and encoder
latency improved 4.8% from 83.60 to 79.62 ms. Median inference was effectively
flat/slightly better (855.93 versus 855.15 ms), and P95 changed only 0.3%
(1021.91 versus 1025.16 ms). All 54 transcripts matched exactly. Run `041`
shows that adding layer 0 crosses the transcript gate, so retain layer 0
attention in BF16 absent QAT. Use `045` as the next encoder comparison point,
without changing the overall `025` baseline.
