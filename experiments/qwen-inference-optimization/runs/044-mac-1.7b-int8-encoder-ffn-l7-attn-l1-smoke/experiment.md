# 044-mac-1.7b-int8-encoder-ffn-l7-attn-l1-smoke: Qwen inference experiment

- Status: complete; smoke gate passed and promoted to repeated run 045
- Date (UTC): 2026-08-13T07:03:21+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 040-mac-1.7b-int8-encoder-ffn-attn-l7-23

## Hypothesis

Leaving only encoder attention layer 0 in BF16 isolates the observed all-layer quality regression while capturing ~24 MiB beyond run 040.

## Change

Kept FFN INT8 at layers 7-23 and quantized attention projections at layers 1-23.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-attn-l1`
- Detected variant: 1.7B
- Model inventory fingerprint: `d86b979db69b78bf8b579ad5b80fd01b43f1d1a9ecd65fd51b84bf6f93cc6ad6`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 867.622 ms |
| P95 inference latency | 7074.578 ms |
| Median model load time | 134.386 ms |
| First-launch model load time | 134.386 ms |
| RSS after model load | 49.31 MiB |
| Peak RSS | 2286.41 MiB |
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

Promote to the repeated benchmark in run `045`.

## Conclusion

Leaving attention layer 0 in BF16 preserved all six exact transcripts while
reducing model files to 2026.42 MiB. Together with failed run `041`, this
isolates layer 0 as the observed attention PTQ quality boundary. Run `045`
performs the full three-launch validation.
