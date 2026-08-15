# 048-mac-m4-kleidiai-control: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-15T08:59:52+00:00
- Platform: Mac mini / Apple M4
- Git revision: `18391d69addf19f8fffa8e124856740263c3dfc2` (dirty)
- Baseline: experiments/qwen-inference-optimization/runs/047-mac-1.7b-static-prefix-kv-cache

## Hypothesis

Same-binary tiled-F32 control for packed INT8 prefill

## Change

KleidiAI linked but QWEN_KLEIDIAI_PREFILL=0

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
| Median inference latency | 817.841 ms |
| P95 inference latency | 964.695 ms |
| Median first-visible-token latency | 512.705 ms |
| Median decoder prefill | 257.183 ms |
| Median model load time | 15.341 ms |
| First-launch model load time | 157.448 ms |
| RSS after model load | 49.12 MiB |
| Peak RSS | 2323.81 MiB |
| Model files | 2026.42 MiB |
| Weight files | 2022.10 MiB |
| Runtime binary | 1.70 MiB |

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

Exact-transcript gate passed. Decide keep/reject from the performance thresholds.

## Conclusion

Same-binary control for the KleidiAI experiments. The feature is linked, but
`QWEN_KLEIDIAI_PREFILL=0` retains the accepted tiled-F32 prefill path. All 54
transcripts are stable and provide the comparison point for runs 049 and 052.
