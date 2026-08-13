# 004-mac-openblas-v15-int8-keep-embeddings-baseline: Qwen inference experiment

- Status: complete; accepted baseline
- Date (UTC): 2026-08-12T22:06:14+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: none

## Hypothesis

Establish the OpenBLAS baseline for the v15 INT8 mobile model with redundant BF16 tied embeddings retained

## Change

None; existing v15 INT8 export with token embeddings retained

## Fixed setup

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000`
- Detected variant: 0.6B
- Model inventory fingerprint: `562a1fc9db1c5ea03cfc675ed06b55681623a7aef89744737962516cf76c8d2c`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 480.398 ms |
| P95 inference latency | 658.124 ms |
| Median model load time | 7.129 ms |
| First-launch model load time | 315.530 ms |
| RSS after model load | 37.77 MiB |
| Peak RSS | 1210.78 MiB |
| Model files | 1226.96 MiB |
| Weight files | 1222.71 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `我急求方法啊`
- `唔做嘢嘅占多數囉`
- `證據嘅話真係冇啊`
- `你睇下呢部劇`
- `仲要提供乜嘢資料？`
- `hello，你好嗎？已經坐低咗啦`

Full normalized measurements: [`metrics.json`](metrics.json).

## Acceptance gates

| Gate | Result |
| --- | --- |
| Exact transcript equality | **PASS** |
| Candidate repetitions identical | PASS |
| Baseline repetitions identical | n/a |
| Same fixed input set | n/a |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Baseline transcript is stable; use this experiment as the named baseline for candidates.

## Conclusion

All six transcripts were stable across 54 runs. This is the controlled OpenBLAS
baseline for evaluating tied-embedding storage changes in the v15 0.6B INT8
mobile model. Decoder projection weights and the LM head are INT8; token
embeddings and audio-encoder weights remain BF16.
