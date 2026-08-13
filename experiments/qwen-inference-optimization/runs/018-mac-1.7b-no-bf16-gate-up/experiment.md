# 018-mac-1.7b-no-bf16-gate-up: Qwen inference experiment

- Status: complete; accepted Mac candidate; Android validation pending
- Date (UTC): 2026-08-12T22:52:54+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 015-mac-1.7b-prefill-scratch-ab

## Hypothesis

Removing the owned BF16 gate/up fusion buffer further lowers 1.7B memory and startup work while separate BF16 prefill projections preserve exact output

## Change

On ARM64 quantize gate/up separately into the fused INT8 decode format; during BF16 prefill project gate/up separately and interleave activations before SwiGLU

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B`
- Detected variant: 1.7B
- Model inventory fingerprint: `9fb23a12239e6a3917c36d17f0951731e5bacfdbe90bcb080fe8f421908da043`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 918.755 ms |
| P95 inference latency | 1058.608 ms |
| Median model load time | 544.268 ms |
| First-launch model load time | 540.840 ms |
| RSS after model load | 4970.80 MiB |
| Peak RSS | 5875.38 MiB |
| Model files | 4485.24 MiB |
| Weight files | 4480.86 MiB |
| Runtime binary | 1.67 MiB |

Unique transcripts:

- `我急求方法啊！`
- `唔做嘢嘅佔多數咯。`
- `證據嘅話，真係冇啊！`
- `你睇下呢部剧。`
- `仲要提供乜嘢資料？`
- `Hello，你好嘛？已经坐低咗啦。`

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

Accept the Mac candidate. Keep both scratch prefill and zero-copy BF16 gate/up loading.

## Conclusion

Against the accepted scratch-only run 015, RSS after load improved **21.1%**,
peak RSS improved **19.1%**, and load time improved **5.1%**. Median inference
regressed only **1.73%**, within the 2% guardrail, while P95 was unchanged.

Against the same-binary legacy-cache control in run 014, the combined candidate
improved median load time **96.2%**, RSS after load **42.7%**, peak RSS **41.1%**,
median inference **6.4%**, and P95 inference **50.0%**. All 54 candidate outputs
exactly matched the control. This is the accepted Mac result; Android ARM64
measurement remains required before calling it a production win.
