# 017-mac-1.7b-no-bf16-gate-up-smoke: Qwen inference experiment

- Status: complete; smoke passed; promoted to run 018
- Date (UTC): 2026-08-12T22:51:51+00:00
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
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 938.971 ms |
| P95 inference latency | 1055.730 ms |
| Median model load time | 540.660 ms |
| First-launch model load time | 540.660 ms |
| RSS after model load | 4970.75 MiB |
| Peak RSS | 5841.27 MiB |
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

Promote to the full 3 x 3 comparison in run 018.

## Conclusion

After fixing the undersized temporary buffer found by the preceding failed
prototype, this smoke passed exact transcripts and reduced peak RSS enough to
justify full measurement.
