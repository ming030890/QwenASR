# 015-mac-1.7b-prefill-scratch-ab: Qwen inference experiment

- Status: complete; accepted; extended by run 018
- Date (UTC): 2026-08-12T22:48:51+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 014-mac-1.7b-prefill-persistent-cache-ab

## Hypothesis

On-demand BF16-to-F32 conversion into reusable prefill scratch eliminates persistent decoder F32 caches, substantially improving 1.7B load time and memory while preserving exact transcripts

## Change

Default path keeps no persistent F32 decoder prefill weights and converts each BF16 projection into one reusable scratch buffer immediately before GEMM

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
| Median inference latency | 903.143 ms |
| P95 inference latency | 1059.097 ms |
| Median model load time | 573.738 ms |
| First-launch model load time | 8511.836 ms |
| RSS after model load | 6302.67 MiB |
| Peak RSS | 7265.38 MiB |
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

Accept. Keep the scratch-prefill path and continue by removing the BF16 gate/up duplicate.

## Conclusion

Against the same-binary persistent-cache control, median load time improved
**95.9%**, RSS after load improved **27.3%**, peak RSS improved **27.1%**, and
median inference improved **8.0%**. P95 inference improved **50.0%**. All 54
transcripts matched exactly. Run 018 extends this accepted change by removing
the remaining owned BF16 gate/up fusion buffer.
