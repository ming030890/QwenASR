# 014-mac-1.7b-prefill-persistent-cache-ab: Qwen inference experiment

- Status: complete; accepted same-binary control
- Date (UTC): 2026-08-12T22:47:42+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: none

## Hypothesis

Establish a same-revision warm-cache control using the legacy persistent F32 prefill cache

## Change

Diagnostic QWEN_PREFILL_CACHE_WEIGHTS=1 restores the legacy persistent decoder F32 prefill cache in the candidate binary

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
| Median inference latency | 981.774 ms |
| P95 inference latency | 2117.088 ms |
| Median model load time | 14158.364 ms |
| First-launch model load time | 7245.615 ms |
| RSS after model load | 8667.69 MiB |
| Peak RSS | 9972.69 MiB |
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
| Baseline repetitions identical | n/a |
| Same fixed input set | n/a |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Use this run as the same-revision control for the scratch-prefill candidate.

## Conclusion

The diagnostic environment switch restored the legacy persistent F32 prefill
cache in the same OpenBLAS binary used by run 015. All 54 outputs were stable.
This removes revision and warm-filesystem-cache ambiguity from the comparison.
