# 013-mac-1.7b-prefill-scratch: Qwen inference experiment

- Status: complete; promising diagnostic; superseded by run 015
- Date (UTC): 2026-08-12T22:44:36+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 002-mac-openblas-baseline

## Hypothesis

On-demand BF16-to-F32 conversion into reusable prefill scratch removes persistent decoder F32 caches, sharply reducing 1.7B memory while preserving exact transcripts with an acceptable latency tradeoff

## Change

Do not materialize persistent F32 decoder prefill weights; convert each BF16 projection into one reusable scratch buffer immediately before GEMM

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
| Median inference latency | 912.108 ms |
| P95 inference latency | 1063.993 ms |
| Median model load time | 1048.156 ms |
| First-launch model load time | 594.681 ms |
| RSS after model load | 6302.66 MiB |
| Peak RSS | 7210.05 MiB |
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

Keep the implementation for a same-binary A/B against a warm-cache legacy control.

## Conclusion

All 54 transcripts matched, median inference improved, and memory fell, but the
historical baseline had different filesystem-cache conditions. Run 014/015 was
created to isolate the optimization in one binary before acceptance.
