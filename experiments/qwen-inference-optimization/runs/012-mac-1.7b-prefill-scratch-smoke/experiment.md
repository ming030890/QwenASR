# 012-mac-1.7b-prefill-scratch-smoke: Qwen inference experiment

- Status: complete; smoke passed; superseded by run 015
- Date (UTC): 2026-08-12T22:43:16+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 002-mac-openblas-baseline

## Hypothesis

On-demand BF16-to-F32 conversion into reusable prefill scratch removes persistent decoder F32 caches, sharply reducing 1.7B load time and memory while preserving exact transcripts

## Change

Do not materialize persistent F32 decoder prefill weights; convert each BF16 projection into one reusable scratch buffer immediately before GEMM

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
| Median inference latency | 974.986 ms |
| P95 inference latency | 3991.526 ms |
| Median model load time | 20099.947 ms |
| First-launch model load time | 20099.947 ms |
| RSS after model load | 4463.03 MiB |
| Peak RSS | 7217.28 MiB |
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

Promote to a controlled 3 x 3 measurement. Do not accept from this smoke alone.

## Conclusion

The exact-transcript gate passed and the smoke showed enough memory promise to
justify a full run. Its single cold/warm load measurement was not comparable to
the older baseline, so load-time acceptance was deferred to a same-binary A/B.
