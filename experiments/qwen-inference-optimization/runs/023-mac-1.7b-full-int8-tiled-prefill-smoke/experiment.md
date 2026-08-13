# 023-mac-1.7b-full-int8-tiled-prefill-smoke: Qwen inference experiment

- Status: complete; hypothesis supported
- Date (UTC): 2026-08-13T05:55:46+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 022-mac-1.7b-full-int8-smoke

## Hypothesis

Tiling INT8-to-F32 prefill dequantization to 1024 output rows cuts live scratch pages substantially while preserving most optimized GEMM throughput and identical INT8 transcripts.

## Change

Added strided output-tile GEMM and QWEN_INT8_PREFILL_TILE_ROWS=1024; only one output-row tile is dequantized into reusable F32 scratch at a time.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-full`
- Detected variant: 1.7B
- Model inventory fingerprint: `e2b965c2ccbcda3776e8555aae2885fdb6ad89c9ddb99b9b3b2f66396d4b9f82`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 936.793 ms |
| P95 inference latency | 1066.119 ms |
| Median model load time | 23.974 ms |
| First-launch model load time | 23.974 ms |
| RSS after model load | 41.64 MiB |
| Peak RSS | 2553.45 MiB |
| Model files | 2253.71 MiB |
| Weight files | 2249.39 MiB |
| Runtime binary | 1.67 MiB |

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

Keep the tiling mechanism and test a smaller tile.

## Conclusion

A 1024-row tile reduced peak RSS from 2600.14 to 2553.45 MiB (1.8%) while
median latency moved from 907.74 to 936.79 ms (+3.2%). INT8 transcripts remained
exact. Try 256 rows to determine whether more scratch can be removed cheaply.
