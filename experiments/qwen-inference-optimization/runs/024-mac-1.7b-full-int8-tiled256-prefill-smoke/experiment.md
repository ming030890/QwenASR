# 024-mac-1.7b-full-int8-tiled256-prefill-smoke: Qwen inference experiment

- Status: complete; hypothesis supported
- Date (UTC): 2026-08-13T05:56:07+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 023-mac-1.7b-full-int8-tiled-prefill-smoke

## Hypothesis

A 256-row dequantization tile further reduces touched F32 scratch pages without enough GEMM fragmentation to exceed the allowed modest latency regression.

## Change

Set QWEN_INT8_PREFILL_TILE_ROWS=256 for the full-decoder INT8 model.

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
| Median inference latency | 915.414 ms |
| P95 inference latency | 1034.083 ms |
| Median model load time | 15.441 ms |
| First-launch model load time | 15.441 ms |
| RSS after model load | 41.59 MiB |
| Peak RSS | 2514.36 MiB |
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

Keep and repeat at 3 launches × 3 runs.

## Conclusion

The 256-row tile reduced smoke peak RSS another 39.09 MiB versus the 1024-row
tile and improved the noisy median to 915.41 ms. All INT8 transcripts remained
exact. A repeated run is required for the final decision.
