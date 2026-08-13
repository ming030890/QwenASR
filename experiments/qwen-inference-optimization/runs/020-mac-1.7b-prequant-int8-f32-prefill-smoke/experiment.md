# 020-mac-1.7b-prequant-int8-f32-prefill-smoke: Qwen inference experiment

- Status: complete; hypothesis supported
- Date (UTC): 2026-08-13T05:49:43+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 019-mac-1.7b-prequant-int8-smoke

## Hypothesis

Dequantizing one mmap-backed INT8 projection at a time into reusable scratch enables optimized F32 GEMM and removes most prefill latency regression while keeping peak RSS close to the native INT8 path.

## Change

Added QWEN_INT8_PREFILL_F32=1 diagnostic path for decoder prefill; weights remain prequantized mmap INT8 and only the active matrix is materialized as F32 in reusable scratch.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-decoder`
- Detected variant: 1.7B
- Model inventory fingerprint: `845c489e4edf58a5b1ee1cd8428bf18c4e5b3c52f87f81b8b0bc6a84efe98419`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 931.957 ms |
| P95 inference latency | 1071.016 ms |
| Median model load time | 15.443 ms |
| First-launch model load time | 15.443 ms |
| RSS after model load | 41.73 MiB |
| Peak RSS | 2602.95 MiB |
| Model files | 2847.21 MiB |
| Weight files | 2842.89 MiB |
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

Keep and repeat. It preserves the established INT8 transcripts and removes the
native INT8 prefill bottleneck for a small RSS cost.

## Conclusion

Versus `019`, median latency improved 44.2% (1670.43 to 931.96 ms) and prefill
improved 67.4% (1084.39 to 353.48 ms). Peak RSS increased 3.0% (2526.95 to
2602.95 MiB). The path reproduced all six INT8 transcripts exactly; the known
one-clip difference from BF16 remains QAT debt. Proceed to repeated measurement.
