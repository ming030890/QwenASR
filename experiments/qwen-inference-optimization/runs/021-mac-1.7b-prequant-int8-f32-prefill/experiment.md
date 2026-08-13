# 021-mac-1.7b-prequant-int8-f32-prefill: Qwen inference experiment

- Status: complete; hypothesis supported
- Date (UTC): 2026-08-13T05:50:57+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 019-mac-1.7b-prequant-int8-smoke

## Hypothesis

Repeated runs confirm that one-matrix-at-a-time INT8 dequantization plus optimized F32 GEMM removes most of the prefill latency regression for under 128 MiB extra peak RSS.

## Change

Enabled QWEN_INT8_PREFILL_F32=1 for repeated 1.7B prequantized INT8 inference; mmap INT8 remains the stored representation and F32 exists only in reusable projection scratch.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-decoder`
- Detected variant: 1.7B
- Model inventory fingerprint: `845c489e4edf58a5b1ee1cd8428bf18c4e5b3c52f87f81b8b0bc6a84efe98419`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 899.679 ms |
| P95 inference latency | 1059.476 ms |
| Median model load time | 14.605 ms |
| First-launch model load time | 14.815 ms |
| RSS after model load | 41.61 MiB |
| Peak RSS | 2654.41 MiB |
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

Keep as the optimized prequantized-INT8 compute baseline.

## Conclusion

Across 54 inferences, median latency was 899.68 ms and prefill 351.04 ms,
confirming that one-matrix F32 scratch recovers BF16-class prefill speed. Versus
`019`, median latency improved 46.1% at a 5.0% peak-RSS cost (2654.41 MiB).
Repetitions were stable and exactly matched the INT8 baseline. The BF16
`Hello`/`哈喽` mismatch remains explicit QAT debt.
