# 022-mac-1.7b-full-int8-smoke: Qwen inference experiment

- Status: complete; storage hypothesis supported
- Date (UTC): 2026-08-13T05:53:20+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 021-mac-1.7b-prequant-int8-f32-prefill

## Hypothesis

Dropping the duplicate BF16 token embedding and reusing the tied rowwise INT8 LM head reduces model size and peak RSS by roughly another 0.5 GiB with negligible latency cost; transcript changes are tracked as expected QAT debt.

## Change

Streaming exporter --drop-embed-tokens omits the BF16 embedding matrix; runtime token lookup falls back to the tied mmap-backed rowwise INT8 LM head.

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
| Median inference latency | 907.738 ms |
| P95 inference latency | 7959.002 ms |
| Median model load time | 337.343 ms |
| First-launch model load time | 337.343 ms |
| RSS after model load | 41.80 MiB |
| Peak RSS | 2600.14 MiB |
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

Keep the full-decoder INT8 export; continue optimizing runtime scratch.

## Conclusion

Omitting the duplicate BF16 embedding reduced model files 20.8% (2847.21 to
2253.71 MiB) and reproduced the prior INT8 transcripts exactly. Peak RSS only
fell about 2.0% in this smoke because the omitted mmap was sparsely faulted; this
is mainly a package/storage win. The cold-run P95 is not decision-grade.
