# 019-mac-1.7b-prequant-int8-smoke: Qwen inference experiment

- Status: measured; retained as QAT target, not quality-accepted
- Date (UTC): 2026-08-13T05:46:45+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 018-mac-1.7b-no-bf16-gate-up

## Hypothesis

A pre-quantized mmap-backed INT8 decoder and LM head eliminate owned runtime quantization buffers and BF16 decoder residency, greatly reducing 1.7B RSS

## Change

Stream-export all decoder projections and tied LM head as row-wise INT8; retain BF16 token embeddings and audio encoder; load INT8 weights zero-copy from mmap

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
| Median inference latency | 1670.429 ms |
| P95 inference latency | 8525.306 ms |
| Median model load time | 143.616 ms |
| First-launch model load time | 143.616 ms |
| RSS after model load | 41.84 MiB |
| Peak RSS | 2526.95 MiB |
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
| Exact transcript equality | **FAIL** |
| Candidate repetitions identical | PASS |
| Baseline repetitions identical | PASS |
| Same fixed input set | PASS |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Do not quality-graduate: the BF16 exact-transcript gate failed. Retain this run
as the prequantized INT8 memory and latency baseline for QAT-directed work.

## Conclusion

Prequantized mmap INT8 cut peak RSS from 5875.38 MiB (`018`) to 2526.95 MiB
(57.0%) and model files from 4485.24 MiB to 2847.21 MiB (36.5%). Native INT8
prefill regressed median latency from 918.76 ms to 1670.43 ms, concentrated in
prefill (371.92 ms to 1084.39 ms). Five clips matched BF16; `ptt_guardrail.wav`
changed `Hello` to `哈喽`. Per project direction, this mismatch is recorded as
QAT debt rather than a reason to abandon the INT8 representation.
