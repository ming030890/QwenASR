# 025-mac-1.7b-full-int8-tiled256-prefill: Qwen inference experiment

- Status: complete; current 1.7B optimization baseline
- Date (UTC): 2026-08-13T05:57:12+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: 021-mac-1.7b-prequant-int8-f32-prefill

## Hypothesis

Repeated runs confirm that full decoder INT8 plus a 256-row prefill dequantization tile reduces peak RSS and storage materially while keeping latency near the optimized BF16-prefill path.

## Change

Omitted duplicate BF16 token embeddings, reused tied rowwise INT8 LM head for lookup, and limited prefill F32 dequantization scratch to 256 output rows.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-full`
- Detected variant: 1.7B
- Model inventory fingerprint: `e2b965c2ccbcda3776e8555aae2885fdb6ad89c9ddb99b9b3b2f66396d4b9f82`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 869.264 ms |
| P95 inference latency | 1030.208 ms |
| Median model load time | 15.900 ms |
| First-launch model load time | 15.900 ms |
| RSS after model load | 41.64 MiB |
| Peak RSS | 2560.17 MiB |
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

Adopt as the new baseline for subsequent Qwen3-ASR 1.7B optimization
experiments. Future candidates must name
`025-mac-1.7b-full-int8-tiled256-prefill` as their baseline and preserve its
configuration. Do not production-graduate until QAT restores the separate BF16
quality reference and Android measurements pass.

## Conclusion

Across 54 inferences, median latency was 869.26 ms, P95 1030.21 ms, and peak RSS
2560.17 MiB. Against optimized BF16 `018`, this is 5.4% faster median, 2.7%
faster P95, 56.4% lower peak RSS, and 49.8% smaller model files. Against the
prior optimized INT8 model `021`, it is 3.4% faster median, 3.6% lower peak RSS,
and 20.8% smaller on disk. All candidate repetitions were stable and matched
the prior INT8 transcript baseline; one of six clips still differs from BF16
and remains QAT debt. Next: run the same candidate on the target Android device,
then use QAT to recover the BF16 transcript before production acceptance.

## Baseline declaration

- Performance and RSS baseline: `025-mac-1.7b-full-int8-tiled256-prefill`
- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-full`
- Required environment: `QWEN_INT8_PREFILL_F32=1`,
  `QWEN_INT8_PREFILL_TILE_ROWS=256`
- BF16 quality reference only: `018-mac-1.7b-no-bf16-gate-up`
