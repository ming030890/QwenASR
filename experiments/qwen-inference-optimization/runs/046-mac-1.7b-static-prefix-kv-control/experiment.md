# 046-mac-1.7b-static-prefix-kv-control: Qwen inference experiment

- Status: complete; same-revision control
- Date (UTC): 2026-08-15T08:26:23+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: none

## Hypothesis

A same-revision control establishes cold and warm first-visible-token latency without static-prefix KV reuse

## Change

Add first-visible-token instrumentation; leave static-prefix KV reuse disabled

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-attn-l1`
- Detected variant: 1.7B
- Model inventory fingerprint: `d86b979db69b78bf8b579ad5b80fd01b43f1d1a9ecd65fd51b84bf6f93cc6ad6`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 865.276 ms |
| P95 inference latency | 1035.119 ms |
| Median model load time | 14.420 ms |
| First-launch model load time | 14.550 ms |
| RSS after model load | 49.12 MiB |
| Peak RSS | 2338.11 MiB |
| Model files | 2026.42 MiB |
| Weight files | 2022.10 MiB |
| Runtime binary | 1.70 MiB |

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
| Baseline repetitions identical | n/a |
| Same fixed input set | n/a |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

Use as the same-revision instrumentation control for run `047`.

## Conclusion

This run establishes the no-cache control using the same binary, model, and
first-visible-token instrumentation as the candidate. Across 54 inferences,
median prefill was 319.68 ms, median first-visible-token latency was 576.52 ms,
median total inference was 865.28 ms, and P95 inference was 1035.12 ms. The
three cold first-visible-token measurements were 792.86, 786.36, and 783.87 ms.
