# 038-mac-1.7b-sized-decoder-scratch-smoke: Qwen inference experiment

- Status: complete; retained as allocation hygiene, rejected as an RSS optimization
- Date (UTC): 2026-08-13T06:52:41+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 035-mac-1.7b-int8-encoder-ffn-l7-23

## Hypothesis

Sizing decoder F32 prefill scratch to the active 256-row INT8 tile removes about 90 MiB of reserved scratch and reduces peak RSS without changing computation or transcripts.

## Change

DecoderBuffers now derives F32 scratch capacity from loaded precision and QWEN_INT8_PREFILL_TILE_ROWS; baseline INT8 capacity is 6 MiB instead of the legacy 96 MiB.

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-encoder-ffn-l7-23`
- Detected variant: 1.7B
- Model inventory fingerprint: `e8c08bf628a1dcafe2f16db54ee7a52d3f65415c3f6643b646c69b3e42c35f4a`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 887.892 ms |
| P95 inference latency | 1043.156 ms |
| Median model load time | 16.171 ms |
| First-launch model load time | 16.171 ms |
| RSS after model load | 48.31 MiB |
| Peak RSS | 2390.75 MiB |
| Model files | 2118.05 MiB |
| Weight files | 2113.73 MiB |
| Runtime binary | 1.69 MiB |

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

Reject the hypothesis as an RSS optimization. Retain the correctly sized
allocation because it removes unnecessary virtual capacity without changing
the inference path.

## Conclusion

Exact transcripts were preserved, but peak RSS was 2390.75 MiB versus 2380.97
MiB in the comparable one-launch `034` smoke run. Median inference was also
within smoke-run noise (887.89 versus 880.07 ms). The old 96 MiB zero-filled
buffer was largely backed by untouched lazy pages, so reducing its capacity to
6 MiB did not translate into resident-memory savings. Keep the bounded sizing
for allocation hygiene; do not attribute an RSS win to it.
