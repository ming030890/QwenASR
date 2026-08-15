# 036-mac-1.7b-mmap-phase-evict-smoke: Qwen inference experiment

- Status: complete; rejected
- Date (UTC): 2026-08-13T06:46:33+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 035-mac-1.7b-int8-encoder-ffn-l7-23

## Hypothesis

Evicting inactive encoder and decoder mmap pages at phase boundaries prevents their resident working sets from overlapping, materially reducing peak RSS with unchanged transcripts and acceptable page-fault latency.

## Change

Enabled QWEN_MMAP_PHASE_EVICT=1: MADV_DONTNEED decoder/LM tensor pages before encoding and audio-encoder tensor pages before decoder prefill.

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
| Median inference latency | 992.515 ms |
| P95 inference latency | 1191.534 ms |
| Median model load time | 15.567 ms |
| First-launch model load time | 15.567 ms |
| RSS after model load | 42.27 MiB |
| Peak RSS | 2382.56 MiB |
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

Reject full bidirectional phase eviction. The runtime implementation was
reverted after measurement; keep this result as historical evidence.

## Conclusion

Exact transcripts passed, but the hypothesis was not supported. Against the
repeated `035` baseline, peak RSS improved only 1.6% (2420.86 to 2382.56 MiB)
while median latency regressed 15.0% (863.29 to 992.52 ms). Against the closer
single-launch `034` smoke, peak RSS was effectively unchanged. The decoder
working set dominates the peak, and evicting decoder pages before every encoder
call forces costly refaults without removing meaningful peak overlap. Test
encoder-only eviction before abandoning this direction.
