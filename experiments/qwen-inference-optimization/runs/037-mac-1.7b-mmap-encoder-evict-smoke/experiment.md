# 037-mac-1.7b-mmap-encoder-evict-smoke: Qwen inference experiment

- Status: complete; rejected
- Date (UTC): 2026-08-13T06:47:17+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 035-mac-1.7b-int8-encoder-ffn-l7-23

## Hypothesis

Evicting only audio-encoder mmap pages after encoder forward captures any phase-overlap RSS reduction without repeatedly refaulting the much larger decoder.

## Change

Set QWEN_MMAP_PHASE_EVICT=encoder; apply MADV_DONTNEED only to audio-encoder tensor pages after encoding.

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
| Median inference latency | 926.784 ms |
| P95 inference latency | 1077.710 ms |
| Median model load time | 15.370 ms |
| First-launch model load time | 15.370 ms |
| RSS after model load | 42.25 MiB |
| Peak RSS | 2369.22 MiB |
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

Reject as a default optimization. The runtime implementation was reverted after
measurement; retain this run for diagnostics and future platform comparison.

## Conclusion

Exact transcripts passed. Against repeated baseline `035`, peak RSS improved
only 2.1% (2420.86 to 2369.22 MiB) while median latency regressed 7.4%. Against
the comparable `034` smoke, peak RSS improved just 0.5% and latency regressed
5.3%. macOS already keeps little encoder/decoder mmap overlap at the observed
peak, so explicit encoder eviction is not worthwhile. Proceed to reducing
allocated decoder scratch or the decoder working set itself.
