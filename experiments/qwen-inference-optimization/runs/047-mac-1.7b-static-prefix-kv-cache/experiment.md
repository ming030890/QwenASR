# 047-mac-1.7b-static-prefix-kv-cache: Qwen inference experiment

- Status: complete; accepted on Mac pending Android validation
- Date (UTC): 2026-08-15T08:27:22+00:00
- Platform: Mac mini / Apple M4
- Git revision: `fe1d46a8b50d25ff80c47b89ca95478babdc54f5` (dirty)
- Baseline: 046-mac-1.7b-static-prefix-kv-control

## Hypothesis

Reusing an exact compact KV snapshot for the invariant prompt prefix reduces warm first-visible-token and decoder-prefill latency by at least 5% without changing transcripts

## Change

Restore cached K/V for PREFIX_HEAD plus prompt plus PREFIX_TAIL and prefill only the audio-dependent suffix after the first utterance

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
| Median inference latency | 819.599 ms |
| P95 inference latency | 968.708 ms |
| Median model load time | 13.964 ms |
| First-launch model load time | 13.919 ms |
| RSS after model load | 49.17 MiB |
| Peak RSS | 2344.59 MiB |
| Model files | 2026.42 MiB |
| Weight files | 2022.10 MiB |
| Runtime binary | 1.70 MiB |

Phase comparison:

| Metric | Control `046` | Candidate `047` | Delta |
| --- | ---: | ---: | ---: |
| Median first-visible-token latency | 576.52 ms | 509.91 ms | **-11.6%** |
| Warm first-visible-token latency | 574.99 ms | 510.25 ms | **-11.3%** |
| Median decoder prefill | 319.68 ms | 256.69 ms | **-19.7%** |
| Warm decoder prefill | 320.13 ms | 257.84 ms | **-19.5%** |
| Median total inference | 865.28 ms | 819.60 ms | **-5.3%** |
| P95 total inference | 1035.12 ms | 968.71 ms | **-6.4%** |
| Peak RSS | 2338.11 MiB | 2344.59 MiB | +0.3% |

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

Accept and enable by default. Set `QWEN_STATIC_PREFIX_KV_CACHE=0` only for a
diagnostic rollback comparison.

## Conclusion

The hypothesis is supported. Reusing the exact static prompt-prefix KV state
reduced median first-visible-token latency by 11.6%, decoder prefill by 19.7%,
median total inference by 5.3%, and P95 inference by 6.4%. All 54 transcripts
matched the control exactly. The compact K/V snapshot added 6.48 MiB to peak
RSS (0.3%), which is well below the experiment's 10% memory threshold.

Cold measurements do not claim a cache benefit: the median cold
first-visible-token result changed only from 786.36 to 781.27 ms, within normal
noise. The measured gain is deliberately a same-session warm-utterance gain.
The benchmark's new `first_visible_token_ms` starts before normalization and
silence compaction and stops at the first complete UTF-8 text piece, so it is a
closer user-facing TTFT metric than the older `prefill_ms` proxy.

The cache is enabled by default. The next Android run should compare the default
against `QWEN_STATIC_PREFIX_KV_CACHE=0`; retain the environment flag as a
diagnostic escape hatch.
