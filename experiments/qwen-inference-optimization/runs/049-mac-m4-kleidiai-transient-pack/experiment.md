# 049-mac-m4-kleidiai-transient-pack: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-15T09:01:32+00:00
- Platform: Mac mini / Apple M4
- Git revision: `18391d69addf19f8fffa8e124856740263c3dfc2` (dirty)
- Baseline: experiments/qwen-inference-optimization/runs/048-mac-m4-kleidiai-control

## Hypothesis

Packed INT8 DotProd projections reduce first-visible-token latency even with transient RHS packing

## Change

Enable opt-in KleidiAI F32-activation x per-channel-INT8 prefill with transient weight packing

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
| Median inference latency | 1530.542 ms |
| P95 inference latency | 1841.908 ms |
| Median first-visible-token latency | 1302.562 ms |
| Median decoder prefill | 1048.879 ms |
| Median model load time | 14.911 ms |
| First-launch model load time | 14.911 ms |
| RSS after model load | 49.11 MiB |
| Peak RSS | 2355.17 MiB |
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
| Baseline repetitions identical | PASS |
| Same fixed input set | PASS |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

**REJECT transient runtime packing.** Exact transcripts pass, but the target
latency regresses far beyond the acceptance threshold.

## Conclusion

Packing every decoder projection for every prefill is not practical. Against
run 048, median first-visible-token latency regressed 154.1% (512.705 to
1302.562 ms), prefill regressed 307.8%, median inference regressed 87.1%, and
P95 regressed 90.9%. Decode-loop latency remained effectively flat and every
transcript matched, isolating repeated RHS packing as the failure. Run 052
moves packing offline into the model artifact.
