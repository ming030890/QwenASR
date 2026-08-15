# 052-mac-m4-kleidiai-offline-packed: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): 2026-08-15T09:11:53+00:00
- Platform: Mac mini / Apple M4
- Git revision: `18391d69addf19f8fffa8e124856740263c3dfc2` (dirty)
- Baseline: experiments/qwen-inference-optimization/runs/048-mac-m4-kleidiai-control

## Hypothesis

Offline-packed KleidiAI weights eliminate runtime packing and reduce first-visible-token latency

## Change

Load a 1.35 GiB DotProd-packed safetensors shard and use it directly for all decoder prefill projections

## Fixed setup

- Model: `/Volumes/SSD/Qwen3-ASR-1.7B-int8-kleidiai-dotprod`
- Detected variant: 1.7B
- Model inventory fingerprint: `c66474786ecece26a914a358bac08958ebde045c332b72d04b50fe682732a96e`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets`
- Threads: 1
- Launches × runs: 3 × 3
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 790.448 ms |
| P95 inference latency | 1112.087 ms |
| Median first-visible-token latency | 571.548 ms |
| Median decoder prefill | 313.353 ms |
| Median model load time | 21.223 ms |
| First-launch model load time | 21.223 ms |
| RSS after model load | 49.22 MiB |
| Peak RSS | 3668.75 MiB |
| Model files | 3377.00 MiB |
| Weight files | 3372.69 MiB |
| Runtime binary | 1.72 MiB |

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

**REJECT as the Mac first-token default.** Offline packing works and exact
transcripts pass, but TTFT, prefill, P95, memory, and model size regress.

## Conclusion

The packed model successfully eliminates runtime RHS packing and preserves all
54 transcripts. It does not improve the target metric on Apple M4: versus run
048, median first-visible-token latency regressed 11.5% (512.705 to 571.548
ms), prefill regressed 21.8%, and P95 inference regressed 15.3%. Median total
inference improved 3.3%, but that mixed-input aggregate does not offset the
direct TTFT regression and fails the 5% acceptance threshold. Peak RSS rose
57.9% (2323.81 to 3668.75 MiB), and model files rose 66.6% because the original
row-major INT8 layout is retained for single-token decode alongside the 1.35
GiB packed shard.

Hot single-projection microbenchmarks overstated the benefit by repeatedly
serving the same weights from cache. Full decoder prefill streams 168 distinct
projections and is the authoritative measurement. KleidiAI remains an opt-in
research path for Android hardware, where the historical custom SDOT result was
positive, but the accepted Mac default remains tiled F32/OpenBLAS.
