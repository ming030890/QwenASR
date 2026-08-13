# infrastructure-mac-collector-0.6b: Qwen inference experiment

- Status: complete (infrastructure validation only)
- Date (UTC): 2026-08-12T07:14:06+00:00
- Platform: Mac mini / Apple M4
- Git revision: `46e7d709eddd479782bc24460f95bd2cc4716f75` (dirty)
- Baseline: none

## Hypothesis

The Mac collector records all four target metric groups and preserves the transcript

## Change

Infrastructure-only validation using the locally available 0.6B model; this is not the 1.7B performance baseline

## Fixed setup

- Model: `/Volumes/SSD/qwen3-asr-0.6b`
- Detected variant: 0.6B
- Model inventory fingerprint: `dfb01692182f4dc6b9784380bd00b106a5b908340d811a2913368c693dcfdcb8`
- Input: `/Users/tony/Documents/keyboard/QwenASR/bench/android-assets/ptt_guardrail.wav`
- Threads: 1
- Launches × runs: 1 × 1
- Language: Auto
- Prompt: 音频是粤语，可能夹杂英语

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | 493.177 ms |
| P95 inference latency | 493.177 ms |
| Median model load time | 332.191 ms |
| First-launch model load time | 332.191 ms |
| RSS after model load | 3753.70 MiB |
| Peak RSS | 4236.97 MiB |
| Model files | 1793.42 MiB |
| Weight files | 1789.18 MiB |
| Runtime binary | 1.65 MiB |

Unique transcripts:

- `哈喽，你好吗？已经坐低咗啦。`

Full normalized measurements: [`metrics.json`](metrics.json).

This smoke predates the exact-baseline transcript gate and therefore cannot be
used as an accepted optimization result. It only validates metric collection.

## Decision

Keep the collector infrastructure. Do not use these 0.6B measurements as the
1.7B optimization baseline.

## Conclusion

The collector successfully recorded inference and phase latency, model loading
time, memory, model size, runtime size, host/revision metadata, and the expected
guardrail transcript. This was a single-launch smoke test on the locally
available 0.6B model, so its numbers are only proof that the loop works. The next
experiment is the controlled Mac mini baseline using Qwen3-ASR 1.7B.
