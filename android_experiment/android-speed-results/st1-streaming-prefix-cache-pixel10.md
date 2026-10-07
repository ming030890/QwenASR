# ST1: Streaming Static-Prefix KV Cache on Pixel 10

## Decision

- **Accept** cross-utterance reuse of the invariant nine-token prompt KV cache.
- **Reject** pinning the one-thread streaming worker to the highest-frequency CPU tier.

## Setup

- Device: Pixel 10 (`53241FDCR00141`), normal performance mode.
- Model: `/Volumes/SSD/Confucius4-R2T2-kleidiai-packed`.
- Runtime: one thread, OpenBLAS + KleidiAI, production VAD/streaming defaults.
- Quality sample: 35 evenly selected clips from the 359-clip ASR2 field-speech corpus.

## Prime-core affinity screen

The Pixel topology exposed CPU 7 as the sole 3.782 GHz tier. A same-binary A/B/B/A test compared normal scheduling with `taskset 80`:

| Mode | Mean RTF | Result |
| --- | ---: | --- |
| Unpinned | 1.370 | Control |
| CPU 7 only | 1.394 | 1.7% slower |

Transcripts were identical. The source candidate was reverted.

## Static-prefix cache screen

The cache snapshots/restores the decoder KV for the invariant prompt prefix and survives no-op `set_prompt("")` calls. It is invalidated when the prompt changes.

| Metric | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Warm first-chunk prefill | 388 ms | 279 ms | **-28.1%** |
| Four-clip mean RTF | 1.334 | 1.309 | -1.9% |
| Cache memory | 0 | 1.97 MiB | +0.09% process RSS |

### Quality gate

- Hypotheses: byte-identical on all 35 clips.
- CER: 42 / 479 = **8.77%** for both.
- Exact match: **18 / 35** for both.
- Host tests: 14 library tests passed; CLI benchmark check passed.

The first-token improvement clears the 5% target without a CER change or significant memory increase.

## Follow-up screens

Only candidates that improved the four-clip speed screen were advanced to the 35-clip quality gate.

| Candidate | Speed result | Quality/latency result | Decision |
| --- | ---: | --- | --- |
| Stream chunk 0.72 s | RTF 1.209 vs 1.308 | Same 35-clip aggregate CER, but speech-relative TTFC regressed by 110 ms and hypotheses changed | Reject |
| Reused stream scratch buffers | RTF 1.341 and 1.411 vs 1.309 | Four-clip hypotheses unchanged | Reject |
| Encode every other chunk | RTF 1.019 and 1.055 vs 1.309 | Stable-text TTFC regressed by 304 ms | Reject |
| Two inference threads | RTF 2.008 vs about 1.309 | Four-clip CER unchanged | Reject |

All rejected source/configuration experiments were reverted. The accepted implementation has no Pixel model, CPU number, or device-specific constant.
