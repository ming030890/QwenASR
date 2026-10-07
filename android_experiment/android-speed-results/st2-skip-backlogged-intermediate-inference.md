# ST2: Skip Backlogged Intermediate Inference

## Decision

**Accept** deferring replaceable intermediate ASR inference while the Android worker has at least 200 ms of newer audio queued. VAD and endpointing still process every frame, and the newest chunk or finalization always runs inference.

The threshold is expressed as an audio duration (`SAMPLE_RATE / 5`), with no Pixel, CPU, or device-model hardcoding.

## Pixel 10 speed screen

Four ASR2 clips were replayed in real time so audio continued to accumulate while inference ran.

| Metric | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Real-time factor | 1.713 | 1.664 | **-2.9%** |
| CER | 12.12% | 12.12% | No change |
| First visible hypothesis, mean wall time | 2,337 ms | 2,342 ms | +5 ms |
| Second visible hypothesis, mean wall time | 3,152 ms | 3,160 ms | +8 ms |
| Persistent memory | 0 | 0 | No change |

The first two visible updates were within 13 ms on every clip. The candidate avoids inference only after meaningful queue growth; a full-chunk 600 ms threshold did not trigger enough to improve speed, while deferring for any non-empty queue made updates unnecessarily coarse.

## 35-clip quality gate

The candidate was replayed in real time on the same evenly selected ASR2 sample used by ST1.

| Metric | ST1 control | ST2 candidate |
| --- | ---: | ---: |
| Edits / reference characters | 42 / 479 | **40 / 479** |
| CER | 8.77% | **8.35%** |
| Exact match | 18 / 35 | **19 / 35** |

Four final hypotheses changed. Aggregate CER improved rather than regressed. The implementation adds only an atomic queue-length load and comparison; it introduces no persistent allocation.
