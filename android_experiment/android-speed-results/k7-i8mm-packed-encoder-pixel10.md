# K7: I8MM-packed INT8 encoder on Pixel 10

Purpose:

- Move the encoder matrices that were already quality-approved as per-row INT8
  onto the accepted KleidiAI I8MM backend.
- Pack them into the model artifact offline and remove their row-major weights
  and scale tensors, avoiding runtime packing and duplicate layouts.
- Leave the quality-sensitive BF16 encoder matrices unchanged: attention layer
  0 and FFN layers 0–6 still use their existing BF16/OpenBLAS path.

## Implementation

The packed-only model now contains 294 I8MM matrices: 168 decoder matrices and
126 encoder matrices. The encoder set is 92 attention projections from layers
1–23 and 34 FFN projections from layers 7–23. Encoder BF16 biases are decoded
to F32 before being embedded into the packed RHS.

Model detection recognizes the packed layer-18 attention key, because the
row-major tensor used by the previous 1.7B/0.6B shape probe is intentionally
absent from the packed-only artifact.

No experiment flag was added. The packed layout is selected by the model keys,
and packing never occurs at runtime.

## Controlled Pixel 10 result

The A/B used the same candidate binary, thermal status 0, fixed-performance
mode, one engine thread, three repetitions, five Common Voice clips, and the
push-to-talk guardrail. Only the model layout changed between runs.

| Metric | Row-major encoder | I8MM-packed encoder | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | 1765.520 ms | 1579.490 ms | **-10.5%** |
| Common Voice median encoder | 436.549 ms | 237.719 ms | **-45.5%** |
| Common Voice median first-visible token | 1287.001 ms | 1087.225 ms | **-15.5%** |
| Guardrail inference | 2748.832 ms | 2435.752 ms | **-11.4%** |
| Guardrail encoder | 778.586 ms | 456.749 ms | **-41.3%** |
| Guardrail first-visible token | 1851.295 ms | 1541.462 ms | **-16.7%** |
| Peak RSS, Common Voice | 2263656 KiB | 2264236 KiB | +0.03% |
| Peak RSS, guardrail | 2277084 KiB | 2278560 KiB | +0.06% |
| RSS after model load, Common Voice | 44068 KiB | 34584 KiB | **-21.5%** |

The peak-RSS movement is normal run noise and does not indicate a memory
regression. The packed-only artifact is 2027.83 MiB versus approximately
2026.36 MiB for decoder-only packing: +1.47 MiB (+0.07%) from packed alignment
and embedded encoder biases.

All three guardrail runs produced `哈喽，你好嘛？已经坐低咗啦。`. Four of
five Common Voice transcripts matched exactly. The remaining clip changed from
simplified `证据嘅话，真系冇啊！` to traditional `證據嘅話，真係冇啊！`,
which is closer to its reference `證據嘅話真係冇呀`; this is not treated as a
quality regression.

## Decision

Accept. This removes roughly 42–46% of encoder execution time and 15–17% of
first-visible-token latency without a meaningful memory, model-size, or quality
regression.
