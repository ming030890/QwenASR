# K8: full encoder linear INT8 on Pixel 10

Purpose:

- Quantize every remaining BF16 encoder matrix that maps to the accepted
  KleidiAI I8MM backend.
- Stop expanding the scope as soon as an exact transcript changes.

## Accepted scope

K8 adds 21 offline-packed matrices to K7:

- layer-0 attention Q/K/V/output projections;
- FFN FC1/FC2 projections in layers 0–6;
- the convolution-output projection; and
- final encoder projections 1 and 2.

The artifact contains 315 packed matrices: 147 encoder and 168 decoder. All
encoder linear weights are now INT8 and packed offline. The three convolution
filters, normalization vectors, and biases remain BF16/F32; they are not linear
matrix weights.

## Controlled Pixel 10 result

The A/B used one binary, thermal status 0, fixed-performance mode, one engine
thread, three repetitions, five Common Voice clips, and the push-to-talk
guardrail. The K7 artifact was the control.

| Metric | K7 partial encoder INT8 | K8 full linear INT8 | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | 1574.075 ms | 1500.350 ms | **-4.7%** |
| Common Voice median encoder | 238.371 ms | 193.163 ms | **-19.0%** |
| Common Voice median first-visible token | 1088.979 ms | 1017.995 ms | **-6.5%** |
| Guardrail inference | 2428.365 ms | 2314.875 ms | **-4.7%** |
| Guardrail encoder | 457.682 ms | 339.112 ms | **-25.9%** |
| Guardrail first-visible token | 1536.008 ms | 1416.941 ms | **-7.8%** |
| Peak RSS, Common Voice | 2264128 KiB | 2145332 KiB | **-5.2%** |
| Peak RSS, guardrail | 2277172 KiB | 2205788 KiB | **-3.1%** |
| Artifact size | 2027.83 MiB | 1957.83 MiB | **-3.5%** |

All five Common Voice transcripts and the guardrail transcript match K7
exactly. K8 is accepted.

## Rejected convolution extension

A follow-up candidate also quantized and packed all three convolution filters.
All six outputs became empty, so the experiment was stopped immediately under
the exact-transcript rule. Its runtime and packing changes were reverted. The
BF16 convolution filters total about 8 MiB and remain in the accepted model.

## Decision

Accept full encoder **linear** quantization. Reject convolution-filter
quantization. The accepted artifact is:

`/Volumes/SSD/Qwen3-ASR-1.7B-int8-kleidiai-i8mm-full-encoder-packed`
