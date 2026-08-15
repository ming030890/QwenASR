# K9: full linear INT8 for the fine-tuned 0.6B model

Purpose:

- Verify that the accepted offline I8MM-packed decoder and encoder format is
  shape-safe for Qwen3-ASR 0.6B.
- Convert the keyboard's v20 fine-tuned mobile model and gate it on a connected
  Pixel 10 before Play release.

## Compatibility audit

The 0.6B checkpoint has 28 decoder layers and an 18-layer, 896-wide encoder.
The runtime detects those dimensions from model tensors; the packer discovers
weights by layer path and suffix and computes each packed size from its actual
shape. No 1.7B layer count or matrix dimension is reused.

The accepted artifact contains 279 offline-packed matrices:

- 168 decoder projections (`28 × 6`); and
- 111 encoder linear matrices (`18 × 6 + 3` projections).

The tied BF16 embedding is absent. Token lookup and output projection reuse the
existing INT8 `lm_head`. Convolution filters, norms, and biases remain BF16/F32.

## Pixel 10 result

Both candidates used the same KleidiAI I8MM/OpenBLAS 0.3.34 binary, thermal
status 0, fixed-performance mode, one engine thread, and three repetitions.
The row-major v20 mobile model was the control.

| Metric | Row-major control | Decoder packed | Full linear packed | Full delta |
| --- | ---: | ---: | ---: | ---: |
| Common Voice median inference | `1669.854 ms` | `1089.978 ms` | `841.064 ms` | **-49.6%** |
| Common Voice median encoder | `479.797 ms` | `458.855 ms` | `223.477 ms` | **-53.4%** |
| Common Voice median first-visible token | `1584.453 ms` | `1022.319 ms` | `807.342 ms` | **-49.1%** |
| Guardrail inference | `2726.442 ms` | `1831.451 ms` | `1350.083 ms` | **-50.5%** |
| Peak RSS across both suites | `1138656 KiB` | `1137312 KiB` | `961580 KiB` | **-15.6%** |
| Artifact size | `931 MiB` | `933 MiB` | `783 MiB` | **-15.9%** |

## Transcript gate

Decoder-only and full-linear candidates produced identical transcripts on all
six inputs. Therefore encoder quantization introduced no additional output
change. Relative to the row-major control, the packed decoder changed:

- `唔做嘢嘅占多數囉` to `唔做嘢嘅佔多數囉`, exactly matching the
  traditional-Chinese reference; and
- guardrail `坐低咗喇` to `坐低咗啦`, matching the reference particle.

The remaining four Common Voice outputs were unchanged. These are accepted
reference improvements, not semantic regressions.

## Decision

Accept full encoder and decoder linear INT8 packing for the fine-tuned 0.6B
release model. The accepted artifact is:

`/Volumes/SSD/qwen-asr-v20-0.6b-kleidiai-i8mm-full-encoder-packed`
