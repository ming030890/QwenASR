# Sequential v18 INT8 measurements — 2026-09-06

Android Pixel 10 (53241FDCR00141) is the acceptance target. Mac M4 measurements are development checks. Language is Cantonese, custom prompt is absent, Chinese variant conversion is disabled, and segmentation is disabled. Experiments are evaluated individually. Existing uncommitted work was preserved in starting-worktree.patch; it must not be discarded with a blanket checkout.

## Initial baseline

Exact model: `/Volumes/SSD/qwen3_asr_17b_staged_qat_v18_pack/int8_exports/stage2a_qat_projector_checkpoint-500`.
The model SHA-256 is `037e71b7a2efb9a5132ab1ac0b6ac4ac4468c12c80256a303111b57eb7e74092`, verified on host and device. The pre-existing device model was a different export, so these runs use `/data/local/tmp/qwen-sequential-v18/model`.

Android build: NDK 29.0.14206865, API 35, `-C target-feature=+dotprod`, `android-blas`, static OpenBLAS v0.3.34 built for ARMV8 with USE_THREAD=0. Runtime reports OpenBLAS, NEON, DotProd. Control switches: CONV_REUSE=0, LM_ROWS=2, QKV_REUSE=0.

Four repetitions per sample; the first is excluded from warm medians.

| Metric | Pixel 10, 1 thread, ptt_guardrail.wav | Mac M4, 4 threads, audio.wav |
|---|---:|---:|
| Warm inference ms | 6775.957 | 2261.039 |
| Encoder ms | 1147.002 | 319.542 |
| Prefill ms | 3998.255 | 792.030 |
| Decode loop ms | 1625.570 | 1145.985 |
| Decode ms/token | 147.779079 | 26.045109 |
| Warm median peak RSS KiB | 2207036 | 2366176 |
| Startup ms | 47.042 | 16.305 |
| First inference phase ms | 7651.412 | 2450.619 |
| First wall inference plus startup ms | 7997.197 | 2544.852 |

These are process-first timings, not cold-cache timings; filesystem caches were not evicted. The five Common Voice samples were also run as a smoke test, but a model hash verification overlapped part of that suite, so it is not a clean latency acceptance reference. The ptt_guardrail run followed after verification completed.

Pixel transcript was identical on all four repetitions: `Hello 你好嘛，已經坐低咗喇`. This differs from the pasted reference string. The Mac likewise produces a longer transcript under the requested settings. The historical timing and exact-text references are not directly interchangeable with this measured configuration.

Raw Pixel results: `../android-results/sequential-b0/ptt_guardrail-t1.json`. Raw Mac results: `b0-mac.json`. Summaries: `baseline-summary.json`; provenance: `baseline-provenance.json`.

## Export compatibility audit

The actual export contains 313 I8 matrices with matching positive finite F32 per-output-row scales: 144 encoder projections, 168 decoder projections, and the tied output head. All matrix shapes and signed integer ranges (-127 through 127) passed validation. There are also 366 BF16 tensors. Convolutions, conv_out, audio projector matrices, biases, and norms remain floating point.

The runtime maps the I8 matrices directly. Encoder and decoder prefill dequantize matrix tiles to F32 and call floating-point GEMM. Decoder token projections and output-head selection use dynamic activation quantization plus INT8 dot products. Missing tied token embeddings are reconstructed row-wise from the quantized output head. The exporter writes gate/up rows and scales in alternating order, matching the decoder's SwiGLU layout.

Therefore the current export is supported; this is not end-to-end integer-only inference. Arbitrary future exports that also quantize the BF16 conv_out/projector matrices need additional support: the current encoder load sites discard the unpacked I8 alternatives for those matrices.

`crates/qwen-asr/tests/int8_export.rs` is an opt-in real-model test. It checks all encoder and decoder layer storage paths, then compares actual-model embedding lookup, Q/K/V, SwiGLU, and output-head argmax against scalar references. It passed on the Mac; the Android test executable has been compiled and awaits the end of the device's current baseline run.

## Accuracy baseline and E2 status

The supplied manifests actually contain 363 pure Cantonese and 126 mixed-script samples, all with unique IDs and present audio files. Both manifest hashes and all audio hashes are retained. Scoring follows the staged evaluator: Chinese characters and punctuation preserved, Latin casefolded, English/alphanumeric words counted as single units. Raw predictions are retained.

The complete Android baseline is in progress (`b0-eval.log`, `b0-eval/`). No accuracy acceptance decision has been made.

E2 is the only optimization under examination. The existing once-per-utterance convolution conversion has been enabled by default while preserving CONV_REUSE=0 as the control. `conv_reuse.rs` passed exact encoder-output comparisons on seven input lengths, including full chunks, single-token tails, and reused scratch buffers across shorter subsequent utterances.

Mac E2 preliminary warm medians: total 2254.429 ms, encoder 311.029 ms, prefill 792.181 ms, decode 1143.628 ms. Transcript unchanged. Encoder improvement: 2.66%; total improvement: 0.29%. Mac RSS varied upward; Android measurements decide acceptance. Full host B0/E2 eval comparisons are running. E2 has not been accepted or committed. A1 and later experiments have not started.
