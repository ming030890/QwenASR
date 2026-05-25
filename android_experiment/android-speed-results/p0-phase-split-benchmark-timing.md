# P0 — Phase-split benchmark timing

## Decision

Accept as infrastructure. P0 is a diagnostic experiment: the goal is to expose
phase-level timings (`mel_ms`, `encoder_ms`, `prefill_ms`, `decode_loop_ms`,
`decode_loop_ms_per_token`, and generated token count) so future kernel/BLAS
work can be attributed to the right pipeline phase. There is no speed
acceptance threshold for P0 itself; the controlled fast-gate confirmed no
regression versus the current baseline.

## Change

- `crates/qwen-asr/src/context.rs`: added `perf_mel_ms`, `perf_encoder_ms`,
  `perf_prefill_ms`, `perf_decode_loop_ms` to `QwenCtx`, initialized in
  `QwenCtx::load` and cleared in `reset_perf()`.
- `crates/qwen-asr/src/transcribe.rs`: accumulate the four phase fields
  alongside the existing `perf_encode_ms` / `perf_decode_ms` in
  `transcribe_segment`, the streaming v1 chunk loop, and the streaming v2
  chunk loop. Mel and encoder forward are now timed independently inside the
  streaming encoder loop so we can attribute audio→encoder cost cleanly.
- `crates/qwen-asr/src/align.rs`: same accumulation for the alignment path.
- `crates/qwen-asr-cli/src/bin/bench_android.rs`: emit `mel_ms`, `encoder_ms`,
  `prefill_ms`, `decode_loop_ms`, `decode_loop_ms_per_token`, and
  `median_tokens` in both the per-run and per-file JSON.

The accumulation is additive (each phase sums into both the legacy bucket and
the new bucket), so existing TSV/JSON consumers continue to work and the new
fields are purely additional.

## Stage 0 / Stage 1 results

Host gate (clean):

```
cargo test --release --test kernels --test audio
4 audio tests pass, 9 kernel tests pass.
```

Android Stage 1 fast gate, label
`android-fast-p0-phase-split-20260525-165425`, fixed-performance, threads=1,
runs=1, fast assets:

| Clip            | inference_ms | mel_ms | encoder_ms | prefill_ms | decode_loop_ms | tokens | ms/token |
| --------------- | -----------: | -----: | ---------: | ---------: | -------------: | -----: | -------: |
| cv_yue_003      |       1672.5 |    3.3 |      403.9 |      893.2 |          371.9 |      7 |     53.1 |
| ptt_guardrail   |       2852.2 |    4.8 |      788.0 |     1459.5 |          599.7 |     13 |     46.1 |

Cross-check against the best Android baseline median for cv_yue_003
(`1677.491 ms`): this run is `1672.5 ms`, well within run-to-run noise.
Guardrail transcript matched (`'哈喽，你好吗？已经坐低咗啦。'`).

JSON:
`bench/android-results/android-fast-p0-phase-split-20260525-165425/`

## Host vs Android phase comparison (threads=1)

Same binary built for darwin-aarch64 and run on the host with the same fast
assets, threads=1, language=Auto, prompt=`音频是粤语，可能夹杂英语`. Median
of 3 runs (run 1 dropped for cold-cache noise).

| Phase                          | macOS cv_yue_003 | Android cv_yue_003 | Android/macOS |
| ------------------------------ | ---------------: | -----------------: | ------------: |
| total inference                |            250.9 |             1672.5 |          6.7x |
| mel                            |              0.2 |                3.3 |         16.5x |
| encoder (bf16_matvec NEON)     |             42.1 |              403.9 |          9.6x |
| prefill (F32 SGEMM via OpenBLAS) |          93.4 |              893.2 |          9.6x |
| decode loop (INT8 SDOT matvec) |            114.2 |              371.9 |          3.3x |

| Phase                          | macOS guardrail | Android guardrail | Android/macOS |
| ------------------------------ | --------------: | ----------------: | ------------: |
| total                          |           385.8 |            2852.2 |          7.4x |
| encoder                        |            75.3 |             788.0 |         10.5x |
| prefill                        |           129.2 |            1459.5 |         11.3x |
| decode loop                    |           181.4 |             599.7 |          3.3x |

Key observation: the gap is **not uniform across phases**.

- `decode_loop` is only `3.3x` slower on Android — the INT8/SDOT matvec path
  is essentially hitting the hardware single-core ceiling on Pixel 9.
- `encoder` (bf16_matvec NEON without BFDOT) and `prefill` (F32 SGEMM via
  OpenBLAS) are `9.6-11.3x` slower — way past raw hardware.

On macOS, `decode_loop` is the *largest* phase (more tokens × faster
per-token math means the loop dominates). On Android, `prefill` is the
largest by far. This is **not the same bottleneck shape**: optimizations
proven on macOS will not translate proportionally to Android.

The dramatic delta between the INT8/SDOT path (~3x) and the
BF16-NEON / F32-SGEMM paths (~10x) is a direct signal: arithmetic
throughput on Pixel 9 single-thread is roughly fine when we use the
right SIMD path, and badly under-utilized when we use the BF16-shift +
F32-FMA path or OpenBLAS F32 SGEMM. The INT8 SDOT kernel proves the
hardware can do better — the encoder/prefill kernels just don't tap it.

## Findings that should steer the next experiments

1. **Mel cost is negligible** (~3–5 ms out of ~400–800 ms of encode). The
   encoder forward pass is essentially all of "encode_ms". Any future encoder
   work (K1/K3/E1) should treat encoder_ms ≈ encode_ms.
2. **Prefill dominates "decode_ms"**, not the autoregressive token loop. On
   cv_yue_003 prefill is `893 ms` (70.6% of decode_ms `1265 ms`) while the
   decode loop is `372 ms`. On the longer guardrail clip prefill is `1459 ms`
   (70.9% of decode_ms `2059 ms`).
3. **Token-loop is fast per token** (~46–53 ms/token), but the absolute decode
   loop cost scales with the number of generated tokens, while prefill scales
   with prefix+encoder-output sequence length, which is the bigger lever right
   now on short ASR utterances.
4. The encoder-forward share of total inference (`404 / 1673 = 24%` on
   cv_yue_003, `788 / 2852 = 28%` on guardrail) is meaningful but smaller than
   prefill (`53%` and `51%` respectively).

Implication for the backlog (revised after host comparison):

- **K4 (new) — INT8 SDOT packed matmat for decoder prefill** is the new
  rank-1 item. Decoder weights are already INT8 on disk (Q2B); single-token
  decode already runs an INT8 SDOT matvec at only `3.3x` macOS speed; the
  prefill path instead dequantizes to F32 and routes through OpenBLAS SGEMM,
  which is `9.6-11.3x` macOS speed. K4 reuses the proven SDOT machinery to
  attack `~50%` of total inference with no weight-format change.
- K1 (BFDOT/BFMMLA BF16 kernel) drops to rank 2. It attacks `encoder_ms`
  (~25% of total) and requires both runtime BF16 feature detection and a new
  kernel; K4 attacks a larger slice with code that closely mirrors the
  already-shipped INT8 decode-loop matvec.
- E1 (encoder INT8 weight pre-quantization + INT8 GEMM) becomes a possible
  K1 alternative: if K1 brings encoder within ~2x of macOS, E1 is unneeded;
  if K1 leaves a gap, E1 + K4's same INT8 GEMM machinery is a natural
  follow-up.
- P4 (decoder token-loop matvec) stays demoted. Token loop is only ~22% of
  total inference, and the Android/macOS ratio for that phase is already
  `3.3x` — close to raw hardware. Little room to win without changing the
  algorithm.

## Followups

- Future experiments should record the phase split alongside the headline
  inference time. `bench/android_check_gate.py` is unchanged; phase-split is
  additive in the JSON, so no gate behavior changes.
