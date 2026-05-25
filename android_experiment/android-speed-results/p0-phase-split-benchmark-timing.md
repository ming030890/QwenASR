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

Implication for the backlog:

- K1/K3 (BF16 NEON GEMM kernels) and E1 (encoder INT8 GEMM) attack both the
  `encoder_ms` slice (~25%) and the `prefill_ms` slice (~50%+) because both
  phases run the same BF16-weight projections through `bf16_matvec` /
  `bf16_matmat`. So K1/K3 is the highest-leverage next item, as the backlog
  already lists.
- P4 (decoder token-loop matvec work) attacks only the `decode_loop_ms` slice
  (~22% on cv_yue_003, ~21% on guardrail). With prefill more than 2× the
  token-loop cost, P4 is correctly demoted: optimize prefill via K1/K3 first.

## Followups

- Future experiments should record the phase split alongside the headline
  inference time. `bench/android_check_gate.py` is unchanged; phase-split is
  additive in the JSON, so no gate behavior changes.
