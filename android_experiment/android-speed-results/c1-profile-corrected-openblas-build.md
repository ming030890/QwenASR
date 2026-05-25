# C1: profile corrected OpenBLAS build

Purpose:
- Profile the prompt-correct Android/OpenBLAS build to decide whether the next
  work should prioritize BLAS/thread tuning or decoder-only microkernels.

Command:

```bash
OPENBLAS_ANDROID_LIB_DIR=/private/tmp/OpenBLAS \
OPENBLAS_ANDROID_STATIC=1 \
CARGO_FEATURES_EXTRA=android-blas \
RUSTFLAGS_EXTRA="-C target-feature=+dotprod" \
BENCH_LANGUAGE=Auto \
BENCH_PROMPT="音频是粤语，可能夹杂英语" \
bench/android_run.sh \
  --label android-profile-openblas-auto-zhprompt-20260524-214023 \
  --assets-dir /Volumes/SSD/qwen-asr-android-candidate-assets \
  --threads "1" \
  --runs 1 \
  --profile
```

Setup notes:
- Added `bench/android_run.sh --profile` passthrough to expose the benchmark
  binary's existing kernel profiling counters.
- The run used static OpenBLAS linking through
  `OPENBLAS_ANDROID_STATIC=1` and `/private/tmp/OpenBLAS/libopenblas.a`.
  This is the closer Android shipping shape for a single Rust/JNI native
  artifact than relying on an extra runtime `libopenblas.so`.
- Thermal status was `0` before and after the run.

Result:
- Label: `android-profile-openblas-auto-zhprompt-20260524-214023`
  (`bench/android-results/android-profile-openblas-auto-zhprompt-20260524-214023`).
- Flags: `["BLAS", "NEON", "DotProd"]`; threads/runs: `1` / `1`;
  profile enabled.
- Candidate Common Voice median inference: `1715.593 ms`; median encode
  `346.939 ms`; median decode `1365.187 ms`.
- `cv_yue_001.wav`: `1966.265 ms` inference, `401.729 ms` encode,
  `1557.680 ms` decode, transcript `唔做嘢嘅佔多數咯。`.
- `cv_yue_003.wav`: `1464.920 ms` inference, `292.149 ms` encode,
  `1172.693 ms` decode, transcript `你睇下呢部劇。`.
- Guardrail: `2641.500 ms`, transcript `哈喽，你好吗？已经坐低咗啦。`.
  This is semantically/phonetically stable but fails the old exact string check.

Profile counters:
- `cv_yue_001.wav`: `sgemm` `1115.066 ms` (`62.2%` of profiled counter
  time), `bf16_matvec` `426.782 ms` (`23.8%`), `conv2d_op` `152.037 ms`
  (`8.5%`), `attention_causal` `73.733 ms` (`4.1%`).
- `cv_yue_003.wav`: `sgemm` `890.748 ms` (`65.6%`), `bf16_matvec`
  `308.251 ms` (`22.7%`), `conv2d_op` `96.587 ms` (`7.1%`),
  `attention_causal` `45.663 ms` (`3.4%`).
- Guardrail: `sgemm` `1560.111 ms` (`63.4%`), `bf16_matvec` `484.377 ms`
  (`19.7%`), `conv2d_op` `248.036 ms` (`10.1%`), `attention_causal`
  `123.751 ms` (`5.0%`).

Decision:
- The remaining Android gap is still BLAS/encoder-heavy enough that C2
  should be run next: a corrected-prompt OpenBLAS thread sweep.
- Decoder single-token work is still relevant because `bf16_matvec` is the
  second largest measured bucket, but C3 should wait until C2 shows whether a
  production thread setting gives an immediate win.
- Keep static OpenBLAS as the primary benchmark/packaging path for now. Shared
  OpenBLAS can be compared later only if packaging or binary-size tradeoffs
  justify it.
