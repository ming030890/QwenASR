# QwenASR Android Improvement Experiments Index

This ledger lists completed and active on-device optimization experiments on Android.

## Completed Result Index

Each completed or evaluated experiment has its own markdown file in the `android-speed-results/` folder detailing the changes, benchmarks, and target metrics.

| ID | Summary | Decision | Result File |
| --- | --- | --- | --- |
| **A0** | Android benchmark binary, fixed assets, and `android_run.py` harness. | Infrastructure | [a0-android-benchmark-harness-and-assets.md](android-speed-results/a0-android-benchmark-harness-and-assets.md) |
| **H1** | macOS host reference baseline measurements. | Reference | [h1-macos-host-baseline-reference.md](android-speed-results/h1-macos-host-baseline-reference.md) |
| **A1** | Legal no-dotprod Android build. | Build fix | [a1-legal-no-dotprod-android-baseline-build.md](android-speed-results/a1-legal-no-dotprod-android-baseline-build.md) |
| **A2** | Dotprod compiler architecture flags (`+dotprod`). | Keep flag | [a2-dotprod-build-feature-controlled-check.md](android-speed-results/a2-dotprod-build-feature-controlled-check.md) |
| **A3** | Baseline controlled rerun (identifies thermal noise). | Infrastructure | [a3-baseline-controlled-rerun-thermal-contaminated.md](android-speed-results/a3-baseline-controlled-rerun-thermal-contaminated.md) |
| **B1** | Clean fast-gate baseline vs dotprod check. | Filter test | [b1-clean-fast-gate-baseline-vs-dotprod.md](android-speed-results/b1-clean-fast-gate-baseline-vs-dotprod.md) |
| **B11**| OpenBLAS static linking setup. | Keep static link | [b11-android-openblas-static-benchmark-arm.md](android-speed-results/b11-android-openblas-static-benchmark-arm.md) |
| **B2** | Candidate-gate baseline vs dotprod check. | Filter test | [b2-candidate-gate-baseline-vs-dotprod.md](android-speed-results/b2-candidate-gate-baseline-vs-dotprod.md) |
| **C1** | Profiled corrected Android/OpenBLAS build. | Diagnostic | [c1-profile-corrected-openblas-build.md](android-speed-results/c1-profile-corrected-openblas-build.md) |
| **C2** | Fixed-performance baseline and thread sweeps. | Keep threads=1 | [c2-openblas-engine-thread-sweep-fixedperf.md](android-speed-results/c2-openblas-engine-thread-sweep-fixedperf.md) |
| **C3** | ARMV8 static OpenBLAS build parameter isolation. | Reject dynamic-arch | [c3-openblas-ndk29-armv8-rebuild.md](android-speed-results/c3-openblas-ndk29-armv8-rebuild.md) |
| **Q1** | Sidecar caching logic for key tokens. | Reject | [q1-decoder-int8-sidecar-cache.md](android-speed-results/q1-decoder-int8-sidecar-cache.md) |
| **Q2** | Pre-quantize decoder weights to INT8 on host. | Reject | [q2-decoder-int8-weights.md](android-speed-results/q2-decoder-int8-weights.md) |
| **Q2B**| Zero-copy mmap INT8 decoder weights. | Accept | [q2b-int8-mmap-zero-copy.md](android-speed-results/q2b-int8-mmap-zero-copy.md) |
| **M1** | Transient prefill dequantization into shared scratch. | Accept | [m1-transient-prefill-dequant.md](android-speed-results/m1-transient-prefill-dequant.md) |
| **M2** | Lazy/dynamic KV cache allocation. | Reject | [m2-lazy-kv-cache.md](android-speed-results/m2-lazy-kv-cache.md) |
| **M3** | Transient BF16 encoder conversion. | Reject | [m3-transient-bf16-encoder.md](android-speed-results/m3-transient-bf16-encoder.md) |
| **M4** | Tied `lm_head` host-side pre-quantization (zero-copy lm_head).| Accept | [m4-tied-lm-head-int8.md](android-speed-results/m4-tied-lm-head-int8.md) |
| **P0** | Phase-split benchmark timing instrumentation. | Infrastructure | [p0-phase-split-benchmark-timing.md](android-speed-results/p0-phase-split-benchmark-timing.md) |
| **K4** | INT8 SDOT packed matmat prefill activation quantization. | Accept | [k4-int8-prefill-sdot.md](android-speed-results/k4-int8-prefill-sdot.md) |
| **K5** | KleidiAI offline-packed decoder on Pixel 10. | Accept | [k5-kleidiai-offline-packed-pixel10.md](android-speed-results/k5-kleidiai-offline-packed-pixel10.md) |
| **K6** | I8MM decoder, fat LTO, and OpenBLAS 0.3.34 follow-ups. | Accept | [k6-i8mm-lto-openblas-pixel10.md](android-speed-results/k6-i8mm-lto-openblas-pixel10.md) |
| **K7** | Offline I8MM packing for already-INT8 encoder matrices. | Accept | [k7-i8mm-packed-encoder-pixel10.md](android-speed-results/k7-i8mm-packed-encoder-pixel10.md) |
| **K8** | Quantize and offline-pack every encoder linear matrix; convolution extension failed transcript gate. | Accept linear / reject convolution | [k8-full-encoder-linear-int8-pixel10.md](android-speed-results/k8-full-encoder-linear-int8-pixel10.md) |
| **K9** | Apply the complete offline I8MM-packed format to the fine-tuned 0.6B mobile model. | Accept full linear INT8 | [k9-full-linear-int8-06b-pixel10.md](android-speed-results/k9-full-linear-int8-06b-pixel10.md) |

---

## Active Backlog (Idea Bank)

This list acts as a prioritized backlog of upcoming optimization experiments. Each item targets one or more core metrics (Latency, First Token Latency, Binary Size, Memory Usage).

### Priority Backlog

1. **INT8/FP16 KV Cache Quantization (KV1)**
   - **Targets**: 1. Latency, 4. Memory usage (`peak_rss_kb`, `decode_loop_ms_per_token`)
   - **Why**: The 2048-token KV cache allocates up to ~470 MB of F32 buffer space across 28 layers. Quantizing KV cache keys and values to FP16 or INT8 (group-wise/channel-wise) reduces maximum cache memory overhead by 50-75% and improves memory bandwidth during the autoregressive decode loop.

2. **Cargo Release Profile Tweaks: Debug Stripping & Panic Abort (CT2)**
   - **Targets**: 3. Binary size, 1. Latency (`binary_size_bytes`)
   - **Why**: Currently, the release profile does not strip symbols or abort on panic. Adding `strip = true` and `panic = "abort"` in `Cargo.toml` strips debugging information and removes panic unwinding tables from the dynamic library. This can reduce the dynamic library (.so) binary footprint on Android by 50-80% and generate tighter, faster assembly.

3. **FP16 Activation/Quantization for Encoder (E2)**
   - **Targets**: 1. Latency, 4. Memory usage (`encoder_ms`)
   - **Why**: Keep intermediate activation buffers in FP16 instead of F32. On CPUs with native FP16 (`+fp16`), this cuts memory bandwidth usage in half, reduces cache line pressure, and doubles FLOPS throughput.

4. **Cache line alignment for weight buffers (CL1)**
   - **Targets**: 1. Latency (`encoder_ms`, `prefill_ms`)
   - **Why**: Align memory-mapped weights and internal buffers to 64/128-byte boundaries to prevent split cache line loads and cache bank conflicts on the CPU.

5. **Custom vectorized Softmax kernel (S1)**
   - **Targets**: 1. Latency (`encoder_ms`, `decode_loop_ms_per_token`)
   - **Why**: Attention layers perform row-wise softmax. Implementing a custom vectorized Softmax kernel with ARM NEON intrinsics (and math approximations for exp) will speed up attention overhead compared to scalar code.

---

### Punted / Irrelevant Ideas

- **Android SGEMM shape microbench (P3)** (OpenBLAS Diagnostic)
  - *Reason for punting*: We are moving away from OpenBLAS F32 SGEMM for critical hot paths (prefill uses INT8 SDOT via K4, encoder will use native BF16 or INT8). Spending time profiling OpenBLAS SGEMM shapes is no longer a high-leverage activity.
- **Isolate threaded OpenBLAS runtime settings (P1)** (Threading Diagnostic)
  - *Reason for punting*: The thread sweeps in C2/C3 already isolated single-thread execution (`threads=1`) as optimal and rejected multi-threading. With custom kernels, OpenBLAS is being bypassed anyway.
- **Overlapped Mel extraction and encoder forward (OL1)** (Latency)
  - *Reason for punting*: P0 phase timing proved that Mel extraction takes < 5 ms (under 1% of total inference time). Overlapping it would yield no noticeable performance gains.
