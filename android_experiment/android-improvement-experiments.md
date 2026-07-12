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

---

## Active Backlog (Idea Bank)

This list acts as a plan of upcoming optimizations to be evaluated. Once run, each will be documented in a corresponding `.md` result file.

1. **NEON BFDOT/BFMMLA BF16 matvec/matmul kernel (K1 / K3)**
   - **Target**: `encoder_ms`
   - **Why**: Use native Armv8.6 hardware instructions for BF16 math to bypass F32 software-unpacking.

2. **Android SGEMM shape microbench (P3)**
   - **Target**: BLAS diagnostic
   - **Why**: Identify exact shapes with poor packing efficiency under OpenBLAS on Tensor G4.

3. **Isolate threaded OpenBLAS runtime settings (P1)**
   - **Target**: Threading diagnostic
   - **Why**: Diagnose exact overhead of OpenBLAS dynamic selection under single-thread constraints.

4. **KleidiAI (Arm) GEMM library integration (K2)**
   - **Target**: `encoder_ms` and `prefill_ms`
   - **Why**: Test Arm's optimized transformer GEMM library as an alternative to OpenBLAS.

5. **Encoder INT8 weights + native INT8 GEMM kernel (E1)**
   - **Target**: `encoder_ms`
   - **Why**: Quantize encoder weights to INT8 and execute via hardware `SDOT`.

6. **Compile-time LTO and codegen units optimization (CT1)**
   - **Target**: `binary_size_bytes` & E2E Latency
   - **Why**: Enable fat link-time optimization (LTO) to force inter-crate inlining.

7. **Cache line alignment for weight buffers (CL1)**
   - **Target**: Latency
   - **Why**: Align weights to 64/128-byte boundaries to reduce cache misses.

8. **Overlapped Mel extraction and encoder forward (OL1)**
   - **Target**: Latency
   - **Why**: Move Mel spectrogram computation to a background thread to overlap ingestion and network feed-forward.
