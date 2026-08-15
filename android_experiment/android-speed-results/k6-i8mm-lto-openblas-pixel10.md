# K6: I8MM, release-profile, and OpenBLAS follow-ups on Pixel 10

Purpose:

- Evaluate the three remaining low-complexity optimization candidates after the
  packed-only DotProd decoder was accepted.
- Run latency candidates on the connected Pixel 10 with thermal status 0,
  fixed-performance mode, one engine thread, three repetitions, five Common
  Voice clips, and the push-to-talk guardrail.

## I8MM-packed KleidiAI decoder

The decoder model was repacked for KleidiAI's
`qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` kernel. The base shard is unchanged;
the packed shard remains approximately 1.35 GiB. A same-state rerun compared
the saved DotProd binary/model with the I8MM binary/model.

| Metric | DotProd | I8MM | Delta |
| --- | ---: | ---: | ---: |
| Common Voice median inference | 1927.116 ms | 1782.147 ms | **-7.5%** |
| Common Voice median first-visible token | 1399.114 ms | 1295.949 ms | **-7.4%** |
| Common Voice median prefill | 470.161 ms | 386.702 ms | **-17.8%** |
| Common Voice median decode loop | 1020.246 ms | 956.791 ms | **-6.2%** |
| Guardrail inference | 2986.146 ms | 2755.311 ms | **-7.7%** |
| Guardrail first-visible token | 2031.835 ms | 1861.499 ms | **-8.4%** |
| Peak RSS, Common Voice | 2263716 KiB | 2263712 KiB | unchanged |

All six transcripts matched the accepted packed output. I8MM is accepted. The
packed format is now explicitly named in the safetensors keys, and model load
fails clearly on an Android Arm CPU without I8MM; there is no fallback.

## Fat LTO and one codegen unit

The release profile changed from thin LTO to fat LTO with one codegen unit. The
back-to-back result directories are:

- `android-pixel10-i8mm-fat-lto-ab-ctrl-20260815-102523`
- `android-pixel10-i8mm-fat-lto-ab-cand-20260815-102523`

Every Common Voice clip improved. Median inference changed from 2637.1 to
2593.2 ms (**-1.7%**), guardrail inference from 4067.1 to 4048.9 ms
(**-0.4%**), and binary size from 1,962,744 to 1,735,648 bytes (**-11.6%**).
Transcripts and peak RSS matched. The release-profile change is accepted.

## OpenBLAS 0.3.34

OpenBLAS 0.3.34 was rebuilt with the accepted NDK 29, API 35, static ARMV8,
CBLAS-only, single-thread configuration. The back-to-back result directories
are:

- `android-pixel10-openblas-034-ab-ctrl-20260815-102850`
- `android-pixel10-openblas-034-ab-cand-20260815-102850`

Common Voice median inference changed from 2605.9 to 2599.0 ms (**-0.3%**) and
guardrail inference from 4040.0 to 4028.8 ms (**-0.3%**). Encoder latency moved
about 0.2%. This is measurement noise, so 0.3.34 is rejected as a performance
change and the existing OpenBLAS revision remains pinned.

## Decision

- Accept the I8MM-packed decoder.
- Accept fat LTO with one codegen unit.
- Reject the OpenBLAS 0.3.34 bump as a latency optimization.

