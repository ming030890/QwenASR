# Qwen inference optimization experiments

This folder is the experiment log and runner for improving **Qwen3-ASR 1.7B**
inference on macOS first, then validating accepted changes on Android. Every run
gets its own folder containing raw measurements, a machine-readable summary,
and an `experiment.md` conclusion.

## Goal

Reduce Qwen3-ASR 1.7B-parameter inference latency, model load time, peak memory,
and deployed file size without changing transcript quality.

Target model: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

## Current 1.7B optimization baseline

Run `025-mac-1.7b-full-int8-tiled256-prefill` is the new baseline for all
subsequent Qwen3-ASR 1.7B optimization experiments. New candidates must pass
`--baseline 025-mac-1.7b-full-int8-tiled256-prefill` unless an experiment is
explicitly establishing a different platform baseline.

Decoder projections and the tied LM head are stored mmap-ready as INT8, the
duplicate BF16 token embedding is omitted, and prefill dequantizes only 256
output rows at a time into reusable F32 scratch for OpenBLAS.

The accepted prefill configuration is now the runtime default. Set
`QWEN_INT8_PREFILL_F32=0` only for a diagnostic rollback; override
`QWEN_INT8_PREFILL_TILE_ROWS` only for controlled tile-size experiments.

Against the optimized BF16 candidate (`018`), it reduced peak RSS by 56.4%,
model files by 49.8%, median inference latency by 5.4%, and P95 latency by 2.7%.
All 54 repetitions were stable. The INT8 result differs from BF16 on one of six
guardrail clips (`Hello` versus `哈喽`), so this is an accepted memory/format
direction with explicit QAT debt, not a production quality acceptance. Against
the prior INT8 representation (`021`), removing BF16 embeddings and tiling
scratch reduced peak RSS by another 3.6% and model size by 20.8% while improving
median latency by 3.4%. Android validation and QAT recovery are still required.
The optimized BF16 run `018` remains the quality reference used to measure the
known QAT debt; it is no longer the performance baseline.

## Latest encoder result

Run `045-mac-1.7b-int8-encoder-ffn-l7-attn-l1` stores encoder FC1/FC2 in layers
7-23 and Q/K/V/output attention projections in layers 1-23 as mmap-backed
per-row INT8. Layer 0 attention, layers 0-6 FFN, the convolutional stem, and
final projections remain BF16. Against encoder comparison run `040`, it reduced
peak RSS by another 1.0% (22.5 MiB), model files by 1.2% (23.9 MiB), and encoder
latency by 4.8%, with effectively flat total latency and all 54 exact
transcripts preserved. Cumulatively versus baseline `025`, the encoder work
reduces peak RSS by 9.1%, model files by 10.1%, and encoder latency by 19.3%.
Runs `041`-`044` isolated the attention quality boundary: quantizing layer 0
changed one transcript's final punctuation, while starting at layer 1 passed.
Run `032` separately established the FFN boundary: including layer 6 changed one
transcript. Use `025` as requested for the overall experiment baseline; use
`045` as the accepted comparison point for subsequent encoder-specific work.

Run `038` sized decoder F32 prefill scratch to the active INT8 tile (6 MiB
instead of 96 MiB). Exact transcripts passed, but peak RSS did not improve
against the comparable `034` smoke run (2390.75 versus 2380.97 MiB), because
the untouched zero-filled capacity was mostly lazy virtual memory. The bounded
allocation is retained as hygiene, not counted as an RSS optimization.

Phase-aware mmap eviction was implemented and measured in runs `036` and `037`.
Full encoder/decoder eviction preserved transcripts but regressed latency 15.0%
for only a 1.6% peak-RSS improvement versus `035`; encoder-only eviction
regressed latency 7.4% for a 2.1% peak-RSS improvement. Both are rejected as
defaults, and the runtime implementation was reverted. The measurements remain
recorded so the approach is not repeated without evidence of different VM
behavior on the target platform.

## Latest first-token result

Run `047-mac-1.7b-static-prefix-kv-cache` reuses a compact, exact K/V snapshot
for `PREFIX_HEAD + prompt + PREFIX_TAIL` across utterances in the same context.
Against same-revision control `046`, it reduced median first-visible-token
latency by 11.6% (576.52 to 509.91 ms), decoder prefill by 19.7%, median total
inference by 5.3%, and P95 inference by 6.4%. All 54 transcripts matched
exactly. Peak RSS increased by 6.48 MiB (0.3%). The optimization is enabled by
default; set `QWEN_STATIC_PREFIX_KV_CACHE=0` only for a diagnostic rollback
comparison.

The benchmark now records `first_visible_token_ms` separately from
`prefill_ms`. It starts at transcription entry and ends when the decoder emits
the first complete UTF-8 text piece; `prefill_ms` remains a decoder phase metric
and must no longer be presented as user-visible TTFT.

The runner defaults to requiring a model detected by the engine as `1.7B`; this
prevents a 0.6B model with a similarly named or sized weight file from becoming
the baseline by accident. Use `--expect-model-variant` only when intentionally
testing the collector or another supported variant. The exact model path,
inventory fingerprint, detected variant, and byte size are recorded in every
result.

## Metrics

| Metric | Unit | Definition |
| --- | --- | --- |
| Inference latency | ms | Median engine inference time across all measured launches and samples. |
| Model load time | ms | Median time reported around `QwenCtx::load()`; the first launch is also retained. |
| Memory after model load | MiB | Resident high-water mark immediately after loading (macOS) or current RSS (Android/Linux). |
| Peak memory | MiB | Maximum resident set size sampled by the benchmark process (`getrusage` on macOS, `/proc/self/status` on Android/Linux). |
| Model file size | MiB | Sum of logical file sizes under `--model-dir`; weight files are reported separately. |
| Runtime file size | MiB | Size of the release benchmark executable (Android should record the `.so`/APK asset equivalent). |

The raw benchmark also retains phase timings (`mel`, `encoder`, `prefill`, and
decode loop), RTF, thread count, transcript, Git revision, host details, and a
SHA-256 fingerprint of model filenames and sizes. File contents are not hashed,
so setup remains fast even for multi-gigabyte weights.

## Hard transcript gate

Transcript equality is a mandatory acceptance condition. A candidate is
eligible for performance acceptance only when:

- every repeated candidate run produces exactly the same transcript;
- every repeated baseline run produces exactly the same transcript;
- candidate and baseline use the same audio-file set; and
- every candidate transcript exactly equals its baseline transcript, including
  Unicode characters, capitalization, whitespace, and punctuation.

There is no fuzzy, phonetic, semantic, or CER tolerance in this gate. The runner
records per-run transcripts and emits `exact_transcript_gate=PASS|FAIL`. A FAIL
automatically writes a reject decision into the experiment Markdown.

Quantization research may retain a failed run as a measured QAT target when the
quality debt is explicit, but it must not be described as production-accepted
or used to weaken this gate.

The exact gate is a sensitive regression alarm, not a replacement for a
ground-truth corpus evaluation. Before changing or graduating INT8 token
embeddings, read
[INT8_EMBEDDING_CONCLUSION.md](INT8_EMBEDDING_CONCLUSION.md), which records the
post-training quantization result and the quantization-aware recovery plan.

## Standard Mac mini loop

Use the same model, WAV, prompt, thread count, and power/thermal conditions for
baseline and candidate. Three fresh process launches and three in-process runs
per launch are the default. The first inference of each launch is retained but
the summary uses medians.

OpenBLAS is mandatory for all future experiments. Install the Homebrew library
once (`brew install openblas`). The runner builds with
`QWEN_BLAS_BACKEND=openblas` and rejects any binary whose runtime optimization
flags do not contain `OpenBLAS`. This also applies with `--skip-build`, preventing
an old Accelerate-only binary from entering the experiment record.

The recorded `001-mac-qwen3-asr-1.7b-baseline` predates this rule and is retained
only as historical evidence. It is not an eligible comparison baseline.

Use the current baseline configuration when evaluating every new 1.7B
candidate:

```bash
QWEN_INT8_PREFILL_F32=1 QWEN_INT8_PREFILL_TILE_ROWS=256 \
  python3 experiments/qwen-inference-optimization/run_experiment.py \
  --id 026-candidate \
  --baseline 025-mac-1.7b-full-int8-tiled256-prefill \
  --model-dir /Volumes/SSD/Qwen3-ASR-1.7B-int8-full \
  --input bench/android-assets \
  --threads 1 \
  --hypothesis "The focused change improves the target metric" \
  --change "Describe one change"
```

For a quick correctness check while iterating:

```bash
python3 experiments/qwen-inference-optimization/run_experiment.py \
  --id smoke \
  --model-dir /Volumes/SSD/Qwen3-ASR-1.7B \
  --input bench/android-assets/ptt_guardrail.wav \
  --launches 1 --runs 1 --threads 1 \
  --skip-build
```

Results are created under `experiments/qwen-inference-optimization/runs/<id>/`.
An ID is never overwritten; choose a new ID for every experiment.

## Experiment protocol

1. Record one hypothesis and make one focused change.
2. Run the Mac smoke gate (`--launches 1 --runs 1`). Verify the transcript.
3. Run the standard Mac measurement (defaults: 3 launches × 3 runs).
4. Pass `--baseline <experiment-id>` to compare with the controlled baseline.
5. Fill in the Decision and Conclusion sections of that run's `experiment.md`.
6. Keep a change only when the target metric improves outside normal run-to-run
   noise and transcript output remains acceptable.
7. Run the existing Android gates in `android_experiment/android-autoresearch-program.md`
   before treating a Mac win as a product win.

Recommended acceptance thresholds for a controlled comparison:

- latency or model load time: at least 5% better;
- peak memory: at least 10% better;
- model/runtime size: at least 5% better;
- no inference latency regression above 2% when optimizing another metric;
- exact transcript gate passes and Android tests remain valid.

Mac results are directional. Apple silicon, virtual memory, and the filesystem
cache differ substantially from Android. Both Mac screening and Android
acceptance must use OpenBLAS; never accept a change for production from the Mac
result alone.

## Result layout

```text
runs/<experiment-id>/
├── experiment.md       # hypothesis, measured result, decision, conclusion
├── metrics.json        # normalized metrics and full run metadata
└── raw/
    ├── launch-001.json # benchmark output from one fresh process
    └── launch-001.log  # timing/runtime diagnostic output
```

Use [`EXPERIMENT_TEMPLATE.md`](EXPERIMENT_TEMPLATE.md) when documenting a
manual or Android-only experiment that was not launched by this runner.
