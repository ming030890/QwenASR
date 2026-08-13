# INT8 token-embedding experiment conclusion

## Decision

The affine groupwise INT8 token-embedding format and inference path are
implemented, but the current post-training-quantized weights must **not** be
graduated as the default model. The next step is quantization-aware recovery,
not another search over post-training group sizes.

This conclusion concerns the tied decoder token embeddings. The current v15
model still has BF16 audio-encoder and other non-decoder tensors, so this is one
step toward full INT8 rather than a claim that the whole model is already INT8.

## Baseline to resume from

The last accepted baseline for this work is
`004-mac-openblas-v15-int8-keep-embeddings-baseline`. Future QAT exports must be
compared against this run, not against one of the rejected groupwise candidates.

- Model: `/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000`
- Model fingerprint:
  `562a1fc9db1c5ea03cfc675ed06b55681623a7aef89744737962516cf76c8d2c`
- Precision: INT8 decoder projections and tied LM head; BF16 token embeddings,
  audio encoder, and remaining non-decoder tensors
- Backend: OpenBLAS, one inference thread
- Inputs: `bench/android-assets`, language `Auto`, prompt
  `音频是粤语，可能夹杂英语`
- Measurement: three fresh launches × three runs per input
- Median inference: 480.398 ms
- Model size: 1226.96 MiB
- Peak RSS: 1210.78 MiB
- Quality gate: all six transcripts stable across 54 measurements

Its canonical transcripts are:

```text
我急求方法啊
唔做嘢嘅占多數囉
證據嘅話真係冇啊
你睇下呢部劇
仲要提供乜嘢資料？
hello，你好嗎？已經坐低咗啦
```

The corresponding run record is
[`runs/004-mac-openblas-v15-int8-keep-embeddings-baseline/experiment.md`](runs/004-mac-openblas-v15-int8-keep-embeddings-baseline/experiment.md).

## Inference support now present

`bench/prepare_int8_model.py` can export tied token embeddings as either
symmetric or affine groupwise INT8. The affine representation contains:

- `thinker.model.embed_tokens.weight`: I8 `[vocab, hidden]`;
- `thinker.model.embed_tokens.weight_int8_group_scale`: F32
  `[vocab, hidden / group_size]`;
- `thinker.model.embed_tokens.weight_int8_group_min`: F32
  `[vocab, hidden / group_size]`.

The runtime detects this representation, validates the I8 matrix and metadata
lengths, keeps all three tensors mmap-backed, and dequantizes only the selected
token row into the decoder's existing F32 input buffer:

```text
q_u8 = q_i8 + 128
embedding = q_u8 * group_scale + group_min
```

No full embedding matrix is materialized at load time. Models without an INT8
embedding tensor retain the existing BF16 or tied-LM-head fallback paths.

Opt-in `QWEN_LOGIT_TRACE=1` and `QWEN_LOGIT_IDS=id,id` diagnostics report
selected INT8 LM-head scores using the same activation quantizer and weight
scales as greedy inference.

## Measured result

All runs used the matched v15 0.6B model, OpenBLAS, one thread, three fresh
launches and three runs per input. Run `004` is the controlled baseline.

| Run | Embedding representation | Model size | Median inference | Exact baseline gate |
| --- | --- | ---: | ---: | --- |
| 004 | BF16 | 1226.96 MiB | 480.398 ms | PASS |
| 005 | tied rowwise INT8 LM head | 930.21 MiB | 471.514 ms | FAIL (2 inputs) |
| 006 | symmetric INT8, group 16 | 1115.68 MiB | 476.019 ms | FAIL (1 input) |
| 007 | symmetric INT8, group 8, mmap | 1152.77 MiB | 469.484 ms | FAIL (2 inputs) |
| 008 | symmetric INT8, group 4, mmap | 1226.96 MiB | 471.567 ms | FAIL (3 inputs) |
| 009 | affine INT8, group 16, mmap | 1152.77 MiB | 485.544 ms | FAIL (3 inputs) |
| 010 | affine INT8, group 8, mmap | 1226.96 MiB | 480.766 ms | FAIL (1 input) |
| 011 | affine INT8, group 4, mmap | 1375.33 MiB | 491.008 ms | FAIL (3 inputs) |

Affine group 16 saves 74.19 MiB (6.0%) relative to the BF16-embedding model,
without a material peak-RSS improvement. Smaller groups do not monotonically
improve decoded output and eventually make the model larger than the baseline.

## What the logit diagnostic established

The `占`/`佔` change is a genuine near tie. At the first divergent step:

| Model | `占` | `佔` | Greedy margin |
| --- | ---: | ---: | ---: |
| BF16 embeddings | 23.640965 | 23.377634 | `占` by 0.263331 |
| affine INT8 group 8 | 23.490849 | 23.864792 | `佔` by 0.373943 |

That orthographic change alone would not prove serious quality loss. The
`仲要`/`重要` result does:

| Model | `仲要` first token | `重要` token | Greedy margin |
| --- | ---: | ---: | ---: |
| BF16 embeddings | 27.565145 | 25.515118 | `仲要` by 2.050027 |
| affine INT8 group 16 | 26.272001 | 26.868698 | `重要` by 0.596697 |

The reference is `仲要提供乜嘢資料？`. INT8 therefore overturned a healthy
baseline margin; the problem is accumulated decoder-state drift, not only a
strict gate objecting to equivalent characters.

## How to regain the loss

Use quantization-aware fine-tuning with the exact deployment quantizers:

1. Keep an F32/BF16 shadow copy of the tied embedding weight during training.
2. In the embedding lookup forward pass, fake-quantize it with affine group-16
   INT8 using the same `round`, clamp, scale and minimum formula as the exporter.
3. In the tied LM-head forward pass, fake-quantize the same shadow weight with
   the runtime's rowwise symmetric INT8 quantizer. Use a straight-through
   estimator for both rounding operations.
4. Start with the encoder and decoder projection weights frozen. Train the tied
   weight and decoder normalization parameters at a low learning rate. Only
   unfreeze adapters or the upper decoder blocks if this cannot recover the
   validation loss.
5. Distil from the accepted run-004 computation graph: combine normal ASR
   cross-entropy with KL divergence between teacher and fake-quantized student
   logits. A practical first trial is temperature 2 and equal CE/KL weighting;
   tune from validation results rather than treating those values as fixed.
6. Oversample Cantonese homophones/orthographic contrasts, mixed English case,
   and punctuation examples, but do not add token-specific inference patches.
7. Export the trained shadow weight through `prepare_int8_model.py` and run the
   exact same OpenBLAS experiment. Then run a larger held-out CER/WER evaluation
   and the Android gate before graduation.

The recovery target is not merely reproducing all six baseline strings. It is:

- restore `仲要` on the diagnostic clip and preserve other critical phrases;
- no meaningful held-out Cantonese CER/WER regression;
- internally stable greedy output;
- retain the affine group-16 size saving; and
- pass the Android device quality and performance gates.

After the tied embedding/LM-head pair is recovered, apply fake quantization to
encoder tensor families incrementally. Do not attempt every remaining BF16
tensor in one training run: doing so would make regressions hard to attribute.
