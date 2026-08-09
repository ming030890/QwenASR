# V20 mobile export difference investigation

Status: open; the production-sized export is the one currently staged for the
Android release. This note records why the larger export must not be used and
what remains to investigate after the release.

## Artifacts

All three files are the same merged V20 checkpoint except for INT8 export
format:

| Export | Bytes | Relevant tensor layout |
|---|---:|---|
| Production v16 INT8 | 970,943,746 | 169 I8 + 169 F32 scales; tied BF16 embedding dropped |
| V20 mobile INT8 | 970,943,746 | Same mobile layout; tied BF16 embedding dropped |
| V20 first INT8 export (do not ship) | 1,282,109,361 | Retained one redundant BF16 tied embedding |

The exporter now defaults to dropping `thinker.model.embed_tokens.weight` for
tied checkpoints. The larger research export is available only through the
explicit `--keep-embed-tokens` option.

## Held-out test result

Evaluation used the same 34 full-input WAVs in
`/private/tmp/keyboard-v20-final-eval/manifest.txt`, greedy local QwenASR
inference, and the existing character CER scorer.

| Model | CER (no punctuation) | Macro CER | CER (with punctuation) |
|---|---:|---:|---:|
| Production v16 mobile | 20.11% (113/562) | 11.13% | 20.54% (122/594) |
| V20 mobile | 19.57% (110/562) | 13.14% | 20.71% (123/594) |

The V20 mobile export changes 6 of 34 decoded strings relative to the first
V20 export that retained the BF16 embedding. The difference is not explained
by file size; it indicates that the runtime's tied-embedding path is not
numerically equivalent to retaining the BF16 input embedding for this model.

## Follow-up experiments

1. Confirm the Rust loader's tied-embedding path: identify whether it uses the
   quantized `thinker.lm_head.weight` for input embeddings and whether the
   dequantization/transpose path matches the BF16 embedding layout.
2. Run an offline logits comparison on identical token prefixes between the
   BF16-embedding and dropped-embedding exports. Find the first layer/token
   where logits diverge and quantify max/mean error.
3. Compare production v16 and V20 with the same decoder/runtime settings on
   Common Voice test, new test, and a short mixed Chinese/English set. Report
   both corpus and macro CER; the macro score is sensitive to short clips.
4. Verify code-switch formatting separately: inspect cases where an English
   span touches Chinese text and ensure the app insertion normalizer supplies
   a space at the boundary.

Do not promote the `--keep-embed-tokens` export to Android merely because its
greedy output differs; it violates the production asset-size contract and
requires an unplanned third asset pack.
