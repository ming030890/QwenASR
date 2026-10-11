# Cantonese candidate LM (Rust)

Independent CPU-only Rust library shared by Quick and the BiLSTM handwriting recognizer.
It has its own Cargo workspace and no dependency on the Qwen ASR engine, PyTorch or ONNX Runtime.

## Context policy

| Available known tokens in the current sentence | Ranking prior |
|---|---|
| None, or only one right token | Unigram |
| One preceding character in the count vocabulary | Unigram plus character bigram |
| At least two known left/right tokens | Neural n-gram |

PAD/BOS are zero context; UNK does not count toward the neural threshold. The neural window
is up to eight left and four right tokens; it does not require a full window. The bigram
uses the immediately preceding token only when it maps to a character in the count model.
A single English word without a character bigram uses unigram.
Terminal punctuation clears preceding sentence context; commas remain context.

Quick's existing dictionaries are already sorted by unigram frequency, and empty-context
lookup returns that exact order. Characters outside a scorer's vocabulary remain in their
original Quick slots. No candidate is removed. Handwriting retains the existing count
priors for unsupported neural characters. Its explicit precomputed-prior API remains available.

## API

- `LanguageModel::from_bytes(&bytes)`: load the validated CNG1 neural artifact.
- `HybridLanguageModel::new(neural, counts, labels)`: shared tokenizer/fallback policy.
- `hybrid.rank(left, right, candidates)`: candidate permutation for Quick.
- `hybrid.scores(left, right, candidates)`: additive priors for handwriting.
- `QuickDictionary::from_bytes(bytes, version)` / `QuickEngine::candidates(...)`:
  Rust Quick3/Quick5 dictionary lookup and context ranking.

Matrices use symmetric INT8 per-row weights with FP32 scales. Activations, LayerNorm,
biases and reductions are FP32; ARM64 row dot products use NEON. This is quantized weight
storage with floating-point inference. The tied output table is stored once. Quick scores
only dictionary candidates, and the last context projection is cached. Handwriting scores
its recognition labels and normalizes over neural-supported labels. Each engine owns its
model instance; sharing this crate does not imply sharing resident tensors across JNI libraries.

Android Kotlin handles composition and UI. Rust handles dictionary lookup, tokenization,
context policy and ranking. Quick loading/inference runs on a worker and stale results are
rejected. The Quick JNI handle registry keeps in-flight calls alive during destruction.
Handwriting's existing synchronized JNI wrapper serializes predictions and destruction.

## Artifact and integration

Selected checkpoint: `nested-curriculum-128-256-512-v7/train-full/best-natural-width256.pt`
under `/Volumes/SSD/cantonese_ngram/experiments/`.

Canonical Rust export: `export-width256/rust/model.cng` in that experiment.
Packaged copy: `cantonese-voice-keyboard/app/src/main/assets/language_model/model.cng`.
The artifact includes vocabulary and tensors: 4,980,849 bytes (29,762 tokens, embedding128,
residual256, two ReLU/LayerNorm blocks). It is produced by
`dataset/cantonese-ngram/export_rust.py`; it is not an ONNX file.

Both engines also load the existing `handwriting/lm_csr.bin` and its `labels.txt`.
`cantonese-voice-keyboard/deploy.sh` rebuilds both native libraries and stages this model;
`LM_SOURCE` and `NATIVE_LM_LINKER` can override the artifact and NDK linker paths.
Android native builds use API32 ARM64.

## Verification

Run Cargo with `--manifest-path` from a directory outside the ASR workspace to avoid
inheriting its OpenBLAS linker configuration. `cargo test --all-targets --features android`
checks tokenizer, dictionary validation and fallback behavior. The handwriting crate tests
also check that count fallback is applied once and raw recognizer scores are preserved.

`dataset/cantonese-ngram/verify_rust.py` compares quantized Python/Rust scores and training
context IDs, verifies all 1,404 empty-context dictionary keys, and tests nine mode transitions.
The current export passed with maximum score error 0.00000954 over sixteen contexts and
187 candidate scores per context. This is numerical/behavioral acceptance, not a candidate
quality benchmark. Handwriting now loads separately calibrated weights (unigram1.0, bigram1.5, neural1.25)
from `language_model/handwriting_fusion.json`, bound to exact asset hashes. The training-only
audit improved top-1 from68.5% to77.4% over six short-context conditions; see
`handwriting-recognizer/RUST_LM_CALIBRATION_V7.md`. This is not a fresh production accuracy test.

## Next-token predictions

`LanguageModel::predict_next(left, limit)` searches the entire neural vocabulary;
`predict(left, right, limit)` also supports insertion with right context.
Results contain `token`, `token_id` and `score`, sorted descending, with deterministic
ties. Scores are logits, not probabilities. Special/control tokens are excluded;
ordinary punctuation is retained. Limit is capped at 100; zero returns no results.
The vocabulary predicts Han characters, whole normalized English words and punctuation.

`HybridLanguageModel::predict_next` / `predict` return `(Mode, predictions)` and
apply the shared policy: empty/unknown context uses count unigrams, a usable single
preceding character uses count bigrams, and two known context tokens enable neural
prediction. Fallback scores are count ranking scores, not comparable to neural logits;
count-only labels may have `token_id: None`. A missing count model yields no fallback
predictions. Existing candidate-only scoring remains unchanged.

Quick displays three predictions while no code is active, refreshes after insertion,
editing or cursor movement, and switches back to dictionary candidates during composition.
English predictions add a boundary space if adjacent to English letters/digits.

Probe JSONL: `{"left":"我想","limit":3}`. Supply the existing count/labels/Quick
asset arguments to `ngram_probe` for hybrid fallback, or just model.cng for raw neural.
