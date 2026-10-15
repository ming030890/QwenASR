# Cantonese candidate LM (Rust)

Independent CPU-only Rust library shared by Quick and the BiLSTM handwriting recognizer.
It has its own Cargo workspace and no dependency on the Qwen ASR engine, PyTorch or ONNX Runtime.

## Context policy

Current keyboard selection is the completed compact two-block model: one CNG1 token-input network for
zero-, one-, and longer-context predictions. Embedding128, residual256, FFN512,
two blocks, 34,081 tokens, and 5,860,722 bytes. The export sets
`neural_short_contexts`; the shared policy discards counts for this model, and
empty Quick candidates also use learned neural unigram scores. No count table or
complete-context memory is packaged. Canonical documentation is
`cantonese-voice-keyboard/LANGUAGE_MODEL_SELECTED.md`. The sections describing v9 below
are historical and explain compatibility with earlier artifacts.

The historical v9 CNG2 selection used: vocabulary/context-feature
dimensions128, memory MLP width128, frozen v7 contextual base width256. Both JNI
integrations load either CNG1 or CNG2 through `NeuralLanguageModel::from_bytes`.
CNG2 uses the neural model from ONE known left/right token (bigram and above).
Known full keys blend95%learned memory with5%v7; unseen keys use v7. No count
backoff is applied to supported bigram+ candidates. Empty-context Quick lookup
retains dictionary order, and characters unavailable in a context's neural head
retain their existing count priors/Quick slots. This includes newly added tokens
when an unseen context uses v7's smaller vocabulary.

The compatibility policy for older CNG1 artifacts remains:

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
- `NeuralLanguageModel::from_bytes(&bytes)`: auto-detect CNG1/CNG2 for keyboard use.
- `HybridLanguageModel::with_neural(neural, counts, labels)`: shared format-aware policy.
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

Experimental CNG1 exports can also use `row_i16` matrices with FP32 row scales.
The parser and ARM64 NEON path support both precisions; unsupported kinds and
truncated payloads are rejected. Exporting INT16 requires a matching fresh native
library and does not select a new production model. Sequence evidence additionally
exposes `logicalUnits` and `unitLogProbability`: English words, Han characters
and grapheme clusters count before UTF-8 byte fallback. The existing encoded-token
mean remains unchanged for previously calibrated callers.

The experimental English decoder uses admissible minimum/maximum terminal-suffix
length bounds to prioritize its trie search. These two runtime bytes per node do
not alter the dictionary format, edit costs, candidate widths or 8 ms budget.
Literal input remains reserved. Exhaustive weighted-edit tests verify the bound;
`dataset/keyboard-lm/audit_search_v35.py` separately measures frozen development
and recorded-touch retrieval. Candidate recall does not establish correction
precision, and these changes do not enable an unqualified replacement policy.

Android Kotlin handles composition and UI. Rust handles dictionary lookup, tokenization,
context policy and ranking. Quick loading/inference runs on a worker and stale results are
rejected. The Quick JNI handle registry keeps in-flight calls alive during destruction.
Handwriting's existing synchronized JNI wrapper serializes predictions and destruction.

## Artifact and integration

Selected checkpoint: `width128/selected/model.pt` under
`/Volumes/SSD/cantonese_ngram/experiments/count-distillation-128-256-v9/paired-width-comparison`.
Canonical CNG2 export: `width128/export/model.cng` in that comparison.
Packaged copy: `cantonese-voice-keyboard/app/src/main/assets/language_model/model.cng`.
The selected artifact occupies 31,840,749 bytes and has SHA256
`0f0f74ee928da0a92de6e201118d17131e2f9be1d709425609c7cb0e940e79d6`.
It is produced by `dataset/cantonese-ngram/export_count_memory.py`.
The embedded contextual base is the earlier CNG1 width256 model with
29,762 tokens and 128-dimensional vocabulary embeddings.

Both engines also load the existing `handwriting/lm_csr.bin` and its `labels.txt`.
`cantonese-voice-keyboard/deploy.sh` rebuilds both native libraries and stages this model;
`LM_SOURCE` and `NATIVE_LM_LINKER` can override the artifact and NDK linker paths.
Android native builds use API32 ARM64.

## Verification

Run Cargo with `--manifest-path` from a directory outside the ASR workspace to avoid
inheriting its OpenBLAS linker configuration. `cargo test --all-targets --features android`
checks tokenizer, dictionary validation and fallback behavior. The handwriting crate tests
also check that count fallback is applied once and raw recognizer scores are preserved.

`tests/keyboard_memory.rs` checks CNG1 compatibility, CNG2 routing, unseen contexts,
new vocabulary availability, and preservation of unsupported Quick candidates.
Actual artifact verification and count/context quality reports are beside the
selected SSD export. Handwriting loads separately calibrated weights (unigram1.0, bigram1.5, neural1.25)
from `language_model/handwriting_fusion.json`, bound to exact asset hashes. The training-only
audit improved top-1 from77.4% with v7 to78.233% with v9 over six short-context conditions;
see `cantonese-voice-keyboard/LANGUAGE_MODEL_V9.md`. This reused training audit is not
a fresh production accuracy test.

## Next-token predictions

`LanguageModel::predict_next(left, limit)` searches the entire neural vocabulary;
`predict(left, right, limit)` also supports insertion with right context.
Results contain `token`, `token_id` and `score`, sorted descending, with deterministic
ties. CNG1 scores are logits; CNG2 scores are normalized log probabilities.
Special/control tokens are excluded;
ordinary punctuation is retained. Limit is capped at 100; zero returns no results.
The vocabulary predicts Han characters, whole normalized English words and punctuation.

`HybridLanguageModel::predict_next` / `predict` return `(Mode, predictions)` and
apply the format-specific context policy above. CNG2 enables neural prediction
from one supported left/right context token; CNG1 retains the earlier two-token
threshold. Fallback scores are count ranking scores, not comparable to neural scores;
count-only labels may have `token_id: None`. A missing count model yields no fallback
predictions. Existing candidate-only scoring remains unchanged.

Quick displays three predictions while no code is active, refreshes after insertion,
editing or cursor movement, and switches back to dictionary candidates during composition.
English predictions add a boundary space if adjacent to English letters/digits.

Probe JSONL: `{"left":"我想","limit":3}`. Supply the existing count/labels/Quick
asset arguments to `ngram_probe` for hybrid fallback, or just model.cng for raw neural.
## Learned count memory

`memory::MemoryLanguageModel` loads the separate CNG2 format produced by
`dataset/cantonese-ngram/export_count_memory.py`. It combines learned complete
context feature embeddings and the selected 128-wide residual MLP with the frozen neural
base embedded as CNG1. Scores are normalized neural log probabilities; empty and
one-token contexts also use learned neural features. No raw count scores are
loaded or consulted. Unlisted complete contexts use the embedded neural base.

`memory_probe MODEL.cng` accepts JSONL text contexts or explicit `context_ids`
with `candidate_ids`, and returns scores. `ngram_probe` accepts both CNG1 and CNG2
and can exercise the shared keyboard policy with actual Quick/count assets.
