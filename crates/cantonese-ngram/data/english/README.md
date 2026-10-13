# English dictionary and search reference

`en.txt` is the unmodified wmkeyboard English seed dictionary from commit
`74bf16e27ff72403bcd55eb942894d2158e9bd2b`:
https://github.com/wasi-master/wmkeyboard/blob/74bf16e27ff72403bcd55eb942894d2158e9bd2b/app/dictionaries-src/en.txt

The English search and Gaussian touch model are Rust adaptations of
`FuzzyBeamSearch.kt`, `TouchModel.kt`, and QWERTY `KeyProximity.kt` from the same
revision. Copyright (c) 2026 Wasi Master; MIT license in `LICENSE`.

Generate the Android asset from the pinned source with:

```sh
cargo run --manifest-path crates/cantonese-ngram/Cargo.toml --bin pack_english -- \
  crates/cantonese-ngram/data/english/en.txt \
  ../cantonese-voice-keyboard/app/src/main/assets/english/en.bin
```

EAC1 stores little-endian node and edge counts, then 16-byte nodes
(first edge, edge count, terminal frequency, maximum subtree frequency) and
8-byte edges (ASCII label, child index). The loader validates topology and
sorted edges before use. Weights are preserved; repeated source words use the
maximum frequency, matching upstream Trie.insert.

Personal lexicons, accents, ambiguous T9 keys, and learned edit habits are
outside this English QWERTY implementation. Search constants and touch costs
follow upstream. Automatic replacement uses the keyboard's conservative gate.
Optional context ranking reuses the existing bilingual NN through the Quick
engine registry, with bounded relative logits and no bonus for unsupported
vocabulary. It never introduces words outside spelling-search results.

`reference-cases.json` records results produced by the pinned Kotlin
FuzzyBeamSearch and TouchModel, using a walker over the same EAC1 dictionary.
The 30 cases cover ASCII QWERTY with and without touch evidence. Touch cases
use row centers with offsets 0/.5/1.5 and tap offsets +.45/+.15 in Kotlin Float
arithmetic. T9 keys, accents and learned habits are disabled. Rust tests compare
every candidate's word, score, edit cost, edit count and completion count.
