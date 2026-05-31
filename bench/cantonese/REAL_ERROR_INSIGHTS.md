# Cantonese ASR Real Error Eyeball Notes

This pass ignores punctuation and folds common orthographic variants when
deciding whether an utterance has a substantive error. The point is to separate
"reference spelling preference" from mistakes a keyboard user would notice.

## Eval Error Buckets

| bucket | utterances | raw edits | folded edits | note |
|---|---:|---:|---:|---|
| perfect | 2824 | 0 | 0 | exact after normalization |
| variant-only | 638 | 710 | 0 | spelling/particle variants only |
| real, non-severe | 1572 | 2927 | 2682 | mostly one-to-few char slips |
| severe | 96 | 583 | 554 | short collapse or unrelated phrase |

The rule miner was not just unlucky, but the earlier rules were too conservative
and too blind to topical vocabulary. The useful rule space is mostly phrase and
domain specific, not context-free character swaps.

## Main Patterns

### 1. Topic vocabulary errors are real and rule-friendly

The eval split has repeated errors around keyboard / Cantonese-writing topics:

| intended | common bad output | eval misses |
|---|---|---:|
| `粤拼` | `月拼`, `月平`, `月聘` | 14 |
| `仓颉` | `创杰`, `创铅`, `仓杰` | 6 |
| `正字` | `政治` | 6 |
| `讯息` | `信息` | 5 |
| `入面` / `里面` | `入边` / `里边` | 14 |
| `叉电` | `插电` | 3 |
| `黐线` | `痴线` | 2 |

Examples:

- `我想添加月拼` -> `我想添加粤拼`
- `咁我推荐你用月平` -> `噉我推荐你用粤拼`
- `好多香港人都用创杰` -> `好多香港人都用仓颉`
- `我系政治特警睇住你哋讲咗咩` -> `我系正字特警睇住你哋讲咗咩`
- `一个月前收过一模一样嘅信息` -> `一个月前收过一模一样嘅讯息`
- `原来手提电话无线插电有啲标准` -> `原来手提电话无线叉电有啲标准`

Insight: a keyboard-specific correction layer should include a small domain
lexicon / phrase table. These are not generic Cantonese rules; they are topic
priors from this corpus and product domain.

### 2. Many high-frequency substitutions are homophone choices

Top non-variant substitutions include:

- `有 -> 又` (15)
- `月 -> 粤` (15)
- `他 -> 它` (13)
- `依 -> 而` (13)
- `边 -> 面` (12)
- `信 -> 讯` (9)
- `读 -> 毒` (8)
- `治 -> 字` (7)
- `发 -> 法` (7)
- `不 -> 笔` (7)
- `平/聘 -> 拼` (12 combined)

Examples:

- `我对你系有爱有恨` -> `我对你系又爱又恨`
- `月平其实冇创铅意打字` -> `粤拼其实冇仓颉易打字`
- `请问下涉亲系咪好少人讲啦政治系` -> `请问下摄亲系咪好少人讲啦正字系`
- `平心而论佢有几靓嘅` -> `平心而论佢又几靓嘅`

Insight: these need context. A blanket `有 -> 又` or `依 -> 而` will absolutely
break good output. This is where a char/word n-gram LM or phrase-level rescoring
beats hand rules.

### 3. Severe failures are a different problem

There are 96 severe eval utterances. Many are short collapse outputs, often just
`系`, or a fluent but unrelated phrase:

- `系` -> `具体点做我唔知不过你可以直接问下佢`
- `系` -> `个女仔噉读文章我一定唔会模仿`
- `系` -> `我唔中意自拍或者类似嘅嘢`
- `你都记啊` -> `你妒忌呀`
- `广州珠海广场条边...` -> `广州海珠广场条片...`
- `非常顺彩我在平远地方...` -> `乡村车或者偏远地方...`

Insight: no post-correction table should try to fix these. They need ASR
confidence, retry/redecode, beam alternatives, or model-side improvements.

### 4. Manual topical rules help, but less than broad dev-pruned rules

A small hand-readable topical list built from train/dev observations:

- eval baseline: 4220 edits, CER 0.0807
- eval corrected: 4133 edits, CER 0.0791
- delta: -87 edits
- fixed/broke: 84 / 20

The broader `rules_pruned_dev.json` does better:

- eval corrected: 4099 edits, CER 0.0784
- delta: -121 edits
- fixed/broke: 171 / 55

Insight: the broad rules are stronger but noisier. A production keyboard should
prefer a two-tier strategy:

1. high-confidence phrase/domain lexicon rules (`粤拼`, `仓颉`, `正字`, `讯息`,
   `叉电`, `黐线`, etc.);
2. an LM/context scorer for ambiguous homophones and particles.

The current JSON rules are a useful experiment artifact, not the final product
shape.

## Upper Bound For Candidate Reranking

I mined a candidate confusion table from **train+dev only** by aligning ASR output
to references, extracting local diff spans with up to 3 chars of context, and
keeping recurring spans. Then I evaluated on eval with an oracle scorer: for each
eval sentence, generate candidates from that train+dev table and pretend the
scorer always picks the candidate closest to the reference.

This is intentionally optimistic. It measures the ceiling of the approach, not
the current shippable result.

| eval condition | edits | CER | gain vs raw |
|---|---:|---:|---:|
| raw ASR | 4220 | 0.0807 | — |
| variant-fold only | 3232 | 0.0618 | −988 |
| perfect train+dev candidate reranker | 3290 | 0.0629 | −930 |
| variant-fold + perfect candidate reranker | 3057 | 0.0585 | −1163 |

So, with the current data and candidate-table approach, the optimistic ceiling is
roughly **8.07% → 5.85% CER**. That is a **27.6% relative edit reduction**.

The oracle candidate reranker fixes:

- 821 eval utterances
- 930 total edits
- 553 edits from variant-only utterances
- 346 edits from non-severe real-error utterances
- 31 edits from severe utterances

Interpretation:

- A candidate/scorer layer can recover a meaningful chunk of the benchmark.
- Most of the recoverable gain is spelling / orthography / repeated phrase
  confusion.
- Severe recognition failures remain almost untouched; they need ASR confidence,
  retry/redecode, beam alternatives, or model-side changes.

## Follow-up: `係` / `系` Collapse Cases Are Mostly Low-Volume Audio

I inspected the eval utterances where the model output only `係。` / `系`.
There are 22 short-collapse cases in eval. Their raw MP3s are real Common Voice
clips with normal durations, but the waveform level is extremely low:

- typical collapse RMS: `0.001` to `0.003`
- typical good eval speech RMS sampled at random: `0.03` to `0.13`
- most collapse clips have almost no 20ms frames above `0.01` amplitude

Example raw stats:

| clip | raw hyp | ref length | duration | RMS | peak |
|---|---|---:|---:|---:|---:|
| `common_voice_yue_32270428.mp3` | `係。` | 17 | 5.08s | 0.0026 | 0.025 |
| `common_voice_yue_32270841.mp3` | `係。` | 12 | 5.18s | 0.0013 | 0.009 |
| `common_voice_yue_31935717.mp3` | `係。` | 14 | 5.83s | 0.0009 | 0.007 |

I amplified all 22 clips by `+30dB` and retranscribed with the same model. The
subset result:

| condition | edits on 22 clips |
|---|---:|
| raw audio | 188 |
| `+30dB` audio | 28 |

21 of 22 clips improved; none regressed.

Representative fixes:

- `係。` -> `具體點做我唔知，不過你可以直接問下佢。`
- `係。` -> `我唔中意自拍或者类似嘅嘢。`
- `係。` -> `大多數平台都冇法用。`
- `係。` -> `要转个用户名。`
- `係。` -> `正月十五係咩節日？`

Conclusion: many of the severe `係` collapses are not language-model errors and
not correctable by text postprocessing. They are caused by very quiet input
reaching the ASR frontend. The fix should be audio-side gain normalization /
AGC before mel extraction or before native ASR transcription.
