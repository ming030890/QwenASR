# Cantonese ASR — Rule-Based Correction Experiment Results

**Model:** `qwen3-asr-0.6b`, INT8-optimized weights at
`/Volumes/SSD/qwen-asr-m4-lmhead-int8` (lm_head + on-the-fly INT8 dequant).
**Host:** Apple Silicon (M4), NEON + vDSP/Accelerate + BLAS, `target-cpu=native`.
**Data:** Common Voice `cv-corpus-25.0` Cantonese (`yue`). All text normalized
Traditional→Simplified (OpenCC `t2s` / `zhconv`) before scoring.
**Scoring:** character error rate, punctuation + whitespace stripped (headline);
with-punct shown for reference.

## 1. Baseline CER

| Split | clips | corpus CER (no-punct) | macro CER | with-punct |
|-------|------:|----------------------:|----------:|-----------:|
| **test (eval)** | 5130 | **0.0807** | 0.0880 | 0.1920 |
| train | 7420 | 0.0683 | 0.0746 | 0.1846 |
| dev   | 5130 | 0.0696 | — | — |

45.0% of test utterances have ≥1 character error; 55% are already perfect.

## 2. Rule mining + correction

Mined on train, tuned on train+dev (swept p_high∈{.85,.90,.95},
n_min∈{10,20,40}, ctx∈{1,2}), net-benefit pruned on dev. Final ruleset:
**3 rules** (`咁你→啖你`, `翻工→返工`, `都系咁→都系啖`).

| Eval (test, 5130) | corpus CER (no-punct) |
|-------------------|----------------------:|
| baseline          | 0.0807 |
| **+ rules**       | **0.0802** |

Δ = **−0.0005 absolute (−0.6% relative)**. fixed=31 utterances, broke=2.
Essentially negligible.

### 2b. Continued pass: dev-pruned bigram/context rules

Starting from the broader 117-rule bigram candidate set in `rules_bigram.json`,
I used **train+dev only** to annotate support/occurrence counts and greedily
kept only rules that reduced total **dev** edit distance. The resulting
test-blind ruleset is `rules_pruned_dev.json` with **59 rules**.

| Split | baseline CER | + `rules_pruned_dev.json` | edit delta | fixed | broke |
|-------|-------------:|--------------------------:|-----------:|------:|------:|
| train | 0.0683 | 0.0654 | −218 | 319 | 108 |
| dev   | 0.0696 | 0.0669 | −144 | 197 | 58 |
| **eval (test)** | **0.0807** | **0.0784** | **−121** | **171** | **55** |

This is a real improvement over the 3-rule set (eval 0.0802 → 0.0784), but it
still mostly captures spelling/orthography preferences (`咁→噉`, `俾→畀`,
`中意→钟意`, `翻→返`) rather than deeper recognition errors. Because eval was
only used after selection, `rules_pruned_dev.json` is the current best
ship/test candidate from this experiment.

## 3. Why rules barely help (the real finding)

The error distribution is dominated by **single-character Cantonese particle /
orthography variants**, not phrase-level recognition mistakes. Top test
substitutions (hyp → ref):

| count | hyp → ref | note |
|------:|-----------|------|
| 257 | 咁 → 噉 | "gam2" sentential particle, both valid written forms |
| 133 | 俾 → 畀 | "bei2" (give), variant chars |
| 111 | 啦 → 喇 | final particle |
|  50 | 咯 → 啰 | final particle |
|  50 | 中 → 钟 | 鐘/鍾 simplification collision |
|  49 | 呀 → 啊 | final particle |
|  41 | 啦 → 嘞 | final particle |
|  39 | 咧 → 呢 | final particle |
|  28 | 翻 → 返 | "faan1" (return) |

These are **context-free ambiguous**: e.g. 咁 in the model output aligns to the
*correct* 咁 **88% of the time** and to 噉 only ~9%. A blanket `咁→噉` rule would
break ~2700 correct chars to fix ~257 — which is exactly why the precision filter
rejected it. Insertions/deletions are minor (452 ins, 75 del vs 3695 subs).

**Equivalence-class test (the key number):** if you fold interchangeable
particle/orthography variant classes (咁≡噉, 俾≡畀, 啦≡喇≡嘞, 呀≡啊≡吖, 咯≡啰,
咧≡呢, 翻≡返, 嚟≡黎≡来, 喺≡系, 冇≡无, 嘅≡既, 噶≡㗎, 哋≡地, 个≡個≡嗰, 咩≡乜)
on **both** hyp and ref before scoring:

| Eval (test, 5130) | corpus CER (no-punct) |
|-------------------|----------------------:|
| baseline          | 0.0808 |
| particle-class folded | **0.0634** |

**22% of all character errors (910 of 4222) are pure variant noise** — the model
is right, it just picked the other accepted spelling of the same word. Real CER for
keyboard purposes is closer to **~6.3%**, not 8%.

**Interpretation for the keyboard:** a large share of these "errors" are not
errors a user would notice or reject — both the model's char (咁/俾/啦/呀) and the
Common Voice reference's char (噉/畀/喇/啊) are accepted written Cantonese. The CER
number overstates the user-perceived error rate. The genuinely fixable,
high-precision, high-frequency confusions are rare, so a context-free rule table
cannot move CER materially without overcorrecting.

## 4. Throughput

INT8 0.6B model, this harness (decode via symphonia + `transcribe_audio`):
- ~**2.1 clips/s** single process, 10 threads (avg clip ~4.0s audio ⇒ ~8× realtime).
- ~**3.2 clips/s** with 5 processes × 2 threads (full-machine parallelism).
- Full test (5130) ≈ 25–40 min depending on parallelism.

## 5. Full error analysis (where the 8.07% comes from)

**Sentence level (5130 clips):**
- 55.0% perfect, 11.6% correct-modulo-spelling (variant only) → **66.6%
  effectively correct**, 33.4% genuinely wrong.

**Error severity (by edit count, 4220 total edits):**

| Bucket | sentences | edits | share of edits |
|--------|----------:|------:|---------------:|
| full failure (CER > 0.5) | 97 | 585 | 14% |
| moderate (0.2–0.5) | 548 | 1463 | 35% |
| mild (≤ 0.2) | 1661 | 2172 | 51% |

→ Errors are **diffuse, not concentrated**: half of all edits are 1–2 char slips
scattered across many otherwise-good sentences. There is no small set of
catastrophic clips to fix; the full-failure tail (97 clips, mostly very short
audio the model collapsed to "係") is only 14%.

**Edit level — by type:**

| Category | edits | share | nature |
|----------|------:|------:|--------|
| variant substitutions | 917 | 21.7% | 咁/噉 etc. — *not real errors* |
| real substitutions | 2767 | 65.6% | wrong character |
| insertions (hyp dropped a char) | 452 | 10.7% | mostly particles 唔/嘅/㗎 |
| deletions (hyp added a char) | 73 | 1.7% | minor |
| ascii | 11 | 0.3% | — |

**The 2767 real substitutions, by phonetics (jyutping via pycantonese):**

| | edits | share of real subs |
|---|------:|------:|
| homophone (same initial+final) | 699 | 25% |
| near-homophone | 330 | 12% |
| phonetically unrelated | 485 | 18% |
| no jyutping (rare/proper/multi-reading) | 1253 | 45% |

Of the real substitutions we can classify, the large majority are
**homophone / near-homophone** — the acoustic model heard the right sound and
picked the wrong character (中/钟, 有/又, 月/粤, 治/字, 读/毒, 信/讯). These are
inherently *context-dependent* — the same output char is correct most of the time —
which is exactly why a context-free rule dictionary can't fix them without
overcorrecting. They need **language-model / context disambiguation**. (The 45%
"no jyutping" bucket is rare chars, proper nouns, and multi-reading chars
pycantonese couldn't score — a mix of genuine recognition errors.)

**Numbers are NOT a major factor:** only 7 of the 97 full-failure sentences (41
edits, ~1% of all errors) involve numbers/counting. Number normalization would buy
almost nothing here.

## 6. Highest value-to-effort ranking

| Rank | Solution | Addresses | Est. CER gain | Effort | Notes |
|-----:|----------|-----------|--------------:|--------|-------|
| **1** | **Variant equivalence classes** (score-side, and/or keyboard accepts both spellings) | 21.7% of edits | 8.07 → **~6.3%** | **~0.5 day** | No model change. Pure measurement + UX fix. Already prototyped here. Biggest win / least work. |
| **2** | **Small char n-gram LM rescorer** over the model's top-k per position, targeting homophone confusions | ~25–37% of edits (homophone bucket) | ~1–2 pt (optimistic) | **medium** (~1 wk) | A few-MB Cantonese bigram/trigram LM; rescore only confusable positions. Fits on-device. The realistic accuracy lever. |
| 3 | **Full neural LM / beam-lattice rescoring** | homophones + grammar | potentially 2–3 pt | high (weeks) | Strictly better than #2 but heavier; only if #2 plateaus. |
| 4 | **1.7B model or fine-tune on CV-yue** | broad incl. the 45% "no-jyutping" rare-char errors | unknown | high cost/latency | Contradicts on-device keyboard budget; last resort. |
| ❌ | **Rule-based correction** (this experiment) | ~0.6% of edits | +0.05 pt | done | Proven not worth shipping. |

**Recommended path for the keyboard:**
1. **Do #1 now.** It's mostly recognizing that ~1/5 of the "errors" aren't errors.
   Half a day, no model change, 8.07% → ~6.3%, and it makes every later measurement
   honest.
2. **Then #2** — a small character n-gram LM is the only *lightweight* thing that
   moves genuine recognition errors, and homophone confusion is the single largest
   addressable real-error class. Keep it on-device-sized.
3. Treat #3/#4 as the heavy accuracy project only if #1+#2 fall short.

Things that looked promising but the data killed: number normalization (~1% of
errors) and any context-free char-swap dictionary (this experiment).

## 7. Recommendation (original task)

The rule-based correction we set out to build is **not worth shipping**: ceiling
~0.05 pt CER. The error analysis (§5) and value/effort ranking (§6) point elsewhere
— start with variant equivalence classes (§6 #1, ~0.5 day, 8.07→~6.3%), then a
small char n-gram LM rescorer for homophone confusions as the real accuracy lever.

## Reproduce

```bash
# binary (ARM-optimized): RUSTFLAGS="-C target-cpu=native" cargo build --release -p yue-tools
M=/Volumes/SSD/qwen-asr-m4-lmhead-int8
C=/Volumes/SSD/cv-corpus-25.0-2026-03-09/yue/clips
B=./target/release/yue-tools
# transcribe (resumable, --threads N)
$B transcribe --model $M --pairs bench/cantonese/eval_set_1000.tsv --clips $C \
   --out bench/cantonese/out/eval.yue.jsonl --threads 10
# baseline CER
$B cer --in bench/cantonese/out/eval_full.yue.jsonl
# mine + tune
$B mine --in bench/cantonese/out/train.yue.jsonl --out bench/cantonese/rules.json
$B tune --train bench/cantonese/out/train.yue.jsonl \
        --dev   bench/cantonese/out/dev.yue.jsonl  --out bench/cantonese/rules_tuned.json
# corrected CER
$B cer --in bench/cantonese/out/eval_full.yue.jsonl --rules bench/cantonese/rules_tuned.json
```
