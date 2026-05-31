"""Shared helpers for the Cantonese ASR correction experiment.

Normalization to Simplified Chinese (the model's default output script) and
character-level CER, reusing the same Levenshtein idea as bench/wer.py.
"""
import json
import re
import unicodedata

import opencc

_t2s = opencc.OpenCC("t2s")  # Traditional -> Simplified


def to_simplified(text: str) -> str:
    return _t2s.convert(text)


# Strip everything that isn't a CJK ideograph or alphanumeric for the
# punctuation-insensitive CER. Keeps Latin letters/digits (some refs mix them).
_PUNCT_RE = re.compile(
    r"[\s　-〿＀-￯ -⁯"
    r"!-/:-@\[-`{-~。，、？！；：「」『』（）《》〈〉…—·．]+"
)


def normalize(text: str, strip_punct: bool) -> str:
    """Normalize for scoring: NFKC (unifies full/half width), T->S, optional
    punctuation removal, and whitespace removal (CER is computed over chars)."""
    text = unicodedata.normalize("NFKC", text)
    text = to_simplified(text)
    if strip_punct:
        text = _PUNCT_RE.sub("", text)
    else:
        text = re.sub(r"\s+", "", text)
    return text


def levenshtein(a, b) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def cer_stats(pairs, strip_punct: bool):
    """pairs: list of (hyp, ref). Returns dict with corpus + macro CER."""
    total_edits = 0
    total_chars = 0
    macro_sum = 0.0
    n = 0
    for hyp, ref in pairs:
        h = list(normalize(hyp, strip_punct))
        r = list(normalize(ref, strip_punct))
        if not r:
            continue
        d = levenshtein(h, r)
        total_edits += d
        total_chars += len(r)
        macro_sum += d / len(r)
        n += 1
    return {
        "corpus_cer": total_edits / max(total_chars, 1),
        "macro_cer": macro_sum / max(n, 1),
        "edits": total_edits,
        "ref_chars": total_chars,
        "n": n,
    }


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_pairs_tsv(path):
    """Read a 2-col TSV: clip_path<TAB>sentence (no header)."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    return out
