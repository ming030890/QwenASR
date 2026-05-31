#!/usr/bin/env python3
"""Join hypotheses with references and report CER (with/without punctuation).

Usage:
  python3 score.py \
    --hyp bench/cantonese/out/eval.hyp.jsonl \
    --refs bench/cantonese/out/eval.refs.jsonl \
    [--rules bench/cantonese/rules.json]   # optional: apply correction first

Outputs corpus + macro CER. If --rules is given, also prints corrected CER and
the delta, plus a few example fixes/breaks.
"""
import argparse
import json

from lib import cer_stats, read_jsonl, normalize


def load_rules(path):
    data = json.load(open(path))
    rules = data["rules"] if isinstance(data, dict) else data
    # longest-match-first, then highest precision
    rules.sort(key=lambda r: (-len(r["from"]), -r.get("precision", 0.0)))
    return rules


def apply_rules(text, rules):
    # single left-to-right pass, longest-match at each position
    out = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for r in rules:
            f = r["from"]
            if f and text.startswith(f, i):
                out.append(r["to"])
                i += len(f)
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--rules")
    args = ap.parse_args()

    hyp = {r["path"]: r.get("text", "") for r in read_jsonl(args.hyp)}
    refs = {r["path"]: r["ref"] for r in read_jsonl(args.refs)}
    keys = [k for k in refs if k in hyp]
    print(f"matched {len(keys)} / {len(refs)} clips")

    base_pairs = [(hyp[k], refs[k]) for k in keys]
    for sp in (True, False):
        s = cer_stats(base_pairs, strip_punct=sp)
        tag = "no-punct" if sp else "with-punct"
        print(f"[baseline {tag:9}] corpus_cer={s['corpus_cer']:.4f} "
              f"macro_cer={s['macro_cer']:.4f} edits={s['edits']} chars={s['ref_chars']}")

    if not args.rules:
        return

    rules = load_rules(args.rules)
    print(f"\napplying {len(rules)} rules")
    corr_pairs = [(apply_rules(hyp[k], rules), refs[k]) for k in keys]
    for sp in (True, False):
        s = cer_stats(corr_pairs, strip_punct=sp)
        tag = "no-punct" if sp else "with-punct"
        print(f"[corrected {tag:9}] corpus_cer={s['corpus_cer']:.4f} "
              f"macro_cer={s['macro_cer']:.4f} edits={s['edits']} chars={s['ref_chars']}")

    # show examples where correction changed edit distance (no-punct)
    from lib import levenshtein
    fixed, broke = [], []
    for k in keys:
        r = list(normalize(refs[k], True))
        h0 = list(normalize(hyp[k], True))
        h1 = list(normalize(apply_rules(hyp[k], rules), True))
        d0, d1 = levenshtein(h0, r), levenshtein(h1, r)
        if d1 < d0:
            fixed.append((k, hyp[k], apply_rules(hyp[k], rules), refs[k]))
        elif d1 > d0:
            broke.append((k, hyp[k], apply_rules(hyp[k], rules), refs[k]))
    print(f"\nfixed={len(fixed)} broke={len(broke)}")
    for tag, lst in (("FIX", fixed[:5]), ("BREAK", broke[:5])):
        for k, h0, h1, ref in lst:
            print(f"  [{tag}] hyp={h0!r} -> {h1!r} | ref={ref!r}")


if __name__ == "__main__":
    main()
