#!/usr/bin/env python3
"""Freeze a reproducible eval subset from a Common Voice split.

Samples N rows (default 1000) from a CV .tsv with a fixed seed and writes a
2-column TSV: clip_filename<TAB>sentence. The output is committed so every eval
run uses the exact same clips.

Usage:
  python3 pick_eval_set.py \
    --tsv /Volumes/SSD/cv-corpus-25.0-2026-03-09/yue/test.tsv \
    --n 1000 --seed 1234 \
    --out bench/cantonese/eval_set_1000.tsv
"""
import argparse
import csv
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            path = (r.get("path") or "").strip()
            sent = (r.get("sentence") or "").strip()
            if path and sent:
                rows.append((path, sent))

    if len(rows) < args.n:
        print(f"warning: only {len(rows)} usable rows < n={args.n}", file=sys.stderr)

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    sample.sort(key=lambda x: x[0])  # stable on-disk order

    with open(args.out, "w") as f:
        for path, sent in sample:
            f.write(f"{path}\t{sent}\n")

    print(f"wrote {len(sample)} rows to {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
