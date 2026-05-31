#!/usr/bin/env python3
"""Run the qwen-asr binary in batch (--manifest) mode and cache hypotheses.

Usage:
  python3 run_asr.py \
    --bin target/release/qwen-asr \
    --model /Volumes/SSD/qwen-asr-m4-lmhead-int8 \
    --manifest bench/cantonese/out/eval.manifest.txt \
    --out bench/cantonese/out/eval.hyp.jsonl \
    [--language Cantonese]

Resumable: if --out already has a line for every manifest path, it is reused.
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/qwen-asr")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="Cantonese")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    n_paths = sum(1 for ln in open(args.manifest) if ln.strip())
    if os.path.exists(args.out) and not args.force:
        have = sum(1 for ln in open(args.out) if ln.strip())
        if have >= n_paths:
            print(f"reuse cached {args.out} ({have} rows)")
            return

    cmd = [args.bin, "-d", args.model, "--manifest", args.manifest, "--silent"]
    if args.language:
        cmd += ["--language", args.language]

    print("running:", " ".join(cmd), file=sys.stderr)
    with open(args.out, "w") as out:
        proc = subprocess.run(cmd, stdout=out)
    if proc.returncode != 0:
        print(f"qwen-asr exited {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

    have = sum(1 for ln in open(args.out) if ln.strip())
    print(f"wrote {have} hyp rows to {args.out}")


if __name__ == "__main__":
    main()
