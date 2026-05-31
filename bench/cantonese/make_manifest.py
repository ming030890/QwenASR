#!/usr/bin/env python3
"""Transcode CV mp3 clips to 16kHz mono WAV and build a transcription manifest.

Given a 2-col TSV (clip_filename<TAB>sentence), this:
  - transcodes each mp3 in <clips_dir> to a cached 16kHz mono 16-bit WAV
    (dedup by clip id; parallel ffmpeg),
  - writes <out>.manifest.txt  (newline-separated WAV paths, in TSV order),
  - writes <out>.refs.jsonl    ({"path": wav_path, "ref": sentence}).

Usage:
  python3 make_manifest.py \
    --pairs bench/cantonese/eval_set_1000.tsv \
    --clips /Volumes/SSD/cv-corpus-25.0-2026-03-09/yue/clips \
    --wav-cache bench/cantonese/out/wav_cache \
    --out bench/cantonese/out/eval
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def transcode(args):
    mp3, wav = args
    if os.path.exists(wav) and os.path.getsize(wav) > 0:
        return wav, True
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", mp3, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav,
    ]
    rc = subprocess.run(cmd).returncode
    return wav, rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--wav-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.wav_cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pairs = []
    with open(args.pairs) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                pairs.append((p[0], p[1]))

    jobs = []
    entries = []  # (wav, ref)
    for fname, sent in pairs:
        stem = os.path.splitext(os.path.basename(fname))[0]
        mp3 = os.path.join(args.clips, fname)
        wav = os.path.join(args.wav_cache, stem + ".wav")
        jobs.append((mp3, wav))
        entries.append((wav, sent))

    ok = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for wav, good in ex.map(transcode, jobs):
            ok += 1 if good else 0

    manifest = args.out + ".manifest.txt"
    refs = args.out + ".refs.jsonl"
    with open(manifest, "w") as mf, open(refs, "w") as rf:
        for wav, ref in entries:
            if os.path.exists(wav) and os.path.getsize(wav) > 0:
                mf.write(wav + "\n")
                rf.write(json.dumps({"path": wav, "ref": ref}, ensure_ascii=False) + "\n")

    print(f"transcoded ok={ok}/{len(jobs)}; wrote {manifest} and {refs}")


if __name__ == "__main__":
    main()
