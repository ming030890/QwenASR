#!/usr/bin/env python3
"""Prepare deterministic Android benchmark WAV assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "ming030890/common_voice_21_0_yue"
API = "https://datasets-server.huggingface.co"
TRANSCRIPT_BASE = "https://huggingface.co/datasets/ming030890/common_voice_21_0/resolve/main/transcript/yue"
PTT_SOURCE = Path(os.environ.get("PTT_SOURCE", os.path.expanduser("~/Downloads/PTT-20241208-WA0011.opus")))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def hf_url_to_https(url: str) -> str:
    if not url.startswith("hf://datasets/"):
        return url
    rest = url[len("hf://datasets/") :]
    namespace, repo_rev_name, path = rest.split("/", 2)
    repo_rev = f"{namespace}/{repo_rev_name}"
    repo, rev = repo_rev.split("@", 1)
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def convert_to_wav(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dest),
        ],
        check=True,
    )


def choose_split() -> tuple[str, str]:
    url = f"{API}/splits?dataset={urllib.parse.quote(DATASET, safe='')}"
    payload = get_json(url)
    splits = payload.get("splits", [])
    if not splits:
        raise RuntimeError(f"no splits returned for {DATASET}")

    preferred = ["validated", "validation", "dev", "test", "train"]
    for split_name in preferred:
        for split in splits:
            if split.get("split") == split_name:
                return split.get("config", "default"), split_name
    split = splits[0]
    return split.get("config", "default"), split["split"]


def row_text(row: dict) -> str:
    for key in ("sentence", "text", "transcription", "normalized_text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    for value in row.values():
        if isinstance(value, dict):
            nested = row_text(value)
            if nested:
                return nested
    return ""


def audio_src(row: dict) -> str | None:
    for key in ("audio", "mp3", "wav"):
        value = row.get(key)
        if isinstance(value, dict):
            src = value.get("src") or value.get("path")
            if isinstance(src, str) and src.startswith(("http://", "https://")):
                return src
    return None


def source_split(row: dict, fallback: str) -> str:
    source = row.get("__url__", "")
    parts = source.split("/")
    if "audio" in parts:
        index = parts.index("audio")
        if index + 2 < len(parts):
            return parts[index + 2]
    key = row.get("__key__", "")
    if key.startswith("yue_"):
        return key.split("_", 2)[1]
    return {"validation": "dev"}.get(fallback, fallback)


def transcript_map(split: str, cache_dir: Path) -> dict[str, str]:
    tsv_path = cache_dir / f"{split}.tsv"
    if not tsv_path.exists():
        download(f"{TRANSCRIPT_BASE}/{split}.tsv", tsv_path)

    refs = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            path = row.get("path", "")
            sentence = row.get("sentence", "")
            if path and sentence:
                if not path.endswith(".mp3"):
                    path += ".mp3"
                refs[path] = sentence
    return refs


def extract_from_tar(row: dict, cache_dir: Path) -> Path:
    tar_url = hf_url_to_https(row["__url__"])
    tar_path = cache_dir / Path(urllib.parse.urlparse(tar_url).path).name
    if not tar_path.exists():
        download(tar_url, tar_path)

    key = row["__key__"]
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        match = next((m for m in members if m.name.startswith(key) and m.isfile()), None)
        if match is None:
            stem = Path(key).name
            match = next((m for m in members if Path(m.name).stem == stem and m.isfile()), None)
        if match is None:
            raise RuntimeError(f"could not find {key} in {tar_path}")

        out = cache_dir / Path(match.name).name
        extracted = tar.extractfile(match)
        if extracted is None:
            raise RuntimeError(f"could not extract {match.name} from {tar_path}")
        with out.open("wb") as fh:
            shutil.copyfileobj(extracted, fh)
        return out


def fetch_common_voice(output_dir: Path, cache_dir: Path, count: int) -> list[dict]:
    config, split = choose_split()
    url = (
        f"{API}/rows?dataset={urllib.parse.quote(DATASET, safe='')}"
        f"&config={urllib.parse.quote(config, safe='')}"
        f"&split={urllib.parse.quote(split, safe='')}&offset=0&length={max(count, 10)}"
    )
    payload = get_json(url)
    rows = [item.get("row", {}) for item in payload.get("rows", [])]
    if len(rows) < count:
        raise RuntimeError(f"only {len(rows)} rows returned for {DATASET} {config}/{split}")

    records = []
    refs_by_split: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows[:count]):
        src = audio_src(row)
        raw_path = cache_dir / f"cv_yue_{index:03d}.audio"
        if src:
            download(src, raw_path)
        else:
            raw_path = extract_from_tar(row, cache_dir)

        wav_path = output_dir / f"cv_yue_{index:03d}.wav"
        txt_path = output_dir / f"cv_yue_{index:03d}.txt"
        convert_to_wav(raw_path, wav_path)
        text = row_text(row)
        split_for_refs = source_split(row, split)
        if not text:
            refs = refs_by_split.setdefault(split_for_refs, transcript_map(split_for_refs, cache_dir))
            text = refs.get(Path(row.get("__key__", "")).name + ".mp3", "")
        txt_path.write_text(text + "\n", encoding="utf-8")
        records.append(
            {
                "name": wav_path.name,
                "reference": text,
                "dataset": DATASET,
                "config": config,
                "split": split,
                "transcript_split": split_for_refs,
                "source_key": row.get("__key__", ""),
                "source_url": row.get("__url__", src or ""),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("bench/android-assets"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/Volumes/SSD/qwen-asr-android-assets"))
    parser.add_argument("--ptt-source", type=Path, default=PTT_SOURCE)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    ptt_opus = args.output_dir / "ptt_guardrail.opus"
    ptt_wav = args.output_dir / "ptt_guardrail.wav"
    shutil.copyfile(args.ptt_source, ptt_opus)
    convert_to_wav(ptt_opus, ptt_wav)
    (args.output_dir / "ptt_guardrail.txt").write_text("Hello,你好吗？已经坐低咗啦。\n", encoding="utf-8")

    records = fetch_common_voice(args.output_dir, args.cache_dir, args.count)
    metadata = {
        "ptt_guardrail": {
            "name": ptt_wav.name,
            "reference": "Hello,你好吗？已经坐低咗啦。",
            "source": str(args.ptt_source),
        },
        "common_voice": records,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
