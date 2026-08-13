#!/usr/bin/env python3
"""Run and document a repeatable QwenASR inference experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
DEFAULT_BINARY = PROJECT_DIR / "target/release/qwen-asr-bench-android"
DEFAULT_RUNS_DIR = EXPERIMENT_DIR / "runs"


def fail(message: str) -> None:
    raise SystemExit(message)


def run_text(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def git_value(*args: str) -> str:
    return run_text(["git", "-C", str(PROJECT_DIR), *args])


def directory_inventory(root: Path) -> tuple[int, int, str, list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    total = 0
    weights = 0
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if ".cache" in path.relative_to(root).parts:
            continue
        size = path.stat().st_size
        relative = path.relative_to(root).as_posix()
        total += size
        if path.suffix.lower() in {".safetensors", ".bin", ".gguf", ".onnx"}:
            weights += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        files.append({"path": relative, "bytes": size})
    return total, weights, digest.hexdigest(), files


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def host_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        hardware = run_text(["/usr/sbin/system_profiler", "SPHardwareDataType"])
        for source, key in (("Model Name", "model"), ("Chip", "chip"), ("Memory", "memory")):
            match = re.search(rf"^\s*{re.escape(source)}:\s*(.+)$", hardware, re.MULTILINE)
            if match:
                metadata[key] = match.group(1).strip()
        metadata["os_version"] = run_text(["sw_vers", "-productVersion"])
    return metadata


def build_binary() -> None:
    openblas_lib_dir = Path(
        os.environ.get("OPENBLAS_LIB_DIR", "/opt/homebrew/opt/openblas/lib")
    ).expanduser()
    if not any(openblas_lib_dir.glob("libopenblas.*")):
        fail(
            f"OpenBLAS library not found in {openblas_lib_dir}; install it with "
            "`brew install openblas` or set OPENBLAS_LIB_DIR"
        )
    command = [
        "cargo", "build", "--release", "-p", "qwen-asr-cli", "--bin",
        "qwen-asr-bench-android",
    ]
    build_environment = os.environ.copy()
    build_environment.update({
        "QWEN_BLAS_BACKEND": "openblas",
        "OPENBLAS_LIB_DIR": str(openblas_lib_dir.resolve()),
    })
    print("Building release benchmark binary with OpenBLAS...", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, env=build_environment, check=True)


def audio_key(audio_path: str, input_path: Path) -> str:
    path = Path(audio_path)
    if input_path.is_file():
        return input_path.name
    try:
        return path.resolve().relative_to(input_path).as_posix()
    except ValueError:
        return path.name


def collect_values(payloads: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    inference: list[float] = []
    wall_inference: list[float] = []
    model_load: list[float] = []
    phase_names = ("mel_ms", "encoder_ms", "prefill_ms", "decode_loop_ms")
    phases: dict[str, list[float]] = {name: [] for name in phase_names}
    transcripts: list[str] = []
    transcript_runs: dict[str, list[str]] = {}
    for payload in payloads:
        model_load.append(float(payload["model_load_ms"]))
        for file_result in payload["files"]:
            transcripts.append(file_result.get("transcript", ""))
            key = audio_key(file_result["audio_path"], input_path)
            transcript_runs.setdefault(key, []).extend(
                str(run.get("transcript", ""))
                for run in file_result.get("run_details", [])
            )
            for run in file_result.get("run_details", []):
                inference.append(float(run["inference_ms"]))
                wall_inference.append(float(run["wall_inference_ms"]))
            for name in phase_names:
                if name in file_result:
                    phases[name].append(float(file_result[name]))
    transcripts_by_input = {
        key: {
            "canonical": values[0] if values else "",
            "all_runs_equal": bool(values) and len(set(values)) == 1,
            "samples": len(values),
            "values": values,
        }
        for key, values in sorted(transcript_runs.items())
    }
    return {
        "inference_latency_ms": {
            "median": statistics.median(inference),
            "p95": percentile(inference, 0.95),
            "min": min(inference),
            "max": max(inference),
            "samples": len(inference),
        },
        "wall_inference_latency_ms": {
            "median": statistics.median(wall_inference),
            "p95": percentile(wall_inference, 0.95),
        },
        "model_load_ms": {
            "median": statistics.median(model_load),
            "first_launch": model_load[0],
            "min": min(model_load),
            "max": max(model_load),
            "samples": len(model_load),
        },
        "phase_medians_ms": {
            name.removesuffix("_ms"): statistics.median(values) if values else None
            for name, values in phases.items()
        },
        "transcripts": list(dict.fromkeys(transcripts)),
        "transcripts_by_input": transcripts_by_input,
    }


def load_baseline(reference: str, runs_dir: Path) -> tuple[Path, dict[str, Any]]:
    supplied = Path(reference).expanduser()
    if supplied.exists():
        path = supplied / "metrics.json" if supplied.is_dir() else supplied
    else:
        path = runs_dir / reference / "metrics.json"
    if not path.is_file():
        fail(f"baseline metrics not found: {path}")
    try:
        return path.resolve(), json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid baseline metrics in {path}: {error}")


def transcript_acceptance(
    candidate: dict[str, Any], baseline: dict[str, Any] | None
) -> dict[str, Any]:
    candidate_files = candidate.get("transcripts_by_input", {})
    candidate_stable = bool(candidate_files) and all(
        item.get("all_runs_equal", False) for item in candidate_files.values()
    )
    result: dict[str, Any] = {
        "rule": "Every transcript must exactly equal the baseline, including case, spacing, and punctuation.",
        "candidate_runs_internally_equal": candidate_stable,
        "baseline_runs_internally_equal": None,
        "same_input_set": None,
        "exact_transcript_match": None,
        "passed": candidate_stable,
        "mismatches": [],
    }
    if baseline is None:
        result["mode"] = "baseline-establishment"
        return result

    baseline_files = baseline.get("summary", {}).get("transcripts_by_input")
    if not isinstance(baseline_files, dict) or not baseline_files:
        fail(
            "baseline lacks per-input transcript data; rerun it with the current "
            "collector before evaluating a candidate"
        )
    baseline_stable = all(
        item.get("all_runs_equal", False) for item in baseline_files.values()
    )
    same_inputs = set(candidate_files) == set(baseline_files)
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(candidate_files) | set(baseline_files)):
        expected = baseline_files.get(key, {}).get("canonical")
        actual_values = candidate_files.get(key, {}).get("values", [])
        if expected is None or not actual_values or any(value != expected for value in actual_values):
            mismatches.append({
                "input": key,
                "baseline": expected,
                "candidate_values": list(dict.fromkeys(actual_values)),
            })
    exact = candidate_stable and baseline_stable and same_inputs and not mismatches
    result.update({
        "mode": "candidate-comparison",
        "baseline_runs_internally_equal": baseline_stable,
        "same_input_set": same_inputs,
        "exact_transcript_match": exact,
        "passed": exact,
        "mismatches": mismatches,
    })
    return result


def mib(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value / 1024 / 1024:.2f}"


def write_experiment_md(
    path: Path, args: argparse.Namespace, metrics: dict[str, Any]
) -> None:
    summary = metrics["summary"]
    size = metrics["file_size"]
    peak = metrics["memory"]["peak_rss_bytes"]
    transcript_gate = metrics["acceptance"]["transcript_gate"]
    gate_label = "PASS" if transcript_gate["passed"] else "FAIL"
    transcripts = "\n".join(f"- `{text}`" for text in summary["transcripts"]) or "- n/a"
    content = f"""# {args.id}: Qwen inference experiment

- Status: measured; conclusion pending
- Date (UTC): {metrics['timestamp_utc']}
- Platform: {metrics['host'].get('model', metrics['host']['system'])} / {metrics['host'].get('chip', metrics['host']['machine'])}
- Git revision: `{metrics['git']['revision']}` ({'dirty' if metrics['git']['dirty'] else 'clean'})
- Baseline: {args.baseline or 'none'}

## Hypothesis

{args.hypothesis}

## Change

{args.change}

## Fixed setup

- Model: `{metrics['model']['path']}`
- Detected variant: {metrics['model']['detected_variant']}
- Model inventory fingerprint: `{metrics['model']['inventory_sha256']}`
- Input: `{metrics['input']}`
- Threads: {args.threads}
- Launches × runs: {args.launches} × {args.runs}
- Language: {args.language or 'default'}
- Prompt: {args.prompt or 'none'}

## Results

| Metric | Result |
| --- | ---: |
| Median inference latency | {summary['inference_latency_ms']['median']:.3f} ms |
| P95 inference latency | {summary['inference_latency_ms']['p95']:.3f} ms |
| Median model load time | {summary['model_load_ms']['median']:.3f} ms |
| First-launch model load time | {summary['model_load_ms']['first_launch']:.3f} ms |
| RSS after model load | {mib(metrics['memory']['after_model_load_bytes_median'])} MiB |
| Peak RSS | {mib(peak)} MiB |
| Model files | {mib(size['model_total_bytes'])} MiB |
| Weight files | {mib(size['model_weight_bytes'])} MiB |
| Runtime binary | {mib(size['runtime_binary_bytes'])} MiB |

Unique transcripts:

{transcripts}

Full normalized measurements: [`metrics.json`](metrics.json).

## Acceptance gates

| Gate | Result |
| --- | --- |
| Exact transcript equality | **{gate_label}** |
| Candidate repetitions identical | {'PASS' if transcript_gate['candidate_runs_internally_equal'] else 'FAIL'} |
| Baseline repetitions identical | {('n/a' if transcript_gate['baseline_runs_internally_equal'] is None else ('PASS' if transcript_gate['baseline_runs_internally_equal'] else 'FAIL'))} |
| Same fixed input set | {('n/a' if transcript_gate['same_input_set'] is None else ('PASS' if transcript_gate['same_input_set'] else 'FAIL'))} |

Exact means byte-for-byte Unicode string equality, including capitalization,
spacing, and punctuation. An experiment cannot be accepted when this gate fails.

## Decision

{('Reject: the exact-transcript acceptance gate failed.' if not transcript_gate['passed'] else ('Baseline transcript is stable; use this experiment as the named baseline for candidates.' if not args.baseline else 'Exact-transcript gate passed. Decide keep/reject from the performance thresholds.'))}

## Conclusion

{('Rejected automatically because at least one transcript differed from the baseline or was unstable across repetitions.' if not transcript_gate['passed'] else 'Transcript correctness is preserved. State whether the performance hypothesis was supported, the percentage deltas from baseline, important caveats, and the next experiment.')}
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="Unique filesystem-safe experiment ID")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="WAV file or directory of WAV files")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--launches", type=int, default=3, help="Fresh process launches")
    parser.add_argument("--runs", type=int, default=3, help="Inference runs per input per launch")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--expect-model-variant", default="1.7B",
        help="Fail unless the engine detects this variant (default: 1.7B)",
    )
    parser.add_argument(
        "--require-backend", default="OpenBLAS",
        help="Fail unless the binary reports this optimization backend (default: OpenBLAS)",
    )
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--prompt", default="音频是粤语，可能夹杂英语")
    parser.add_argument("--hypothesis", default="Not recorded")
    parser.add_argument("--change", default="Not recorded")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.id):
        fail("--id must contain only letters, digits, dot, underscore, and hyphen")
    if args.launches < 1 or args.runs < 1 or args.threads < 1:
        fail("--launches, --runs, and --threads must be positive")

    args.model_dir = args.model_dir.expanduser().resolve()
    args.input = args.input.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    args.runs_dir = args.runs_dir.expanduser().resolve()
    if not args.model_dir.is_dir():
        fail(f"model directory not found: {args.model_dir}")
    if not args.input.exists():
        fail(f"input not found: {args.input}")

    # Capture repository state before creating the untracked result directory.
    git_metadata = {
        "revision": git_value("rev-parse", "HEAD"),
        "short_revision": git_value("rev-parse", "--short", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
    }

    output_dir = args.runs_dir / args.id
    if output_dir.exists():
        fail(f"experiment already exists (results are never overwritten): {output_dir}")

    baseline_path: Path | None = None
    baseline_metrics: dict[str, Any] | None = None
    if args.baseline:
        baseline_path, baseline_metrics = load_baseline(args.baseline, args.runs_dir)
        baseline_flags = baseline_metrics.get("configuration", {}).get(
            "benchmark_optimization_flags", []
        )
        if args.require_backend not in baseline_flags:
            fail(
                f"baseline {args.baseline!r} does not report required backend "
                f"{args.require_backend!r} (flags: {baseline_flags!r}); establish a "
                "new OpenBLAS baseline first"
            )

    if not args.skip_build:
        build_binary()
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        fail(f"benchmark binary not found or not executable: {args.binary}")

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    payloads: list[dict[str, Any]] = []
    peaks: list[int] = []
    benchmark_args = [
        "--model-dir", str(args.model_dir), "--input", str(args.input),
        "--runs", str(args.runs), "--threads", str(args.threads),
    ]
    if args.language:
        benchmark_args.extend(["--language", args.language])
    if args.prompt:
        benchmark_args.extend(["--prompt", args.prompt])

    for launch in range(1, args.launches + 1):
        json_path = raw_dir / f"launch-{launch:03d}.json"
        log_path = raw_dir / f"launch-{launch:03d}.log"
        command = [str(args.binary), *benchmark_args]
        print(f"Launch {launch}/{args.launches}...", flush=True)
        with json_path.open("w", encoding="utf-8") as stdout, log_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=False)
        if completed.returncode != 0:
            fail(f"benchmark launch {launch} failed; inspect {log_path}")
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            fail(f"invalid benchmark JSON in {json_path}: {error}")
        detected_variant = payload.get("model_variant")
        if detected_variant != args.expect_model_variant:
            fail(
                f"expected model variant {args.expect_model_variant!r}, but the engine "
                f"reported {detected_variant!r}; results remain in {output_dir} for diagnosis"
            )
        optimization_flags = payload.get("optimization_flags", [])
        if args.require_backend not in optimization_flags:
            fail(
                f"required backend {args.require_backend!r} is absent from reported "
                f"optimization flags {optimization_flags!r}; results remain in "
                f"{output_dir} for diagnosis"
            )
        payloads.append(payload)
        peak_kb = max(
            [payload.get("memory_after_model_load_kb", 0)]
            + [
                run.get("peak_rss_kb", 0)
                for file_result in payload.get("files", [])
                for run in file_result.get("run_details", [])
            ]
        )
        if peak_kb:
            peaks.append(int(peak_kb) * 1024)

    model_total, model_weights, fingerprint, model_files = directory_inventory(args.model_dir)
    summary = collect_values(payloads, args.input)
    transcript_gate = transcript_acceptance(summary, baseline_metrics)
    memory_after_load = [
        int(payload.get("memory_after_model_load_kb", 0)) * 1024
        for payload in payloads
        if payload.get("memory_after_model_load_kb", 0)
    ]
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": args.id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": args.hypothesis,
        "change": args.change,
        "baseline": {
            "reference": args.baseline,
            "metrics_path": str(baseline_path) if baseline_path else None,
            "experiment_id": baseline_metrics.get("experiment_id") if baseline_metrics else None,
        } if args.baseline else None,
        "git": git_metadata,
        "host": host_metadata(),
        "model": {
            "path": str(args.model_dir),
            "detected_variant": payloads[0]["model_variant"],
            "expected_variant": args.expect_model_variant,
            "inventory_sha256": fingerprint,
            "files": model_files,
        },
        "input": str(args.input),
        "configuration": {
            "launches": args.launches,
            "runs_per_input": args.runs,
            "threads": args.threads,
            "language": args.language or None,
            "prompt": args.prompt or None,
            "binary": str(args.binary),
            "benchmark_optimization_flags": payloads[0].get("optimization_flags", []),
            "required_backend": args.require_backend,
        },
        "summary": summary,
        "acceptance": {
            "transcript_gate": transcript_gate,
            "eligible_for_performance_acceptance": transcript_gate["passed"],
        },
        "memory": {
            "after_model_load_bytes_median": (
                int(statistics.median(memory_after_load)) if memory_after_load else None
            ),
            "peak_rss_bytes": max(peaks) if peaks else None,
            "peak_rss_bytes_by_launch": peaks,
            "source": "benchmark process sampler",
        },
        "file_size": {
            "model_total_bytes": model_total,
            "model_weight_bytes": model_weights,
            "runtime_binary_bytes": args.binary.stat().st_size,
        },
        "raw_launch_files": [f"raw/launch-{i:03d}.json" for i in range(1, args.launches + 1)],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_experiment_md(output_dir / "experiment.md", args, metrics)

    print(f"Wrote {output_dir}")
    print(
        f"inference={summary['inference_latency_ms']['median']:.3f} ms  "
        f"load={summary['model_load_ms']['median']:.3f} ms  "
        f"peak_rss={mib(metrics['memory']['peak_rss_bytes'])} MiB  "
        f"model={mib(model_total)} MiB  binary={mib(args.binary.stat().st_size)} MiB"
    )
    print(f"exact_transcript_gate={'PASS' if transcript_gate['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
