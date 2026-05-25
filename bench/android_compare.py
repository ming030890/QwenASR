#!/usr/bin/env python3
"""Run baseline and candidate binaries back-to-back for a same-session gate check.

Same-session comparison eliminates between-session thermal/scheduling noise
(observed at 6-13% per clip).  Both binaries run under identical device state
so the tight gate thresholds (5-10%) are meaningful.

Usage:
  bench/android_compare.py --baseline-binary PATH --label LABEL [options]

The baseline binary must be a pre-built aarch64-linux-android release binary.
Save one after an accepted run with:
  bench/android_run.py --save-binary bench/android-baselines/my-baseline [...]

This script:
  1. Waits for device to reach Thermal Status 0.
  2. Pushes and runs the baseline binary → saves to {label}-ctrl-TIMESTAMP/
  3. Builds the candidate from source.
  4. Pushes and runs the candidate binary → saves to {label}-cand-TIMESTAMP/
  5. Runs android_check_gate.py --result cand --baseline ctrl --stage STAGE.

Env vars (same defaults as android_run.py):
  ADB, MODEL_DIR, NDK_HOME / ANDROID_NDK_HOME / ANDROID_NDK_ROOT, NDK_BIN,
  RUST_TOOLCHAIN_BIN, RUSTFLAGS_EXTRA, CARGO_FEATURES_EXTRA,
  BENCH_LANGUAGE, BENCH_PROMPT, BENCH_REMOTE_ENV,
  OPENBLAS_ANDROID_LIB_DIR, OPENBLAS_ANDROID_STATIC,
  OPENBLAS_ANDROID_RUNTIME_LIB, THERMAL_TIMEOUT
"""

import argparse
import atexit
import datetime
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_DIR  = SCRIPT_DIR.parent
GATE_SCRIPT  = SCRIPT_DIR / "android_check_gate.py"

GUARDRAIL_KEYWORDS = ["你好吗", "坐低咗"]

TSV_HEADER = "\t".join([
    "timestamp", "label", "threads", "input",
    "model_load_ms", "memory_after_model_load_kb", "peak_inference_rss_kb",
    "median_inference_ms", "median_encode_ms", "median_decode_ms",
    "rtf", "guardrail_ok", "json_path",
])


def env(key, default=""):
    return os.environ.get(key, default)


# ── adb helpers ───────────────────────────────────────────────────────────────

def adb_output(adb, *args):
    return subprocess.run(
        [adb] + [str(a) for a in args],
        capture_output=True, text=True, check=False,
    ).stdout


def wait_for_thermal_ready(adb, timeout):
    interval = 15
    elapsed = 0
    while True:
        out = adb_output(adb, "shell", "dumpsys", "thermalservice")
        status_line = next((l for l in out.splitlines() if "Thermal Status" in l), "")
        status = status_line.split()[-1].strip() if status_line else "?"
        if status == "0":
            print("Thermal Status: 0 — device ready.")
            return
        print(f"Thermal Status: {status} — waiting ({elapsed}s elapsed, timeout {timeout}s)...")
        if elapsed >= timeout:
            print(f"WARNING: thermal wait timed out after {timeout}s (status {status}). Proceeding.", file=sys.stderr)
            return
        time.sleep(interval)
        elapsed += interval


def prepare_device(adb):
    original = adb_output(adb, "shell", "settings", "get", "global",
                          "stay_on_while_plugged_in").strip().replace("\r", "") or None
    subprocess.run([adb, "shell", "input", "keyevent", "KEYCODE_WAKEUP"],
                   check=False, capture_output=True)
    subprocess.run([adb, "shell", "svc", "power", "stayon", "true"],
                   check=False, capture_output=True)
    subprocess.run([adb, "shell", "cmd", "power", "set-fixed-performance-mode-enabled", "true"],
                   check=False, capture_output=True)
    print("Device: screen on, stay-on, fixed-performance mode enabled.")
    return original


def cleanup_device(adb, original_stay_on):
    subprocess.run([adb, "shell", "cmd", "power", "set-fixed-performance-mode-enabled", "false"],
                   check=False, capture_output=True)
    if original_stay_on and original_stay_on != "null":
        subprocess.run([adb, "shell", "settings", "put", "global",
                        "stay_on_while_plugged_in", original_stay_on],
                       check=False, capture_output=True)
    else:
        subprocess.run([adb, "shell", "settings", "delete", "global",
                        "stay_on_while_plugged_in"], check=False, capture_output=True)


# ── TSV ───────────────────────────────────────────────────────────────────────

def ensure_tsv(path):
    path = Path(path)
    if not path.exists():
        path.write_text(TSV_HEADER + "\n", encoding="utf-8")
        return
    first = path.read_text(encoding="utf-8").splitlines()[0]
    if "memory_after_model_load_kb" not in first:
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([TSV_HEADER] + lines[1:]) + "\n", encoding="utf-8")


def append_tsv(tsv_path, json_path, timestamp, label, threads, input_name):
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    files = payload["files"]
    model_load_ms = payload.get("model_load_ms", 0.0)
    mem_kb = payload.get("memory_after_model_load_kb", 0)
    peak_rss_kb = max(
        (run.get("peak_rss_kb", 0) for f in files for run in f.get("run_details", [])),
        default=0,
    )
    median_infer  = statistics.median(f["inference_ms"] for f in files)
    median_encode = statistics.median(f["encode_ms"] for f in files)
    median_decode = statistics.median(f["decode_ms"] for f in files)
    median_rtf    = statistics.median(f["rtf"] for f in files)

    guardrail_ok = ""
    if input_name == "ptt_guardrail":
        t = files[0]["transcript"]
        ok = all(kw in t for kw in GUARDRAIL_KEYWORDS)
        guardrail_ok = str(ok).lower()
        mark = "✓" if ok else "✗"
        print(f"  guardrail {mark}: {t!r}")

    print(f"  {input_name}: inference {median_infer:.1f} ms  "
          f"encode {median_encode:.1f}  decode {median_decode:.1f}  "
          f"model_load {model_load_ms:.1f}  mem {mem_kb} KB")

    with open(tsv_path, "a", encoding="utf-8") as fh:
        fh.write(
            f"{timestamp}\t{label}\t{threads}\t{input_name}\t"
            f"{model_load_ms:.3f}\t{mem_kb}\t{peak_rss_kb}\t"
            f"{median_infer:.3f}\t{median_encode:.3f}\t{median_decode:.3f}\t"
            f"{median_rtf:.6f}\t{guardrail_ok}\t{json_path}\n"
        )


# ── Benchmark run (one binary) ────────────────────────────────────────────────

def run_binary_on_device(
    adb, remote_dir, remote_bin, remote_model, remote_assets, remote_cv,
    label, threads_list, runs, bench_lang, bench_prompt, remote_env,
    out_dir, tsv_path, timestamp,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for threads in threads_list:
        for input_name in ("common_voice", "ptt_guardrail"):
            remote_input = remote_cv if input_name == "common_voice" else f"{remote_assets}/ptt_guardrail.wav"
            json_path = out_dir / f"{input_name}-t{threads}.json"

            parts = [remote_bin,
                     "--model-dir", remote_model,
                     "--input",     remote_input,
                     "--runs",      str(runs),
                     "--threads",   str(threads)]
            if bench_lang:
                parts += ["--language", bench_lang]
            if bench_prompt:
                parts += ["--prompt", bench_prompt]

            remote_cmd = " ".join(shlex.quote(p) for p in parts)
            if remote_env:
                remote_cmd = f"{remote_env} {remote_cmd}"

            print(f"\n  [{label} | {input_name}, threads={threads}]")
            subprocess.run([adb, "shell", "input", "keyevent", "KEYCODE_WAKEUP"],
                           check=False, capture_output=True)
            result = subprocess.run([adb, "shell", remote_cmd],
                                    capture_output=True, text=True, check=True)
            json_path.write_text(result.stdout, encoding="utf-8")
            append_tsv(tsv_path, json_path, timestamp, label, threads, input_name)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Run baseline + candidate back-to-back for a same-session gate check.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--baseline-binary", required=True,
                   help="Path to the pre-built baseline aarch64-linux-android binary")
    p.add_argument("--label",      required=True,
                   help="Short experiment name, e.g. 'my-openblas-experiment'")
    p.add_argument("--stage",      type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--runs",       type=int, default=3)
    p.add_argument("--threads",    default="1")
    p.add_argument("--model-dir",  default=env("MODEL_DIR", "/Volumes/SSD/qwen3-asr-0.6b"))
    p.add_argument("--assets-dir", default=str(PROJECT_DIR / "bench" / "android-assets"))
    p.add_argument("--remote-dir", default="/data/local/tmp/qwen-asr-bench")
    p.add_argument("--adb",        default=env("ADB", "/Users/tony/Library/Android/sdk/platform-tools/adb"))
    args = p.parse_args()

    baseline_binary = Path(args.baseline_binary)
    if not baseline_binary.is_file():
        sys.exit(f"baseline binary not found: {baseline_binary}\n"
                 f"Save one with: bench/android_run.py --save-binary bench/android-baselines/NAME [...]")

    ndk_home    = env("NDK_HOME", env("ANDROID_NDK_HOME", env("ANDROID_NDK_ROOT",
                      "/Users/tony/Library/Android/sdk/ndk/29.0.14206865")))
    ndk_bin     = env("NDK_BIN", str(Path(ndk_home) / "toolchains/llvm/prebuilt/darwin-x86_64/bin"))
    rust_bin    = env("RUST_TOOLCHAIN_BIN",
                      "/Users/tony/.rustup/toolchains/stable-aarch64-apple-darwin/bin")
    rustflags   = env("RUSTFLAGS_EXTRA", "-C target-feature=+dotprod")
    cargo_feats = env("CARGO_FEATURES_EXTRA", "android-blas")
    bench_lang  = env("BENCH_LANGUAGE", "Auto")
    bench_prompt = env("BENCH_PROMPT", "音频是粤语，可能夹杂英语")
    remote_env  = env("BENCH_REMOTE_ENV", "")
    openblas_dir     = env("OPENBLAS_ANDROID_LIB_DIR", "/private/tmp/OpenBLAS")
    openblas_static  = env("OPENBLAS_ANDROID_STATIC", "1")
    openblas_runtime = env("OPENBLAS_ANDROID_RUNTIME_LIB", "")
    thermal_timeout  = int(env("THERMAL_TIMEOUT", "600"))
    linker = env("CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER",
                 str(PROJECT_DIR / "android-aarch64-clang"))

    # Validate
    if not os.access(args.adb, os.X_OK):
        sys.exit(f"adb not found or not executable: {args.adb}")
    if not Path(args.model_dir).is_dir():
        sys.exit(f"model directory not found: {args.model_dir}")
    if not Path(args.assets_dir).is_dir():
        sys.exit(f"assets directory not found: {args.assets_dir}\nRun: bench/android_prepare_assets.py")

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    ctrl_label = f"{args.label}-ctrl-{ts}"
    cand_label = f"{args.label}-cand-{ts}"

    threads_list = args.threads.split()
    results_dir  = PROJECT_DIR / "bench" / "android-results"
    tsv_path     = PROJECT_DIR / "results-android.tsv"
    ensure_tsv(tsv_path)

    remote_dir    = args.remote_dir
    remote_bin    = f"{remote_dir}/qwen-asr-bench-android"
    remote_model  = f"{remote_dir}/model"
    remote_assets = f"{remote_dir}/assets"
    remote_cv     = f"{remote_dir}/common_voice"

    # ── Thermal + device setup ────────────────────────────────────────────────
    print("\n── Thermal check ────────────────────────────────────────")
    wait_for_thermal_ready(args.adb, thermal_timeout)
    original_stay_on = prepare_device(args.adb)
    atexit.register(cleanup_device, args.adb, original_stay_on)

    # ── Push shared assets (model, wavs) — done once ──────────────────────────
    print("\n── Push shared assets ───────────────────────────────────")
    subprocess.run([args.adb, "shell",
        f"mkdir -p '{remote_dir}' && "
        f"rm -rf '{remote_assets}' '{remote_cv}' && "
        f"mkdir -p '{remote_assets}' '{remote_cv}'"
    ], check=True)
    if openblas_runtime:
        subprocess.run([args.adb, "push", openblas_runtime, f"{remote_dir}/"],
                       check=True, capture_output=True)
    print(f"  pushing model ({args.model_dir})...")
    subprocess.run([args.adb, "push", f"{args.model_dir}/.", remote_model],
                   check=True, capture_output=True)
    print(f"  pushing assets ({args.assets_dir})...")
    subprocess.run([args.adb, "push", f"{args.assets_dir}/.", remote_assets],
                   check=True, capture_output=True)
    subprocess.run([args.adb, "shell",
        f"rm -rf '{remote_cv}' && mkdir -p '{remote_cv}' && "
        f"cp '{remote_assets}'/cv_yue_*.wav '{remote_cv}/'"
    ], check=True, capture_output=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── 1. Run baseline binary ────────────────────────────────────────────────
    print(f"\n── Baseline run: {ctrl_label} ───────────────────────────")
    subprocess.run([args.adb, "push", str(baseline_binary), remote_bin],
                   check=True, capture_output=True)
    subprocess.run([args.adb, "shell", f"chmod 755 '{remote_bin}'"],
                   check=True, capture_output=True)
    run_binary_on_device(
        args.adb, remote_dir, remote_bin, remote_model, remote_assets, remote_cv,
        ctrl_label, threads_list, args.runs, bench_lang, bench_prompt, remote_env,
        results_dir / ctrl_label, tsv_path, timestamp,
    )

    # ── 2. Build candidate ────────────────────────────────────────────────────
    print(f"\n── Build candidate ──────────────────────────────────────")
    build_env = os.environ.copy()
    build_env.update({
        "CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER": linker,
        "CC_aarch64_linux_android":     build_env.get("CC_aarch64_linux_android", linker),
        "AR_aarch64_linux_android":     build_env.get("AR_aarch64_linux_android", f"{ndk_bin}/llvm-ar"),
        "RANLIB_aarch64_linux_android": build_env.get("RANLIB_aarch64_linux_android", f"{ndk_bin}/llvm-ranlib"),
        "RUSTFLAGS": rustflags,
        "PATH": f"{rust_bin}:{ndk_bin}:{build_env['PATH']}",
    })
    if openblas_dir:
        build_env["OPENBLAS_ANDROID_LIB_DIR"] = openblas_dir
    if openblas_static:
        build_env["OPENBLAS_ANDROID_STATIC"] = openblas_static

    build_cmd = [
        "cargo", "build", "--release",
        "--target", "aarch64-linux-android",
        "-p", "qwen-asr-cli",
        "--bin", "qwen-asr-bench-android",
    ]
    if cargo_feats:
        build_cmd += ["--features", cargo_feats]
    subprocess.run(build_cmd, env=build_env, cwd=str(PROJECT_DIR), check=True)

    candidate_bin = PROJECT_DIR / "target/aarch64-linux-android/release/qwen-asr-bench-android"

    # ── 3. Run candidate binary ───────────────────────────────────────────────
    print(f"\n── Candidate run: {cand_label} ──────────────────────────")
    subprocess.run([args.adb, "push", str(candidate_bin), remote_bin],
                   check=True, capture_output=True)
    subprocess.run([args.adb, "shell", f"chmod 755 '{remote_bin}'"],
                   check=True, capture_output=True)
    run_binary_on_device(
        args.adb, remote_dir, remote_bin, remote_model, remote_assets, remote_cv,
        cand_label, threads_list, args.runs, bench_lang, bench_prompt, remote_env,
        results_dir / cand_label, tsv_path, timestamp,
    )

    # ── 4. Gate check ─────────────────────────────────────────────────────────
    print(f"\n── Gate check (stage {args.stage}) ───────────────────────────")
    gate_cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--result",   cand_label,
        "--baseline", ctrl_label,
        "--stage",    str(args.stage),
        "--threads",  threads_list[0],
    ]
    result = subprocess.run(gate_cmd)

    print(f"\nBaseline results : bench/android-results/{ctrl_label}/")
    print(f"Candidate results: bench/android-results/{cand_label}/")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
