#!/usr/bin/env python3
"""Build, push, and run the Android benchmark.

Usage:
  bench/android_run.py [options]

Collect numbers with this script, then evaluate with bench/android_check_gate.py.

Env vars (all have defaults matching the accepted baseline config):
  ADB                         adb binary path
  MODEL_DIR                   host model directory
  NDK_HOME / ANDROID_NDK_HOME / ANDROID_NDK_ROOT
  NDK_BIN                     prebuilt toolchain bin dir (overrides NDK_HOME derivation)
  RUST_TOOLCHAIN_BIN          rustup toolchain bin dir
  RUSTFLAGS_EXTRA             extra RUSTFLAGS  (default: -C target-feature=+dotprod)
  CARGO_FEATURES_EXTRA        cargo --features  (default: android-blas)
  BENCH_LANGUAGE              benchmark language  (default: Auto)
  BENCH_PROMPT                benchmark prompt text
  BENCH_REMOTE_ENV            space-separated VAR=VAL pairs prepended to the device command
  OPENBLAS_ANDROID_LIB_DIR    dir containing libopenblas.a or libopenblas.so
  OPENBLAS_ANDROID_STATIC     set 1 when linking libopenblas.a  (default: 1)
  OPENBLAS_ANDROID_RUNTIME_LIB  exact .so path to push when using shared linking
  THERMAL_TIMEOUT             seconds to wait for Thermal Status 0  (default: 600)
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent

TSV_HEADER = "\t".join([
    "timestamp", "label", "threads", "input",
    "model_load_ms", "memory_after_model_load_kb", "peak_inference_rss_kb",
    "median_inference_ms", "median_encode_ms", "median_decode_ms",
    "rtf", "guardrail_ok", "json_path",
])

def env(key, default=""):
    return os.environ.get(key, default)


def log(msg=""):
    print(msg, flush=True)


# ── adb helpers ───────────────────────────────────────────────────────────────

def adb_run(adb, *args, capture=False, check=True, silent=False):
    cmd = [adb] + [str(a) for a in args]
    if not silent:
        print(f"  adb {' '.join(shlex.quote(str(a)) for a in args)}", flush=True)
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=capture,
        check=check,
    )


def adb_output(adb, *args):
    return subprocess.run(
        [adb] + [str(a) for a in args],
        capture_output=True, text=True, check=False,
    ).stdout


def run_step(label, cmd, **kwargs):
    log(f"  {label}...")
    start = time.monotonic()
    try:
        result = subprocess.run(cmd, check=True, **kwargs)
        log(f"  {label} done in {time.monotonic() - start:.1f}s")
        return result
    except Exception:
        log(f"  {label} failed after {time.monotonic() - start:.1f}s")
        raise


# ── Thermal wait ──────────────────────────────────────────────────────────────

def wait_for_thermal_ready(adb, timeout):
    interval = 15
    elapsed = 0
    while True:
        out = adb_output(adb, "shell", "dumpsys", "thermalservice")
        status_line = next((l for l in out.splitlines() if "Thermal Status" in l), "")
        status = status_line.split()[-1].strip() if status_line else "?"
        if status == "0":
            log("Thermal Status: 0 — device ready.")
            return
        log(f"Thermal Status: {status} — waiting ({elapsed}s elapsed, timeout {timeout}s)...")
        for line in out.splitlines():
            if any(k in line for k in ("VIRTUAL-SKIN", "soc_therm", "BIG", "MID", "LITTLE", "battery")):
                log(f"  {line.strip()}")
        if elapsed >= timeout:
            print(f"WARNING: thermal wait timed out after {timeout}s (status {status}). Proceeding.", file=sys.stderr)
            return
        time.sleep(interval)
        elapsed += interval


# ── Device lifecycle ──────────────────────────────────────────────────────────

def prepare_device(adb):
    original = adb_output(adb, "shell", "settings", "get", "global", "stay_on_while_plugged_in").strip().replace("\r", "") or None
    subprocess.run([adb, "shell", "input", "keyevent", "KEYCODE_WAKEUP"], check=False, capture_output=True)
    subprocess.run([adb, "shell", "svc", "power", "stayon", "true"], check=False, capture_output=True)
    subprocess.run([adb, "shell", "cmd", "power", "set-fixed-performance-mode-enabled", "true"], check=False, capture_output=True)
    log("Device: screen on, stay-on, fixed-performance mode enabled.")
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
        has_nihao = any(text in t for text in ("你好吗", "你好嗎", "你好嘛"))
        has_zuodi = "坐低咗" in t
        ok = has_nihao and has_zuodi
        guardrail_ok = str(ok).lower()
        mark = "✓" if ok else "✗"
        log(f"  guardrail {mark}: {t!r}")

    log(f"  {input_name}: inference {median_infer:.1f} ms  "
        f"encode {median_encode:.1f}  decode {median_decode:.1f}  "
        f"model_load {model_load_ms:.1f}  mem {mem_kb} KB")

    with open(tsv_path, "a", encoding="utf-8") as fh:
        fh.write(
            f"{timestamp}\t{label}\t{threads}\t{input_name}\t"
            f"{model_load_ms:.3f}\t{mem_kb}\t{peak_rss_kb}\t"
            f"{median_infer:.3f}\t{median_encode:.3f}\t{median_decode:.3f}\t"
            f"{median_rtf:.6f}\t{guardrail_ok}\t{json_path}\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Build, push, and run the Android benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model-dir",  default=env("MODEL_DIR", "/Volumes/SSD/qwen3-asr-0.6b"))
    p.add_argument("--assets-dir", default=str(PROJECT_DIR / "bench" / "android-assets"))
    p.add_argument("--label",      default=f"android-baseline-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--runs",       type=int, default=3)
    p.add_argument("--threads",    default="1", help='space-separated thread counts, e.g. "1 2 4"')
    p.add_argument("--remote-dir", default="/data/local/tmp/qwen-asr-bench")
    p.add_argument(
        "--remote-model-dir",
        help="Reuse an explicit on-device model directory (implies --skip-model-push)",
    )
    p.add_argument("--adb",        default=env("ADB", os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")))
    p.add_argument("--profile",     action="store_true", help="Enable kernel profiling counters")
    p.add_argument("--save-binary", metavar="PATH",
                   help="Copy the compiled binary to PATH after build (e.g. bench/android-baselines/my-baseline)")
    p.add_argument(
        "--skip-model-push",
        action="store_true",
        help="Reuse the model already present under REMOTE_DIR/model",
    )
    args = p.parse_args()

    ndk_home     = env("NDK_HOME", env("ANDROID_NDK_HOME", env("ANDROID_NDK_ROOT",
                       os.path.expanduser("~/Library/Android/sdk/ndk/29.0.14206865"))))
    ndk_bin      = env("NDK_BIN", str(Path(ndk_home) / "toolchains/llvm/prebuilt/darwin-x86_64/bin"))
    rust_bin     = env("RUST_TOOLCHAIN_BIN",
                       os.path.expanduser("~/.rustup/toolchains/stable-aarch64-apple-darwin/bin"))
    rustflags    = env("RUSTFLAGS_EXTRA", "-C target-feature=+dotprod")
    cargo_feats  = env("CARGO_FEATURES_EXTRA", "android-blas")
    bench_lang   = env("BENCH_LANGUAGE", "Auto")
    bench_prompt = env("BENCH_PROMPT", "音频是粤语，可能夹杂英语")
    remote_env   = env("BENCH_REMOTE_ENV", "")
    openblas_dir = env("OPENBLAS_ANDROID_LIB_DIR", "/private/tmp/OpenBLAS")
    openblas_static  = env("OPENBLAS_ANDROID_STATIC", "1")
    openblas_runtime = env("OPENBLAS_ANDROID_RUNTIME_LIB", "")
    thermal_timeout  = int(env("THERMAL_TIMEOUT", "600"))
    linker = env("CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER",
                 str(PROJECT_DIR / "android-aarch64-clang"))

    # ── Validate ──────────────────────────────────────────────────────────────
    if not os.access(args.adb, os.X_OK):
        sys.exit(f"adb not found or not executable: {args.adb}")
    if not Path(args.model_dir).is_dir():
        sys.exit(f"model directory not found: {args.model_dir}")
    if not Path(args.assets_dir).is_dir():
        sys.exit(f"assets directory not found: {args.assets_dir}\nRun: bench/android_prepare_assets.py")
    if openblas_dir:
        if not (Path(openblas_dir) / "libopenblas.a").exists() and \
           not (Path(openblas_dir) / "libopenblas.so").exists():
            sys.exit(
                f"OpenBLAS library not found in: {openblas_dir}\n"
                f"Rebuild: bench/build-openblas-android-baseline.sh --force\n"
                f"Or set OPENBLAS_ANDROID_LIB_DIR to a compatible ARMV8 USE_THREAD=0 build."
            )

    # ── Thermal + device setup ────────────────────────────────────────────────
    log("\n── Thermal check ────────────────────────────────────────")
    wait_for_thermal_ready(args.adb, thermal_timeout)
    original_stay_on = prepare_device(args.adb)
    atexit.register(cleanup_device, args.adb, original_stay_on)

    # ── Build ─────────────────────────────────────────────────────────────────
    log("\n── Build ────────────────────────────────────────────────")
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

    run_step("cargo build", build_cmd, env=build_env, cwd=str(PROJECT_DIR))

    if args.save_binary:
        import shutil
        dest = Path(args.save_binary)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(PROJECT_DIR / "target/aarch64-linux-android/release/qwen-asr-bench-android"), dest)
        log(f"  binary saved to: {dest}")

    # ── Push ─────────────────────────────────────────────────────────────────
    log("\n── Push ─────────────────────────────────────────────────")
    bin_path      = PROJECT_DIR / "target/aarch64-linux-android/release/qwen-asr-bench-android"
    remote_bin    = f"{args.remote_dir}/qwen-asr-bench-android"
    remote_model  = args.remote_model_dir or f"{args.remote_dir}/model"
    remote_assets = f"{args.remote_dir}/assets"
    remote_cv     = f"{args.remote_dir}/common_voice"

    run_step("prepare remote dirs", [args.adb, "shell",
        f"mkdir -p '{args.remote_dir}' && "
        f"rm -rf '{remote_assets}' '{remote_cv}' && "
        f"mkdir -p '{remote_assets}' '{remote_cv}'"
    ])
    run_step("push benchmark binary", [args.adb, "push", str(bin_path), remote_bin])
    run_step("chmod benchmark binary", [args.adb, "shell", f"chmod 755 '{remote_bin}'"],
             capture_output=True)
    if openblas_runtime:
        run_step("push OpenBLAS runtime", [args.adb, "push", openblas_runtime, f"{args.remote_dir}/"])
    if args.skip_model_push or args.remote_model_dir:
        log(f"  reusing model already present at {remote_model}")
    else:
        run_step(f"push model ({args.model_dir})", [args.adb, "push", f"{args.model_dir}/.", remote_model])
    run_step(f"push assets ({args.assets_dir})", [args.adb, "push", f"{args.assets_dir}/.", remote_assets])
    run_step("prepare Common Voice input", [args.adb, "shell",
        f"rm -rf '{remote_cv}' && mkdir -p '{remote_cv}' && "
        f"cp '{remote_assets}'/cv_yue_*.wav '{remote_cv}/'"
    ], capture_output=True)
    log("  push complete.")

    # ── Run ───────────────────────────────────────────────────────────────────
    log(f"\n── Run: {args.label} ────────────────────────────────────")
    results_tsv = PROJECT_DIR / "results-android.tsv"
    ensure_tsv(results_tsv)
    out_dir = PROJECT_DIR / "bench" / "android-results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for threads in args.threads.split():
        for input_name in ("common_voice", "ptt_guardrail"):
            remote_input = remote_cv if input_name == "common_voice" else f"{remote_assets}/ptt_guardrail.wav"
            json_path = out_dir / f"{input_name}-t{threads}.json"

            parts = [remote_bin,
                     "--model-dir", remote_model,
                     "--input",     remote_input,
                     "--runs",      str(args.runs),
                     "--threads",   threads]
            if args.profile:
                parts.append("--profile")
            if bench_lang:
                parts += ["--language", bench_lang]
            if bench_prompt:
                parts += ["--prompt", bench_prompt]

            remote_cmd = " ".join(shlex.quote(part) for part in parts)
            if remote_env:
                remote_cmd = f"{remote_env} {remote_cmd}"

            log(f"\n  [{input_name}, threads={threads}]")
            subprocess.run([args.adb, "shell", "input", "keyevent", "KEYCODE_WAKEUP"],
                           check=False, capture_output=True)
            result = run_step(
                f"run {input_name} threads={threads}",
                [args.adb, "shell", remote_cmd],
                capture_output=True,
                text=True,
            )
            try:
                payload = json.loads(result.stdout)
                if bin_path.exists():
                    payload["binary_size_bytes"] = bin_path.stat().st_size
                json_path.write_text(json.dumps(payload), encoding="utf-8")
            except Exception as e:
                json_path.write_text(result.stdout, encoding="utf-8")
            append_tsv(results_tsv, json_path, timestamp, args.label, threads, input_name)

    log(f"\nJSON results : {out_dir}")
    log(f"TSV          : {results_tsv}")
    log(f"\nNext — same-session gate check (eliminates thermal noise):")
    log(f"  bench/android_compare.py --baseline-binary bench/android-baselines/NAME --label LABEL --stage 1")
    log(f"\nOr store this build as the new baseline binary for future comparisons:")
    log(f"  cp target/aarch64-linux-android/release/qwen-asr-bench-android bench/android-baselines/NAME")


if __name__ == "__main__":
    main()
