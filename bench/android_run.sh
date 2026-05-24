#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="/Volumes/SSD/qwen3-asr-0.6b"
ASSETS_DIR="$PROJECT_DIR/bench/android-assets"
RESULTS_TSV="$PROJECT_DIR/results-android.tsv"
REMOTE_DIR="/data/local/tmp/qwen-asr-bench"
RUNS=3
THREADS="1 2 4 6 8"
LABEL="android-baseline-$(date -u +%Y%m%d-%H%M%S)"
ADB="${ADB:-/Users/tony/Library/Android/sdk/platform-tools/adb}"
LINKER="${CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER:-$PROJECT_DIR/android-aarch64-clang}"
NDK_HOME="${NDK_HOME:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-/Users/tony/Library/Android/sdk/ndk/29.0.14206865}}}"
NDK_BIN="${NDK_BIN:-$NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin}"
RUST_TOOLCHAIN_BIN="${RUST_TOOLCHAIN_BIN:-/Users/tony/.rustup/toolchains/stable-aarch64-apple-darwin/bin}"
RUSTFLAGS_EXTRA="${RUSTFLAGS_EXTRA:-}"
CARGO_FEATURES_EXTRA="${CARGO_FEATURES_EXTRA:-}"
OPENBLAS_ANDROID_LIB_DIR="${OPENBLAS_ANDROID_LIB_DIR:-}"
OPENBLAS_ANDROID_STATIC="${OPENBLAS_ANDROID_STATIC:-}"
OPENBLAS_ANDROID_RUNTIME_LIB="${OPENBLAS_ANDROID_RUNTIME_LIB:-}"

usage() {
    cat >&2 <<EOF
Usage: bench/android_run.sh [options]

  --model-dir DIR     Host model directory to push (default: /Volumes/SSD/qwen3-asr-0.6b)
  --assets-dir DIR    Host WAV assets directory (default: bench/android-assets)
  --label NAME        Label for results-android.tsv
  --runs N            Runs per file inside the device harness (default: 3)
  --threads LIST      Quoted space-separated thread counts (default: "1 2 4 6 8")
  --remote-dir DIR    Device directory (default: /data/local/tmp/qwen-asr-bench)
  --adb PATH          adb path
  -h, --help          Show this help

Set RUSTFLAGS_EXTRA for experiments, e.g. RUSTFLAGS_EXTRA="-C target-cpu=cortex-a76 -C target-feature=+dotprod".
Set NDK_HOME or NDK_BIN to override the Android NDK. Default NDK is 29.0.14206865 because this install's NDK 30 clang shim drops compiler arguments.
Set CARGO_FEATURES_EXTRA for opt-in benchmark features, e.g. CARGO_FEATURES_EXTRA="android-blas".
For Android BLAS, set OPENBLAS_ANDROID_LIB_DIR to the directory containing libopenblas.so or libopenblas.a.
Set OPENBLAS_ANDROID_STATIC=1 when linking libopenblas.a.
If using a shared library, optionally set OPENBLAS_ANDROID_RUNTIME_LIB to the exact .so path to push beside the benchmark binary.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2;;
        --assets-dir) ASSETS_DIR="$2"; shift 2;;
        --label) LABEL="$2"; shift 2;;
        --runs) RUNS="$2"; shift 2;;
        --threads) THREADS="$2"; shift 2;;
        --remote-dir) REMOTE_DIR="$2"; shift 2;;
        --adb) ADB="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown option: $1" >&2; usage;;
    esac
done

if [[ ! -x "$ADB" ]]; then
    echo "adb not found or not executable: $ADB" >&2
    exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "model directory not found: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! -d "$ASSETS_DIR" ]]; then
    echo "assets directory not found: $ASSETS_DIR" >&2
    echo "Run: bench/android_prepare_assets.py" >&2
    exit 1
fi

cd "$PROJECT_DIR"

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$LINKER"
export CC_aarch64_linux_android="${CC_aarch64_linux_android:-$LINKER}"
export AR_aarch64_linux_android="${AR_aarch64_linux_android:-$NDK_BIN/llvm-ar}"
export RANLIB_aarch64_linux_android="${RANLIB_aarch64_linux_android:-$NDK_BIN/llvm-ranlib}"
export RUSTFLAGS="${RUSTFLAGS_EXTRA}"
export PATH="$RUST_TOOLCHAIN_BIN:$NDK_BIN:$PATH"
if [[ -n "$OPENBLAS_ANDROID_LIB_DIR" ]]; then
    export OPENBLAS_ANDROID_LIB_DIR
fi
if [[ -n "$OPENBLAS_ANDROID_STATIC" ]]; then
    export OPENBLAS_ANDROID_STATIC
fi

build_args=(build --release --target aarch64-linux-android -p qwen-asr-cli --bin qwen-asr-bench-android)
if [[ -n "$CARGO_FEATURES_EXTRA" ]]; then
    build_args+=(--features "$CARGO_FEATURES_EXTRA")
fi
cargo "${build_args[@]}"

BIN="$PROJECT_DIR/target/aarch64-linux-android/release/qwen-asr-bench-android"
REMOTE_BIN="$REMOTE_DIR/qwen-asr-bench-android"
REMOTE_MODEL="$REMOTE_DIR/model"
REMOTE_ASSETS="$REMOTE_DIR/assets"
REMOTE_CV="$REMOTE_DIR/common_voice"

"$ADB" shell "mkdir -p '$REMOTE_DIR' && rm -rf '$REMOTE_ASSETS' '$REMOTE_CV' && mkdir -p '$REMOTE_ASSETS' '$REMOTE_CV'"
"$ADB" push "$BIN" "$REMOTE_BIN" >/dev/null
"$ADB" shell "chmod 755 '$REMOTE_BIN'"
if [[ -n "$OPENBLAS_ANDROID_RUNTIME_LIB" ]]; then
    "$ADB" push "$OPENBLAS_ANDROID_RUNTIME_LIB" "$REMOTE_DIR/" >/dev/null
fi
"$ADB" push "$MODEL_DIR/." "$REMOTE_MODEL" >/dev/null
"$ADB" push "$ASSETS_DIR/." "$REMOTE_ASSETS" >/dev/null
"$ADB" shell "rm -rf '$REMOTE_CV' && mkdir -p '$REMOTE_CV' && cp '$REMOTE_ASSETS'/cv_yue_*.wav '$REMOTE_CV'/"

if [[ ! -f "$RESULTS_TSV" ]]; then
    printf 'timestamp\tlabel\tthreads\tinput\tmedian_inference_ms\tmedian_encode_ms\tmedian_decode_ms\trtf\tguardrail_ok\tjson_path\n' >"$RESULTS_TSV"
fi

OUT_DIR="$PROJECT_DIR/bench/android-results/$LABEL"
mkdir -p "$OUT_DIR"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for threads in $THREADS; do
    for input_name in common_voice ptt_guardrail; do
        if [[ "$input_name" == "common_voice" ]]; then
            remote_input="$REMOTE_CV"
        else
            remote_input="$REMOTE_ASSETS/ptt_guardrail.wav"
        fi

        json_path="$OUT_DIR/${input_name}-t${threads}.json"
        "$ADB" shell "'$REMOTE_BIN' --model-dir '$REMOTE_MODEL' --input '$remote_input' --runs '$RUNS' --threads '$threads'" >"$json_path"

        python3 - "$json_path" "$RESULTS_TSV" "$TIMESTAMP" "$LABEL" "$threads" "$input_name" <<'PY'
import json
import statistics
import sys

json_path, results_tsv, timestamp, label, threads, input_name = sys.argv[1:]
payload = json.load(open(json_path, "r", encoding="utf-8"))
files = payload["files"]
median_infer = statistics.median(item["inference_ms"] for item in files)
median_encode = statistics.median(item["encode_ms"] for item in files)
median_decode = statistics.median(item["decode_ms"] for item in files)
median_rtf = statistics.median(item["rtf"] for item in files)
guardrail_ok = ""
if input_name == "ptt_guardrail":
    guardrail_ok = str(files[0]["transcript"] == "Hello,你好吗？已经坐低咗啦。").lower()
with open(results_tsv, "a", encoding="utf-8") as fh:
    fh.write(
        f"{timestamp}\t{label}\t{threads}\t{input_name}\t"
        f"{median_infer:.3f}\t{median_encode:.3f}\t{median_decode:.3f}\t"
        f"{median_rtf:.6f}\t{guardrail_ok}\t{json_path}\n"
    )
PY
    done
done

echo "Wrote $RESULTS_TSV"
echo "JSON results in $OUT_DIR"
