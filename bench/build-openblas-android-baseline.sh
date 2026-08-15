#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OPENBLAS_REPO="${OPENBLAS_REPO:-https://github.com/OpenMathLib/OpenBLAS.git}"
OPENBLAS_REF="${OPENBLAS_REF:-e0166008be8e466242aa76b2ff75ce3f0fbf574a}"
OPENBLAS_DIR="${OPENBLAS_DIR:-/private/tmp/OpenBLAS}"
NDK_VERSION="${NDK_VERSION:-29.0.14206865}"
ANDROID_API="${ANDROID_API:-35}"
JOBS="${JOBS:-10}"

ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}"
NDK_HOME="${NDK_HOME:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-$ANDROID_SDK_ROOT/ndk/$NDK_VERSION}}}"
NDK_BIN="$NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
export PATH="$PATH:$NDK_BIN"
CC="${CC:-$NDK_BIN/aarch64-linux-android${ANDROID_API}-clang}"
AR="${AR:-$NDK_BIN/llvm-ar}"
RANLIB="${RANLIB:-$NDK_BIN/llvm-ranlib}"


usage() {
    cat >&2 <<EOF
Usage: bench/build-openblas-android-baseline.sh [--force]

Builds the accepted Android OpenBLAS baseline archive:
  TARGET=ARMV8, OSNAME=Android, BINARY=64, ONLY_CBLAS=1, NOFORTRAN=1,
  USE_THREAD=0, NUM_THREADS=1

Defaults:
  OPENBLAS_DIR=$OPENBLAS_DIR
  OPENBLAS_REF=$OPENBLAS_REF
  NDK_HOME=$NDK_HOME
  ANDROID_API=$ANDROID_API
  JOBS=$JOBS

The benchmark runner expects:
  \$OPENBLAS_DIR/libopenblas.a
EOF
    exit 1
}

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift;;
        -h|--help) usage;;
        *) echo "Unknown option: $1" >&2; usage;;
    esac
done

for tool in git make clang "$CC" "$AR" "$RANLIB"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "required tool not found: $tool" >&2
        exit 1
    fi
done

if [[ "$FORCE" == "1" && -e "$OPENBLAS_DIR" ]]; then
    rm -rf "$OPENBLAS_DIR"
fi

if [[ ! -d "$OPENBLAS_DIR/.git" ]]; then
    if [[ -e "$OPENBLAS_DIR" ]]; then
        echo "OPENBLAS_DIR exists but is not a git checkout: $OPENBLAS_DIR" >&2
        echo "Rerun with --force to replace it." >&2
        exit 1
    fi
    git clone "$OPENBLAS_REPO" "$OPENBLAS_DIR"
fi

cd "$OPENBLAS_DIR"
git fetch --tags origin
git checkout --detach "$OPENBLAS_REF"

make clean >/dev/null 2>&1 || true

make -j"$JOBS" \
    TARGET=ARMV8 \
    OSNAME=Android \
    BINARY=64 \
    CC="$CC" \
    HOSTCC=clang \
    AR="$AR" \
    RANLIB="$RANLIB" \
    NOFORTRAN=1 \
    ONLY_CBLAS=1 \
    USE_THREAD=0 \
    NUM_THREADS=1

if [[ ! -e "$OPENBLAS_DIR/libopenblas.a" ]]; then
    echo "OpenBLAS build completed but libopenblas.a was not created" >&2
    exit 1
fi

cat >"$OPENBLAS_DIR/android-baseline-build-info.txt" <<EOF
openblas_repo=$OPENBLAS_REPO
openblas_ref=$OPENBLAS_REF
openblas_dir=$OPENBLAS_DIR
ndk_home=$NDK_HOME
android_api=$ANDROID_API
cc=$CC
ar=$AR
ranlib=$RANLIB
target=ARMV8
osname=Android
binary=64
nofortran=1
only_cblas=1
use_thread=0
num_threads=1
EOF

echo "Built accepted Android OpenBLAS baseline archive:"
echo "  $OPENBLAS_DIR/libopenblas.a"
echo "Build metadata:"
echo "  $OPENBLAS_DIR/android-baseline-build-info.txt"
echo
echo "Use with:"
echo "  OPENBLAS_ANDROID_LIB_DIR=$OPENBLAS_DIR OPENBLAS_ANDROID_STATIC=1 bench/android_run.py --threads '1' --runs 3"
