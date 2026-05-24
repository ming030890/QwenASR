#!/usr/bin/env bash
# build-android.sh — build qwen-asr-bench-android for arm64-v8a
#
# One-time setup (run once per machine):
#   bench/build-android.sh --setup
#
# Build only:
#   bench/build-android.sh
#
# NDK used: 29.0.14206865 (stable). Installed via:
#   android sdk install ndk/29.0.14206865
#
# NDK note: The NDK ships stub wrapper scripts (e.g. clang → clang-21) that
# don't forward arguments. --setup replaces them with proper symlinks.
# This is a one-time fix per NDK installation.

set -euo pipefail

NDK_VERSION="29.0.14206865"
NDK_ROOT="$HOME/Library/Android/sdk/ndk/$NDK_VERSION"
NDK_BIN="$NDK_ROOT/toolchains/llvm/prebuilt/darwin-x86_64/bin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── setup ────────────────────────────────────────────────────────────────────

do_setup() {
    echo "=== Installing Rust Android targets ==="
    rustup target add aarch64-linux-android armv7-linux-androideabi

    echo "=== Installing cargo-ndk ==="
    cargo install cargo-ndk

    echo "=== Downloading NDK $NDK_VERSION ==="
    android sdk install "ndk/$NDK_VERSION"

    echo "=== Fixing NDK wrapper scripts → symlinks ==="
    fix_ndk_wrappers
}

# Replace broken single-word stub scripts with symlinks to the real binary.
# The NDK ships e.g. clang as a plain text file containing just "clang-21",
# which does not forward arguments. A symlink works correctly.
fix_ndk_wrappers() {
    local fixed=0
    for f in "$NDK_BIN"/*; do
        [[ -f "$f" && ! -L "$f" ]] || continue
        local content
        content=$(cat "$f" 2>/dev/null)
        # Match a single bare word — no spaces, no shebang, no $@
        if [[ "$content" =~ ^[a-zA-Z0-9._+-]+$ ]]; then
            local target="$content"
            if [[ -e "$NDK_BIN/$target" ]]; then
                ln -sf "$target" "$f"
                echo "  fixed: $(basename "$f") -> $target"
                ((fixed++)) || true
            fi
        fi
    done
    echo "  $fixed wrapper(s) replaced with symlinks"
}

# ── build ─────────────────────────────────────────────────────────────────────

do_build() {
    if [[ ! -d "$NDK_ROOT" ]]; then
        echo "NDK $NDK_VERSION not found at $NDK_ROOT"
        echo "Run: bench/build-android.sh --setup"
        exit 1
    fi

    export ANDROID_NDK_HOME="$NDK_ROOT"
    export PATH="$NDK_BIN:$PATH"

    cd "$PROJECT_DIR"
    cargo ndk -t arm64-v8a build --release -p qwen-asr-cli --bin qwen-asr-bench-android

    local bin="$PROJECT_DIR/target/aarch64-linux-android/release/qwen-asr-bench-android"
    echo ""
    echo "Built: $bin"
    echo "Push to device:  adb push $bin /data/local/tmp/"
}

# ── main ──────────────────────────────────────────────────────────────────────

case "${1:-}" in
    --setup)
        do_setup
        echo ""
        echo "Setup complete. Run bench/build-android.sh to build."
        ;;
    "")
        do_build
        ;;
    *)
        echo "Usage: bench/build-android.sh [--setup]"
        exit 1
        ;;
esac
