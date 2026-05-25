#!/usr/bin/env bash
# Thin wrapper — the real script is android_run.py.
exec python3 "$(dirname "$0")/android_run.py" "$@"
