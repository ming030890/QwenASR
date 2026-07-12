#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bench/android_snapshot.sh --label LABEL --phase before|after [--adb PATH]

Saves Android battery, thermal, thermal-zone, and CPU-frequency snapshots to:
  bench/android-results/<LABEL>/thermal/

The script intentionally uses one adb entrypoint so Codex/user approval can be
granted once for this helper instead of separately for each snapshot command.
EOF
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
LABEL=""
PHASE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2;;
        --phase) PHASE="$2"; shift 2;;
        --adb) ADB="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown option: $1" >&2; usage;;
    esac
done

if [[ -z "$LABEL" || -z "$PHASE" ]]; then
    usage
fi
if [[ "$PHASE" != "before" && "$PHASE" != "after" ]]; then
    echo "--phase must be before or after" >&2
    exit 1
fi
if [[ ! -x "$ADB" ]]; then
    echo "adb not found or not executable: $ADB" >&2
    exit 1
fi

SNAP_DIR="$PROJECT_DIR/bench/android-results/$LABEL/thermal"
mkdir -p "$SNAP_DIR"

"$ADB" shell dumpsys battery > "$SNAP_DIR/battery-$PHASE.txt"
"$ADB" shell dumpsys thermalservice > "$SNAP_DIR/thermal-$PHASE.txt"
"$ADB" shell 'for f in /sys/class/thermal/thermal_zone*/temp; do echo "$f=$(cat "$f" 2>/dev/null)"; done' \
    > "$SNAP_DIR/thermal-zones-$PHASE.txt"
"$ADB" shell 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do echo "$f=$(cat "$f" 2>/dev/null)"; done' \
    > "$SNAP_DIR/cpu-freq-$PHASE.txt"

grep 'Thermal Status' "$SNAP_DIR/thermal-$PHASE.txt" || true
grep -E 'VIRTUAL-SKIN|soc_therm|BIG|MID|LITTLE|battery' "$SNAP_DIR/thermal-$PHASE.txt" | head -20 || true
echo "Saved Android $PHASE snapshot in $SNAP_DIR"
