#!/usr/bin/env python3
"""Evaluate a benchmark result against a stage gate.

Usage:
  bench/android_check_gate.py --result LABEL --stage 1|2|3 [--baseline LABEL] [--threads N]

Exits 0 = PASS (promote/accept), 1 = FAIL (reject).

IMPORTANT: baseline and result must come from the same on-device session (run
back-to-back via android_compare.py) so that thermal/scheduling noise cancels
out.  Comparing results from different sessions produces 6-13% phantom variance
that violates the tight regression limits below.

Stage 1 (fast gate, 1 CV clip):
  - Guardrail must be phonetically stable.
  - cv_yue_003 inference must not regress >10% vs baseline.

Stage 2 (candidate gate, 2 CV clips):
  - Guardrail must be phonetically stable.
  - No clip may regress >5% vs baseline.
  - Promotes if median inference is >=3% faster.

Stage 3 (acceptance gate, 5 CV clips):
  - Guardrail must be phonetically stable.
  - Accepts if ANY of:
      inference_ms improves >=5%
      model_load_ms improves >=5%
      memory_after_model_load_kb improves >=10%
      peak_inference_rss_kb improves >=10%
  - Startup/memory wins must not regress inference by more than 2%.
"""

import argparse
import json
import os
import statistics
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "android-results")

GUARDRAIL_KEYWORDS = ["你好吗", "坐低咗"]

STAGE1_REGRESSION_LIMIT    = 0.10
STAGE2_REGRESSION_LIMIT    = 0.05
STAGE2_PROMOTION_THRESHOLD = 0.03
STAGE3_INFERENCE_THRESHOLD = 0.05
STAGE3_LOAD_THRESHOLD      = 0.05
STAGE3_MEMORY_THRESHOLD    = 0.10
STAGE3_FIRST_TOKEN_THRESHOLD = 0.05
STAGE3_BINARY_SIZE_THRESHOLD = 0.05
STAGE3_INFERENCE_REGRESSION_LIMIT = 0.02


def load_json(label, filename):
    path = os.path.join(RESULTS_DIR, label, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_guardrail(label, threads):
    data = load_json(label, f"ptt_guardrail-t{threads}.json")
    if data is None:
        return False, f"ptt_guardrail-t{threads}.json not found in {label}"
    if not data.get("files"):
        return False, "ptt_guardrail JSON has no files"
    transcript = data["files"][0]["transcript"]
    has_nihao = "你好吗" in transcript or "你好嗎" in transcript
    has_zuodi = "坐低咗" in transcript
    if not (has_nihao and has_zuodi):
        missing = []
        if not has_nihao:
            missing.append("你好吗/你好嗎")
        if not has_zuodi:
            missing.append("坐低咗")
        return False, f"FAIL — transcript '{transcript}' missing keywords {missing}"
    return True, f"OK — '{transcript}'"


def get_cv_results(label, threads):
    """Returns ({clip_name: inference_ms}, {clip_name: prefill_ms}, data) from common_voice JSON."""
    data = load_json(label, f"common_voice-t{threads}.json")
    if not data:
        return {}, {}, data
    clips = {}
    prefills = {}
    for f in data["files"]:
        name = os.path.basename(f["audio_path"]).replace(".wav", "")
        clips[name] = f["inference_ms"]
        if "prefill_ms" in f:
            prefills[name] = f["prefill_ms"]
    return clips, prefills, data


def get_peak_rss_kb(data):
    vals = [
        run["peak_rss_kb"]
        for f in data.get("files", [])
        for run in f.get("run_details", [])
        if "peak_rss_kb" in run
    ]
    return statistics.median(vals) if vals else None


def pct(new, old):
    """Signed relative change. Negative = improvement."""
    return (new - old) / old


def fmt_pct(v):
    return f"{v*100:+.1f}%"


def sep():
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a benchmark result passes a stage gate."
    )
    parser.add_argument("--result",   required=True, help="Result label to evaluate")
    parser.add_argument("--baseline", help="Baseline label to compare against")
    parser.add_argument("--stage",    type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--threads",  type=int, default=1, help="Thread count suffix used in JSON filenames (default: 1)")
    args = parser.parse_args()

    t = args.threads
    sep()
    print(f"Stage {args.stage} gate check")
    print(f"  Result:   {args.result}")
    if args.baseline:
        print(f"  Baseline: {args.baseline}")
    sep()

    verdict_ok = True
    reasons_fail = []
    reasons_pass = []

    # ── Guardrail ─────────────────────────────────────────────────────────────
    gok, gmsg = check_guardrail(args.result, t)
    print(f"\n[guardrail] {gmsg}")
    if not gok:
        verdict_ok = False
        reasons_fail.append(f"Guardrail failed: {gmsg}")

    # ── Inference numbers ─────────────────────────────────────────────────────
    result_cv, result_prefills, result_data = get_cv_results(args.result, t)
    
    if args.baseline:
        baseline_cv, baseline_prefills, baseline_data = get_cv_results(args.baseline, t)
    else:
        latest_path = os.path.join(os.path.dirname(__file__), "latest_baseline.json")
        if os.path.exists(latest_path):
            with open(latest_path, encoding="utf-8") as f:
                base_obj = json.load(f)
            baseline_data = base_obj.get("common_voice")
            baseline_cv = {}
            baseline_prefills = {}
            if baseline_data:
                for f in baseline_data.get("files", []):
                    name = os.path.basename(f["audio_path"]).replace(".wav", "")
                    baseline_cv[name] = f["inference_ms"]
                    if "prefill_ms" in f:
                        baseline_prefills[name] = f["prefill_ms"]
            print(f"Loaded latest baseline from standalone file: {latest_path} (label: {base_obj.get('label')})")
        else:
            baseline_cv, baseline_prefills, baseline_data = {}, {}, None

    print(f"\n[inference_ms]")
    for clip, ms in sorted(result_cv.items()):
        if clip in baseline_cv:
            d = pct(ms, baseline_cv[clip])
            direction = "faster ✓" if d < 0 else "slower"
            print(f"  {clip}: {ms:.1f} ms  (baseline {baseline_cv[clip]:.1f} ms, {fmt_pct(d)} {direction})")
        else:
            print(f"  {clip}: {ms:.1f} ms")

    if result_prefills:
        print(f"\n[first_token_latency_ms (prefill)]")
        for clip, ms in sorted(result_prefills.items()):
            if clip in baseline_prefills:
                d = pct(ms, baseline_prefills[clip])
                direction = "faster ✓" if d < 0 else "slower"
                print(f"  {clip}: {ms:.1f} ms  (baseline {baseline_prefills[clip]:.1f} ms, {fmt_pct(d)} {direction})")
            else:
                print(f"  {clip}: {ms:.1f} ms")

    if result_data:
        load_new = result_data.get("model_load_ms")
        mem_new  = result_data.get("memory_after_model_load_kb")
        rss_new  = get_peak_rss_kb(result_data)
        bin_size_new = result_data.get("binary_size_bytes")
        print(f"\n[startup/memory/binary_size]")
        print(f"  model_load_ms:            {load_new:.1f}" if load_new else "  model_load_ms:            n/a")
        print(f"  memory_after_model_load:  {mem_new} KB" if mem_new else "  memory_after_model_load:  n/a")
        print(f"  peak_inference_rss:       {rss_new:.0f} KB" if rss_new else "  peak_inference_rss:       n/a")
        if bin_size_new:
            print(f"  binary_size_bytes:        {bin_size_new} bytes ({bin_size_new / 1024 / 1024:.2f} MB)")

    if not baseline_data:
        print("\n[no baseline data found — skipping comparison checks]")
        sep()
        print(f"RESULT: {'PASS' if verdict_ok else 'FAIL'}")
        if reasons_fail:
            print("Reasons failed:")
            for r in reasons_fail:
                print(f"  - {r}")
        sep()
        sys.exit(0 if verdict_ok else 1)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.stage >= 1:
        print(f"\n[stage 1 — regression limit +{STAGE1_REGRESSION_LIMIT*100:.0f}%]")
        cv003 = result_cv.get("cv_yue_003")
        base003 = baseline_cv.get("cv_yue_003")
        if cv003 and base003:
            d = pct(cv003, base003)
            if d > STAGE1_REGRESSION_LIMIT:
                msg = f"cv_yue_003 regressed {fmt_pct(d)} (limit +{STAGE1_REGRESSION_LIMIT*100:.0f}%)"
                print(f"  FAIL — {msg}")
                verdict_ok = False
                reasons_fail.append(msg)
            else:
                msg = f"cv_yue_003 within threshold ({fmt_pct(d)})"
                print(f"  PASS — {msg}")
                reasons_pass.append(msg)
        else:
            msg = "cv_yue_003 missing from result or baseline"
            print(f"  WARN — {msg}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if args.stage >= 2:
        print(f"\n[stage 2 — per-clip regression limit +{STAGE2_REGRESSION_LIMIT*100:.0f}%, promotion threshold -{STAGE2_PROMOTION_THRESHOLD*100:.0f}%]")
        common = sorted(set(result_cv) & set(baseline_cv))
        deltas = [pct(result_cv[c], baseline_cv[c]) for c in common]
        if deltas:
            worst_clip = common[deltas.index(max(deltas))]
            worst_d = max(deltas)
            if worst_d > STAGE2_REGRESSION_LIMIT:
                msg = f"{worst_clip} regressed {fmt_pct(worst_d)} (limit +{STAGE2_REGRESSION_LIMIT*100:.0f}%)"
                print(f"  FAIL — {msg}")
                verdict_ok = False
                reasons_fail.append(msg)
            else:
                print(f"  worst regression: {worst_clip} {fmt_pct(worst_d)} — within limit")

            median_new  = statistics.median(result_cv[c] for c in common)
            median_base = statistics.median(baseline_cv[c] for c in common)
            median_d = pct(median_new, median_base)
            if median_d <= -STAGE2_PROMOTION_THRESHOLD:
                msg = f"median inference {fmt_pct(median_d)} — promotes to stage 3"
                print(f"  PROMOTE — {msg}")
                reasons_pass.append(msg)
            else:
                print(f"  median: {median_new:.1f} ms vs {median_base:.1f} ms ({fmt_pct(median_d)}) — below promotion threshold")
        else:
            print(f"  WARN — no common clips between result and baseline")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    if args.stage == 3:
        print(f"\n[stage 3 — acceptance thresholds]")

        load_base = baseline_data.get("model_load_ms") if baseline_data else None
        mem_base  = baseline_data.get("memory_after_model_load_kb") if baseline_data else None
        rss_base  = get_peak_rss_kb(baseline_data) if baseline_data else None
        bin_size_base = baseline_data.get("binary_size_bytes") if baseline_data else None

        common5 = sorted(set(result_cv) & set(baseline_cv))
        inf_ok = False
        if common5:
            median_new  = statistics.median(result_cv[c] for c in common5)
            median_base = statistics.median(baseline_cv[c] for c in common5)
            inf_d = pct(median_new, median_base)
            inf_ok = inf_d <= -STAGE3_INFERENCE_THRESHOLD
            label_inf = "MEETS" if inf_ok else "does not meet"
            print(f"  inference_ms:    {median_new:.1f} vs {median_base:.1f} ({fmt_pct(inf_d)}) — {label_inf} >={STAGE3_INFERENCE_THRESHOLD*100:.0f}% threshold")
        else:
            print(f"  inference_ms:    insufficient clips ({len(common5)}/5)")
            inf_d = 0.0

        prefill_ok = False
        common_pre = sorted(set(result_prefills) & set(baseline_prefills))
        if common_pre:
            median_pre_new  = statistics.median(result_prefills[c] for c in common_pre)
            median_pre_base = statistics.median(baseline_prefills[c] for c in common_pre)
            prefill_d = pct(median_pre_new, median_pre_base)
            prefill_ok = prefill_d <= -STAGE3_FIRST_TOKEN_THRESHOLD
            label_pre = "MEETS" if prefill_ok else "does not meet"
            print(f"  first_token_ms:  {median_pre_new:.1f} vs {median_pre_base:.1f} ({fmt_pct(prefill_d)}) — {label_pre} >={STAGE3_FIRST_TOKEN_THRESHOLD*100:.0f}% threshold")
        else:
            print(f"  first_token_ms:  missing/insufficient data")

        load_ok = False
        if load_new and load_base:
            load_d = pct(load_new, load_base)
            load_ok = load_d <= -STAGE3_LOAD_THRESHOLD
            print(f"  model_load_ms:   {load_new:.1f} vs {load_base:.1f} ({fmt_pct(load_d)}) — {'MEETS' if load_ok else 'does not meet'} >={STAGE3_LOAD_THRESHOLD*100:.0f}% threshold")
        else:
            print(f"  model_load_ms:   missing data")

        mem_ok = False
        if mem_new and mem_base:
            mem_d = pct(mem_new, mem_base)
            mem_ok = mem_d <= -STAGE3_MEMORY_THRESHOLD
            print(f"  memory_kb:       {mem_new} vs {mem_base} ({fmt_pct(mem_d)}) — {'MEETS' if mem_ok else 'does not meet'} >={STAGE3_MEMORY_THRESHOLD*100:.0f}% threshold")
        else:
            print(f"  memory_kb:       missing data")

        rss_ok = False
        if rss_new and rss_base:
            rss_d = pct(rss_new, rss_base)
            rss_ok = rss_d <= -STAGE3_MEMORY_THRESHOLD
            print(f"  peak_rss_kb:     {rss_new:.0f} vs {rss_base:.0f} ({fmt_pct(rss_d)}) — {'MEETS' if rss_ok else 'does not meet'} >={STAGE3_MEMORY_THRESHOLD*100:.0f}% threshold")
        else:
            print(f"  peak_rss_kb:     missing data")

        bin_ok = False
        if bin_size_new and bin_size_base:
            bin_d = pct(bin_size_new, bin_size_base)
            bin_ok = bin_d <= -STAGE3_BINARY_SIZE_THRESHOLD
            print(f"  binary_size:     {bin_size_new} vs {bin_size_base} ({fmt_pct(bin_d)}) — {'MEETS' if bin_ok else 'does not meet'} >={STAGE3_BINARY_SIZE_THRESHOLD*100:.0f}% threshold")
        else:
            print(f"  binary_size:     missing data")

        any_meets = inf_ok or prefill_ok or load_ok or mem_ok or rss_ok or bin_ok
        # Check if a memory/load/binary size win comes with an inference regression
        inf_regressed = inf_d > STAGE3_INFERENCE_REGRESSION_LIMIT if common5 else False

        if not any_meets:
            msg = "no metric meets acceptance threshold"
            print(f"\n  REJECT — {msg}")
            verdict_ok = False
            reasons_fail.append(msg)
        elif inf_regressed and not inf_ok:
            msg = f"win on secondary metric but inference latency regressed {fmt_pct(inf_d)} (limit +{STAGE3_INFERENCE_REGRESSION_LIMIT*100:.0f}%)"
            print(f"\n  REJECT — {msg}")
            verdict_ok = False
            reasons_fail.append(msg)
        else:
            wins = [m for ok, m in [
                (inf_ok, "inference"),
                (prefill_ok, "first_token_prefill"),
                (load_ok, "model_load"),
                (mem_ok, "memory"),
                (rss_ok, "peak_rss"),
                (bin_ok, "binary_size")
            ] if ok]
            msg = f"ACCEPT — {', '.join(wins)} threshold(s) met"
            print(f"\n  {msg}")
            reasons_pass.append(msg)

    # ── Save new baseline if accepted ──────────────────────────────────────────
    if args.stage == 3 and verdict_ok:
        latest_path = os.path.join(os.path.dirname(__file__), "latest_baseline.json")
        common_voice_path = os.path.join(RESULTS_DIR, args.result, f"common_voice-t{t}.json")
        ptt_guardrail_path = os.path.join(RESULTS_DIR, args.result, f"ptt_guardrail-t{t}.json")
        
        if os.path.exists(common_voice_path):
            with open(common_voice_path, encoding="utf-8") as f:
                common_voice_data = json.load(f)
            ptt_guardrail_data = None
            if os.path.exists(ptt_guardrail_path):
                with open(ptt_guardrail_path, encoding="utf-8") as f:
                    ptt_guardrail_data = json.load(f)
            
            timestamp = common_voice_data.get("timestamp", "")
            if not timestamp:
                import datetime
                timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                
            new_baseline = {
                "label": args.result,
                "timestamp": timestamp,
                "common_voice": common_voice_data,
                "ptt_guardrail": ptt_guardrail_data
            }
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(new_baseline, f, indent=2)
            print(f"\nUpdated latest baseline in standalone file: {latest_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    sep()
    print(f"RESULT: {'PASS' if verdict_ok else 'FAIL'}")
    if reasons_fail:
        print("Failed:")
        for r in reasons_fail:
            print(f"  ✗ {r}")
    if reasons_pass:
        print("Passed:")
        for r in reasons_pass:
            print(f"  ✓ {r}")
    sep()

    sys.exit(0 if verdict_ok else 1)


if __name__ == "__main__":
    main()
