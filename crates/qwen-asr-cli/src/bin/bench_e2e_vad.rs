use qwen_asr::audio;
use qwen_asr::context::QwenCtx;
use qwen_asr::vad_stream::{simulate_vad_stream, VadStreamConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct ManifestItem {
    audio: String,
    reference: Option<String>,
}

#[derive(Debug, Clone)]
struct SampleEntry {
    path: String,
    reference: String,
}

fn normalize_text(text: &str) -> String {
    let converted = zhconv::zhconv(text, zhconv::Variant::ZhHans);
    let mut out = String::new();
    for c in converted.chars() {
        if c.is_alphanumeric() {
            for lc in c.to_lowercase() {
                out.push(lc);
            }
        }
    }
    out
}

fn levenshtein(a: &str, b: &str) -> usize {
    let s1: Vec<char> = a.chars().collect();
    let s2: Vec<char> = b.chars().collect();
    let n1 = s1.len();
    let n2 = s2.len();
    if n1 == 0 {
        return n2;
    }
    if n2 == 0 {
        return n1;
    }

    let mut dp = vec![vec![0usize; n2 + 1]; n1 + 1];
    for i in 0..=n1 {
        dp[i][0] = i;
    }
    for j in 0..=n2 {
        dp[0][j] = j;
    }

    for i in 1..=n1 {
        for j in 1..=n2 {
            let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[n1][n2]
}

#[derive(Debug, Clone, Serialize)]
struct EvalSummary {
    config_name: String,
    threshold: f32,
    neg_threshold: f32,
    min_silence_ms: u32,
    speech_pad_ms: u32,
    speech_tail_pad_ms: u32,
    stream_chunk_sec: f32,
    stream_rollback: i32,
    adaptive_drain: bool,
    samples_count: usize,
    total_ref_chars: usize,
    total_edits: usize,
    cer: f64,
    exact_matches: usize,
    exact_match_pct: f64,
    avg_ttfc_audio_ms: f64,
    avg_ttfc_speech_ms: f64,
    avg_tail_latency_ms: f64,
    avg_char_latency_1: f64,
    avg_char_latency_2: f64,
    avg_char_latency_3: f64,
    avg_char_latency_5: f64,
    avg_char_latency_8: f64,
    avg_char_latency_10: f64,
    avg_rtf: f64,
}

fn evaluate_dataset(
    ctx: &mut QwenCtx,
    samples: &[SampleEntry],
    config: &VadStreamConfig,
    config_name: &str,
    silent: bool,
) -> EvalSummary {
    let mut total_ref_chars = 0;
    let mut total_edits = 0;
    let mut exact_matches = 0;

    let mut ttfc_audio_sum = 0.0;
    let mut ttfc_audio_count = 0;

    let mut ttfc_speech_sum = 0.0;
    let mut ttfc_speech_count = 0;

    let mut tail_latency_sum = 0.0;
    let mut rtf_sum = 0.0;

    // Per character latency sums
    let mut char_lat_sums: HashMap<usize, (f64, usize)> = HashMap::new();

    for (idx, entry) in samples.iter().enumerate() {
        let pcm = match audio::load_wav(&entry.path) {
            Some(s) => s,
            None => {
                eprintln!("Warning: failed to load {}", entry.path);
                continue;
            }
        };

        ctx.set_prompt("").ok();
        let metrics = simulate_vad_stream(ctx, &pcm, config);

        let norm_ref = normalize_text(&entry.reference);
        let norm_hyp = normalize_text(&metrics.text);

        let edits = levenshtein(&norm_ref, &norm_hyp);
        total_ref_chars += norm_ref.chars().count();
        total_edits += edits;
        if edits == 0 {
            exact_matches += 1;
        }

        if let Some(ttfc_a) = metrics.ttfc_from_audio_start_ms {
            ttfc_audio_sum += ttfc_a;
            ttfc_audio_count += 1;
        }
        if let Some(ttfc_s) = metrics.ttfc_from_speech_start_ms {
            ttfc_speech_sum += ttfc_s;
            ttfc_speech_count += 1;
        }
        tail_latency_sum += metrics.tail_latency_ms;
        rtf_sum += metrics.real_time_factor;

        for cl in &metrics.char_latencies {
            let entry = char_lat_sums.entry(cl.char_index).or_insert((0.0, 0));
            entry.0 += cl.audio_time_ms;
            entry.1 += 1;
        }

        if !silent {
            println!(
                "[{:>2}] ref: \"{}\"\n     hyp: \"{}\"\n     edits: {}, utt: {}, ttfc: {:.0}ms, tail: {:.0}ms",
                idx + 1,
                entry.reference,
                metrics.text,
                edits,
                metrics.utterances_count,
                metrics.ttfc_from_speech_start_ms.unwrap_or(0.0),
                metrics.tail_latency_ms
            );
        }
    }

    let cer = if total_ref_chars > 0 {
        (total_edits as f64 / total_ref_chars as f64) * 100.0
    } else {
        0.0
    };

    let n = samples.len().max(1) as f64;
    let avg_char_lat = |k: usize| -> f64 {
        if let Some(&(sum, cnt)) = char_lat_sums.get(&k) {
            if cnt > 0 {
                sum / cnt as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    EvalSummary {
        config_name: config_name.to_string(),
        threshold: config.threshold,
        neg_threshold: config.neg_threshold,
        min_silence_ms: config.min_silence_ms,
        speech_pad_ms: config.speech_pad_ms,
        speech_tail_pad_ms: config.speech_tail_pad_ms,
        stream_chunk_sec: config.stream_chunk_sec,
        stream_rollback: config.stream_rollback,
        adaptive_drain: config.adaptive_drain,
        samples_count: samples.len(),
        total_ref_chars,
        total_edits,
        cer,
        exact_matches,
        exact_match_pct: (exact_matches as f64 / n) * 100.0,
        avg_ttfc_audio_ms: if ttfc_audio_count > 0 {
            ttfc_audio_sum / ttfc_audio_count as f64
        } else {
            0.0
        },
        avg_ttfc_speech_ms: if ttfc_speech_count > 0 {
            ttfc_speech_sum / ttfc_speech_count as f64
        } else {
            0.0
        },
        avg_tail_latency_ms: tail_latency_sum / n,
        avg_char_latency_1: avg_char_lat(1),
        avg_char_latency_2: avg_char_lat(2),
        avg_char_latency_3: avg_char_lat(3),
        avg_char_latency_5: avg_char_lat(5),
        avg_char_latency_8: avg_char_lat(8),
        avg_char_latency_10: avg_char_lat(10),
        avg_rtf: rtf_sum / n,
    }
}

fn load_samples(manifest_path: &str, max_clips: usize) -> Vec<SampleEntry> {
    let mut ref_map: HashMap<String, String> = HashMap::new();

    // Check if master manifest is available
    let master_manifest = "/Volumes/SSD/canto_keyboard_field_speech/manifest.jsonl";
    if Path::new(master_manifest).exists() {
        if let Ok(file) = File::open(master_manifest) {
            for line in BufReader::new(file).lines().flatten() {
                if let Ok(item) = serde_json::from_str::<ManifestItem>(&line) {
                    if let Some(ref r) = item.reference {
                        ref_map.insert(item.audio.clone(), r.clone());
                        let p = Path::new(&item.audio);
                        if let Some(parent) = p.parent().and_then(|p| p.file_name()) {
                            ref_map.insert(parent.to_string_lossy().to_string(), r.clone());
                        }
                    }
                }
            }
        }
    }

    let mut samples = Vec::new();
    let file = File::open(manifest_path).expect("failed to open manifest file");
    for line in BufReader::new(file).lines().flatten() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Try parsing as JSONL first
        if let Ok(item) = serde_json::from_str::<ManifestItem>(trimmed) {
            let ref_text = item
                .reference
                .unwrap_or_else(|| ref_map.get(&item.audio).cloned().unwrap_or_default());
            samples.push(SampleEntry {
                path: item.audio,
                reference: ref_text,
            });
        } else {
            // Text line with path
            let path_str = trimmed.to_string();
            let mut ref_text = ref_map.get(&path_str).cloned().unwrap_or_default();
            if ref_text.is_empty() {
                let p = Path::new(&path_str);
                if let Some(parent) = p.parent().and_then(|p| p.file_name()) {
                    ref_text = ref_map
                        .get(&parent.to_string_lossy().to_string())
                        .cloned()
                        .unwrap_or_default();
                }
            }
            samples.push(SampleEntry {
                path: path_str,
                reference: ref_text,
            });
        }

        if max_clips > 0 && samples.len() >= max_clips {
            break;
        }
    }

    samples
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = "/Volumes/SSD/models/netease-youdao/Confucius4-R2T2".to_string();
    let mut manifest_path = "/Users/tony/Documents/keyboard/dataset/qwen3_asr_finetune/offline_eval_keyboard_359_confucius4_r2t2/audio_manifest_40.txt".to_string();
    let mut max_clips = 0usize;
    let mut grid_search = false;
    let mut language = "Cantonese".to_string();

    let mut custom_threshold: Option<f32> = None;
    let mut custom_neg_threshold: Option<f32> = None;
    let mut custom_min_silence_ms: Option<u32> = None;
    let mut custom_speech_pad_ms: Option<u32> = None;
    let mut custom_stream_chunk_sec: Option<f32> = None;
    let mut custom_stream_rollback: Option<i32> = None;
    let mut custom_adaptive_drain: Option<bool> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--model" => {
                i += 1;
                model_dir = args.get(i).expect("missing model dir").clone();
            }
            "--manifest" => {
                i += 1;
                manifest_path = args.get(i).expect("missing manifest path").clone();
            }
            "--max-clips" => {
                i += 1;
                max_clips = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--language" => {
                i += 1;
                language = args.get(i).expect("missing language").clone();
            }
            "--grid-search" => {
                grid_search = true;
            }
            "--threshold" => {
                i += 1;
                custom_threshold = args.get(i).and_then(|s| s.parse().ok());
            }
            "--neg-threshold" => {
                i += 1;
                custom_neg_threshold = args.get(i).and_then(|s| s.parse().ok());
            }
            "--min-silence-ms" => {
                i += 1;
                custom_min_silence_ms = args.get(i).and_then(|s| s.parse().ok());
            }
            "--speech-pad-ms" => {
                i += 1;
                custom_speech_pad_ms = args.get(i).and_then(|s| s.parse().ok());
            }
            "--stream-chunk-sec" => {
                i += 1;
                custom_stream_chunk_sec = args.get(i).and_then(|s| s.parse().ok());
            }
            "--stream-rollback" => {
                i += 1;
                custom_stream_rollback = args.get(i).and_then(|s| s.parse().ok());
            }
            "--adaptive-drain" => {
                custom_adaptive_drain = Some(true);
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!("=== Qwen3-ASR End-to-End VAD + Streaming Benchmark ===");
    eprintln!("Model: {}", model_dir);
    eprintln!("Manifest: {}", manifest_path);

    let mut ctx = QwenCtx::load(&model_dir).expect("failed to load model");
    ctx.set_force_language(&language)
        .expect("unsupported language");
    ctx.set_chinese_variant("zh-Hant").ok();

    let samples = load_samples(&manifest_path, max_clips);
    eprintln!("Loaded {} sample clips with reference text.", samples.len());

    if grid_search {
        eprintln!("\n>>> Running Hill-Climbing Grid Search over VAD & Streaming Constants <<<\n");

        let mut configs = Vec::new();

        // 1. Min silence variation
        for &ms in &[300u32, 400, 500, 600, 800] {
            let mut c = VadStreamConfig::default();
            c.min_silence_ms = ms;
            configs.push((format!("min_silence_{}ms", ms), c));
        }

        // 2. VAD threshold variation
        for &th in &[0.35f32, 0.45, 0.50, 0.55, 0.65] {
            let mut c = VadStreamConfig::default();
            c.threshold = th;
            c.neg_threshold = (th - 0.15).max(0.15);
            configs.push((format!("threshold_{:.2}", th), c));
        }

        // 3. Pre-speech pad variation
        for &pad in &[30u32, 60, 90, 120] {
            let mut c = VadStreamConfig::default();
            c.speech_pad_ms = pad;
            configs.push((format!("pad_{}ms", pad), c));
        }

        // 4. Stream chunk size variation
        for &chunk in &[0.36f32, 0.40, 0.44, 0.48, 0.60] {
            let mut c = VadStreamConfig::default();
            c.stream_chunk_sec = chunk;
            configs.push((format!("chunk_{:.2}s", chunk), c));
        }

        // 5. Rollback variation (1 vs 2)
        for &rb in &[1i32, 2] {
            let mut c = VadStreamConfig::default();
            c.stream_rollback = rb;
            configs.push((format!("rollback_{}", rb), c));
        }

        // 6. Adaptive drain on vs off
        for &ad in &[false, true] {
            let mut c = VadStreamConfig::default();
            c.adaptive_drain = ad;
            configs.push((format!("adaptive_drain_{}", ad), c));
        }

        let mut results = Vec::new();
        for (name, cfg) in configs {
            eprintln!("Testing config: {} ...", name);
            let t0 = Instant::now();
            let summary = evaluate_dataset(&mut ctx, &samples, &cfg, &name, true);
            eprintln!(
                "  --> CER: {:.2}%, Exact: {:.1}%, TTFC: {:.0}ms, Tail: {:.0}ms, RTF: {:.3} ({:.1}s)",
                summary.cer, summary.exact_match_pct, summary.avg_ttfc_speech_ms, summary.avg_tail_latency_ms, summary.avg_rtf, t0.elapsed().as_secs_f64()
            );
            results.push(summary);
        }

        // Print Markdown Summary Table
        println!("\n## Hill-Climbing Grid Search Results\n");
        println!("| Configuration | CER (%) | Exact Match (%) | TTFC (Speech ms) | Tail Latency (ms) | 1st Char Lat (ms) | 5th Char Lat (ms) | RTF |");
        println!("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |");
        for r in &results {
            println!(
                "| **{}** | **{:.2}%** | {:.1}% | {:.0} ms | {:.0} ms | {:.0} ms | {:.0} ms | {:.3} |",
                r.config_name, r.cer, r.exact_match_pct, r.avg_ttfc_speech_ms, r.avg_tail_latency_ms, r.avg_char_latency_1, r.avg_char_latency_5, r.avg_rtf
            );
        }
    } else {
        let mut cfg = VadStreamConfig::default();
        if let Some(th) = custom_threshold {
            cfg.threshold = th;
        }
        if let Some(nth) = custom_neg_threshold {
            cfg.neg_threshold = nth;
        }
        if let Some(ms) = custom_min_silence_ms {
            cfg.min_silence_ms = ms;
        }
        if let Some(pad) = custom_speech_pad_ms {
            cfg.speech_pad_ms = pad;
        }
        if let Some(chunk) = custom_stream_chunk_sec {
            cfg.stream_chunk_sec = chunk;
        }
        if let Some(rb) = custom_stream_rollback {
            cfg.stream_rollback = rb;
        }
        if let Some(ad) = custom_adaptive_drain {
            cfg.adaptive_drain = ad;
        }

        eprintln!("\nEvaluating configuration:");
        eprintln!(
            "  Threshold: {:.2} (neg: {:.2})",
            cfg.threshold, cfg.neg_threshold
        );
        eprintln!("  Min Silence: {} ms", cfg.min_silence_ms);
        eprintln!(
            "  Speech Pad: {} ms (tail pad: {} ms)",
            cfg.speech_pad_ms, cfg.speech_tail_pad_ms
        );
        eprintln!("  Stream Chunk: {:.2} s", cfg.stream_chunk_sec);
        eprintln!("  Rollback: {}", cfg.stream_rollback);
        eprintln!("  Adaptive Drain: {}", cfg.adaptive_drain);

        let summary = evaluate_dataset(&mut ctx, &samples, &cfg, "single_run", false);

        println!("\n=== Evaluation Results ===");
        println!("Total Samples:       {}", summary.samples_count);
        println!("Total Ref Chars:     {}", summary.total_ref_chars);
        println!("Total Edits:         {}", summary.total_edits);
        println!("CER:                 {:.2}%", summary.cer);
        println!(
            "Exact Match:         {}/{} ({:.2}%)",
            summary.exact_matches, summary.samples_count, summary.exact_match_pct
        );
        println!("TTFC (Audio Start):  {:.1} ms", summary.avg_ttfc_audio_ms);
        println!("TTFC (Speech Start): {:.1} ms", summary.avg_ttfc_speech_ms);
        println!("Tail Latency:        {:.1} ms", summary.avg_tail_latency_ms);
        println!("Real-Time Factor:    {:.3}", summary.avg_rtf);
        println!("\n--- Per-Character Latency (Audio Time) ---");
        println!("1st Character:       {:.1} ms", summary.avg_char_latency_1);
        println!("2nd Character:       {:.1} ms", summary.avg_char_latency_2);
        println!("3rd Character:       {:.1} ms", summary.avg_char_latency_3);
        println!("5th Character:       {:.1} ms", summary.avg_char_latency_5);
        println!("8th Character:       {:.1} ms", summary.avg_char_latency_8);
        println!("10th Character:      {:.1} ms", summary.avg_char_latency_10);
    }
}
