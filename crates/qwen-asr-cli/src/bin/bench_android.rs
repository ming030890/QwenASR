use qwen_asr::{audio, context::QwenCtx, kernels, transcribe};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone)]
struct RunResult {
    run: usize,
    inference_ms: f64,
    wall_inference_ms: f64,
    encode_ms: f64,
    decode_ms: f64,
    tokens: i32,
    rtf: f64,
    transcript: String,
    profile: Vec<kernels::ProfileCounterSnapshot>,
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

fn median_by<F>(runs: &[RunResult], value: F) -> f64
where
    F: Fn(&RunResult) -> f64,
{
    let mut values: Vec<f64> = runs.iter().map(value).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn usage() -> ! {
    eprintln!(
        "Usage: qwen-asr-bench-android --model-dir DIR --input WAV_OR_DIR [--runs N] [--threads N] [--profile]\n\
         Backward-compatible positional form: qwen-asr-bench-android <model_dir> <audio.wav|dir> [runs] [threads]"
    );
    std::process::exit(2);
}

fn collect_wavs(input: &Path) -> Vec<PathBuf> {
    if input.is_file() {
        return vec![input.to_path_buf()];
    }
    if !input.is_dir() {
        eprintln!("input is not a file or directory: {}", input.display());
        std::process::exit(2);
    }

    let mut wavs = Vec::new();
    let entries = match std::fs::read_dir(input) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!(
                "failed to read input directory {}: {}",
                input.display(),
                err
            );
            std::process::exit(2);
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("wav") {
            wavs.push(path);
        }
    }
    wavs.sort();
    if wavs.is_empty() {
        eprintln!("no .wav files found in {}", input.display());
        std::process::exit(2);
    }
    wavs
}

fn parse_args() -> (String, PathBuf, usize, i32, bool) {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }

    if !args[1].starts_with('-') {
        let model_dir = args[1].clone();
        let input = PathBuf::from(&args[2]);
        let runs = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
        let threads = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        return (model_dir, input, runs, threads, false);
    }

    let mut model_dir = None;
    let mut input = None;
    let mut runs = 3usize;
    let mut threads = 0i32;
    let mut profile = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = args.get(i).cloned();
            }
            "--input" => {
                i += 1;
                input = args.get(i).map(PathBuf::from);
            }
            "--runs" => {
                i += 1;
                runs = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(3);
            }
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--profile" => {
                profile = true;
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown option: {}", other);
                usage();
            }
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(|| usage());
    let input = input.unwrap_or_else(|| usage());
    (model_dir, input, runs, threads, profile)
}

fn profile_json(counters: &[kernels::ProfileCounterSnapshot]) -> String {
    counters
        .iter()
        .map(|counter| {
            format!(
                "{{\"name\":\"{}\",\"total_ms\":{:.3},\"calls\":{},\"avg_ms\":{:.6}}}",
                json_escape(counter.name),
                counter.total_ms,
                counter.calls,
                counter.avg_ms
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn run_details_json(runs: &[RunResult]) -> String {
    runs.iter()
        .map(|run| {
            format!(
                "{{\"run\":{},\"inference_ms\":{:.3},\"wall_inference_ms\":{:.3},\"encode_ms\":{:.3},\"decode_ms\":{:.3},\"tokens\":{},\"rtf\":{:.6},\"transcript\":\"{}\",\"profile\":[{}]}}",
                run.run,
                run.inference_ms,
                run.wall_inference_ms,
                run.encode_ms,
                run.decode_ms,
                run.tokens,
                run.rtf,
                json_escape(&run.transcript),
                profile_json(&run.profile)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let (model_dir, input, runs, mut n_threads, profile) = parse_args();
    if runs == 0 {
        eprintln!("--runs must be greater than zero");
        std::process::exit(2);
    }

    let wavs = collect_wavs(&input);
    let process_start = Instant::now();
    kernels::set_verbose(0);
    kernels::set_profile(profile);
    if n_threads <= 0 {
        n_threads = kernels::get_num_cpus() as i32;
    }
    kernels::set_threads(n_threads as usize);
    let setup_ms = elapsed_ms(process_start);

    let t0 = Instant::now();
    let mut ctx = match QwenCtx::load(&model_dir) {
        Some(ctx) => ctx,
        None => {
            eprintln!("failed to load model from {}", model_dir);
            std::process::exit(1);
        }
    };
    let model_load_ms = elapsed_ms(t0);
    let startup_ms = elapsed_ms(process_start);

    let flags = qwen_asr::optimization_flags()
        .iter()
        .map(|flag| format!("\"{}\"", json_escape(flag)))
        .collect::<Vec<_>>()
        .join(",");

    let mut file_json = Vec::with_capacity(wavs.len());
    for wav in wavs {
        let audio_path = wav.to_string_lossy().to_string();
        let t0 = Instant::now();
        let samples = match audio::load_wav(&audio_path) {
            Some(samples) => samples,
            None => {
                eprintln!("failed to load wav from {}", audio_path);
                std::process::exit(1);
            }
        };
        let audio_load_ms = elapsed_ms(t0);
        let audio_s = samples.len() as f64 / 16000.0;
        let mut file_runs = Vec::with_capacity(runs);

        for run in 1..=runs {
            ctx.reset_perf();
            if profile {
                kernels::profile_reset();
            }
            let t0 = Instant::now();
            let transcript = match transcribe::transcribe_audio(&mut ctx, &samples) {
                Some(text) => text,
                None => {
                    eprintln!("transcription failed on {} run {}", audio_path, run);
                    std::process::exit(1);
                }
            };
            let wall_inference_ms = elapsed_ms(t0);
            let inference_ms = if ctx.perf_total_ms > 0.0 {
                ctx.perf_total_ms
            } else {
                wall_inference_ms
            };
            let rtf = if inference_ms > 0.0 {
                audio_s / (inference_ms / 1000.0)
            } else {
                0.0
            };
            let profile_snapshot = if profile {
                kernels::profile_snapshot()
            } else {
                Vec::new()
            };
            file_runs.push(RunResult {
                run,
                inference_ms,
                wall_inference_ms,
                encode_ms: ctx.perf_encode_ms,
                decode_ms: ctx.perf_decode_ms,
                tokens: ctx.perf_text_tokens,
                rtf,
                transcript,
                profile: profile_snapshot,
            });
        }

        let mut sorted_runs = file_runs.clone();
        sorted_runs.sort_by(|a, b| a.inference_ms.partial_cmp(&b.inference_ms).unwrap());
        let median_run = &sorted_runs[sorted_runs.len() / 2];
        let median_inference_ms = median_by(&file_runs, |run| run.inference_ms);
        let median_encode_ms = median_by(&file_runs, |run| run.encode_ms);
        let median_decode_ms = median_by(&file_runs, |run| run.decode_ms);
        let median_rtf = if median_inference_ms > 0.0 {
            audio_s / (median_inference_ms / 1000.0)
        } else {
            0.0
        };

        file_json.push(format!(
            "{{\"audio_path\":\"{}\",\"audio_load_ms\":{:.3},\"audio_s\":{:.6},\"median_run\":{},\"inference_ms\":{:.3},\"encode_ms\":{:.3},\"decode_ms\":{:.3},\"rtf\":{:.6},\"transcript\":\"{}\",\"run_details\":[{}]}}",
            json_escape(&audio_path),
            audio_load_ms,
            audio_s,
            median_run.run,
            median_inference_ms,
            median_encode_ms,
            median_decode_ms,
            median_rtf,
            json_escape(&median_run.transcript),
            run_details_json(&file_runs)
        ));
    }

    println!(
        "{{\"model_dir\":\"{}\",\"input\":\"{}\",\"runs\":{},\"threads\":{},\"setup_ms\":{:.3},\"model_load_ms\":{:.3},\"startup_ms\":{:.3},\"optimization_flags\":[{}],\"target_arch\":\"{}\",\"profile_enabled\":{},\"files\":[{}]}}",
        json_escape(&model_dir),
        json_escape(&input.to_string_lossy()),
        runs,
        n_threads,
        setup_ms,
        model_load_ms,
        startup_ms,
        flags,
        std::env::consts::ARCH,
        if profile { "true" } else { "false" },
        file_json.join(",")
    );
}
