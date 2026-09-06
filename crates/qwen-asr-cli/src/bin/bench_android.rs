use qwen_asr::{audio, context::QwenCtx, kernels, transcribe};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone)]
struct RunResult {
    run: usize,
    inference_ms: f64,
    wall_inference_ms: f64,
    encode_ms: f64,
    decode_ms: f64,
    mel_ms: f64,
    encoder_ms: f64,
    prefill_ms: f64,
    first_visible_token_ms: f64,
    decode_loop_ms: f64,
    rss_before_kb: u64,
    rss_after_kb: u64,
    peak_rss_kb: u64,
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

#[cfg(any(target_os = "android", target_os = "linux"))]
fn proc_status_kb(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let mut parts = rest.split_whitespace();
            return parts.next()?.parse().ok();
        }
    }
    None
}

#[cfg(any(target_os = "android", target_os = "linux"))]
fn current_rss_kb() -> u64 {
    proc_status_kb("VmRSS:").unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn current_rss_kb() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the supplied rusage struct on success.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    // macOS reports ru_maxrss in bytes; Linux reports KiB and uses /proc above.
    (unsafe { usage.assume_init() }.ru_maxrss as u64) / 1024
}

#[cfg(not(any(target_os = "android", target_os = "linux", target_os = "macos")))]
fn current_rss_kb() -> u64 {
    0
}

fn peak_rss_during<F, R>(operation: F) -> (R, u64)
where
    F: FnOnce() -> R,
{
    let running = Arc::new(AtomicBool::new(true));
    let sampler_running = Arc::clone(&running);
    let initial_rss = current_rss_kb();
    let sampler = thread::spawn(move || {
        let mut peak = initial_rss;
        while sampler_running.load(Ordering::Relaxed) {
            peak = peak.max(current_rss_kb());
            thread::sleep(Duration::from_millis(2));
        }
        peak.max(current_rss_kb())
    });

    let result = operation();
    running.store(false, Ordering::Relaxed);
    let peak_rss = sampler.join().unwrap_or(initial_rss);
    (result, peak_rss)
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
        "Usage: qwen-asr-bench-android --model-dir DIR --input WAV_OR_DIR [--runs N] [--threads N] [--language LANG] [--prompt TEXT] [--profile]\n\
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

fn parse_args() -> (
    String,
    PathBuf,
    usize,
    i32,
    Option<String>,
    Option<String>,
    bool,
) {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }

    if !args[1].starts_with('-') {
        let model_dir = args[1].clone();
        let input = PathBuf::from(&args[2]);
        let runs = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
        let threads = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        return (model_dir, input, runs, threads, None, None, false);
    }

    let mut model_dir = None;
    let mut input = None;
    let mut runs = 3usize;
    let mut threads = 0i32;
    let mut language = None;
    let mut prompt = None;
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
            "--language" => {
                i += 1;
                language = args.get(i).cloned();
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
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
    (model_dir, input, runs, threads, language, prompt, profile)
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
            let decode_loop_ms_per_token = if run.tokens > 0 {
                run.decode_loop_ms / run.tokens as f64
            } else {
                0.0
            };
            format!(
                "{{\"run\":{},\"inference_ms\":{:.3},\"wall_inference_ms\":{:.3},\"encode_ms\":{:.3},\"decode_ms\":{:.3},\"mel_ms\":{:.3},\"encoder_ms\":{:.3},\"prefill_ms\":{:.3},\"first_visible_token_ms\":{:.3},\"decode_loop_ms\":{:.3},\"decode_loop_ms_per_token\":{:.6},\"rss_before_kb\":{},\"rss_after_kb\":{},\"peak_rss_kb\":{},\"tokens\":{},\"rtf\":{:.6},\"transcript\":\"{}\",\"profile\":[{}]}}",
                run.run,
                run.inference_ms,
                run.wall_inference_ms,
                run.encode_ms,
                run.decode_ms,
                run.mel_ms,
                run.encoder_ms,
                run.prefill_ms,
                run.first_visible_token_ms,
                run.decode_loop_ms,
                decode_loop_ms_per_token,
                run.rss_before_kb,
                run.rss_after_kb,
                run.peak_rss_kb,
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
    let (model_dir, input, runs, mut n_threads, language, prompt, profile) = parse_args();
    if runs == 0 {
        eprintln!("--runs must be greater than zero");
        std::process::exit(2);
    }

    let wavs = collect_wavs(&input);
    let process_start = Instant::now();
    kernels::set_verbose(0);
    kernels::set_profile(profile);
    if n_threads <= 0 {
        n_threads = kernels::default_num_threads() as i32;
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
    if let Some(ref language) = language {
        if ctx.set_force_language(language).is_err() {
            eprintln!("unsupported language: {}", language);
            std::process::exit(2);
        }
    }
    if let Some(ref prompt) = prompt {
        if ctx.set_prompt(prompt).is_err() {
            eprintln!("failed to set prompt");
            std::process::exit(2);
        }
    }
    let model_load_ms = elapsed_ms(t0);
    let startup_ms = elapsed_ms(process_start);
    let memory_after_model_load_kb = current_rss_kb();
    let model_variant = if ctx.config.dec_hidden >= 2048 {
        "1.7B"
    } else {
        "0.6B"
    };

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
            let rss_before_kb = current_rss_kb();
            let t0 = Instant::now();
            let (transcript_result, peak_rss_kb) =
                peak_rss_during(|| transcribe::transcribe_audio(&mut ctx, &samples));
            let transcript = match transcript_result {
                Some(text) => text,
                None => {
                    eprintln!("transcription failed on {} run {}", audio_path, run);
                    std::process::exit(1);
                }
            };
            let wall_inference_ms = elapsed_ms(t0);
            let rss_after_kb = current_rss_kb();
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
                mel_ms: ctx.perf_mel_ms,
                encoder_ms: ctx.perf_encoder_ms,
                prefill_ms: ctx.perf_prefill_ms,
                first_visible_token_ms: ctx.perf_first_visible_token_ms,
                decode_loop_ms: ctx.perf_decode_loop_ms,
                rss_before_kb,
                rss_after_kb,
                peak_rss_kb,
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
        let median_mel_ms = median_by(&file_runs, |run| run.mel_ms);
        let median_encoder_ms = median_by(&file_runs, |run| run.encoder_ms);
        let median_prefill_ms = median_by(&file_runs, |run| run.prefill_ms);
        let median_first_visible_token_ms = median_by(&file_runs, |run| run.first_visible_token_ms);
        let median_decode_loop_ms = median_by(&file_runs, |run| run.decode_loop_ms);
        let median_tokens = median_by(&file_runs, |run| run.tokens as f64);
        let median_decode_loop_ms_per_token = if median_tokens > 0.0 {
            median_decode_loop_ms / median_tokens
        } else {
            0.0
        };
        let median_rtf = if median_inference_ms > 0.0 {
            audio_s / (median_inference_ms / 1000.0)
        } else {
            0.0
        };

        file_json.push(format!(
            "{{\"audio_path\":\"{}\",\"audio_load_ms\":{:.3},\"audio_s\":{:.6},\"median_run\":{},\"inference_ms\":{:.3},\"encode_ms\":{:.3},\"decode_ms\":{:.3},\"mel_ms\":{:.3},\"encoder_ms\":{:.3},\"prefill_ms\":{:.3},\"first_visible_token_ms\":{:.3},\"decode_loop_ms\":{:.3},\"decode_loop_ms_per_token\":{:.6},\"median_tokens\":{:.1},\"rtf\":{:.6},\"transcript\":\"{}\",\"run_details\":[{}]}}",
            json_escape(&audio_path),
            audio_load_ms,
            audio_s,
            median_run.run,
            median_inference_ms,
            median_encode_ms,
            median_decode_ms,
            median_mel_ms,
            median_encoder_ms,
            median_prefill_ms,
            median_first_visible_token_ms,
            median_decode_loop_ms,
            median_decode_loop_ms_per_token,
            median_tokens,
            median_rtf,
            json_escape(&median_run.transcript),
            run_details_json(&file_runs)
        ));
    }

    println!(
        "{{\"model_dir\":\"{}\",\"model_variant\":\"{}\",\"input\":\"{}\",\"runs\":{},\"threads\":{},\"language\":{},\"prompt\":{},\"setup_ms\":{:.3},\"model_load_ms\":{:.3},\"startup_ms\":{:.3},\"memory_after_model_load_kb\":{},\"optimization_flags\":[{}],\"target_arch\":\"{}\",\"profile_enabled\":{},\"files\":[{}]}}",
        json_escape(&model_dir),
        model_variant,
        json_escape(&input.to_string_lossy()),
        runs,
        n_threads,
        language
            .as_ref()
            .map(|lang| format!("\"{}\"", json_escape(lang)))
            .unwrap_or_else(|| "null".to_string()),
        prompt
            .as_ref()
            .map(|prompt| format!("\"{}\"", json_escape(prompt)))
            .unwrap_or_else(|| "null".to_string()),
        setup_ms,
        model_load_ms,
        startup_ms,
        memory_after_model_load_kb,
        flags,
        std::env::consts::ARCH,
        if profile { "true" } else { "false" },
        file_json.join(",")
    );
}
