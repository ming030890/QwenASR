//! Offline harness for the Cantonese ASR rule-correction experiment.
//!
//! Subcommands:
//!   transcribe  decode mp3 clips + run qwen-asr, cache {path,hyp,ref} jsonl
//!   cer         report CER (with/without punct); optionally apply a rule set
//!   mine        mine correction rules from a transcribed split -> rules.json
//!   tune        sweep mining constants on train+dev, emit the chosen rule set
//!   zhconv      stream Chinese variant conversion as JSON string lines

mod decode;
mod rules;
mod text;

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{BufRead, Write};

use rules::{MineParams, Rule, RuleSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HypRec {
    path: String,
    hyp: String,
    #[serde(rename = "ref")]
    reference: String,
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
fn arg_f64(args: &[String], flag: &str, default: f64) -> f64 {
    arg(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn arg_u32(args: &[String], flag: &str, default: u32) -> u32 {
    arg(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn arg_usize(args: &[String], flag: &str, default: usize) -> usize {
    arg(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: yue-tools <transcribe|cer|mine|tune|score|zhconv> [opts]");
        std::process::exit(1);
    }
    let cmd = args[1].clone();
    let rest = &args[2..].to_vec();
    match cmd.as_str() {
        "transcribe" => cmd_transcribe(rest),
        "cer" => cmd_cer(rest),
        "mine" => cmd_mine(rest),
        "tune" => cmd_tune(rest),
        "score" => cmd_score(rest),
        "zhconv" => cmd_zhconv(rest),
        other => {
            eprintln!("unknown subcommand: {}", other);
            std::process::exit(1);
        }
    }
}

fn cmd_zhconv(args: &[String]) {
    let variant_name = arg(args, "--variant").unwrap_or_else(|| "zh-HK".to_string());
    let variant = match variant_name.parse::<zhconv::Variant>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("unsupported zh variant: {}", variant_name);
            std::process::exit(1);
        }
    };

    let stdin = std::io::stdin();
    let mut stdout = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let text = match serde_json::from_str::<String>(&line) {
            Ok(text) => text,
            Err(_) => line,
        };
        let converted = zhconv::zhconv(&text, variant);
        writeln!(stdout, "{}", serde_json::to_string(&converted).unwrap()).unwrap();
        stdout.flush().ok();
    }
}

// ---------------------------------------------------------------------------

fn read_pairs_tsv(path: &str) -> Vec<(String, String)> {
    let f = std::fs::File::open(path).expect("open pairs tsv");
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line.unwrap();
        let mut it = line.splitn(2, '\t');
        if let (Some(p), Some(s)) = (it.next(), it.next()) {
            if !p.is_empty() && !s.is_empty() {
                out.push((p.to_string(), s.to_string()));
            }
        }
    }
    out
}

fn read_hyp_jsonl(path: &str) -> Vec<HypRec> {
    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<HypRec>(&line) {
            out.push(r);
        }
    }
    out
}

// ---------------------------------------------------------------------------

fn cmd_transcribe(args: &[String]) {
    let model = arg(args, "--model").expect("--model required");
    let pairs_path = arg(args, "--pairs").expect("--pairs required");
    let clips = arg(args, "--clips").expect("--clips required");
    let out_path = arg(args, "--out").expect("--out required");
    let language = arg(args, "--language").unwrap_or_else(|| "Cantonese".to_string());
    let zh_variant = arg(args, "--zh-variant").unwrap_or_else(|| "zh-HK".to_string());
    let limit = arg(args, "--limit").and_then(|s| s.parse::<usize>().ok());
    let threads = arg(args, "--threads")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(qwen_asr::kernels::get_num_cpus);
    qwen_asr::kernels::set_threads(threads);
    eprintln!("threads = {}", threads);

    let mut pairs = read_pairs_tsv(&pairs_path);
    if let Some(n) = limit {
        pairs.truncate(n);
    }

    // resume: skip paths already present in out
    let done: HashSet<String> = read_hyp_jsonl(&out_path)
        .into_iter()
        .map(|r| r.path)
        .collect();

    let mut ctx = qwen_asr::context::QwenCtx::load(&model).expect("load model");
    if ctx.set_force_language(&language).is_err() {
        eprintln!("unsupported language: {}", language);
        std::process::exit(1);
    }
    if ctx.set_chinese_variant(&zh_variant).is_err() {
        eprintln!("unsupported zh variant: {}", zh_variant);
        std::process::exit(1);
    }

    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("open out");

    let total = pairs.len();
    let mut n_done = 0usize;
    for (i, (fname, sent)) in pairs.iter().enumerate() {
        let stem = std::path::Path::new(fname)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(fname)
            .to_string();
        if done.contains(&stem) {
            continue;
        }
        let mp3 = format!("{}/{}", clips.trim_end_matches('/'), fname);
        let hyp = match decode::decode_to_16k_mono(&mp3) {
            Some(samples) => {
                qwen_asr::transcribe::transcribe_audio(&mut ctx, &samples).unwrap_or_default()
            }
            None => {
                eprintln!("decode failed: {}", mp3);
                String::new()
            }
        };
        let rec = HypRec {
            path: stem,
            hyp,
            reference: sent.clone(),
        };
        writeln!(out, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
        out.flush().ok();
        n_done += 1;
        if (i + 1) % 50 == 0 {
            eprintln!("  {} / {} (newly transcribed {})", i + 1, total, n_done);
        }
    }
    eprintln!(
        "done: {} clips, {} newly transcribed -> {}",
        total, n_done, out_path
    );
}

// ---------------------------------------------------------------------------

fn norm_pairs(recs: &[HypRec], strip_punct: bool) -> Vec<(Vec<char>, Vec<char>)> {
    recs.iter()
        .map(|r| {
            (
                text::normalize(&r.hyp, strip_punct),
                text::normalize(&r.reference, strip_punct),
            )
        })
        .collect()
}

fn corpus_cer_chars(pairs: &[(Vec<char>, Vec<char>)]) -> (f64, usize, usize) {
    let mut edits = 0usize;
    let mut chars = 0usize;
    for (h, r) in pairs {
        if r.is_empty() {
            continue;
        }
        edits += text::levenshtein(h, r);
        chars += r.len();
    }
    (edits as f64 / chars.max(1) as f64, edits, chars)
}

fn cmd_cer(args: &[String]) {
    let in_path = arg(args, "--in").expect("--in required");
    let recs = read_hyp_jsonl(&in_path);
    eprintln!("loaded {} records from {}", recs.len(), in_path);

    for &sp in &[true, false] {
        let tag = if sp { "no-punct" } else { "with-punct" };
        let pairs: Vec<(&str, &str)> = recs
            .iter()
            .map(|r| (r.hyp.as_str(), r.reference.as_str()))
            .collect();
        let s = text::cer(pairs.iter().copied(), sp);
        println!(
            "[baseline {:9}] corpus_cer={:.4} macro_cer={:.4} edits={} chars={} n={}",
            tag, s.corpus_cer, s.macro_cer, s.edits, s.ref_chars, s.n
        );
    }

    if let Some(rules_path) = arg(args, "--rules") {
        let rs: RuleSet =
            serde_json::from_str(&std::fs::read_to_string(&rules_path).expect("read rules"))
                .expect("parse rules");
        let mut ruleset = rs.rules;
        rules::sort_for_apply(&mut ruleset);
        let compiled: Vec<(Vec<char>, Vec<char>)> = ruleset
            .iter()
            .map(|r| (r.from.chars().collect(), r.to.chars().collect()))
            .collect();
        println!("\napplying {} rules", ruleset.len());
        for &sp in &[true, false] {
            let tag = if sp { "no-punct" } else { "with-punct" };
            let np = norm_pairs(&recs, sp);
            let corr: Vec<(Vec<char>, Vec<char>)> = np
                .into_iter()
                .map(|(h, r)| (rules::apply_compiled(&h, &compiled), r))
                .collect();
            let (cer, edits, chars) = corpus_cer_chars(&corr);
            println!(
                "[corrected {:9}] corpus_cer={:.4} edits={} chars={}",
                tag, cer, edits, chars
            );
        }
        report_examples(&recs, &compiled);
    }
}

fn report_examples(recs: &[HypRec], compiled: &[(Vec<char>, Vec<char>)]) {
    let mut fixed = 0usize;
    let mut broke = 0usize;
    let mut samples_fix = Vec::new();
    let mut samples_break = Vec::new();
    for r in recs {
        let h = text::normalize(&r.hyp, true);
        let rf = text::normalize(&r.reference, true);
        if rf.is_empty() {
            continue;
        }
        let hc = rules::apply_compiled(&h, compiled);
        let d0 = text::levenshtein(&h, &rf);
        let d1 = text::levenshtein(&hc, &rf);
        if d1 < d0 {
            fixed += 1;
            if samples_fix.len() < 8 {
                samples_fix.push((
                    h.iter().collect::<String>(),
                    hc.iter().collect::<String>(),
                    rf.iter().collect::<String>(),
                ));
            }
        } else if d1 > d0 {
            broke += 1;
            if samples_break.len() < 8 {
                samples_break.push((
                    h.iter().collect::<String>(),
                    hc.iter().collect::<String>(),
                    rf.iter().collect::<String>(),
                ));
            }
        }
    }
    println!("\nfixed={} broke={}", fixed, broke);
    for (tag, lst) in [("FIX", &samples_fix), ("BREAK", &samples_break)] {
        for (h, hc, rf) in lst {
            println!("  [{}] {} -> {} | ref {}", tag, h, hc, rf);
        }
    }
}

// ---------------------------------------------------------------------------

fn params_from_args(args: &[String]) -> MineParams {
    let d = MineParams::default();
    MineParams {
        p_high: arg_f64(args, "--p-high", d.p_high),
        n_min: arg_u32(args, "--n-min", d.n_min),
        max_ctx: arg_usize(args, "--max-ctx", d.max_ctx),
        p_high_single: arg_f64(args, "--p-high-single", d.p_high_single),
        n_min_single: arg_u32(args, "--n-min-single", d.n_min_single),
    }
}

fn cmd_mine(args: &[String]) {
    let in_path = arg(args, "--in").expect("--in required");
    let out_path = arg(args, "--out").expect("--out required");
    let params = params_from_args(args);
    let recs = read_hyp_jsonl(&in_path);
    let pairs = norm_pairs(&recs, true);
    eprintln!(
        "mining from {} utterances, params={:?}",
        pairs.len(),
        params
    );
    let rules = rules::mine(&pairs, params);
    eprintln!("mined {} rules", rules.len());
    let rs = RuleSet { params, rules };
    std::fs::write(&out_path, serde_json::to_string_pretty(&rs).unwrap()).unwrap();
    eprintln!("wrote {}", out_path);
    // preview top rules by count
    let mut preview = rs.rules.clone();
    preview.sort_by(|a, b| b.count.cmp(&a.count));
    for r in preview.iter().take(20) {
        eprintln!(
            "  {:>5}x p={:.3}  {:?} -> {:?}",
            r.count, r.precision, r.from, r.to
        );
    }
}

// ---------------------------------------------------------------------------

fn net_benefit_prune(mut rules: Vec<Rule>, dev: &[(Vec<char>, Vec<char>)]) -> Vec<Rule> {
    // greedily add rules (by count desc); keep only if it reduces total dev edits.
    rules.sort_by(|a, b| b.count.cmp(&a.count));
    let mut kept: Vec<Rule> = Vec::new();
    let base_edits = |kr: &[Rule]| -> usize {
        let compiled: Vec<(Vec<char>, Vec<char>)> = kr
            .iter()
            .map(|r| (r.from.chars().collect(), r.to.chars().collect()))
            .collect();
        dev.iter()
            .map(|(h, r)| text::levenshtein(&rules::apply_compiled(h, &compiled), r))
            .sum()
    };
    let mut cur = base_edits(&kept);
    for r in rules {
        kept.push(r.clone());
        rules::sort_for_apply(&mut kept);
        let e = base_edits(&kept);
        if e < cur {
            cur = e;
        } else {
            // revert
            kept.retain(|x| !(x.from == r.from && x.to == r.to));
        }
    }
    rules::sort_for_apply(&mut kept);
    kept
}

fn cmd_tune(args: &[String]) {
    let train_path = arg(args, "--train").expect("--train required");
    let dev_path = arg(args, "--dev").expect("--dev required");
    let out_path = arg(args, "--out").expect("--out required");
    let dev_limit = arg(args, "--dev-limit").and_then(|s| s.parse::<usize>().ok());

    let train = norm_pairs(&read_hyp_jsonl(&train_path), true);
    let mut dev = norm_pairs(&read_hyp_jsonl(&dev_path), true);
    if let Some(n) = dev_limit {
        dev.truncate(n);
    }
    let (base_cer, _, _) = corpus_cer_chars(&dev);
    eprintln!(
        "dev baseline corpus_cer(no-punct) = {:.4} over {} utts",
        base_cer,
        dev.len()
    );

    let p_highs = [0.85, 0.90, 0.95];
    let n_mins = [10u32, 20, 40];
    let max_ctxs = [1usize, 2];

    let mut best: Option<(f64, MineParams, Vec<Rule>)> = None;
    for &p in &p_highs {
        for &nm in &n_mins {
            for &k in &max_ctxs {
                let params = MineParams {
                    p_high: p,
                    n_min: nm,
                    max_ctx: k,
                    p_high_single: (p + 0.05).min(0.99),
                    n_min_single: nm * 2,
                };
                let mined = rules::mine(&train, params);
                let compiled: Vec<(Vec<char>, Vec<char>)> = mined
                    .iter()
                    .map(|r| (r.from.chars().collect(), r.to.chars().collect()))
                    .collect();
                let corr: Vec<(Vec<char>, Vec<char>)> = dev
                    .iter()
                    .map(|(h, r)| (rules::apply_compiled(h, &compiled), r.clone()))
                    .collect();
                let (cer, _, _) = corpus_cer_chars(&corr);
                eprintln!(
                    "  p={:.2} n={} k={} -> {} rules, dev_cer={:.4} (delta {:+.4})",
                    p,
                    nm,
                    k,
                    mined.len(),
                    cer,
                    cer - base_cer
                );
                let better = best.as_ref().map(|(bc, _, _)| cer < *bc).unwrap_or(true);
                if better {
                    best = Some((cer, params, mined));
                }
            }
        }
    }

    let (cer, params, mined) = best.expect("no params tried");
    eprintln!("best params={:?} dev_cer={:.4}", params, cer);

    // net-benefit prune on dev (final overcorrection guard)
    let pruned = net_benefit_prune(mined, &dev);
    let compiled: Vec<(Vec<char>, Vec<char>)> = pruned
        .iter()
        .map(|r| (r.from.chars().collect(), r.to.chars().collect()))
        .collect();
    let corr: Vec<(Vec<char>, Vec<char>)> = dev
        .iter()
        .map(|(h, r)| (rules::apply_compiled(h, &compiled), r.clone()))
        .collect();
    let (cer2, _, _) = corpus_cer_chars(&corr);
    eprintln!(
        "after net-benefit prune: {} rules, dev_cer={:.4} (baseline {:.4})",
        pruned.len(),
        cer2,
        base_cer
    );

    if cer2 > base_cer {
        eprintln!("WARNING: pruned set regresses dev; writing empty rule set");
    }
    let final_rules = if cer2 <= base_cer { pruned } else { Vec::new() };
    let rs = RuleSet {
        params,
        rules: final_rules,
    };
    std::fs::write(&out_path, serde_json::to_string_pretty(&rs).unwrap()).unwrap();
    eprintln!("wrote {} ({} rules)", out_path, rs.rules.len());
}

// ---------------------------------------------------------------------------

fn cmd_score(args: &[String]) {
    let model = arg(args, "--model").expect("--model required");
    let audio_path = arg(args, "--audio"); // optional
    let s1 = arg(args, "--s1").unwrap_or_else(|| "我今日用粵拼".to_string());
    let s2 = arg(args, "--s2").unwrap_or_else(|| "我今日用月拼".to_string());

    let mut ctx = qwen_asr::context::QwenCtx::load(&model).expect("load model");
    let vocab_path = format!("{}/vocab.json", model);
    let tokenizer = qwen_asr::tokenizer::QwenTokenizer::load(&vocab_path).expect("load tokenizer");

    ctx.set_force_language("Cantonese").unwrap();
    ctx.prepare_prompt_tokens(&tokenizer);

    println!("Scoring S1: {}", s1);
    println!("Scoring S2: {}\n", s2);

    let samples = audio_path.as_ref().and_then(|path| {
        println!("Loading audio: {}", path);
        decode::decode_to_16k_mono(path)
    });

    let s1_tokens = tokenizer.encode(&s1).expect("encode s1");
    let s2_tokens = tokenizer.encode(&s2).expect("encode s2");

    println!("========================================");
    println!("EXPERIMENT 1: With Audio Context (ASR Rescoring)");
    println!("========================================");
    if let Some(ref s_data) = samples {
        println!("\n--- S1 ('{}') ---", s1);
        let score_s1 = score_sequence(&mut ctx, &tokenizer, Some(s_data), &s1_tokens);
        println!("\n--- S2 ('{}') ---", s2);
        let score_s2 = score_sequence(&mut ctx, &tokenizer, Some(s_data), &s2_tokens);
        println!("\nResult Summary (With Audio):");
        println!("  S1 Total Log-Prob: {:.4}", score_s1);
        println!("  S2 Total Log-Prob: {:.4}", score_s2);
        println!("  Difference (S1 - S2): {:.4}", score_s1 - score_s2);
    } else {
        println!("No audio file provided via --audio. Skipping Experiment 1.\n");
    }

    println!("\n========================================");
    println!("EXPERIMENT 2: Pure Text LM Score (No Audio)");
    println!("========================================");
    println!("\n--- S1 ('{}') ---", s1);
    let score_s1_no = score_sequence(&mut ctx, &tokenizer, None, &s1_tokens);
    println!("\n--- S2 ('{}') ---", s2);
    let score_s2_no = score_sequence(&mut ctx, &tokenizer, None, &s2_tokens);
    println!("\nResult Summary (No Audio):");
    println!("  S1 Total Log-Prob: {:.4}", score_s1_no);
    println!("  S2 Total Log-Prob: {:.4}", score_s2_no);
    println!("  Difference (S1 - S2): {:.4}", score_s1_no - score_s2_no);
}

fn score_sequence(
    ctx: &mut qwen_asr::context::QwenCtx,
    tokenizer: &qwen_asr::tokenizer::QwenTokenizer,
    audio_samples: Option<&[f32]>,
    sent_tokens: &[i32],
) -> f64 {
    use qwen_asr::decoder::{decoder_prefill_logits, tok_embed_to_f32};

    // Reset KV Cache length to 0 to simulate fresh sequence start
    ctx.kv_cache.len = 0;

    let dim = ctx.config.dec_hidden;

    // 1. Build prompt tokens prefix (standard Qwen-ASR transcript prompt layout)
    let prefix_head: &[i32] = &[151644, 8948, 198];
    let prefix_tail: &[i32] = &[151645, 198, 151644, 872, 198, 151669];
    let suffix_base: &[i32] = &[151670, 151645, 198, 151644, 77091, 198];

    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());

    let (enc_output, enc_seq_len) = if let Some(samples) = audio_samples {
        let (mel, mel_frames) = qwen_asr::audio::mel_spectrogram(samples).expect("mel");
        let (win_enc, win_seq) = ctx
            .encoder
            .forward(&ctx.config, &mel, mel_frames, Some(&mut ctx.enc_bufs))
            .expect("encoder forward");
        (win_enc, win_seq)
    } else {
        (Vec::new(), 0)
    };

    let prefix_len = prefix_head.len() + n_prompt_tokens + prefix_tail.len();
    let suffix_len = suffix_base.len() + n_force_prompt_tokens;

    // Total sequence fed to the decoder
    let total_seq = prefix_len + enc_seq_len + suffix_len + sent_tokens.len();

    let mut input_embeds = vec![0.0f32; total_seq * dim];
    let mut off = 0;

    // Embed Prefix Head
    for &tok in prefix_head {
        unsafe {
            tok_embed_to_f32(
                &ctx.decoder,
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok,
                dim,
            )
        };
        off += 1;
    }

    // Embed Prompt Tokens
    if let Some(ref ptoks) = ctx.prompt_tokens {
        for &tok in ptoks {
            unsafe {
                tok_embed_to_f32(
                    &ctx.decoder,
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok,
                    dim,
                )
            };
            off += 1;
        }
    }

    // Embed Prefix Tail
    for &tok in prefix_tail {
        unsafe {
            tok_embed_to_f32(
                &ctx.decoder,
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok,
                dim,
            )
        };
        off += 1;
    }

    // Copy Audio Embeddings
    if enc_seq_len > 0 {
        let n = enc_seq_len * dim;
        input_embeds[off * dim..off * dim + n].copy_from_slice(&enc_output);
        off += enc_seq_len;
    }

    // Embed Suffix Base
    for &tok in suffix_base {
        unsafe {
            tok_embed_to_f32(
                &ctx.decoder,
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok,
                dim,
            )
        };
        off += 1;
    }

    // Embed Force Prompt Tokens
    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for &tok in ftoks {
            unsafe {
                tok_embed_to_f32(
                    &ctx.decoder,
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok,
                    dim,
                )
            };
            off += 1;
        }
    }

    // Embed Sentence Tokens
    let text_start_idx = off;
    for &tok in sent_tokens {
        unsafe {
            tok_embed_to_f32(
                &ctx.decoder,
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok,
                dim,
            )
        };
        off += 1;
    }

    // 2. Prefill and obtain per-position logits
    let logits = decoder_prefill_logits(
        &ctx.decoder,
        &ctx.config,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        &input_embeds,
        total_seq,
    );

    // 3. Accumulate sequence probability
    let out_dim = ctx.config.lm_head_dim();
    let mut total_log_prob = 0.0;

    println!("  Token-by-token log-probs:");
    for i in 0..sent_tokens.len() {
        // The prediction of the token at `sent_tokens[i]` happens at the previous position: `text_start_idx - 1 + i`
        let pred_pos = text_start_idx - 1 + i;
        let logit_slice = &logits[pred_pos * out_dim..(pred_pos + 1) * out_dim];
        let log_probs = log_softmax(logit_slice);

        let target_tok = sent_tokens[i];
        let tok_log_prob = log_probs[target_tok as usize] as f64;
        total_log_prob += tok_log_prob;

        let char_bytes = tokenizer.decode_bytes(target_tok);
        let char_str = String::from_utf8_lossy(char_bytes);
        println!(
            "    pos {:>2} | token: {:>6} ({:^6}) | log_prob: {:.4}",
            i,
            target_tok,
            char_str.trim(),
            tok_log_prob
        );
    }

    total_log_prob
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &val in logits {
        if val > max_val {
            max_val = val;
        }
    }
    let mut sum = 0.0f32;
    let mut out = vec![0.0f32; logits.len()];
    for i in 0..logits.len() {
        let diff = logits[i] - max_val;
        let exp = diff.exp();
        out[i] = diff;
        sum += exp;
    }
    let log_sum = sum.ln();
    for i in 0..logits.len() {
        out[i] -= log_sum;
    }
    out
}
