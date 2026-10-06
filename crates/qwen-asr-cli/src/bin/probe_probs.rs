use qwen_asr::audio;
use qwen_asr::config::*;
use qwen_asr::context::QwenCtx;
use qwen_asr::decoder::{self, tok_embed_to_f32};
use qwen_asr::tokenizer::QwenTokenizer;

const PREFIX_HEAD: &[i32] = &[151644, 8948, 198];
const PREFIX_TAIL: &[i32] = &[151645, 198, 151644, 872, 198, 151669];
const SUFFIX_BASE: &[i32] = &[151670, 151645, 198, 151644, 77091, 198];

fn usage() -> ! {
    eprintln!(
        "Usage: probe_probs -d <model_dir> -i <audio.wav> [--candidate <text> ...] [--next-after <text>] [--language <lang>]"
    );
    std::process::exit(2);
}

fn logprob(row: &[f32], token: i32) -> (f32, f32, usize) {
    let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f64;
    for &v in row {
        denom += ((v - max_logit) as f64).exp();
    }
    let token_logit = row[token as usize];
    let lp = token_logit - max_logit - (denom.ln() as f32);
    let prob = lp.exp();
    let rank = 1 + row.iter().filter(|&&v| v > token_logit).count();
    (lp, prob, rank)
}

fn top_token(row: &[f32]) -> (usize, f32) {
    row.iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

fn build_prefix_embeds(
    ctx: &mut QwenCtx,
    tokenizer: &QwenTokenizer,
    samples: &[f32],
) -> Option<Vec<f32>> {
    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    let (mel, mel_frames) = audio::mel_spectrogram(samples)?;
    let (enc_output, enc_seq_len) =
        ctx.encoder
            .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;

    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len;
    let mut input_embeds = vec![0.0f32; total_seq * dim];

    let mut off = 0;
    for &tok in PREFIX_HEAD {
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

    for &tok in PREFIX_TAIL {
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

    for i in 0..enc_seq_len {
        input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
            .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
    }

    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
        unsafe {
            tok_embed_to_f32(
                &ctx.decoder,
                &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                tok,
                dim,
            )
        };
    }

    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
            unsafe {
                tok_embed_to_f32(
                    &ctx.decoder,
                    &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                        ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                    tok,
                    dim,
                )
            };
        }
    }

    Some(input_embeds)
}

fn score_candidate(
    ctx: &mut QwenCtx,
    tokenizer: &QwenTokenizer,
    prefix_embeds: &[f32],
    candidate: &str,
) -> Option<(Vec<i32>, f32, f32)> {
    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    let out_dim = cfg.lm_head_dim();
    let mut tokens = tokenizer.encode(candidate)?;
    tokens.push(TOKEN_ENDOFTEXT);

    let prefix_len = prefix_embeds.len() / dim;
    let seq_len = prefix_len + tokens.len() - 1;
    let mut input_embeds = vec![0.0f32; seq_len * dim];
    input_embeds[..prefix_embeds.len()].copy_from_slice(prefix_embeds);
    for (i, &tok) in tokens[..tokens.len() - 1].iter().enumerate() {
        unsafe {
            tok_embed_to_f32(
                &ctx.decoder,
                &mut input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim],
                tok,
                dim,
            )
        };
    }

    ctx.kv_cache.len = 0;
    let needed_scratch = cfg.lm_head_dim() * cfg.dec_hidden;
    if ctx.dec_bufs.bf16_scratch.len() < needed_scratch {
        ctx.dec_bufs.bf16_scratch.resize(needed_scratch, 0.0);
    }
    let logits = decoder::decoder_prefill_logits(
        &ctx.decoder,
        &cfg,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        &input_embeds,
        seq_len,
    );

    let mut total_lp = 0.0f32;
    for (i, &tok) in tokens.iter().enumerate() {
        let pos = prefix_len - 1 + i;
        let row = &logits[pos * out_dim..(pos + 1) * out_dim];
        total_lp += logprob(row, tok).0;
    }
    let avg_lp = total_lp / tokens.len() as f32;
    Some((tokens, total_lp, avg_lp))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut model_dir = None;
    let mut audio_path = None;
    let mut candidates = Vec::new();
    let mut next_after = None;
    let mut language = String::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-d" => model_dir = args.next(),
            "-i" => audio_path = args.next(),
            "--candidate" => candidates.push(args.next().unwrap_or_default()),
            "--next-after" => next_after = args.next(),
            "--language" => language = args.next().unwrap_or_default(),
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    let model_dir = model_dir.unwrap_or_else(|| usage());
    let audio_path = audio_path.unwrap_or_else(|| usage());
    if candidates.is_empty() && next_after.is_none() {
        usage();
    }

    qwen_asr::kernels::set_verbose(0);
    let tokenizer = QwenTokenizer::load(&format!("{}/vocab.json", model_dir)).expect("tokenizer");
    let mut ctx = QwenCtx::load(&model_dir).expect("model");
    if !language.is_empty() {
        ctx.set_force_language(&language).expect("valid language");
    }
    let samples = audio::load_wav(&audio_path).expect("audio");
    let prefix_embeds = build_prefix_embeds(&mut ctx, &tokenizer, &samples).expect("prefix");

    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    if !candidates.is_empty() {
        let out_dim = cfg.lm_head_dim();
        ctx.kv_cache.len = 0;
        let needed_scratch = cfg.lm_head_dim() * cfg.dec_hidden;
        if ctx.dec_bufs.bf16_scratch.len() < needed_scratch {
            ctx.dec_bufs.bf16_scratch.resize(needed_scratch, 0.0);
        }
        let first_logits = decoder::decoder_prefill_logits(
            &ctx.decoder,
            &cfg,
            &mut ctx.kv_cache,
            &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            &prefix_embeds,
            prefix_embeds.len() / dim,
        );
        let first_row = &first_logits
            [(prefix_embeds.len() / dim - 1) * out_dim..(prefix_embeds.len() / dim) * out_dim];
        let (top_id, top_logit) = top_token(first_row);
        println!(
            "{{\"kind\":\"first_top\",\"token_id\":{},\"piece\":\"{}\",\"logit\":{:.6}}}",
            top_id,
            String::from_utf8_lossy(tokenizer.decode_bytes(top_id as i32)),
            top_logit
        );

        for ch in ["彌", "尼", "離", "地"] {
            let toks = tokenizer.encode(ch).expect("char token");
            if toks.len() == 1 {
                let (lp, p, rank) = logprob(first_row, toks[0]);
                println!(
                "{{\"kind\":\"first_char\",\"text\":\"{}\",\"token_id\":{},\"logprob\":{:.6},\"prob\":{:.9},\"rank\":{}}}",
                ch, toks[0], lp, p, rank
            );
            } else {
                let (lp, p, rank) = logprob(first_row, toks[0]);
                println!(
                "{{\"kind\":\"first_char\",\"text\":\"{}\",\"tokens\":{:?},\"note\":\"multi_token\",\"first_token_logprob\":{:.6},\"first_token_prob\":{:.9},\"first_token_rank\":{}}}",
                ch, toks, lp, p, rank
            );
            }
        }
    }

    if let Some(text) = next_after {
        let text_tokens = tokenizer.encode(&text).expect("next-after tokens");
        let prefix_len = prefix_embeds.len() / dim;
        let seq_len = prefix_len + text_tokens.len();
        let mut input_embeds = vec![0.0f32; seq_len * dim];
        input_embeds[..prefix_embeds.len()].copy_from_slice(&prefix_embeds);
        for (i, &tok) in text_tokens.iter().enumerate() {
            unsafe {
                tok_embed_to_f32(
                    &ctx.decoder,
                    &mut input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim],
                    tok,
                    dim,
                )
            };
        }
        let symbols = ["？", "?", "。", "，"];
        let mut token_ids = vec![TOKEN_ENDOFTEXT as usize, TOKEN_IM_END as usize];
        for symbol in symbols {
            let tokens = tokenizer.encode(symbol).expect("symbol token");
            assert_eq!(tokens.len(), 1, "expected one token for {symbol}");
            token_ids.push(tokens[0] as usize);
        }
        ctx.kv_cache.len = 0;
        let logits = decoder::decoder_prefill_last_selected_logits(
            &ctx.decoder,
            &cfg,
            &mut ctx.kv_cache,
            &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            &input_embeds,
            seq_len,
            &token_ids,
        );
        let q_logit = logits[2].max(logits[3]);
        let stop_logit = logits[0].max(logits[1]);
        let question_vs_stop = 1.0f32 / (1.0 + (stop_logit - q_logit).exp());
        let terminal_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let terminal_denom: f32 = logits
            .iter()
            .map(|value| (*value - terminal_max).exp())
            .sum();
        let question_terminal_probability =
            ((logits[2] - terminal_max).exp() + (logits[3] - terminal_max).exp()) / terminal_denom;
        println!(
            "{{\"kind\":\"next_after\",\"eos_logit\":{:.6},\"im_end_logit\":{:.6},\"fullwidth_question_logit\":{:.6},\"ascii_question_logit\":{:.6},\"full_stop_logit\":{:.6},\"comma_logit\":{:.6},\"question_vs_stop_probability\":{:.9},\"question_terminal_probability\":{:.9}}}",
            logits[0], logits[1], logits[2], logits[3], logits[4], logits[5], question_vs_stop, question_terminal_probability
        );
    }

    for cand in candidates {
        let (tokens, total_lp, avg_lp) =
            score_candidate(&mut ctx, &tokenizer, &prefix_embeds, &cand).expect("score");
        println!(
            "{{\"kind\":\"candidate\",\"text\":\"{}\",\"tokens\":{:?},\"total_logprob\":{:.6},\"avg_logprob\":{:.6},\"n_tokens\":{}}}",
            cand,
            tokens,
            total_lp,
            avg_lp,
            tokens.len()
        );
    }
}
