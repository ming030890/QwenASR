//! Pure-text (no-audio) LM rescoring to resolve acoustic homophone confusions.
//!
//! The ASR decoder, conditioned on audio, sometimes prefers an acoustically
//! plausible but linguistically nonsensical token (e.g. `月` over `粵`). The
//! *same* decoder weights, run as a plain causal LM with **no audio prefix**,
//! strongly prefer the correct token. This module exploits that: at a decode
//! step where the top-2 acoustic candidates are close ("a confusion"), we do a
//! single-step (K=1) width-2 fork, generate the acoustic next token for each
//! branch, and let the text-only LM pick the branch with the higher joint
//! probability.
//!
//! Design constraints:
//!   * **Zero extra work when there is no confusion** — gated on the acoustic
//!     top-1/top-2 logit gap.
//!   * **Memory ≤ a normal ASR run** — reuses the shared `DecoderBuffers` and
//!     `RopeCache`; the only added state is a *text-only* KV cache (no audio
//!     positions, so far smaller than the ASR cache) plus one reusable
//!     full-vocab logit buffer.
//!   * **Android-first** — the text-LM projection uses the INT8 lm_head path on
//!     aarch64, identical to the ASR argmax.

use crate::config::{QwenConfig, TOKEN_ENDOFTEXT};
use crate::decoder::{self, tok_embed_to_f32, Decoder, DecoderBuffers, KvCache, RopeCache};

/// True for ordinary text tokens; special/control tokens have ids >= `TOKEN_ENDOFTEXT`.
#[inline]
pub fn is_text_token(t: i32) -> bool {
    (0..TOKEN_ENDOFTEXT).contains(&t)
}

/// Numerically stable log-sum-exp over a logit slice.
fn log_sum_exp(logits: &[f32]) -> f32 {
    let mut max_val = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_val {
            max_val = v;
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let mut sum = 0.0f32;
    for &v in logits {
        sum += (v - max_val).exp();
    }
    max_val + sum.ln()
}

pub struct LmRescorer {
    /// Whether rescoring is active. When false, the decode path is untouched.
    pub enabled: bool,
    /// Fork only when `top1_logit - top2_logit < gap_threshold` (acoustic space).
    pub gap_threshold: f32,
    /// Text-only KV cache (no audio); independent from the ASR KV cache.
    kv: KvCache,
    /// Conditioning scaffold prefilled before the transcript history: the same
    /// no-audio Qwen prompt prefix the ASR decoder sees (system prompt, user turn,
    /// `language <lang>`, `<asr_text>`). Without this the text-LM prior is
    /// unconditioned; with it the prior matches the proven offline scoring setup.
    prefix: Vec<i32>,
    /// Number of (prefix ++ history) tokens currently reflected in `kv`.
    synced_len: usize,
    /// Reusable full-vocab logit buffer (len == lm_head_dim) — no per-call alloc.
    logits: Vec<f32>,
    /// Reusable embedding scratch (len == dec_hidden).
    emb: Vec<f32>,
    dim: usize,
    vocab: usize,
}

impl LmRescorer {
    pub fn new(cfg: &QwenConfig) -> Self {
        let enabled = std::env::var("QWEN_LM_RESCORE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        let gap_threshold = std::env::var("QWEN_LM_RESCORE_GAP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(2.5);
        // Text history is short relative to audio; 128 covers a long sentence and
        // KvCache::grow() handles overflow if needed. This cache holds only text
        // positions (no audio), so it is a small fraction of the ASR KV cache.
        let kv = KvCache::new(cfg.dec_layers, 128, cfg.dec_kv_heads, cfg.dec_head_dim);
        LmRescorer {
            enabled,
            gap_threshold,
            kv,
            prefix: Vec::new(),
            synced_len: 0,
            logits: vec![0.0; cfg.lm_head_dim()],
            emb: vec![0.0; cfg.dec_hidden],
            dim: cfg.dec_hidden,
            vocab: cfg.lm_head_dim(),
        }
    }

    /// Reset between utterances/segments.
    pub fn reset(&mut self) {
        self.kv.len = 0;
        self.synced_len = 0;
    }

    /// Set the no-audio conditioning scaffold (Qwen prompt prefix incl. the
    /// `language <lang>` tokens) that precedes the transcript history. Call once
    /// per segment after the prompt tokens are prepared. Resets the cache.
    pub fn set_prefix(&mut self, prefix: &[i32]) {
        self.prefix.clear();
        self.prefix.extend_from_slice(prefix);
        self.kv.len = 0;
        self.synced_len = 0;
    }

    /// Extract the top-2 tokens (id + logit) of the lm_head projection of the
    /// hidden state currently in `bufs.x`. Used for confusion detection.
    pub fn top2_from_hidden(
        &mut self,
        decoder: &Decoder,
        cfg: &QwenConfig,
        bufs: &DecoderBuffers,
    ) -> (i32, f32, i32, f32) {
        decoder::decoder_project_logits(decoder, cfg, &bufs.x[..self.dim], &mut self.logits);
        let (mut i0, mut v0, mut i1, mut v1) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
        for (i, &v) in self.logits[..self.vocab].iter().enumerate() {
            if v > v0 {
                i1 = i0;
                v1 = v0;
                i0 = i;
                v0 = v;
            } else if v > v1 {
                i1 = i;
                v1 = v;
            }
        }
        (i0 as i32, v0, i1 as i32, v1)
    }

    /// Advance the text-only KV cache so it reflects `prefix ++ history`. On
    /// return, `bufs.x` holds the normalized hidden state that predicts the token
    /// *following* the last history token. Returns false if there is nothing to
    /// condition on.
    fn sync(
        &mut self,
        decoder: &Decoder,
        cfg: &QwenConfig,
        rope: &mut RopeCache,
        bufs: &mut DecoderBuffers,
        history: &[i32],
    ) -> bool {
        let plen = self.prefix.len();
        let total = plen + history.len();
        if total == 0 {
            return false;
        }
        // Append-only invariant; rebuild if the caller's history shrank.
        if self.synced_len > total {
            self.kv.len = 0;
            self.synced_len = 0;
        }
        while self.synced_len < total {
            let tok = if self.synced_len < plen {
                self.prefix[self.synced_len]
            } else {
                history[self.synced_len - plen]
            };
            unsafe { tok_embed_to_f32(decoder, &mut self.emb, tok, self.dim) };
            decoder::decoder_forward_hidden(decoder, cfg, &mut self.kv, rope, bufs, &self.emb);
            self.synced_len += 1;
        }
        true
    }

    /// log P(`target` | text history + `prefix`) under the text-only LM, where the
    /// LM cache is currently synced to the history. Feeds `prefix` speculatively,
    /// scores `target`, then rolls the cache back. Returns 0 for non-text
    /// continuations (don't bias the comparison on control tokens).
    fn logprob_after(
        &mut self,
        decoder: &Decoder,
        cfg: &QwenConfig,
        rope: &mut RopeCache,
        bufs: &mut DecoderBuffers,
        prefix: i32,
        target: i32,
    ) -> f32 {
        if !is_text_token(target) {
            return 0.0;
        }
        let base = self.kv.len;
        unsafe { tok_embed_to_f32(decoder, &mut self.emb, prefix, self.dim) };
        decoder::decoder_forward_hidden(decoder, cfg, &mut self.kv, rope, bufs, &self.emb);
        decoder::decoder_project_logits(decoder, cfg, &bufs.x[..self.dim], &mut self.logits);
        let logz = log_sum_exp(&self.logits[..self.vocab]);
        let lp = self.logits[target as usize] - logz;
        self.kv.len = base; // discard speculative token
        lp
    }

    /// Decide between two close acoustic candidates `c0` (top-1) and `c1` (top-2)
    /// using K=1 fork + text-LM scoring. Returns the chosen next token.
    ///
    /// `audio_kv` must be positioned at the fork point (i.e. `len` == the position
    /// the chosen token will occupy). It is left unchanged on return so the normal
    /// decode loop can commit the winner.
    #[allow(clippy::too_many_arguments)]
    pub fn pick(
        &mut self,
        decoder: &Decoder,
        cfg: &QwenConfig,
        audio_kv: &mut KvCache,
        rope: &mut RopeCache,
        bufs: &mut DecoderBuffers,
        history: &[i32],
        c0: i32,
        c1: i32,
    ) -> i32 {
        // 1. Text-LM prior for the candidate token itself: P(c | history).
        if !self.sync(decoder, cfg, rope, bufs, history) {
            return c0;
        }
        decoder::decoder_project_logits(decoder, cfg, &bufs.x[..self.dim], &mut self.logits);
        let logz_h = log_sum_exp(&self.logits[..self.vocab]);
        let lp_c0 = self.logits[c0 as usize] - logz_h;
        let lp_c1 = self.logits[c1 as usize] - logz_h;

        // 2. Acoustic 1-step lookahead: the next token the audio model would emit
        //    after each candidate. This grounds the right-context in the audio.
        let fork = audio_kv.len;
        unsafe { tok_embed_to_f32(decoder, &mut self.emb, c0, self.dim) };
        let d0 = decoder::decoder_forward(decoder, cfg, audio_kv, rope, bufs, &self.emb);
        audio_kv.len = fork;
        unsafe { tok_embed_to_f32(decoder, &mut self.emb, c1, self.dim) };
        let d1 = decoder::decoder_forward(decoder, cfg, audio_kv, rope, bufs, &self.emb);
        audio_kv.len = fork;

        // 3. Text-LM likelihood of each acoustic continuation: P(d | history, c).
        let lp_d0 = self.logprob_after(decoder, cfg, rope, bufs, c0, d0);
        let lp_d1 = self.logprob_after(decoder, cfg, rope, bufs, c1, d1);

        let score0 = lp_c0 + lp_d0;
        let score1 = lp_c1 + lp_d1;
        if score1 > score0 {
            c1
        } else {
            c0
        }
    }
}
