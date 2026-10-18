//! Top-level engine state ([`QwenCtx`]) owning all loaded weights and runtime buffers.

use crate::config::*;
use crate::decoder::*;
use crate::encoder::EncoderBuffers;
use crate::encoder::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;
use crate::tokenizer::QwenTokenizer;

pub type TokenCallback = Box<dyn Fn(&str) + Send>;

pub(crate) struct StaticPrefixKvCache {
    pub tokens: Vec<i32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Top-level ASR engine state owning model weights, KV cache, and scratch buffers.
///
/// Create with [`QwenCtx::load`], then pass to functions in the [`crate::transcribe`] module.
///
/// # Configurable fields
///
/// | Field | Default | Description |
/// |-------|---------|-------------|
/// | `segment_sec` | 0.0 | Segment duration for long audio (0 = no splitting) |
/// | `skip_silence` | false | Drop silent spans before transcription |
/// | `token_cb` | None | Streaming callback invoked for each decoded token |
/// | `prompt` | None | Optional text prompt (set via [`QwenCtx::set_prompt`]) |
/// | `force_language` | None | Force a language (set via [`QwenCtx::set_force_language`]) |
pub struct QwenCtx {
    pub config: QwenConfig,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub _safetensors: MultiSafetensors, // kept alive for mmap'd BF16 pointers
    pub model_dir: String,

    // KV cache
    pub kv_cache: KvCache,
    pub(crate) static_prefix_kv_cache: Option<StaticPrefixKvCache>,

    // Decoder buffers
    pub dec_bufs: DecoderBuffers,

    // Encoder scratch buffers (reusable across calls)
    pub enc_bufs: EncoderBuffers,

    // RoPE cache
    pub rope_cache: RopeCache,

    // Token streaming callback
    pub token_cb: Option<TokenCallback>,

    // Segmentation settings
    pub segment_sec: f32,
    pub search_sec: f32,

    // Streaming settings
    pub stream_chunk_sec: f32,
    pub stream_rollback: i32,
    pub stream_unfixed_chunks: i32,
    pub stream_max_new_tokens: i32,
    pub past_text_conditioning: bool,
    pub skip_silence: bool,
    pub stream_adaptive_drain: bool,
    pub(crate) stream_latency_trace: bool,

    // Optional prompt/language
    pub prompt: Option<String>,
    pub force_language: Option<String>,

    // Optional Simplified->Traditional (or other variant) output conversion.
    // None = no conversion (raw model output).
    pub chinese_variant: Option<zhconv::Variant>,
    /// Optional in-decoder Cantonese Chinese conversion adapter.
    pub adapter: Option<crate::adapter::CantoneseZhconvAdapter>,
    pub prompt_tokens: Option<Vec<i32>>,
    pub force_prompt_tokens: Option<Vec<i32>>,
    pub prompt_tokens_ready: bool,

    // Perf stats
    pub perf_total_ms: f64,
    pub perf_text_tokens: i32,
    pub perf_audio_ms: f64,
    /// Mel spectrogram + encoder forward pass time combined.
    pub perf_encode_ms: f64,
    pub perf_decode_ms: f64,
    /// Phase-split timings (P0). Accumulated alongside perf_encode_ms/perf_decode_ms.
    pub perf_mel_ms: f64,
    pub perf_encoder_ms: f64,
    pub perf_prefill_ms: f64,
    /// Time from the start of transcription to the first complete UTF-8 text piece.
    pub perf_first_visible_token_ms: f64,
    pub perf_decode_loop_ms: f64,
}

impl QwenCtx {
    /// Load a Qwen3-ASR model from `model_dir`.
    ///
    /// The directory must contain `model*.safetensors` and `vocab.json`.
    /// Returns `None` if any required file is missing or malformed.
    ///
    /// ```rust,no_run
    /// use qwen_asr::context::QwenCtx;
    /// let ctx = QwenCtx::load("qwen3-asr-0.6b").expect("failed to load");
    /// ```
    pub fn load(model_dir: &str) -> Option<Self> {
        if kernels::verbose() >= 1 {
            eprintln!("Loading model from {}", model_dir);
        }

        let ms = MultiSafetensors::open(model_dir)?;

        // Detect model variant from tensor shapes
        let info = crate::config::DetectInfo {
            has_enc_layer_18: ms
                .has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight")
                || ms.has_tensor(
                    "thinker.audio_tower.layers.18.self_attn.q_proj.weight_kleidiai_i8mm_packed",
                ),
            lm_head_shape: ms
                .find("thinker.lm_head.weight")
                .map(|(_, t)| t.shape.as_slice()),
            embed_tokens_shape: ms
                .find("thinker.model.embed_tokens.weight")
                .map(|(_, t)| t.shape.as_slice()),
            gate_proj_shape: ms
                .find("thinker.model.layers.0.mlp.gate_proj.weight")
                .map(|(_, t)| t.shape.as_slice()),
        };
        let cfg = QwenConfig::detect(&info);

        if kernels::verbose() >= 1 {
            let variant = if cfg.dec_hidden >= 2048 {
                "1.7B"
            } else {
                "0.6B"
            };
            let model_type = if cfg.is_aligner() {
                "ForcedAligner"
            } else {
                "ASR"
            };
            eprintln!("Detected: Qwen3-{}-{}", model_type, variant);
            if cfg.is_aligner() {
                eprintln!(
                    "  classify_num={}, timestamp_segment_time={:.0}ms",
                    cfg.classify_num, cfg.timestamp_segment_time
                );
                eprintln!(
                    "  encoder: {}d {}L, decoder: {}d {}L",
                    cfg.enc_d_model, cfg.enc_layers, cfg.dec_hidden, cfg.dec_layers
                );
            }
        }

        // Load encoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading encoder weights...");
        }
        let encoder = Encoder::load(&ms, &cfg)?;

        // Load decoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading decoder weights...");
        }
        let decoder = Decoder::load(&ms, &cfg)?;

        let kv_cache = KvCache::new(cfg.dec_layers, 2048, cfg.dec_kv_heads, cfg.dec_head_dim);
        let dec_bufs = DecoderBuffers::new(&cfg, decoder.prefill_scratch_len(&cfg));

        let adapter = crate::adapter::CantoneseZhconvAdapter::load(model_dir);

        if kernels::verbose() >= 1 {
            eprintln!("Model loaded.");
        }

        Some(QwenCtx {
            config: cfg,
            encoder,
            decoder,
            _safetensors: ms,
            model_dir: model_dir.to_string(),
            kv_cache,
            static_prefix_kv_cache: None,
            dec_bufs,
            enc_bufs: EncoderBuffers::new(),
            rope_cache: RopeCache::new(),
            token_cb: None,
            segment_sec: 0.0,
            search_sec: 3.0,
            stream_chunk_sec: 0.48,
            stream_rollback: 1,
            stream_unfixed_chunks: 1,
            stream_max_new_tokens: 16,
            past_text_conditioning: true,
            skip_silence: true,
            stream_adaptive_drain: false,
            stream_latency_trace: false,
            prompt: None,
            force_language: None,
            chinese_variant: None,
            adapter,
            prompt_tokens: None,
            force_prompt_tokens: None,
            prompt_tokens_ready: false,
            perf_total_ms: 0.0,
            perf_text_tokens: 0,
            perf_audio_ms: 0.0,
            perf_encode_ms: 0.0,
            perf_decode_ms: 0.0,
            perf_mel_ms: 0.0,
            perf_encoder_ms: 0.0,
            perf_prefill_ms: 0.0,
            perf_first_visible_token_ms: 0.0,
            perf_decode_loop_ms: 0.0,
        })
    }

    /// Set an optional text prompt to guide transcription. Pass an empty string to clear.
    #[allow(clippy::result_unit_err)]
    pub fn set_prompt(&mut self, prompt: &str) -> Result<(), ()> {
        let next_prompt = if prompt.is_empty() {
            None
        } else {
            Some(prompt.to_string())
        };
        if self.prompt != next_prompt {
            self.prompt = next_prompt;
            self.prompt_tokens_ready = false;
            self.static_prefix_kv_cache = None;
        }
        Ok(())
    }

    /// Force a specific language (e.g. `"English"`, `"Chinese"`). Pass an empty
    /// string for auto-detection. Returns `Err(())` if the language is not recognized.
    #[allow(clippy::result_unit_err)]
    pub fn set_force_language(&mut self, language: &str) -> Result<(), ()> {
        if language.trim().is_empty() || language.trim().eq_ignore_ascii_case("auto") {
            self.force_language = None;
            self.prompt_tokens_ready = false;
            return Ok(());
        }

        match normalize_language(language) {
            Some(normalized) => {
                self.force_language = Some(normalized);
                self.prompt_tokens_ready = false;
                Ok(())
            }
            None => Err(()),
        }
    }

    /// Enable conversion of decoded text to a Chinese variant.
    ///
    /// Qwen3-ASR emits Simplified Chinese; set e.g. `"zh-Hant"`, `"zh-TW"`, or
    /// `"zh-HK"` to convert the output to Traditional Chinese. Pass an empty
    /// string, `"off"`, or `"none"` to disable conversion. Returns `Err(())` if
    /// the variant string is not recognized.
    #[allow(clippy::result_unit_err)]
    pub fn set_chinese_variant(&mut self, variant: &str) -> Result<(), ()> {
        let v = variant.trim();
        if v.is_empty() || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("none") {
            self.chinese_variant = None;
            return Ok(());
        }
        match v.parse::<zhconv::Variant>() {
            Ok(parsed) => {
                self.chinese_variant = Some(parsed);
                Ok(())
            }
            Err(_) => Err(()),
        }
    }

    /// Apply the configured Chinese conversion to `text`. Returns `text`
    /// unchanged when no variant is set.
    pub fn convert_chinese(&self, text: &str) -> String {
        match self.chinese_variant {
            Some(v) => zhconv::zhconv(text, v),
            None => text.to_string(),
        }
    }

    pub fn prepare_prompt_tokens(&mut self, tokenizer: &QwenTokenizer) -> bool {
        if self.prompt_tokens_ready {
            return true;
        }

        self.prompt_tokens = None;
        self.force_prompt_tokens = None;

        if let Some(ref prompt) = self.prompt {
            match tokenizer.encode(prompt) {
                Some(tokens) => self.prompt_tokens = Some(tokens),
                None => {
                    eprintln!("qwen: failed to encode --prompt text");
                    return false;
                }
            }
        }

        if let Some(ref lang) = self.force_language {
            let force_text = format!("language {}", lang);
            match tokenizer.encode(&force_text) {
                Some(mut lang_tokens) => {
                    lang_tokens.push(TOKEN_ASR_TEXT);
                    self.force_prompt_tokens = Some(lang_tokens);
                }
                None => {
                    eprintln!("qwen: failed to encode --language text");
                    return false;
                }
            }
        }

        self.prompt_tokens_ready = true;
        true
    }

    pub fn reset_perf(&mut self) {
        self.perf_total_ms = 0.0;
        self.perf_text_tokens = 0;
        self.perf_audio_ms = 0.0;
        self.perf_encode_ms = 0.0;
        self.perf_decode_ms = 0.0;
        self.perf_mel_ms = 0.0;
        self.perf_encoder_ms = 0.0;
        self.perf_prefill_ms = 0.0;
        self.perf_first_visible_token_ms = 0.0;
        self.perf_decode_loop_ms = 0.0;
    }

    pub fn preload_encoder(&self) {
        self._safetensors.preload_encoder();
    }

    pub fn preload_decoder(&self) {
        self._safetensors.preload_decoder();
    }
}
