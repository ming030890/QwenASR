//! Qwen3 LLM decoder with GQA, KV cache, and generation.

use crate::config::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;

/// INT8 weight slice — either a zero-copy pointer into the mmap region or an
/// owned heap Vec (used when weights are quantized on-device from BF16).
///
/// The `Deref<Target=[i8]>` impl means `&layer.wq_int8` coerces to `&[i8]`
/// at every kernel call site without any change to those call sites.
pub struct Int8Buf {
    ptr: *const i8,
    len: usize,
    _owned: Vec<i8>, // non-empty only for the on-device BF16→INT8 quantization path
}

unsafe impl Send for Int8Buf {}
unsafe impl Sync for Int8Buf {}

impl Int8Buf {
    fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
            _owned: Vec::new(),
        }
    }

    pub(crate) fn from_mmap(ptr: *const i8, len: usize) -> Self {
        Int8Buf {
            ptr,
            len,
            _owned: Vec::new(),
        }
    }
    pub(crate) fn from_owned(v: Vec<i8>) -> Self {
        let ptr = v.as_ptr();
        let len = v.len();
        Int8Buf {
            ptr,
            len,
            _owned: v,
        }
    }
}

impl std::ops::Deref for Int8Buf {
    type Target = [i8];
    fn deref(&self) -> &[i8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

pub(crate) struct PackedInt8Buf {
    ptr: *const i8,
    len: usize,
}

impl PackedInt8Buf {
    #[cfg(feature = "kleidiai")]
    pub(crate) fn from_mmap(ptr: *const i8, len: usize) -> Self {
        Self { ptr, len }
    }
}

impl std::ops::Deref for PackedInt8Buf {
    type Target = [i8];

    fn deref(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

pub struct DecLayer {
    pub wq_weight_bf16: *const u16,
    pub wk_weight_bf16: *const u16,
    pub wv_weight_bf16: *const u16,
    pub wo_weight_bf16: *const u16,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub gate_weight_bf16: *const u16,
    pub up_weight_bf16: *const u16,
    pub down_weight_bf16: *const u16,
    pub gate_up_fused_bf16: Vec<u16>, // owned, interleaved
    /// INT8 quantized attention weights + per-row scales
    pub wq_int8: Int8Buf,
    pub wq_int8_scales: Vec<f32>,
    pub wk_int8: Int8Buf,
    pub wk_int8_scales: Vec<f32>,
    pub wv_int8: Int8Buf,
    pub wv_int8_scales: Vec<f32>,
    pub wo_int8: Int8Buf,
    pub wo_int8_scales: Vec<f32>,
    /// INT8 quantized FFN weights + per-row scales
    pub gate_up_int8: Int8Buf,
    pub gate_up_int8_scales: Vec<f32>,
    pub down_int8: Int8Buf,
    pub down_int8_scales: Vec<f32>,
    #[cfg(feature = "kleidiai")]
    wq_kleidiai: Option<PackedInt8Buf>,
    #[cfg(feature = "kleidiai")]
    wk_kleidiai: Option<PackedInt8Buf>,
    #[cfg(feature = "kleidiai")]
    wv_kleidiai: Option<PackedInt8Buf>,
    #[cfg(feature = "kleidiai")]
    wo_kleidiai: Option<PackedInt8Buf>,
    #[cfg(feature = "kleidiai")]
    gate_up_kleidiai: Option<PackedInt8Buf>,
    #[cfg(feature = "kleidiai")]
    down_kleidiai: Option<PackedInt8Buf>,
}

unsafe impl Send for DecLayer {}
unsafe impl Sync for DecLayer {}

#[cfg(feature = "kleidiai")]
fn shared_qkv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("QWEN_EXPERIMENT_QKV_REUSE").as_deref() == Ok("1"))
}

#[allow(clippy::too_many_arguments)]
fn try_shared_qkv(q: &mut [f32], k: &mut [f32], v: &mut [f32], x: &[f32],
    wq: Option<&[i8]>, wk: Option<&[i8]>, wv: Option<&[i8]>,
    seq_len: usize, dim: usize, q_dim: usize, kv_dim: usize, use_kleidiai: bool) -> bool {
    #[cfg(feature = "kleidiai")]
    if use_kleidiai && shared_qkv_enabled() {
        kernels::linear_int8_kleidiai_shared(&mut [
            (q, wq.expect("packed Q"), q_dim), (k, wk.expect("packed K"), kv_dim),
            (v, wv.expect("packed V"), kv_dim)], x, seq_len, dim);
        return true;
    }
    let _ = (q, k, v, x, wq, wk, wv, seq_len, dim, q_dim, kv_dim, use_kleidiai);
    false
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecoderBackend {
    RowMajorInt8,
    #[cfg(feature = "kleidiai")]
    KleidiAiPacked,
}

pub struct Decoder {
    pub tok_embeddings_bf16: *const u16,
    pub tok_embeddings_int8: Option<Int8Buf>,
    pub tok_embeddings_int8_scales: *const f32,
    pub tok_embeddings_int8_scales_len: usize,
    pub tok_embeddings_int8_mins: *const f32,
    pub tok_embeddings_int8_mins_len: usize,
    pub tok_embeddings_int8_group_size: usize,
    pub layers: Vec<DecLayer>,
    pub norm: Vec<f32>,
    /// Separate lm_head for forced aligner (None = tied weights with tok_embeddings)
    pub lm_head_bf16: Option<*const u16>,
    /// INT8 quantized lm_head weights for fast argmax
    pub lm_head_int8: Option<Int8Buf>,
    pub lm_head_int8_scales: Option<Vec<f32>>,
    #[cfg_attr(not(feature = "kleidiai"), allow(dead_code))]
    backend: DecoderBackend,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

fn load_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let result = ms.get_f32(name);
    if result.is_none() {
        eprintln!("decoder: weight not found: {}", name);
    }
    result
}

fn load_bf16_direct(ms: &MultiSafetensors, name: &str) -> Option<*const u16> {
    let result = ms.get_bf16_direct(name);
    if result.is_none() {
        eprintln!("decoder: weight not found: {}", name);
    }
    result
}

fn load_i8_mmap(ms: &MultiSafetensors, name: &str) -> Option<Int8Buf> {
    match ms.get_i8_ptr(name) {
        Some((ptr, len)) => Some(Int8Buf::from_mmap(ptr, len)),
        None => {
            eprintln!("decoder: I8 weight not found: {}", name);
            None
        }
    }
}

#[cfg(feature = "kleidiai")]
fn load_kleidiai_weight(
    ms: &MultiSafetensors,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Option<PackedInt8Buf> {
    let packed_name = format!("{name}_kleidiai_i8mm_packed");
    let (ptr, len) = ms.get_i8_ptr(&packed_name)?;
    let expected = kernels::kleidiai_packed_rhs_size(in_dim, out_dim);
    if len != expected {
        eprintln!(
            "decoder: invalid packed projection {packed_name}: got {len} bytes, expected {expected}"
        );
        return None;
    }
    Some(PackedInt8Buf::from_mmap(ptr, len))
}

#[cfg(feature = "kleidiai")]
fn packed_decoder_tensor_counts(ms: &MultiSafetensors, cfg: &QwenConfig) -> (usize, usize) {
    const SUFFIXES: [&str; 6] = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_up_fused.weight",
        "mlp.down_proj.weight",
    ];
    let expected = cfg.dec_layers * SUFFIXES.len();
    let found = (0..cfg.dec_layers)
        .flat_map(|layer| {
            SUFFIXES.into_iter().map(move |suffix| {
                format!("thinker.model.layers.{layer}.{suffix}_kleidiai_i8mm_packed")
            })
        })
        .filter(|name| ms.get_i8_ptr(name).is_some())
        .count();
    (found, expected)
}

fn dequantize_int8_to_f32_scratch<'a>(
    int8: &[i8],
    scales: &[f32],
    out_dim: usize,
    in_dim: usize,
    scratch: &'a mut [f32],
) -> &'a [f32] {
    let n = out_dim * in_dim;
    debug_assert!(scratch.len() >= n);
    let out = &mut scratch[..n];
    for (row, &s) in scales.iter().enumerate().take(out_dim) {
        let row_off = row * in_dim;
        for col in 0..in_dim {
            out[row_off + col] = (int8[row_off + col] as f32) * s;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn linear_nobias_int8_f32_scratch(
    y: &mut [f32],
    x: &[f32],
    int8: &[i8],
    scales: &[f32],
    packed: Option<&[i8]>,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    scratch: &mut [f32],
    tile_rows: usize,
    use_kleidiai: bool,
) {
    #[cfg(feature = "kleidiai")]
    if use_kleidiai {
        kernels::linear_nobias_int8_kleidiai(
            y,
            x,
            packed.expect("validated KleidiAI projection missing"),
            seq_len,
            in_dim,
            out_dim,
        );
        return;
    }
    #[cfg(not(feature = "kleidiai"))]
    let _ = (packed, use_kleidiai);

    let tile_rows = tile_rows.max(1).min(out_dim);
    for out_start in (0..out_dim).step_by(tile_rows) {
        let rows = tile_rows.min(out_dim - out_start);
        let weight_start = out_start * in_dim;
        let weight_end = weight_start + rows * in_dim;
        let weights = dequantize_int8_to_f32_scratch(
            &int8[weight_start..weight_end],
            &scales[out_start..out_start + rows],
            rows,
            in_dim,
            scratch,
        );
        kernels::linear_nobias_output_tile(
            y, x, weights, seq_len, in_dim, out_dim, out_start, rows,
        );
    }
}

impl Decoder {
    fn uses_kleidiai(&self) -> bool {
        #[cfg(feature = "kleidiai")]
        {
            self.backend == DecoderBackend::KleidiAiPacked
        }
        #[cfg(not(feature = "kleidiai"))]
        {
            false
        }
    }

    /// F32 conversion scratch required by the active decoder representation.
    /// Prequantized INT8 prefill only materializes one output-row tile at a
    /// time; BF16 and aligner models retain the legacy full-matrix capacity.
    pub fn prefill_scratch_len(&self, cfg: &QwenConfig) -> usize {
        if self.uses_kleidiai() {
            return 0;
        }
        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;
        let full = (2 * intermediate * dim).max(q_dim * dim).max(kv_dim * dim);
        let prequantized = self.layers.iter().all(|layer| {
            layer.wq_weight_bf16.is_null()
                && layer.wk_weight_bf16.is_null()
                && layer.wv_weight_bf16.is_null()
                && layer.wo_weight_bf16.is_null()
                && layer.gate_up_fused_bf16.is_empty()
                && layer.down_weight_bf16.is_null()
        });
        if !prequantized || cfg.is_aligner() {
            return full;
        }

        const TILE_ROWS: usize = 256;
        [
            TILE_ROWS.min(q_dim) * dim,
            TILE_ROWS.min(kv_dim) * dim,
            TILE_ROWS.min(dim) * q_dim,
            TILE_ROWS.min(2 * intermediate) * dim,
            TILE_ROWS.min(dim) * intermediate,
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }

    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig) -> Option<Self> {
        use crate::safetensors::Dtype;

        let tok_embeddings_bf16_opt = ms.get_bf16_direct("thinker.model.embed_tokens.weight");
        let tok_embeddings_int8_raw = ms.get_i8_ptr("thinker.model.embed_tokens.weight");
        let expected_embedding_len = cfg.vocab_size.checked_mul(cfg.dec_hidden)?;
        if let Some((_, len)) = tok_embeddings_int8_raw {
            if len != expected_embedding_len {
                eprintln!(
                    "decoder: invalid INT8 token embedding length: got {}, expected {}",
                    len, expected_embedding_len
                );
                return None;
            }
        }
        let tok_embeddings_int8 =
            tok_embeddings_int8_raw.map(|(ptr, len)| Int8Buf::from_mmap(ptr, len));
        let (tok_embeddings_int8_scales, tok_embeddings_int8_scales_len) = ms
            .get_f32_ptr("thinker.model.embed_tokens.weight_int8_group_scale")
            .unwrap_or((std::ptr::null(), 0));
        let (tok_embeddings_int8_mins, tok_embeddings_int8_mins_len) = ms
            .get_f32_ptr("thinker.model.embed_tokens.weight_int8_group_min")
            .unwrap_or((std::ptr::null(), 0));
        let tok_embeddings_int8_group_size = if tok_embeddings_int8.is_some()
            && tok_embeddings_int8_scales_len > 0
            && tok_embeddings_int8_scales_len % cfg.vocab_size == 0
        {
            let groups_per_row = tok_embeddings_int8_scales_len / cfg.vocab_size;
            if cfg.dec_hidden.is_multiple_of(groups_per_row) {
                cfg.dec_hidden / groups_per_row
            } else {
                0
            }
        } else {
            0
        };
        if tok_embeddings_int8.is_some() {
            if tok_embeddings_int8_group_size == 0 {
                eprintln!("decoder: invalid groupwise INT8 token embedding scales");
                return None;
            }
            if tok_embeddings_int8_mins_len != 0
                && tok_embeddings_int8_mins_len != tok_embeddings_int8_scales_len
            {
                eprintln!(
                    "decoder: affine INT8 token embedding min/scale length mismatch: {} vs {}",
                    tok_embeddings_int8_mins_len, tok_embeddings_int8_scales_len
                );
                return None;
            }
        }

        // Detect whether the model file stores decoder weights as pre-quantized INT8.
        // The Python host script prepare_int8_model.py creates a model where every
        // decoder projection weight has dtype I8 instead of BF16.
        let use_int8_weights = matches!(
            ms.get_dtype("thinker.model.layers.0.self_attn.q_proj.weight"),
            Some(Dtype::I8)
        );

        #[cfg(feature = "kleidiai")]
        let decoder_backend = {
            let (packed_found, packed_expected) = packed_decoder_tensor_counts(ms, cfg);
            if packed_found != 0 && packed_found != packed_expected {
                eprintln!(
                    "decoder: incomplete KleidiAI packed decoder: found {packed_found} of {packed_expected} projections"
                );
                return None;
            }
            #[cfg(target_arch = "aarch64")]
            if packed_found == packed_expected && !std::arch::is_aarch64_feature_detected!("i8mm") {
                eprintln!(
                    "decoder: KleidiAI I8MM-packed model requires an Arm CPU with I8MM support"
                );
                return None;
            }
            if cfg!(target_arch = "aarch64") && packed_found == packed_expected {
                DecoderBackend::KleidiAiPacked
            } else {
                DecoderBackend::RowMajorInt8
            }
        };
        #[cfg(not(feature = "kleidiai"))]
        let decoder_backend = DecoderBackend::RowMajorInt8;
        #[cfg(feature = "kleidiai")]
        let use_packed_weights = decoder_backend == DecoderBackend::KleidiAiPacked;
        #[cfg(not(feature = "kleidiai"))]
        let use_packed_weights = false;

        if use_int8_weights {
            #[cfg(not(target_arch = "aarch64"))]
            {
                eprintln!(
                    "decoder: INT8 pre-quantized model is not supported on this architecture"
                );
                return None;
            }
        }

        let mut layers = Vec::new();
        for i in 0..cfg.dec_layers {
            let lp = format!("thinker.model.layers.{}", i);

            let q_dim = cfg.dec_heads * cfg.dec_head_dim;
            let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
            let hidden = cfg.dec_hidden;
            let inter = cfg.dec_intermediate;

            let q_norm = load_f32(ms, &format!("{}.self_attn.q_norm.weight", lp))?;
            let k_norm = load_f32(ms, &format!("{}.self_attn.k_norm.weight", lp))?;
            let input_norm = load_f32(ms, &format!("{}.input_layernorm.weight", lp))?;
            let post_attn_norm = load_f32(ms, &format!("{}.post_attention_layernorm.weight", lp))?;

            if use_packed_weights || use_int8_weights {
                // INT8 model path: decoder projection weights stored as I8 + F32 scales.
                // Zero-copy: raw pointers point directly into the mmap region — no heap copies.
                // gate_up_fused is pre-interleaved by prepare_int8_model.py, so no runtime fusion.
                let (
                    wq_int8,
                    wq_int8_scales,
                    wk_int8,
                    wk_int8_scales,
                    wv_int8,
                    wv_int8_scales,
                    wo_int8,
                    wo_int8_scales,
                    gate_up_int8,
                    gate_up_int8_scales,
                    down_int8,
                    down_int8_scales,
                ) = if use_packed_weights {
                    (
                        Int8Buf::empty(),
                        Vec::new(),
                        Int8Buf::empty(),
                        Vec::new(),
                        Int8Buf::empty(),
                        Vec::new(),
                        Int8Buf::empty(),
                        Vec::new(),
                        Int8Buf::empty(),
                        Vec::new(),
                        Int8Buf::empty(),
                        Vec::new(),
                    )
                } else {
                    (
                        load_i8_mmap(ms, &format!("{}.self_attn.q_proj.weight", lp))?,
                        load_f32(ms, &format!("{}.self_attn.q_proj.weight_int8_scale", lp))?,
                        load_i8_mmap(ms, &format!("{}.self_attn.k_proj.weight", lp))?,
                        load_f32(ms, &format!("{}.self_attn.k_proj.weight_int8_scale", lp))?,
                        load_i8_mmap(ms, &format!("{}.self_attn.v_proj.weight", lp))?,
                        load_f32(ms, &format!("{}.self_attn.v_proj.weight_int8_scale", lp))?,
                        load_i8_mmap(ms, &format!("{}.self_attn.o_proj.weight", lp))?,
                        load_f32(ms, &format!("{}.self_attn.o_proj.weight_int8_scale", lp))?,
                        load_i8_mmap(ms, &format!("{}.mlp.gate_up_fused.weight", lp))?,
                        load_f32(ms, &format!("{}.mlp.gate_up_fused.weight_int8_scale", lp))?,
                        load_i8_mmap(ms, &format!("{}.mlp.down_proj.weight", lp))?,
                        load_f32(ms, &format!("{}.mlp.down_proj.weight_int8_scale", lp))?,
                    )
                };

                #[cfg(feature = "kleidiai")]
                let (
                    wq_kleidiai,
                    wk_kleidiai,
                    wv_kleidiai,
                    wo_kleidiai,
                    gate_up_kleidiai,
                    down_kleidiai,
                ) = if decoder_backend == DecoderBackend::KleidiAiPacked {
                    (
                        Some(load_kleidiai_weight(
                            ms,
                            &format!("{}.self_attn.q_proj.weight", lp),
                            hidden,
                            q_dim,
                        )?),
                        Some(load_kleidiai_weight(
                            ms,
                            &format!("{}.self_attn.k_proj.weight", lp),
                            hidden,
                            kv_dim,
                        )?),
                        Some(load_kleidiai_weight(
                            ms,
                            &format!("{}.self_attn.v_proj.weight", lp),
                            hidden,
                            kv_dim,
                        )?),
                        Some(load_kleidiai_weight(
                            ms,
                            &format!("{}.self_attn.o_proj.weight", lp),
                            q_dim,
                            hidden,
                        )?),
                        Some(load_kleidiai_weight(
                            ms,
                            &format!("{}.mlp.gate_up_fused.weight", lp),
                            hidden,
                            2 * inter,
                        )?),
                        Some(load_kleidiai_weight(
                            ms,
                            &format!("{}.mlp.down_proj.weight", lp),
                            inter,
                            hidden,
                        )?),
                    )
                } else {
                    (None, None, None, None, None, None)
                };

                layers.push(DecLayer {
                    wq_weight_bf16: std::ptr::null(),
                    wk_weight_bf16: std::ptr::null(),
                    wv_weight_bf16: std::ptr::null(),
                    wo_weight_bf16: std::ptr::null(),
                    q_norm_weight: q_norm,
                    k_norm_weight: k_norm,
                    input_norm,
                    post_attn_norm,
                    gate_weight_bf16: std::ptr::null(),
                    up_weight_bf16: std::ptr::null(),
                    down_weight_bf16: std::ptr::null(),
                    gate_up_fused_bf16: Vec::new(),
                    wq_int8,
                    wq_int8_scales,
                    wk_int8,
                    wk_int8_scales,
                    wv_int8,
                    wv_int8_scales,
                    wo_int8,
                    wo_int8_scales,
                    gate_up_int8,
                    gate_up_int8_scales,
                    down_int8,
                    down_int8_scales,
                    #[cfg(feature = "kleidiai")]
                    wq_kleidiai,
                    #[cfg(feature = "kleidiai")]
                    wk_kleidiai,
                    #[cfg(feature = "kleidiai")]
                    wv_kleidiai,
                    #[cfg(feature = "kleidiai")]
                    wo_kleidiai,
                    #[cfg(feature = "kleidiai")]
                    gate_up_kleidiai,
                    #[cfg(feature = "kleidiai")]
                    down_kleidiai,
                });
            } else {
                // BF16 model path (original): load BF16 then quantize on device.
                let wq = load_bf16_direct(ms, &format!("{}.self_attn.q_proj.weight", lp))?;
                let wk = load_bf16_direct(ms, &format!("{}.self_attn.k_proj.weight", lp))?;
                let wv = load_bf16_direct(ms, &format!("{}.self_attn.v_proj.weight", lp))?;
                let wo = load_bf16_direct(ms, &format!("{}.self_attn.o_proj.weight", lp))?;

                let gate_bf16 = load_bf16_direct(ms, &format!("{}.mlp.gate_proj.weight", lp))?;
                let up_bf16 = load_bf16_direct(ms, &format!("{}.mlp.up_proj.weight", lp))?;
                let down_bf16 = load_bf16_direct(ms, &format!("{}.mlp.down_proj.weight", lp))?;

                // The ARM64 path can project gate/up independently during prefill
                // and uses fused INT8 weights during decode, so retaining another
                // full BF16 copy is unnecessary. Keep it only for the non-ARM64
                // BF16 decode fallback.
                let mut gate_up_fused = Vec::new();
                if !cfg!(target_arch = "aarch64") {
                    gate_up_fused.resize(2 * inter * hidden, 0u16);
                    unsafe {
                        let gate_slice = std::slice::from_raw_parts(gate_bf16, inter * hidden);
                        let up_slice = std::slice::from_raw_parts(up_bf16, inter * hidden);
                        for r in 0..inter {
                            gate_up_fused[2 * r * hidden..(2 * r + 1) * hidden]
                                .copy_from_slice(&gate_slice[r * hidden..(r + 1) * hidden]);
                            gate_up_fused[(2 * r + 1) * hidden..(2 * r + 2) * hidden]
                                .copy_from_slice(&up_slice[r * hidden..(r + 1) * hidden]);
                        }
                    }
                }
                // INT8 quantize all decoder layer weights (on-device, heap-backed)
                let (wq_i8, wq_int8_scales) =
                    unsafe { kernels::quantize_bf16_weights_to_int8(wq, q_dim, hidden) };
                let wq_int8 = Int8Buf::from_owned(wq_i8);
                let (wk_i8, wk_int8_scales) =
                    unsafe { kernels::quantize_bf16_weights_to_int8(wk, kv_dim, hidden) };
                let wk_int8 = Int8Buf::from_owned(wk_i8);
                let (wv_i8, wv_int8_scales) =
                    unsafe { kernels::quantize_bf16_weights_to_int8(wv, kv_dim, hidden) };
                let wv_int8 = Int8Buf::from_owned(wv_i8);
                let (wo_i8, wo_int8_scales) =
                    unsafe { kernels::quantize_bf16_weights_to_int8(wo, hidden, q_dim) };
                let wo_int8 = Int8Buf::from_owned(wo_i8);
                let (gate_up_i8, gate_up_int8_scales) = if gate_up_fused.is_empty() {
                    let (gate_i8, gate_scales) =
                        unsafe { kernels::quantize_bf16_weights_to_int8(gate_bf16, inter, hidden) };
                    let (up_i8, up_scales) =
                        unsafe { kernels::quantize_bf16_weights_to_int8(up_bf16, inter, hidden) };
                    let mut fused_i8 = vec![0i8; 2 * inter * hidden];
                    let mut fused_scales = vec![0.0f32; 2 * inter];
                    for r in 0..inter {
                        fused_i8[2 * r * hidden..(2 * r + 1) * hidden]
                            .copy_from_slice(&gate_i8[r * hidden..(r + 1) * hidden]);
                        fused_i8[(2 * r + 1) * hidden..(2 * r + 2) * hidden]
                            .copy_from_slice(&up_i8[r * hidden..(r + 1) * hidden]);
                        fused_scales[2 * r] = gate_scales[r];
                        fused_scales[2 * r + 1] = up_scales[r];
                    }
                    (fused_i8, fused_scales)
                } else {
                    unsafe {
                        kernels::quantize_bf16_weights_to_int8(
                            gate_up_fused.as_ptr(),
                            2 * inter,
                            hidden,
                        )
                    }
                };
                let gate_up_int8 = Int8Buf::from_owned(gate_up_i8);
                let (down_i8, down_int8_scales) =
                    unsafe { kernels::quantize_bf16_weights_to_int8(down_bf16, hidden, inter) };
                let down_int8 = Int8Buf::from_owned(down_i8);

                layers.push(DecLayer {
                    wq_weight_bf16: wq,
                    wk_weight_bf16: wk,
                    wv_weight_bf16: wv,
                    wo_weight_bf16: wo,
                    q_norm_weight: q_norm,
                    k_norm_weight: k_norm,
                    input_norm,
                    post_attn_norm,
                    gate_weight_bf16: gate_bf16,
                    up_weight_bf16: up_bf16,
                    down_weight_bf16: down_bf16,
                    gate_up_fused_bf16: gate_up_fused,
                    wq_int8,
                    wq_int8_scales,
                    wk_int8,
                    wk_int8_scales,
                    wv_int8,
                    wv_int8_scales,
                    wo_int8,
                    wo_int8_scales,
                    gate_up_int8,
                    gate_up_int8_scales,
                    down_int8,
                    down_int8_scales,
                    #[cfg(feature = "kleidiai")]
                    wq_kleidiai: None,
                    #[cfg(feature = "kleidiai")]
                    wk_kleidiai: None,
                    #[cfg(feature = "kleidiai")]
                    wv_kleidiai: None,
                    #[cfg(feature = "kleidiai")]
                    wo_kleidiai: None,
                    #[cfg(feature = "kleidiai")]
                    gate_up_kleidiai: None,
                    #[cfg(feature = "kleidiai")]
                    down_kleidiai: None,
                });
            }
        }

        let norm = load_f32(ms, "thinker.model.norm.weight")?;

        // Load separate lm_head if present (forced aligner has untied lm_head)
        let lm_head_bf16 = if cfg.classify_num > 0 {
            let ptr = load_bf16_direct(ms, "thinker.lm_head.weight")?;
            Some(ptr)
        } else {
            // For normal ASR, lm_head is tied with tok_embeddings (no separate weight)
            ms.get_bf16_direct("thinker.lm_head.weight")
        };

        let lm_out_dim = cfg.lm_head_dim();
        let lm_in_dim = cfg.dec_hidden;
        let (lm_head_int8, lm_head_int8_scales) = match ms.get_i8_ptr("thinker.lm_head.weight") {
            Some((ptr, len)) if len == lm_out_dim * lm_in_dim => (
                Some(Int8Buf::from_mmap(ptr, len)),
                Some(load_f32(ms, "thinker.lm_head.weight_int8_scale")?),
            ),
            _ => {
                let lm_weight = lm_head_bf16.unwrap_or(tok_embeddings_bf16_opt?);
                let (lm_int8, lm_scales) =
                    unsafe { kernels::quantize_bf16_weights_to_int8(lm_weight, lm_out_dim, lm_in_dim) };
                (Some(Int8Buf::from_owned(lm_int8)), Some(lm_scales))
            }
        };

        let tok_embeddings_bf16 = tok_embeddings_bf16_opt.unwrap_or(std::ptr::null());

        Some(Decoder {
            tok_embeddings_bf16,
            tok_embeddings_int8,
            tok_embeddings_int8_scales,
            tok_embeddings_int8_scales_len,
            tok_embeddings_int8_mins,
            tok_embeddings_int8_mins_len,
            tok_embeddings_int8_group_size,
            layers,
            norm,
            lm_head_bf16,
            lm_head_int8,
            lm_head_int8_scales,
            backend: decoder_backend,
        })
    }
}

// ========================================================================
// KV Cache
// ========================================================================

pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    pub max_seq: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    /// Layout: `[layer][head][pos][head_dim]` — head-contiguous for cache-friendly attention.
    pub fn new(n_layers: usize, max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let total = n_layers * n_kv_heads * max_seq * head_dim;
        KvCache {
            k: vec![0.0f32; total],
            v: vec![0.0f32; total],
            len: 0,
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn grow(&mut self, required: usize) {
        if required <= self.max_seq {
            return;
        }

        let mut new_max = self.max_seq;
        while new_max < required {
            new_max *= 2;
        }

        let old_head_stride = self.max_seq * self.head_dim;
        let new_head_stride = new_max * self.head_dim;
        let total = self.n_layers * self.n_kv_heads * new_head_stride;

        let mut new_k = vec![0.0f32; total];
        let mut new_v = vec![0.0f32; total];

        let copy_len = self.len * self.head_dim;
        for l in 0..self.n_layers {
            for h in 0..self.n_kv_heads {
                let old_off = (l * self.n_kv_heads + h) * old_head_stride;
                let new_off = (l * self.n_kv_heads + h) * new_head_stride;
                new_k[new_off..new_off + copy_len]
                    .copy_from_slice(&self.k[old_off..old_off + copy_len]);
                new_v[new_off..new_off + copy_len]
                    .copy_from_slice(&self.v[old_off..old_off + copy_len]);
            }
        }

        self.k = new_k;
        self.v = new_v;
        self.max_seq = new_max;
    }

    /// Copy the first `len` positions into compact head-contiguous buffers.
    pub fn snapshot_prefix(&self, len: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(len <= self.len);
        let compact_head_stride = len * self.head_dim;
        let total = self.n_layers * self.n_kv_heads * compact_head_stride;
        let mut k = vec![0.0; total];
        let mut v = vec![0.0; total];
        let source_head_stride = self.head_stride();
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                let source = (layer * self.n_kv_heads + head) * source_head_stride;
                let destination = (layer * self.n_kv_heads + head) * compact_head_stride;
                k[destination..destination + compact_head_stride]
                    .copy_from_slice(&self.k[source..source + compact_head_stride]);
                v[destination..destination + compact_head_stride]
                    .copy_from_slice(&self.v[source..source + compact_head_stride]);
            }
        }
        (k, v)
    }

    /// Restore compact prefix buffers into the working cache.
    pub fn restore_prefix(&mut self, len: usize, k: &[f32], v: &[f32]) {
        self.grow(len);
        let compact_head_stride = len * self.head_dim;
        let expected = self.n_layers * self.n_kv_heads * compact_head_stride;
        assert_eq!(k.len(), expected);
        assert_eq!(v.len(), expected);
        let destination_head_stride = self.head_stride();
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                let source = (layer * self.n_kv_heads + head) * compact_head_stride;
                let destination = (layer * self.n_kv_heads + head) * destination_head_stride;
                self.k[destination..destination + compact_head_stride]
                    .copy_from_slice(&k[source..source + compact_head_stride]);
                self.v[destination..destination + compact_head_stride]
                    .copy_from_slice(&v[source..source + compact_head_stride]);
            }
        }
        self.len = len;
    }

    /// Write K for all heads at a given position (from interleaved kv_dim buffer).
    pub fn k_write_pos(&mut self, layer: usize, pos: usize, src: &[f32]) {
        let head_stride = self.max_seq * self.head_dim;
        for h in 0..self.n_kv_heads {
            let dst_off = (layer * self.n_kv_heads + h) * head_stride + pos * self.head_dim;
            let src_off = h * self.head_dim;
            self.k[dst_off..dst_off + self.head_dim]
                .copy_from_slice(&src[src_off..src_off + self.head_dim]);
        }
    }

    /// Write V for all heads at a given position (from interleaved kv_dim buffer).
    pub fn v_write_pos(&mut self, layer: usize, pos: usize, src: &[f32]) {
        let head_stride = self.max_seq * self.head_dim;
        for h in 0..self.n_kv_heads {
            let dst_off = (layer * self.n_kv_heads + h) * head_stride + pos * self.head_dim;
            let src_off = h * self.head_dim;
            self.v[dst_off..dst_off + self.head_dim]
                .copy_from_slice(&src[src_off..src_off + self.head_dim]);
        }
    }

    /// Base pointer for K data of a specific layer (head-contiguous layout).
    /// Layout within layer: `[head][pos][head_dim]`, stride between heads = max_seq * head_dim.
    pub fn k_layer_base(&self, layer: usize) -> *const f32 {
        let off = layer * self.n_kv_heads * self.max_seq * self.head_dim;
        unsafe { self.k.as_ptr().add(off) }
    }

    pub fn v_layer_base(&self, layer: usize) -> *const f32 {
        let off = layer * self.n_kv_heads * self.max_seq * self.head_dim;
        unsafe { self.v.as_ptr().add(off) }
    }

    /// Stride between heads (in floats): max_seq * head_dim.
    pub fn head_stride(&self) -> usize {
        self.max_seq * self.head_dim
    }
}

// ========================================================================
// RoPE Cache
// ========================================================================

pub struct RopeCache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub inv_freq: Vec<f32>,
    pub cap: usize,
    pub head_dim: usize,
}

impl Default for RopeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RopeCache {
    pub fn new() -> Self {
        RopeCache {
            cos: Vec::new(),
            sin: Vec::new(),
            inv_freq: Vec::new(),
            cap: 0,
            head_dim: 0,
        }
    }

    pub fn ensure(&mut self, required_pos: usize, head_dim: usize, theta: f32) {
        if self.head_dim != head_dim || self.inv_freq.is_empty() {
            let half = head_dim / 2;
            self.inv_freq = (0..half)
                .map(|d| 1.0 / theta.powf((2 * d) as f32 / head_dim as f32))
                .collect();
            self.head_dim = head_dim;
        }

        if required_pos <= self.cap {
            return;
        }

        let mut new_cap = if self.cap > 0 { self.cap } else { 1024 };
        while new_cap < required_pos {
            new_cap *= 2;
        }

        self.cos.resize(new_cap * head_dim, 0.0);
        self.sin.resize(new_cap * head_dim, 0.0);

        let half = head_dim / 2;
        for pos in self.cap..new_cap {
            let p = pos as f32;
            for d in 0..half {
                let angle = p * self.inv_freq[d];
                let c = angle.cos();
                let s = angle.sin();
                self.cos[pos * head_dim + d] = c;
                self.cos[pos * head_dim + half + d] = c;
                self.sin[pos * head_dim + d] = s;
                self.sin[pos * head_dim + half + d] = s;
            }
        }

        self.cap = new_cap;
    }

    pub fn cos_at(&self, pos: usize) -> &[f32] {
        &self.cos[pos * self.head_dim..(pos + 1) * self.head_dim]
    }

    pub fn sin_at(&self, pos: usize) -> &[f32] {
        &self.sin[pos * self.head_dim..(pos + 1) * self.head_dim]
    }

    pub fn cos_range(&self, start: usize, len: usize) -> &[f32] {
        &self.cos[start * self.head_dim..(start + len) * self.head_dim]
    }

    pub fn sin_range(&self, start: usize, len: usize) -> &[f32] {
        &self.sin[start * self.head_dim..(start + len) * self.head_dim]
    }
}

// ========================================================================
// Decoder Forward
// ========================================================================

pub struct DecoderBuffers {
    // Single-token decode buffers
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub ffn_out: Vec<f32>,

    // Prefill buffers
    pub pref_x: Vec<f32>,
    pub pref_x_norm: Vec<f32>,
    pub pref_q: Vec<f32>,
    pub pref_k: Vec<f32>,
    pub pref_v: Vec<f32>,
    pub pref_attn_out: Vec<f32>,
    pub pref_proj_out: Vec<f32>,
    pub pref_ffn_out: Vec<f32>,
    pub pref_gate_up: Vec<f32>,
    pub pref_gate: Vec<f32>,
    pub pref_up: Vec<f32>,
    pub pref_seq_cap: usize,

    // Reusable scratch for BF16→F32 conversion in prefill path
    pub bf16_scratch: Vec<f32>,
}

impl DecoderBuffers {
    pub fn new(cfg: &QwenConfig, scratch_len: usize) -> Self {
        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;

        DecoderBuffers {
            x: vec![0.0f32; dim],
            x_norm: vec![0.0f32; dim],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; q_dim],
            proj_out: vec![0.0f32; dim],
            gate_buf: vec![0.0f32; 2 * intermediate],
            ffn_out: vec![0.0f32; intermediate],
            pref_x: Vec::new(),
            pref_x_norm: Vec::new(),
            pref_q: Vec::new(),
            pref_k: Vec::new(),
            pref_v: Vec::new(),
            pref_attn_out: Vec::new(),
            pref_proj_out: Vec::new(),
            pref_ffn_out: Vec::new(),
            pref_gate_up: Vec::new(),
            pref_gate: Vec::new(),
            pref_up: Vec::new(),
            pref_seq_cap: 0,
            bf16_scratch: vec![0.0f32; scratch_len],
        }
    }

    pub fn ensure_prefill(&mut self, seq_len: usize, cfg: &QwenConfig) {
        if seq_len <= self.pref_seq_cap {
            return;
        }

        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;

        let mut new_cap = if self.pref_seq_cap > 0 {
            self.pref_seq_cap
        } else {
            64
        };
        while new_cap < seq_len {
            new_cap *= 2;
        }

        self.pref_x.resize(new_cap * dim, 0.0);
        self.pref_x_norm.resize(new_cap * dim, 0.0);
        self.pref_q.resize(new_cap * q_dim, 0.0);
        self.pref_k.resize(new_cap * kv_dim, 0.0);
        self.pref_v.resize(new_cap * kv_dim, 0.0);
        self.pref_attn_out.resize(new_cap * q_dim, 0.0);
        self.pref_proj_out.resize(new_cap * dim, 0.0);
        self.pref_ffn_out.resize(new_cap * dim, 0.0);
        self.pref_gate_up.resize(new_cap * 2 * intermediate, 0.0);
        self.pref_gate.resize(new_cap * intermediate, 0.0);
        self.pref_up.resize(new_cap * intermediate, 0.0);

        self.pref_seq_cap = new_cap;
    }
}

/// Decoder prefill: process multiple tokens.
pub fn decoder_prefill(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    const INT8_PREFILL_TILE_ROWS: usize = 256;
    let use_kleidiai = decoder.uses_kleidiai();

    // Ensure KV cache
    let needed = kv_cache.len + seq_len;
    if needed > kv_cache.max_seq {
        kv_cache.grow(needed + 1024);
    }

    bufs.ensure_prefill(seq_len, cfg);

    let x = &mut bufs.pref_x[..seq_len * dim];
    x.copy_from_slice(&input_embeds[..seq_len * dim]);

    let start_pos = kv_cache.len;
    rope.ensure(start_pos + seq_len, head_dim, theta);
    let rope_cos = rope.cos_range(start_pos, seq_len);
    let rope_sin = rope.sin_range(start_pos, seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, layer) in decoder.layers.iter().enumerate() {
        #[cfg(feature = "kleidiai")]
        let (wq_packed, wk_packed, wv_packed, wo_packed, gate_up_packed, down_packed) = (
            layer.wq_kleidiai.as_deref(),
            layer.wk_kleidiai.as_deref(),
            layer.wv_kleidiai.as_deref(),
            layer.wo_kleidiai.as_deref(),
            layer.gate_up_kleidiai.as_deref(),
            layer.down_kleidiai.as_deref(),
        );
        #[cfg(not(feature = "kleidiai"))]
        let (wq_packed, wk_packed, wv_packed, wo_packed, gate_up_packed, down_packed) =
            (None::<&[i8]>, None, None, None, None, None);
        let x_norm = &mut bufs.pref_x_norm[..seq_len * dim];
        kernels::rms_norm(
            x_norm,
            &bufs.pref_x[..seq_len * dim],
            &layer.input_norm,
            seq_len,
            dim,
            eps,
        );

        let q = &mut bufs.pref_q[..seq_len * q_dim];
        let k = &mut bufs.pref_k[..seq_len * kv_dim];
        let v = &mut bufs.pref_v[..seq_len * kv_dim];

        if try_shared_qkv(q, k, v, x_norm, wq_packed, wk_packed, wv_packed,
            seq_len, dim, q_dim, kv_dim, use_kleidiai) {
        } else if !layer.wq_weight_bf16.is_null() {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    q,
                    x_norm,
                    layer.wq_weight_bf16,
                    seq_len,
                    dim,
                    q_dim,
                    &mut bufs.bf16_scratch,
                );
                kernels::linear_nobias_bf16_scratch(
                    k,
                    x_norm,
                    layer.wk_weight_bf16,
                    seq_len,
                    dim,
                    kv_dim,
                    &mut bufs.bf16_scratch,
                );
                kernels::linear_nobias_bf16_scratch(
                    v,
                    x_norm,
                    layer.wv_weight_bf16,
                    seq_len,
                    dim,
                    kv_dim,
                    &mut bufs.bf16_scratch,
                );
            }
        } else {
            linear_nobias_int8_f32_scratch(
                q,
                x_norm,
                &layer.wq_int8,
                &layer.wq_int8_scales,
                wq_packed,
                seq_len,
                dim,
                q_dim,
                &mut bufs.bf16_scratch,
                INT8_PREFILL_TILE_ROWS,
                use_kleidiai,
            );
            linear_nobias_int8_f32_scratch(
                k,
                x_norm,
                &layer.wk_int8,
                &layer.wk_int8_scales,
                wk_packed,
                seq_len,
                dim,
                kv_dim,
                &mut bufs.bf16_scratch,
                INT8_PREFILL_TILE_ROWS,
                use_kleidiai,
            );
            linear_nobias_int8_f32_scratch(
                v,
                x_norm,
                &layer.wv_int8,
                &layer.wv_int8_scales,
                wv_packed,
                seq_len,
                dim,
                kv_dim,
                &mut bufs.bf16_scratch,
                INT8_PREFILL_TILE_ROWS,
                use_kleidiai,
            );
        }

        kernels::rms_norm_per_head(q, &layer.q_norm_weight, seq_len, n_heads, head_dim, eps);
        kernels::rms_norm_per_head(k, &layer.k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        kernels::apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        kernels::apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        // Store K, V in cache (scatter to head-contiguous layout)
        for s in 0..seq_len {
            kv_cache.k_write_pos(
                layer_idx,
                start_pos + s,
                &bufs.pref_k[s * kv_dim..(s + 1) * kv_dim],
            );
            kv_cache.v_write_pos(
                layer_idx,
                start_pos + s,
                &bufs.pref_v[s * kv_dim..(s + 1) * kv_dim],
            );
        }

        let total_seq = start_pos + seq_len;
        let k_base = kv_cache.k_layer_base(layer_idx);
        let v_base = kv_cache.v_layer_base(layer_idx);
        let head_stride = kv_cache.head_stride();

        let attn_out = &mut bufs.pref_attn_out[..seq_len * q_dim];
        kernels::causal_attention(
            attn_out,
            q,
            k_base,
            v_base,
            head_stride,
            seq_len,
            total_seq,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            start_pos,
        );

        let proj_out = &mut bufs.pref_proj_out[..seq_len * dim];
        if !layer.wo_weight_bf16.is_null() {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    proj_out,
                    attn_out,
                    layer.wo_weight_bf16,
                    seq_len,
                    q_dim,
                    dim,
                    &mut bufs.bf16_scratch,
                );
            }
        } else {
            linear_nobias_int8_f32_scratch(
                proj_out,
                attn_out,
                &layer.wo_int8,
                &layer.wo_int8_scales,
                wo_packed,
                seq_len,
                q_dim,
                dim,
                &mut bufs.bf16_scratch,
                INT8_PREFILL_TILE_ROWS,
                use_kleidiai,
            );
        }
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], proj_out, seq_len * dim);

        // Post-attention RMSNorm + SwiGLU MLP
        let x_norm2 = &mut bufs.pref_x_norm[..seq_len * dim];
        kernels::rms_norm(
            x_norm2,
            &bufs.pref_x[..seq_len * dim],
            &layer.post_attn_norm,
            seq_len,
            dim,
            eps,
        );

        let gate_up = &mut bufs.pref_gate_up[..seq_len * 2 * intermediate];
        if !layer.gate_up_fused_bf16.is_empty() {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    gate_up,
                    x_norm2,
                    layer.gate_up_fused_bf16.as_ptr(),
                    seq_len,
                    dim,
                    2 * intermediate,
                    &mut bufs.bf16_scratch,
                );
            }
        } else if !layer.gate_weight_bf16.is_null() && !layer.up_weight_bf16.is_null() {
            // Reuse existing activation buffers for the two projections, then
            // interleave their outputs for the established vectorized SwiGLU.
            // This avoids retaining a duplicate BF16 gate/up weight matrix.
            let gate_values = &mut bufs.pref_gate[..seq_len * intermediate];
            let up_values = &mut bufs.pref_up[..seq_len * intermediate];
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    gate_values,
                    x_norm2,
                    layer.gate_weight_bf16,
                    seq_len,
                    dim,
                    intermediate,
                    &mut bufs.bf16_scratch,
                );
                kernels::linear_nobias_bf16_scratch(
                    up_values,
                    x_norm2,
                    layer.up_weight_bf16,
                    seq_len,
                    dim,
                    intermediate,
                    &mut bufs.bf16_scratch,
                );
            }
            for s in 0..seq_len {
                let src = s * intermediate;
                let dst = s * 2 * intermediate;
                for j in 0..intermediate {
                    gate_up[dst + 2 * j] = gate_values[src + j];
                    gate_up[dst + 2 * j + 1] = up_values[src + j];
                }
            }
        } else {
            linear_nobias_int8_f32_scratch(
                gate_up,
                x_norm2,
                &layer.gate_up_int8,
                &layer.gate_up_int8_scales,
                gate_up_packed,
                seq_len,
                dim,
                2 * intermediate,
                &mut bufs.bf16_scratch,
                INT8_PREFILL_TILE_ROWS,
                use_kleidiai,
            );
        }

        let gate = &mut bufs.pref_gate[..seq_len * intermediate];
        kernels::swiglu_multiply(gate, gate_up, seq_len, intermediate);

        let ffn_out = &mut bufs.pref_ffn_out[..seq_len * dim];
        if !layer.down_weight_bf16.is_null() {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    ffn_out,
                    gate,
                    layer.down_weight_bf16,
                    seq_len,
                    intermediate,
                    dim,
                    &mut bufs.bf16_scratch,
                );
            }
        } else {
            linear_nobias_int8_f32_scratch(
                ffn_out,
                gate,
                &layer.down_int8,
                &layer.down_int8_scales,
                down_packed,
                seq_len,
                intermediate,
                dim,
                &mut bufs.bf16_scratch,
                INT8_PREFILL_TILE_ROWS,
                use_kleidiai,
            );
        }
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], ffn_out, seq_len * dim);
    }

    kv_cache.len = start_pos + seq_len;
}

/// Decoder single-token forward: returns greedy token ID.
pub fn decoder_forward(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embed: &[f32],
) -> i32 {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    bufs.x[..dim].copy_from_slice(&input_embed[..dim]);

    let pos = kv_cache.len;

    if pos >= kv_cache.max_seq {
        kv_cache.grow(pos + 1024);
    }

    rope.ensure(pos + 1, head_dim, theta);
    let rope_cos = rope.cos_at(pos);
    let rope_sin = rope.sin_at(pos);

    let scale = 1.0 / (head_dim as f32).sqrt();

    #[cfg(feature = "kleidiai")]
    let use_kleidiai_decode = decoder.uses_kleidiai();

    for (layer_idx, layer) in decoder.layers.iter().enumerate() {
        kernels::rms_norm(
            &mut bufs.x_norm[..dim],
            &bufs.x[..dim],
            &layer.input_norm,
            1,
            dim,
            eps,
        );

        // INT8 fused QKV projection (aarch64 only, BF16 fallback on x86_64)
        #[cfg(all(target_arch = "aarch64", feature = "kleidiai"))]
        if use_kleidiai_decode {
            if !try_shared_qkv(&mut bufs.q[..q_dim], &mut bufs.k[..kv_dim], &mut bufs.v[..kv_dim],
                &bufs.x_norm[..dim], layer.wq_kleidiai.as_deref(), layer.wk_kleidiai.as_deref(),
                layer.wv_kleidiai.as_deref(), 1, dim, q_dim, kv_dim, true) {
            kernels::linear_nobias_int8_kleidiai(
                &mut bufs.q[..q_dim],
                &bufs.x_norm[..dim],
                layer
                    .wq_kleidiai
                    .as_deref()
                    .expect("validated KleidiAI Q projection missing"),
                1,
                dim,
                q_dim,
            );
            kernels::linear_nobias_int8_kleidiai(
                &mut bufs.k[..kv_dim],
                &bufs.x_norm[..dim],
                layer
                    .wk_kleidiai
                    .as_deref()
                    .expect("validated KleidiAI K projection missing"),
                1,
                dim,
                kv_dim,
            );
            kernels::linear_nobias_int8_kleidiai(
                &mut bufs.v[..kv_dim],
                &bufs.x_norm[..dim],
                layer
                    .wv_kleidiai
                    .as_deref()
                    .expect("validated KleidiAI V projection missing"),
                1,
                dim,
                kv_dim,
            );
            }
        } else {
            kernels::linear_nobias_int8_qkv(
                &mut bufs.q[..q_dim],
                &mut bufs.k[..kv_dim],
                &mut bufs.v[..kv_dim],
                &bufs.x_norm[..dim],
                &layer.wq_int8,
                &layer.wq_int8_scales,
                &layer.wk_int8,
                &layer.wk_int8_scales,
                &layer.wv_int8,
                &layer.wv_int8_scales,
                dim,
                q_dim,
                kv_dim,
            );
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "kleidiai")))]
        kernels::linear_nobias_int8_qkv(
            &mut bufs.q[..q_dim],
            &mut bufs.k[..kv_dim],
            &mut bufs.v[..kv_dim],
            &bufs.x_norm[..dim],
            &layer.wq_int8,
            &layer.wq_int8_scales,
            &layer.wk_int8,
            &layer.wk_int8_scales,
            &layer.wv_int8,
            &layer.wv_int8_scales,
            dim,
            q_dim,
            kv_dim,
        );
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            kernels::linear_nobias_bf16(
                &mut bufs.q[..q_dim],
                &bufs.x_norm[..dim],
                layer.wq_weight_bf16,
                1,
                dim,
                q_dim,
            );
            kernels::linear_nobias_bf16(
                &mut bufs.k[..kv_dim],
                &bufs.x_norm[..dim],
                layer.wk_weight_bf16,
                1,
                dim,
                kv_dim,
            );
            kernels::linear_nobias_bf16(
                &mut bufs.v[..kv_dim],
                &bufs.x_norm[..dim],
                layer.wv_weight_bf16,
                1,
                dim,
                kv_dim,
            );
        }

        kernels::rms_norm_per_head(
            &mut bufs.q[..q_dim],
            &layer.q_norm_weight,
            1,
            n_heads,
            head_dim,
            eps,
        );
        kernels::rms_norm_per_head(
            &mut bufs.k[..kv_dim],
            &layer.k_norm_weight,
            1,
            n_kv_heads,
            head_dim,
            eps,
        );

        kernels::apply_rope_neox(
            &mut bufs.q[..q_dim],
            rope_cos,
            rope_sin,
            1,
            n_heads,
            head_dim,
        );
        kernels::apply_rope_neox(
            &mut bufs.k[..kv_dim],
            rope_cos,
            rope_sin,
            1,
            n_kv_heads,
            head_dim,
        );

        kv_cache.k_write_pos(layer_idx, pos, &bufs.k[..kv_dim]);
        kv_cache.v_write_pos(layer_idx, pos, &bufs.v[..kv_dim]);

        let total_seq = pos + 1;
        let k_base = kv_cache.k_layer_base(layer_idx);
        let v_base = kv_cache.v_layer_base(layer_idx);
        let head_stride = kv_cache.head_stride();

        kernels::causal_attention(
            &mut bufs.attn_out[..q_dim],
            &bufs.q[..q_dim],
            k_base,
            v_base,
            head_stride,
            1,
            total_seq,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            pos,
        );

        // O-projection with fused residual add: x += attn_out @ wo (INT8 on aarch64, BF16 on x86_64)
        #[cfg(all(target_arch = "aarch64", feature = "kleidiai"))]
        if use_kleidiai_decode {
            kernels::linear_nobias_int8_kleidiai(
                &mut bufs.proj_out[..dim],
                &bufs.attn_out[..q_dim],
                layer
                    .wo_kleidiai
                    .as_deref()
                    .expect("validated KleidiAI O projection missing"),
                1,
                q_dim,
                dim,
            );
            kernels::add_inplace(&mut bufs.x[..dim], &bufs.proj_out[..dim], dim);
        } else {
            kernels::linear_nobias_int8_addto(
                &mut bufs.x[..dim],
                &bufs.attn_out[..q_dim],
                &layer.wo_int8,
                &layer.wo_int8_scales,
                q_dim,
                dim,
            );
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "kleidiai")))]
        kernels::linear_nobias_int8_addto(
            &mut bufs.x[..dim],
            &bufs.attn_out[..q_dim],
            &layer.wo_int8,
            &layer.wo_int8_scales,
            q_dim,
            dim,
        );
        #[cfg(not(target_arch = "aarch64"))]
        kernels::linear_nobias_bf16_addto(
            &mut bufs.x[..dim],
            &bufs.attn_out[..q_dim],
            layer.wo_weight_bf16,
            q_dim,
            dim,
        );

        kernels::rms_norm(
            &mut bufs.x_norm[..dim],
            &bufs.x[..dim],
            &layer.post_attn_norm,
            1,
            dim,
            eps,
        );

        // gate_up + SwiGLU (INT8 on aarch64, BF16 on x86_64)
        #[cfg(all(target_arch = "aarch64", feature = "kleidiai"))]
        if use_kleidiai_decode {
            kernels::linear_nobias_int8_kleidiai(
                &mut bufs.gate_buf[..2 * intermediate],
                &bufs.x_norm[..dim],
                layer
                    .gate_up_kleidiai
                    .as_deref()
                    .expect("validated KleidiAI gate/up projection missing"),
                1,
                dim,
                2 * intermediate,
            );
            kernels::swiglu_multiply(
                &mut bufs.ffn_out[..intermediate],
                &bufs.gate_buf[..2 * intermediate],
                1,
                intermediate,
            );
        } else {
            kernels::linear_nobias_int8_swiglu(
                &mut bufs.ffn_out[..intermediate],
                &mut bufs.gate_buf[..2 * intermediate],
                &bufs.x_norm[..dim],
                &layer.gate_up_int8,
                &layer.gate_up_int8_scales,
                dim,
                intermediate,
            );
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "kleidiai")))]
        kernels::linear_nobias_int8_swiglu(
            &mut bufs.ffn_out[..intermediate],
            &mut bufs.gate_buf[..2 * intermediate],
            &bufs.x_norm[..dim],
            &layer.gate_up_int8,
            &layer.gate_up_int8_scales,
            dim,
            intermediate,
        );
        #[cfg(not(target_arch = "aarch64"))]
        kernels::linear_nobias_bf16_swiglu(
            &mut bufs.ffn_out[..intermediate],
            &bufs.x_norm[..dim],
            layer.gate_up_fused_bf16.as_ptr(),
            dim,
            intermediate,
        );

        // down-projection with fused residual add: x += ffn_out @ down (INT8 on aarch64, BF16 on x86_64)
        #[cfg(all(target_arch = "aarch64", feature = "kleidiai"))]
        if use_kleidiai_decode {
            kernels::linear_nobias_int8_kleidiai(
                &mut bufs.proj_out[..dim],
                &bufs.ffn_out[..intermediate],
                layer
                    .down_kleidiai
                    .as_deref()
                    .expect("validated KleidiAI down projection missing"),
                1,
                intermediate,
                dim,
            );
            kernels::add_inplace(&mut bufs.x[..dim], &bufs.proj_out[..dim], dim);
        } else {
            kernels::linear_nobias_int8_addto(
                &mut bufs.x[..dim],
                &bufs.ffn_out[..intermediate],
                &layer.down_int8,
                &layer.down_int8_scales,
                intermediate,
                dim,
            );
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "kleidiai")))]
        kernels::linear_nobias_int8_addto(
            &mut bufs.x[..dim],
            &bufs.ffn_out[..intermediate],
            &layer.down_int8,
            &layer.down_int8_scales,
            intermediate,
            dim,
        );
        #[cfg(not(target_arch = "aarch64"))]
        kernels::linear_nobias_bf16_addto(
            &mut bufs.x[..dim],
            &bufs.ffn_out[..intermediate],
            layer.down_weight_bf16,
            intermediate,
            dim,
        );
    }

    kv_cache.len = pos + 1;

    // Final norm + streaming argmax (use x_norm as temp to avoid heap allocation)
    kernels::rms_norm(
        &mut bufs.x_norm[..dim],
        &bufs.x[..dim],
        &decoder.norm,
        1,
        dim,
        eps,
    );
    bufs.x[..dim].copy_from_slice(&bufs.x_norm[..dim]);
    let lm_out_dim = cfg.lm_head_dim();

    // Use INT8 quantized argmax on aarch64 (2x less bandwidth), BF16 on x86_64
    #[cfg(target_arch = "aarch64")]
    if let (Some(ref int8_data), Some(ref scales)) =
        (&decoder.lm_head_int8, &decoder.lm_head_int8_scales)
    {
        let best = kernels::argmax_matvec_int8(&bufs.x[..dim], int8_data, scales, dim, lm_out_dim);
        if std::env::var_os("QWEN_LOGIT_TRACE").is_some() {
            let mut rows = vec![best];
            if let Ok(ids) = std::env::var("QWEN_LOGIT_IDS") {
                rows.extend(
                    ids.split(',')
                        .filter_map(|id| id.trim().parse::<usize>().ok())
                        .filter(|&id| id < lm_out_dim),
                );
                rows.sort_unstable();
                rows.dedup();
            }
            let logits =
                kernels::matvec_int8_selected_logits(&bufs.x[..dim], int8_data, scales, dim, &rows);
            eprintln!(
                "LOGIT_TRACE pos={} best={} selected={:?}",
                pos,
                best,
                rows.iter().copied().zip(logits).collect::<Vec<_>>()
            );
        }
        return best as i32;
    }

    let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);
    if lm_weight.is_null() {
        panic!("decoder: fallback BF16 argmax called but no BF16 weights are loaded!");
    }
    kernels::argmax_matvec_bf16(&bufs.x[..dim], lm_weight, dim, lm_out_dim) as i32
}

/// Decoder prefill that returns per-position logits (for forced aligner).
/// Returns `[seq_len × out_dim]` logits where out_dim = classify_num.
pub fn decoder_prefill_logits(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) -> Vec<f32> {
    let dim = cfg.dec_hidden;
    let eps = cfg.dec_rms_norm_eps;
    let out_dim = cfg.lm_head_dim();

    // Run the standard prefill to get hidden states
    decoder_prefill(decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len);

    // After prefill, pref_x contains the final hidden states for all positions.
    // Apply final RMS norm and lm_head projection.
    let x = &bufs.pref_x[..seq_len * dim];
    let mut x_norm = vec![0.0f32; seq_len * dim];
    kernels::rms_norm(&mut x_norm, x, &decoder.norm, seq_len, dim, eps);

    let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);
    if lm_weight.is_null() {
        panic!("decoder: fallback BF16 prefill logits called but no BF16 weights are loaded!");
    }

    // Project each position through lm_head: [seq_len × dim] × [out_dim × dim]^T → [seq_len × out_dim]
    let mut logits = vec![0.0f32; seq_len * out_dim];
    unsafe {
        kernels::linear_nobias_bf16_scratch(
            &mut logits,
            &x_norm,
            lm_weight,
            seq_len,
            dim,
            out_dim,
            &mut bufs.bf16_scratch,
        );
    }

    logits
}

/// Prefill and project only selected vocabulary rows at the final position.
///
/// This is intended for confidence diagnostics on quantized checkpoints where
/// the BF16 lm-head has deliberately been omitted. Returned logits preserve
/// `rows` order and use the same INT8 activation quantization as greedy decode.
pub fn decoder_prefill_last_selected_logits(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
    rows: &[usize],
) -> Vec<f32> {
    let dim = cfg.dec_hidden;
    decoder_prefill(decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len);
    let final_hidden = &bufs.pref_x[(seq_len - 1) * dim..seq_len * dim];
    let mut normalized = vec![0.0f32; dim];
    kernels::rms_norm(
        &mut normalized,
        final_hidden,
        &decoder.norm,
        1,
        dim,
        cfg.dec_rms_norm_eps,
    );
    if let (Some(int8_data), Some(scales)) =
        (&decoder.lm_head_int8, &decoder.lm_head_int8_scales)
    {
        return kernels::matvec_int8_selected_logits(&normalized, int8_data, scales, dim, rows);
    }
    let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);
    if lm_weight.is_null() {
        panic!("decoder has neither INT8 nor BF16 lm-head weights");
    }
    rows.iter()
        .map(|&row| unsafe {
            let weights = std::slice::from_raw_parts(lm_weight.add(row * dim), dim);
            normalized
                .iter()
                .zip(weights)
                .map(|(&x, &w)| x * f32::from_bits((w as u32) << 16))
                .sum()
        })
        .collect()
}


/// Look up a token embedding and convert/dequantize to f32.
/// If tok_embeddings_bf16 is valid (non-null), reads from it in BF16.
/// Otherwise, dequantizes on-the-fly from lm_head_int8 (INT8) using its scales.
///
/// # Safety
/// The decoder must have either a valid tok_embeddings_bf16 pointer or valid
/// lm_head_int8 + scales pointers.
pub unsafe fn tok_embed_to_f32(decoder: &Decoder, dst: &mut [f32], token_id: i32, dim: usize) {
    if !decoder.tok_embeddings_bf16.is_null() {
        let src = std::slice::from_raw_parts(
            decoder.tok_embeddings_bf16.add(token_id as usize * dim),
            dim,
        );
        kernels::bf16_to_f32_buf(dst, src);
    } else if let Some(ref int8_buf) = decoder.tok_embeddings_int8 {
        let group_size = decoder.tok_embeddings_int8_group_size;
        let groups_per_row = dim / group_size;
        let token = token_id as usize;
        let src = std::slice::from_raw_parts(int8_buf.as_ptr().add(token * dim), dim);
        let scale_base = token * groups_per_row;
        debug_assert!(scale_base + groups_per_row <= decoder.tok_embeddings_int8_scales_len);
        for group in 0..groups_per_row {
            let scale = std::ptr::read_unaligned(
                decoder.tok_embeddings_int8_scales.add(scale_base + group),
            );
            let min = if decoder.tok_embeddings_int8_mins_len
                == decoder.tok_embeddings_int8_scales_len
            {
                std::ptr::read_unaligned(decoder.tok_embeddings_int8_mins.add(scale_base + group))
            } else {
                0.0
            };
            let start = group * group_size;
            for i in start..start + group_size {
                dst[i] = if decoder.tok_embeddings_int8_mins_len
                    == decoder.tok_embeddings_int8_scales_len
                {
                    ((src[i] as i16 + 128) as f32) * scale + min
                } else {
                    (src[i] as f32) * scale
                };
            }
        }
    } else if let (Some(ref int8_buf), Some(ref scales)) =
        (&decoder.lm_head_int8, &decoder.lm_head_int8_scales)
    {
        let src = std::slice::from_raw_parts(int8_buf.as_ptr().add(token_id as usize * dim), dim);
        let scale = scales[token_id as usize];
        for i in 0..dim {
            dst[i] = (src[i] as f32) * scale;
        }
    } else {
        panic!("decoder has no token embeddings!");
    }
}

#[cfg(test)]
mod tests {
    use super::KvCache;

    #[test]
    fn compact_prefix_snapshot_restores_exact_values() {
        let mut cache = KvCache::new(2, 8, 2, 3);
        cache.len = 4;
        for (index, value) in cache.k.iter_mut().enumerate() {
            *value = index as f32 + 0.25;
        }
        for (index, value) in cache.v.iter_mut().enumerate() {
            *value = -(index as f32) - 0.5;
        }

        let (expected_k, expected_v) = cache.snapshot_prefix(3);
        cache.k.fill(0.0);
        cache.v.fill(0.0);
        cache.len = 0;
        cache.restore_prefix(3, &expected_k, &expected_v);

        let (actual_k, actual_v) = cache.snapshot_prefix(3);
        assert_eq!(actual_k, expected_k);
        assert_eq!(actual_v, expected_v);
        assert_eq!(cache.len, 3);
    }
}
