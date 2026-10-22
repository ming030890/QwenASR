//! Small standalone, CPU-only candidate scorer and next-token predictor. No ASR/ONNX runtime dependency.
//! Weights are symmetric row-INT8/INT16, activations and reductions are float32.
use serde::Deserialize;
use std::collections::HashMap;
pub mod counts;
pub mod cnn;
mod cnn_spelling;
pub mod touch;
pub mod hwr;
pub mod hybrid;
#[cfg(feature = "android")]
mod jni_api;
pub mod memory;
pub mod neural;
pub mod phrases;
pub use neural::NeuralLanguageModel;
pub mod quick;
pub mod suggestion_request;
pub mod suggestions;
pub mod tokenizer;

#[derive(Deserialize)]
struct Config {
    vocab_size: usize,
    left: usize,
    right: usize,
    embedding: usize,
    hidden: usize,
    residual_blocks: usize,
    ffn_hidden: usize,
    activation: String,
    layer_norm: bool,
}
#[derive(Deserialize)]
struct Description {
    name: String,
    shape: Vec<usize>,
    kind: String,
}
#[derive(Deserialize)]
struct Metadata {
    version: u32,
    config: Config,
    tokens: Vec<String>,
    tensors: Vec<Description>,
    lookup_normalization: String,
    remove_bos: bool,
    #[serde(default)]
    neural_short_contexts: bool,
    #[serde(default)]
    spelling_cng_hex: Option<String>,
}

struct Tensor {
    shape: Vec<usize>,
    q: Vec<i8>,
    q16: Vec<i16>,
    scale: Vec<f32>,
    values: Vec<f32>,
}
impl Tensor {
    fn linear(&self, x: &[f32], bias: &Self) -> Vec<f32> {
        let columns = self.shape[1];
        assert_eq!(columns, x.len());
        (0..self.shape[0]).map(|i| {
            let score = if self.q16.is_empty() { dot(&self.q[i*columns..(i+1)*columns],x) }
                else { dot16(&self.q16[i*columns..(i+1)*columns],x) };
            score*self.scale[i]+bias.values[i]
        }).collect()
    }
}
fn dot16(q: &[i16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut a=vdupq_n_f32(0.);let mut b=vdupq_n_f32(0.);let n=q.len()/8*8;
        for i in (0..n).step_by(8) {
            let w=vld1q_s16(q.as_ptr().add(i));
            a=vfmaq_f32(a,vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))),vld1q_f32(x.as_ptr().add(i)));
            b=vfmaq_f32(b,vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))),vld1q_f32(x.as_ptr().add(i+4)));
        }
        let mut total=vaddvq_f32(vaddq_f32(a,b));
        for i in n..q.len() {total+=q[i] as f32*x[i];}
        total
    }
    #[cfg(not(target_arch = "aarch64"))]
    {q.iter().zip(x).map(|(&a,&b)|a as f32*b).sum()}
}
fn dot(q: &[i8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut a = vdupq_n_f32(0.);
        let mut b = vdupq_n_f32(0.);
        let n = q.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            let w = vmovl_s8(vld1_s8(q.as_ptr().add(i)));
            a = vfmaq_f32(
                a,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))),
                vld1q_f32(x.as_ptr().add(i)),
            );
            b = vfmaq_f32(
                b,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))),
                vld1q_f32(x.as_ptr().add(i + 4)),
            );
        }
        return vaddvq_f32(vaddq_f32(a, b))
            + q[n..]
                .iter()
                .zip(&x[n..])
                .map(|(&w, &v)| w as f32 * v)
                .sum::<f32>();
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        q.iter().zip(x).map(|(&w, &v)| w as f32 * v).sum()
    }
}
fn norm(x: &[f32], weight: &Tensor, bias: &Tensor) -> Vec<f32> {
    norm_values(x, &weight.values, &bias.values)
}
fn norm_values(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let inverse = (variance + 1e-5).sqrt().recip();
    x.iter()
        .enumerate()
        .map(|(i, &v)| (v - mean) * inverse * weight[i] + bias[i])
        .collect()
}
fn floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let v: Vec<_> = bytes
        .chunks_exact(4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("Nonfinite weights".into());
    }
    Ok(v)
}

/// Ranked token and its model score (logit or count score, not probability).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenPrediction {
    pub token: String,
    pub token_id: Option<usize>,
    pub score: f32,
}

pub(crate) fn displayable(token: &str) -> bool {
    !token.is_empty()
        && token != "<blank>"
        && !matches!(token, "[PAD]" | "[BOS]" | "[UNK]")
        && !token.chars().any(|c| c.is_control() || c.is_whitespace())
}

pub(crate) fn top_predictions(
    mut values: Vec<TokenPrediction>,
    limit: usize,
) -> Result<Vec<TokenPrediction>, String> {
    if values.iter().any(|v| !v.score.is_finite()) {
        return Err("Nonfinite prediction".into());
    }
    let compare = |a: &TokenPrediction, b: &TokenPrediction| {
        b.score.total_cmp(&a.score).then(a.token.cmp(&b.token))
    };
    let limit = limit.min(100).min(values.len());
    if limit < values.len() {
        values.select_nth_unstable_by(limit, compare);
        values.truncate(limit);
    }
    values.sort_by(compare);
    Ok(values)
}

pub struct LanguageModel {
    pub(crate) neural_short_contexts: bool,
    config: Config,
    tokens: Vec<String>,
    tensors: HashMap<String, Tensor>,
    pub tokenizer: tokenizer::Tokenizer,
    cached_ids: Vec<usize>,
    cached_projection: Vec<f32>,
    spelling: Option<Box<NeuralLanguageModel>>,
}
impl LanguageModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::from_bytes_inner(bytes,true)
    }
    fn from_bytes_inner(bytes: &[u8],allow_spelling:bool) -> Result<Self,String> {
        if bytes.len() < 8 || &bytes[..4] != b"CNG1" {
            return Err("Invalid n-gram model magic".into());
        }
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let end = 8usize
            .checked_add(n)
            .filter(|&n| n <= bytes.len())
            .ok_or("Truncated metadata")?;
        let meta: Metadata = serde_json::from_slice(&bytes[8..end]).map_err(|e| e.to_string())?;
        let c = &meta.config;
        if meta.version != 1
            || !meta.remove_bos
            || meta.lookup_normalization != "english_lower_width_quotes_v1"
            || c.activation != "relu"
            || !c.layer_norm
            || c.vocab_size != meta.tokens.len()
            || !(3..=100000).contains(&c.vocab_size)
            || !(1..=512).contains(&c.embedding)
            || !(1..=1024).contains(&c.hidden)
            || c.residual_blocks > 8
            || c.left > 32
            || c.right > 16
            || c.left + c.right == 0
            || c.ffn_hidden > 2048
            || meta.tokens[..3] != ["[PAD]", "[BOS]", "[UNK]"]
        {
            return Err("Unsupported model/tokenizer configuration".into());
        }
        if meta
            .tokens
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != meta.tokens.len()
        {
            return Err("Duplicate vocabulary token".into());
        }
        let mut offset = end;
        let mut tensors = HashMap::new();
        for d in meta.tensors {
            let size = d
                .shape
                .iter()
                .try_fold(1usize, |n, &s| n.checked_mul(s))
                .ok_or("Tensor overflow")?;
            let (q, q16, scale, values, length) = match d.kind.as_str() {
                "row_i8" | "row_i16" if d.shape.len() == 2 => {
                    let scales = d.shape[0].checked_mul(4).ok_or("Scale overflow")?;
                    let width=if d.kind=="row_i16" {2}else{1};
                    let length = scales.checked_add(size.checked_mul(width).ok_or("Tensor overflow")?).ok_or("Tensor overflow")?;
                    let raw = bytes
                        .get(offset..offset.checked_add(length).ok_or("Tensor overflow")?)
                        .ok_or("Truncated weights")?;
                    let scale = floats(&raw[..scales])?;
                    if scale.iter().any(|&s| s <= 0.) {
                        return Err("Invalid quantization scales".into());
                    }
                    (
                        if width==1 {raw[scales..].iter().map(|&b|b as i8).collect()}else{Vec::new()},
                        if width==2 {raw[scales..].chunks_exact(2).map(|b|i16::from_le_bytes([b[0],b[1]])).collect()}else{Vec::new()},
                        scale,
                        Vec::new(),
                        length,
                    )
                }
                "f32" if d.shape.len() == 1 => {
                    let length = size.checked_mul(4).ok_or("Tensor overflow")?;
                    let raw = bytes
                        .get(offset..offset.checked_add(length).ok_or("Tensor overflow")?)
                        .ok_or("Truncated weights")?;
                    (Vec::new(), Vec::new(), Vec::new(), floats(raw)?, length)
                }
                _ => return Err("Unsupported tensor encoding".into()),
            };
            offset += length;
            if tensors
                .insert(
                    d.name,
                    Tensor {
                        shape: d.shape,
                        q,
                        q16,
                        scale,
                        values,
                    },
                )
                .is_some()
            {
                return Err("Duplicate tensor".into());
            }
        }
        if offset != bytes.len() {
            return Err("Trailing model bytes".into());
        }
        let spelling=if let Some(hex)=meta.spelling_cng_hex {
            if !allow_spelling || hex.len()>600_000 || hex.len()%2!=0 || !hex.is_ascii() {return Err("Invalid spelling head".into());}
            let mut payload=Vec::with_capacity(hex.len()/2);
            for pair in hex.as_bytes().chunks_exact(2) {
                let text=std::str::from_utf8(pair).map_err(|_|"Invalid spelling hex")?;
                payload.push(u8::from_str_radix(text,16).map_err(|_|"Invalid spelling hex")?);
            }
            let child=Self::from_bytes_inner(&payload,false)?;
            if child.config.vocab_size!=259 || child.config.right!=0 || child.config.left>32 || child.config.hidden>256 || child.config.embedding>32 || child.config.residual_blocks>2 {
                return Err("Unsupported spelling head dimensions".into());
            }
            Some(Box::new(NeuralLanguageModel::Dense(child)))
        } else {None};
        let tokenizer = tokenizer::Tokenizer::new(&meta.tokens);
        let model = Self {
            neural_short_contexts: meta.neural_short_contexts,
            config: meta.config,
            tokens: meta.tokens,
            tensors,
            tokenizer,
            cached_ids: Vec::new(),
            cached_projection: Vec::new(),
            spelling,
        };
        model.validate_shapes()?;
        Ok(model)
    }
    fn validate_shapes(&self) -> Result<(), String> {
        let c = &self.config;
        let mut expected = vec![
            ("embedding.weight".into(), vec![c.vocab_size, c.embedding]),
            ("bias".into(), vec![c.vocab_size]),
            (
                "hidden.weight".into(),
                vec![c.hidden, (c.left + c.right) * c.embedding],
            ),
            ("hidden.bias".into(), vec![c.hidden]),
            ("project.weight".into(), vec![c.embedding, c.hidden]),
            ("project.bias".into(), vec![c.embedding]),
            ("norm.weight".into(), vec![c.hidden]),
            ("norm.bias".into(), vec![c.hidden]),
        ];
        for i in 0..c.residual_blocks {
            let f = if c.ffn_hidden == 0 {
                c.hidden
            } else {
                c.ffn_hidden
            };
            for (name, shape) in [
                ("norm.weight", vec![c.hidden]),
                ("norm.bias", vec![c.hidden]),
                ("first.weight", vec![f, c.hidden]),
                ("first.bias", vec![f]),
                ("second.weight", vec![c.hidden, f]),
                ("second.bias", vec![c.hidden]),
            ] {
                expected.push((format!("blocks.{i}.{name}"), shape));
            }
        }
        if self.tensors.len() != expected.len() {
            return Err("Incorrect tensor count".into());
        }
        for (name, shape) in expected {
            let t = self
                .tensors
                .get(&name)
                .ok_or_else(|| format!("Missing {name}"))?;
            if t.shape != shape
                || (shape.len() == 2 && t.q.is_empty() && t.q16.is_empty())
                || (shape.len() == 1 && t.values.is_empty())
            {
                return Err(format!("Invalid shape/encoding for {name}"));
            }
        }
        Ok(())
    }
    fn projected(&mut self, ids: &[usize]) -> Result<&[f32], String> {
        let c = &self.config;
        if ids.len() != c.left + c.right || ids.iter().any(|&i| i >= c.vocab_size) {
            return Err("Invalid context IDs".into());
        }
        if self.cached_ids == ids {
            return Ok(&self.cached_projection);
        }
        let embedding = &self.tensors["embedding.weight"];
        let mut x = vec![0.; ids.len() * c.embedding];
        for (position, &id) in ids.iter().enumerate() {
            if id == 0 || id == 1 {
                continue;
            }
            for j in 0..c.embedding {
                let value=if embedding.q16.is_empty() {embedding.q[id*c.embedding+j] as f32}
                    else {embedding.q16[id*c.embedding+j] as f32};
                x[position * c.embedding + j] = value * embedding.scale[id];
            }
        }
        let t = &self.tensors;
        let mut h = t["hidden.weight"].linear(&x, &t["hidden.bias"]);
        for v in &mut h {
            *v = v.max(0.);
        }
        for i in 0..c.residual_blocks {
            let prefix = format!("blocks.{i}.");
            let n = norm(
                &h,
                &t[&(prefix.clone() + "norm.weight")],
                &t[&(prefix.clone() + "norm.bias")],
            );
            let mut z = t[&(prefix.clone() + "first.weight")]
                .linear(&n, &t[&(prefix.clone() + "first.bias")]);
            for v in &mut z {
                *v = v.max(0.);
            }
            let z =
                t[&(prefix.clone() + "second.weight")].linear(&z, &t[&(prefix + "second.bias")]);
            for (v, z) in h.iter_mut().zip(z) {
                *v += z;
            }
        }
        let h = norm(&h, &t["norm.weight"], &t["norm.bias"]);
        self.cached_projection = t["project.weight"].linear(&h, &t["project.bias"]);
        self.cached_ids = ids.to_vec();
        Ok(&self.cached_projection)
    }
    /// Only candidate rows are scored; softmax is unnecessary for ordering.
    pub fn scores(&mut self, ids: &[usize], candidate_ids: &[usize]) -> Result<Vec<f32>, String> {
        self.projected(ids)?;
        let t = &self.tensors["embedding.weight"];
        let d = self.config.embedding;
        let bias=&self.tensors["bias"].values;
        candidate_ids
            .iter()
            .map(|&id| {
                if id < 3 || id >= self.config.vocab_size {
                    return Err("Invalid candidate ID".into());
                }
                Ok(
                    (if t.q16.is_empty() {dot(&t.q[id*d..(id+1)*d],&self.cached_projection)}
                     else {dot16(&t.q16[id*d..(id+1)*d],&self.cached_projection)}) * t.scale[id]
                        + bias[id],
                )
            })
            .collect()
    }
    /// Predict the next token with no right context. At most 100 results.
    /// This is the raw neural API; use HybridLanguageModel for count fallback.
    pub fn predict_next(
        &mut self,
        left: &str,
        limit: usize,
    ) -> Result<Vec<TokenPrediction>, String> {
        self.predict(left, "", limit)
    }
    /// Predict an insertion using optional left and right context.
    pub fn predict(
        &mut self,
        left: &str,
        right: &str,
        limit: usize,
    ) -> Result<Vec<TokenPrediction>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let ids = self.context(left, right);
        let targets: Vec<_> = (3..self.tokens.len())
            .filter(|&id| displayable(&self.tokens[id]))
            .collect();
        let scores = self.scores(&ids, &targets)?;
        top_predictions(
            targets
                .into_iter()
                .zip(scores)
                .map(|(id, score)| TokenPrediction {
                    token: self.tokens[id].clone(),
                    token_id: Some(id),
                    score,
                })
                .collect(),
            limit,
        )
    }
    pub fn context(&self, left: &str, right: &str) -> Vec<usize> {
        self.tokenizer
            .context(left, right, self.config.left, self.config.right)
    }
    pub fn context_lengths(&self) -> (usize, usize) {
        (self.config.left, self.config.right)
    }
    /// Legacy exports preserve empty-fragment order; learned short-context
    /// exports also rank empty input through the same neural network.
    /// Unsupported characters keep their original slots; no candidates are lost.
    pub fn rank(
        &mut self,
        left: &str,
        right: &str,
        candidates: &[String],
    ) -> Result<Vec<usize>, String> {
        let ids = self.context(left, right);
        let mut order: Vec<_> = (0..candidates.len()).collect();
        if !self.neural_short_contexts && ids.iter().all(|&id| id == 0) {
            return Ok(order);
        }
        let supported: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| self.tokenizer.id(c).filter(|&id| id >= 3).map(|id| (i, id)))
            .collect();
        if supported.len() < 2 {
            return Ok(order);
        }
        let scores = self.scores(&ids, &supported.iter().map(|p| p.1).collect::<Vec<_>>())?;
        if scores.iter().any(|s| !s.is_finite()) {
            return Err("Nonfinite prediction".into());
        }
        let mut sorted: Vec<_> = supported
            .iter()
            .enumerate()
            .map(|(i, p)| (p.0, scores[i]))
            .collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for ((slot, _), (index, _)) in supported.iter().zip(sorted) {
            order[*slot] = index;
        }
        Ok(order)
    }
}

#[cfg(test)]
mod prediction_tests {
    use super::*;
    #[test]
    fn int16_dot_matches_scalar_for_signed_values_and_simd_tails() {
        for size in [0,1,7,8,9,17,128] {
            let q:Vec<i16>=(0..size).map(|i|if i%2==0 {32767-(i as i16)*17}else{-32768+(i as i16)*19}).collect();
            let x:Vec<f32>=(0..size).map(|i|(i as f32-13.)/31.).collect();
            let expected:f32=q.iter().zip(&x).map(|(&a,&b)|a as f32*b).sum();
            let actual=dot16(&q,&x);
            assert!((actual-expected).abs()<=1e-4*(1.+expected.abs()),"{size}: {actual} != {expected}");
        }
    }
    #[test]
    fn top_k_is_sorted_bounded_and_keeps_punctuation() {
        let rows = || {
            ["b", "a", "c"]
                .into_iter()
                .map(|token| TokenPrediction {
                    token: token.into(),
                    token_id: None,
                    score: 1.,
                })
                .collect()
        };
        assert_eq!(
            top_predictions(rows(), 2)
                .unwrap()
                .iter()
                .map(|p| p.token.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert!(top_predictions(rows(), 0).unwrap().is_empty());
        assert_eq!(top_predictions(rows(), 99).unwrap().len(), 3);
        for t in ["[", "<", "。", "didn't"] {
            assert!(displayable(t));
        }
        for t in ["", "<blank>", "[PAD]", "[UNK]", "[BOS]", "\n"] {
            assert!(!displayable(t));
        }
    }
}
