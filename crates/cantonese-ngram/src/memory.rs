//! Learned complete-context features with a frozen neural base. No count scores.
use crate::{
    displayable, floats, top_predictions, Description, LanguageModel, Tensor, TokenPrediction,
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Metadata {
    version: u32,
    architecture: String,
    tokens: Vec<String>,
    lookup_normalization: String,
    keys: Vec<[usize; 12]>,
    parent_share: f32,
    base_bytes: usize,
    tensors: Vec<Description>,
}

pub struct MemoryLanguageModel {
    base: LanguageModel,
    tokens: Vec<String>,
    tokenizer: crate::tokenizer::Tokenizer,
    lookup: HashMap<[usize; 12], usize>,
    tensors: HashMap<String, Tensor>,
    parent_share: f32,
    cached_ids: Vec<usize>,
    cached_scores: Vec<f32>,
}

fn log_normalize(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let partition = maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f32>().ln();
    for value in values {
        *value -= partition;
    }
}

fn log_add(a: f32, b: f32) -> f32 {
    let maximum = a.max(b);
    maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
}

impl MemoryLanguageModel {
    pub fn vocabulary_size(&self) -> usize { self.tokens.len() }
    pub fn token(&self, id: usize) -> Option<&str> { self.tokens.get(id).map(String::as_str) }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 || &bytes[..4] != b"CNG2" {
            return Err("Invalid memory model magic".into());
        }
        let size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let mut offset = 8usize
            .checked_add(size)
            .filter(|&n| n <= bytes.len())
            .ok_or("Truncated memory metadata")?;
        let meta: Metadata =
            serde_json::from_slice(&bytes[8..offset]).map_err(|e| e.to_string())?;
        if meta.version != 2
            || meta.architecture != "learned_ngram_feature_memory_v1"
            || meta.lookup_normalization != "english_lower_width_quotes_v1"
            || !(3..=100000).contains(&meta.tokens.len())
            || meta.tokens[..3] != ["[PAD]", "[BOS]", "[UNK]"]
            || meta.keys.is_empty()
            || meta.keys.len() > 250000
            || !meta.parent_share.is_finite()
            || !(0.0..1.0).contains(&meta.parent_share)
            || meta.parent_share == 0.0
        {
            return Err("Unsupported memory configuration".into());
        }
        if meta
            .tokens
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != meta.tokens.len()
        {
            return Err("Duplicate memory vocabulary".into());
        }
        let end = offset
            .checked_add(meta.base_bytes)
            .filter(|&n| n <= bytes.len())
            .ok_or("Truncated base model")?;
        let base = LanguageModel::from_bytes(&bytes[offset..end])?;
        offset = end;
        if base.config.left != 8
            || base.config.right != 4
            || base.config.hidden != 256
            || base.config.embedding != 128
            || !matches!(base.config.ffn_hidden, 0 | 256)
            || meta.tokens.len() < base.tokens.len()
            || meta.tokens[..base.tokens.len()] != base.tokens
        {
            return Err("Memory/base vocabulary or width mismatch".into());
        }
        let mut lookup = HashMap::new();
        for (index, key) in meta.keys.iter().enumerate() {
            if key.iter().any(|&id| id >= meta.tokens.len() || id == 1)
                || lookup.insert(*key, index).is_some()
            {
                return Err("Invalid or duplicate memory context".into());
            }
        }
        let mut tensors = HashMap::new();
        for descriptor in meta.tensors {
            let size = descriptor
                .shape
                .iter()
                .try_fold(1usize, |n, &s| n.checked_mul(s))
                .ok_or("Memory tensor overflow")?;
            let (q, scale, values, length) = match descriptor.kind.as_str() {
                "row_i8" if descriptor.shape.len() == 2 => {
                    let scales = descriptor.shape[0]
                        .checked_mul(4)
                        .ok_or("Memory scale overflow")?;
                    let length = scales.checked_add(size).ok_or("Memory tensor overflow")?;
                    let end = offset.checked_add(length).ok_or("Memory tensor overflow")?;
                    let raw = bytes.get(offset..end).ok_or("Truncated memory weights")?;
                    let scale = floats(&raw[..scales])?;
                    if scale.iter().any(|&v| v <= 0.) {
                        return Err("Invalid memory scales".into());
                    }
                    (
                        raw[scales..].iter().map(|&v| v as i8).collect(),
                        scale,
                        Vec::new(),
                        length,
                    )
                }
                "f32" if descriptor.shape.len() == 1 => {
                    let length = size.checked_mul(4).ok_or("Memory tensor overflow")?;
                    let end = offset.checked_add(length).ok_or("Memory tensor overflow")?;
                    (
                        Vec::new(),
                        Vec::new(),
                        floats(bytes.get(offset..end).ok_or("Truncated memory weights")?)?,
                        length,
                    )
                }
                _ => return Err("Unsupported memory tensor encoding".into()),
            };
            offset += length;
            if tensors
                .insert(
                    descriptor.name,
                    Tensor {
                        shape: descriptor.shape,
                        q,
                        scale,
                        values,
                    },
                )
                .is_some()
            {
                return Err("Duplicate memory tensor".into());
            }
        }
        if offset != bytes.len() {
            return Err("Trailing memory model bytes".into());
        }
        let hidden_width = tensors
            .get("hidden.weight")
            .and_then(|tensor| tensor.shape.first())
            .copied()
            .filter(|width| matches!(width, 128 | 256))
            .ok_or("Memory hidden width must be128or256")?;
        let expected = [
            ("features.weight", vec![lookup.len(), 128]),
            ("hidden.weight", vec![hidden_width, 128]),
            ("hidden.bias", vec![hidden_width]),
            ("project.weight", vec![128, hidden_width]),
            ("project.bias", vec![128]),
            ("output.weight", vec![meta.tokens.len(), 128]),
            ("output.bias", vec![meta.tokens.len()]),
        ];
        if tensors.len() != expected.len() {
            return Err("Incorrect memory tensor count".into());
        }
        for (name, shape) in expected {
            let tensor = tensors
                .get(name)
                .ok_or_else(|| format!("Missing memory {name}"))?;
            if tensor.shape != shape
                || (shape.len() == 2 && tensor.q.is_empty())
                || (shape.len() == 1 && tensor.values.is_empty())
            {
                return Err(format!("Invalid memory tensor {name}"));
            }
        }
        Ok(Self {
            tokenizer: crate::tokenizer::Tokenizer::new(&meta.tokens),
            base,
            tokens: meta.tokens,
            lookup,
            tensors,
            parent_share: meta.parent_share,
            cached_ids: Vec::new(),
            cached_scores: Vec::new(),
        })
    }

    pub fn context(&self, left: &str, right: &str) -> Vec<usize> {
        self.tokenizer.context(left, right, 8, 4)
    }

    pub fn token_id(&self, token: &str) -> Option<usize> {
        self.tokenizer.id(token)
    }

    pub fn supports_candidate(&self, ids: &[usize], id: usize) -> bool {
        id >= 3
            && id < self.tokens.len()
            && (id < self.base.tokens.len()
                || <[usize; 12]>::try_from(ids)
                    .ok()
                    .map(|key| self.lookup.contains_key(&key))
                    .unwrap_or(false))
    }

    pub fn supports_context(&self, ids: &[usize]) -> bool {
        if !ids.iter().any(|&id| id >= 3) {
            return false;
        }
        <[usize; 12]>::try_from(ids)
            .ok()
            .map(|key| self.lookup.contains_key(&key))
            .unwrap_or(false)
            || ids.iter().any(|&id| id >= 3 && id < self.base.tokens.len())
    }

    /// Preserve dictionary slots for unsupported candidates and stable ties.
    pub fn rank(
        &mut self,
        left: &str,
        right: &str,
        candidates: &[String],
    ) -> Result<Vec<usize>, String> {
        let ids = self.context(left, right);
        let mut order: Vec<_> = (0..candidates.len()).collect();
        let supported: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                self.token_id(c)
                    .filter(|&id| self.supports_candidate(&ids, id))
                    .map(|id| (i, id))
            })
            .collect();
        if supported.len() < 2 {
            return Ok(order);
        }
        let scores = self.scores(&ids, &supported.iter().map(|p| p.1).collect::<Vec<_>>())?;
        if scores.iter().any(|s| !s.is_finite()) {
            return Err("Nonfinite memory prediction".into());
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

    /// Normalized neural log probabilities, including the learned empty context.
    pub fn scores(&mut self, ids: &[usize], candidate_ids: &[usize]) -> Result<Vec<f32>, String> {
        if ids.len() != 12
            || ids.iter().any(|&id| id >= self.tokens.len())
            || candidate_ids
                .iter()
                .any(|&id| id < 3 || id >= self.tokens.len())
        {
            return Err("Invalid memory context/candidate IDs".into());
        }
        if self.cached_ids != ids {
            let old = self.base.tokens.len();
            let known: Vec<_> = ids
                .iter()
                .map(|&id| if id < old { id } else { 2 })
                .collect();
            let mut scores = self.base.scores(&known, &(3..old).collect::<Vec<_>>())?;
            log_normalize(&mut scores);
            let mut full = vec![-1e4; self.tokens.len()];
            full[3..old].copy_from_slice(&scores);
            let key: [usize; 12] = ids.try_into().unwrap();
            if let Some(&index) = self.lookup.get(&key) {
                let t = &self.tensors;
                let feature = &t["features.weight"];
                let mut h: Vec<_> = feature.q[index * 128..(index + 1) * 128]
                    .iter()
                    .map(|&v| v as f32 * feature.scale[index])
                    .collect();
                let mut hidden = t["hidden.weight"].linear(&h, &t["hidden.bias"]);
                for value in &mut hidden {
                    *value = value.max(0.);
                }
                let residual = t["project.weight"].linear(&hidden, &t["project.bias"]);
                for (value, residual) in h.iter_mut().zip(residual) {
                    *value += residual;
                }
                let mut memory = t["output.weight"].linear(&h, &t["output.bias"]);
                memory[..3].fill(-1e4);
                log_normalize(&mut memory);
                for (value, memory) in full.iter_mut().zip(memory) {
                    *value = log_add(
                        *value + self.parent_share.ln(),
                        memory + (1. - self.parent_share).ln(),
                    );
                }
            }
            self.cached_scores = full;
            self.cached_ids = ids.to_vec();
        }
        Ok(candidate_ids
            .iter()
            .map(|&id| self.cached_scores[id])
            .collect())
    }

    pub fn predict(
        &mut self,
        left: &str,
        right: &str,
        limit: usize,
    ) -> Result<Vec<TokenPrediction>, String> {
        let ids = self.context(left, right);
        let targets: Vec<_> = (3..self.tokens.len())
            .filter(|&id| displayable(&self.tokens[id]) && self.supports_candidate(&ids, id))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_missing_and_truncated_headers() {
        for bytes in [b"".as_slice(), b"CNG1", b"CNG2\xff\xff\xff\xff"] {
            assert!(MemoryLanguageModel::from_bytes(bytes).is_err());
        }
    }
    #[test]
    fn stable_probability_operations() {
        let mut values = vec![10000., 10000., 9999.];
        log_normalize(&mut values);
        assert!((values.iter().map(|v| v.exp()).sum::<f32>() - 1.).abs() < 0.001);
        assert!((log_add(0.2f32.ln(), 0.3f32.ln()).exp() - 0.5).abs() < 1e-6);
    }
}
