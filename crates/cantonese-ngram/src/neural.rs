//! Format-independent neural inference for the keyboard integrations.
use crate::{memory::MemoryLanguageModel, LanguageModel, TokenPrediction};

pub enum NeuralLanguageModel {
    Dense(LanguageModel),
    Memory(MemoryLanguageModel),
}
impl From<LanguageModel> for NeuralLanguageModel {
    fn from(model: LanguageModel) -> Self {
        Self::Dense(model)
    }
}
impl NeuralLanguageModel {
    pub fn has_byte_fallback(&self)->bool {
        match self {Self::Dense(m)=>m.tokenizer.has_byte_fallback(),Self::Memory(_)=>false}
    }
    pub fn tokenize(&self,text:&str)->Vec<usize> {
        match self { Self::Dense(m)=>m.tokenizer.tokenize(text),Self::Memory(m)=>m.tokenize(text) }
    }
    pub fn tokenize_with_units(&self,text:&str)->(Vec<usize>,usize) {
        match self {
            Self::Dense(m)=>m.tokenizer.tokenize_with_units(text),
            // Legacy memory tokenization has no byte expansion.
            Self::Memory(m)=>{let tokens=m.tokenize(text);let units=tokens.len();(tokens,units)}
        }
    }
    pub fn token(&self, id: usize) -> Option<&str> {
        match self { Self::Dense(m) => m.tokens.get(id).map(String::as_str), Self::Memory(m) => m.token(id) }
    }
    pub fn vocabulary_size(&self) -> usize {
        match self { Self::Dense(m) => m.tokens.len(), Self::Memory(m) => m.vocabulary_size() }
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.starts_with(b"CNG2") {
            MemoryLanguageModel::from_bytes(bytes).map(Self::Memory)
        } else {
            LanguageModel::from_bytes(bytes).map(Self::Dense)
        }
    }
    pub fn minimum_context_tokens(&self) -> usize {
        match self {
            Self::Dense(m) => if m.neural_short_contexts { 0 } else { 2 },
            Self::Memory(_) => 1,
        }
    }
    pub fn supports_context(&self, ids: &[usize]) -> bool {
        match self {
            Self::Dense(m) => m.neural_short_contexts || ids.iter().filter(|&&id| id >= 3).count() >= 2,
            Self::Memory(m) => m.supports_context(ids),
        }
    }
    pub fn token_id(&self, token: &str) -> Option<usize> {
        match self {
            Self::Dense(m) => m.tokenizer.id(token),
            Self::Memory(m) => m.token_id(token),
        }
    }
    pub fn supports_candidate(&self, ids: &[usize], id: usize) -> bool {
        match self {
            Self::Dense(m) => id >= 3 && id < m.tokens.len(),
            Self::Memory(m) => m.supports_candidate(ids, id),
        }
    }
    pub fn context(&self, left: &str, right: &str) -> Vec<usize> {
        match self {
            Self::Dense(m) => m.context(left, right),
            Self::Memory(m) => m.context(left, right),
        }
    }
    pub fn context_lengths(&self) -> (usize, usize) {
        match self {
            Self::Dense(m) => m.context_lengths(),
            Self::Memory(_) => (8, 4),
        }
    }
    pub fn scores(&mut self, ids: &[usize], candidates: &[usize]) -> Result<Vec<f32>, String> {
        match self {
            Self::Dense(m) => m.scores(ids, candidates),
            Self::Memory(m) => m.scores(ids, candidates),
        }
    }
    pub fn predict(
        &mut self,
        left: &str,
        right: &str,
        limit: usize,
    ) -> Result<Vec<TokenPrediction>, String> {
        match self {
            Self::Dense(m) => m.predict(left, right, limit),
            Self::Memory(m) => m.predict(left, right, limit),
        }
    }
    pub fn rank(
        &mut self,
        left: &str,
        right: &str,
        candidates: &[String],
    ) -> Result<Vec<usize>, String> {
        match self {
            Self::Dense(m) => m.rank(left, right, candidates),
            Self::Memory(m) => m.rank(left, right, candidates),
        }
    }
}

/// Exact full-vocabulary-normalized sequence evidence, independent of candidate
/// pool composition. Mean scores are supplied alongside sums and token counts;
/// the decision layer must calibrate English words versus Han/byte sequences.
#[derive(serde::Serialize, Debug)]
#[serde(rename_all="camelCase")]
pub struct SequenceEvidence {
    pub log_probability: f32,
    pub mean_log_probability: f32,
    /// Sequence log probability per word/Han/grapheme before byte fallback.
    /// Keep the encoded-token mean above for existing calibrated callers.
    pub unit_log_probability: f32,
    pub logical_units: usize,
    pub tokens: usize,
    pub supported: bool,
}
pub fn score_sequences(model:&mut NeuralLanguageModel,left:&str,right:&str,candidates:&[String])
    ->Result<Vec<SequenceEvidence>,String> {
    if candidates.len()>128 {return Err("Too many sequence candidates".into());}
    let (nleft,nright)=model.context_lengths();
    let initial=model.context(left,right);
    let all:Vec<_>=(3..model.vocabulary_size()).collect();
    let mut cache=std::collections::HashMap::<Vec<usize>,Vec<f32>>::new();
    let mut result=Vec::with_capacity(candidates.len());
    for text in candidates {
        let (tokens,logical_units)=model.tokenize_with_units(text);
        let mut context=initial.clone();let mut total=0.;let mut supported=!tokens.is_empty() && tokens.len()<=64;
        if supported {
            for &token in &tokens {
                if token<3 || !model.supports_candidate(&context,token) {supported=false;break;}
                if !cache.contains_key(&context) {
                    // Bound temporary probability tables independently of how
                    // many unseen-word byte sequences a caller supplies.
                    if cache.len()>=16 {cache.clear();}
                    let scores=model.scores(&context,&all)?;
                    let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
                    let z=max+scores.iter().map(|v|(v-max).exp()).sum::<f32>().ln();
                    cache.insert(context.clone(),scores.iter().map(|v|v-z).collect());
                }
                total+=cache[&context][token-3];
                if nleft>0 {
                    context[..nleft].rotate_left(1);context[nleft-1]=token;
                }
                debug_assert_eq!(context.len(),nleft+nright);
            }
        }
        // Finite sentinel plus an explicit support bit; never mistake an
        // unsupported candidate's sentinel for a learned LM probability.
        result.push(SequenceEvidence {log_probability:if supported {total} else {-1e4},
            mean_log_probability:if supported {total/tokens.len() as f32} else {-1e4},
            unit_log_probability:if supported {total/logical_units.max(1) as f32} else {-1e4},
            logical_units,tokens:tokens.len(),supported});
    }
    Ok(result)
}
