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
