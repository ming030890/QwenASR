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
    #[serde(skip_serializing_if="Option::is_none")]
    pub spelling_log_probability: Option<f32>,
    #[serde(skip_serializing_if="Option::is_none")]
    pub spelling_characters: Option<usize>,
}
pub fn score_sequences(model:&mut NeuralLanguageModel,left:&str,right:&str,candidates:&[String])
    ->Result<Vec<SequenceEvidence>,String> {
    if candidates.len()>128 {return Err("Too many sequence candidates".into());}
    let (nleft,nright)=model.context_lengths();
    let initial=model.context(left,right);
    let all:Vec<_>=(3..model.vocabulary_size()).collect();
    // Group every requested token by its exact context before inference. The old
    // 16-distribution cache cleared the common initial context whenever many
    // byte-fallback spellings filled it, repeating full-vocabulary projections.
    // Only one full distribution is live now; each distinct prefix is scored
    // once, with bounded O(128*64) context/target metadata.
    let mut indices=std::collections::HashMap::<Vec<usize>,usize>::new();
    let mut groups:Vec<(Vec<usize>,Vec<(usize,usize,usize)>)>=Vec::new();
    let mut paths:Vec<Vec<f32>>=Vec::with_capacity(candidates.len());
    let mut result=Vec::with_capacity(candidates.len());
    for (candidate,text) in candidates.iter().enumerate() {
        let (tokens,logical_units)=model.tokenize_with_units(text);
        let mut context=initial.clone();let mut supported=!tokens.is_empty() && tokens.len()<=64;
        paths.push(vec![0.;tokens.len().min(64)]);
        if supported {
            for (position,&token) in tokens.iter().enumerate() {
                if token<3 || !model.supports_candidate(&context,token) {supported=false;break;}
                let next=groups.len();
                let group=*indices.entry(context.clone()).or_insert_with(|| {
                    groups.push((context.clone(),Vec::new()));next
                });
                groups[group].1.push((candidate,position,token));
                if nleft>0 {
                    context[..nleft].rotate_left(1);context[nleft-1]=token;
                }
                debug_assert_eq!(context.len(),nleft+nright);
            }
        }
        // Finite sentinel plus an explicit support bit; never mistake an
        // unsupported candidate's sentinel for a learned LM probability.
        result.push(SequenceEvidence {log_probability:-1e4,
            mean_log_probability:-1e4,
            unit_log_probability:-1e4,
            logical_units,tokens:tokens.len(),supported,spelling_log_probability:None,spelling_characters:None});
    }
    for (context,targets) in groups {
        let scores=model.scores(&context,&all)?;
        let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let z=max+scores.iter().map(|v|(v-max).exp()).sum::<f32>().ln();
        for (candidate,position,token) in targets {paths[candidate][position]=scores[token-3]-z;}
    }
    for (row,path) in result.iter_mut().zip(paths) {
        if row.supported {
            // Preserve the original left-to-right f32 addition order exactly.
            let total=path.into_iter().fold(0.,|sum,value|sum+value);
            row.log_probability=total;row.mean_log_probability=total/row.tokens as f32;
            row.unit_log_probability=total/row.logical_units.max(1) as f32;
        }
    }
    if let NeuralLanguageModel::Dense(parent)=model {
        if let Some(head)=parent.spelling.as_deref_mut() {
            let mut marked=Vec::new();let mut spans=Vec::new();
            for text in candidates {
                let start=marked.len();let mut valid=!text.trim().is_empty();
                for word in text.split_whitespace() {
                    if word.len()>63 || !word.bytes().all(|c|c.is_ascii_lowercase() || c==b'\'') {valid=false;break;}
                    marked.push(format!("{word}\0"));
                }
                if valid {spans.push(Some(start..marked.len()));}else{marked.truncate(start);spans.push(None);}
            }
            let mut spelling=Vec::new();
            for chunk in marked.chunks(128) {spelling.extend(score_sequences(head,"","",chunk)?);}
            for (row,span) in result.iter_mut().zip(spans) {
                if let Some(span)=span {
                    let part=&spelling[span];
                    if !part.is_empty() && part.iter().all(|r|r.supported) {
                        row.spelling_log_probability=Some(part.iter().map(|r|r.log_probability).sum());
                        row.spelling_characters=Some(part.iter().map(|r|r.tokens).sum());
                    }
                }
            }
        }
    }
    Ok(result)
}
