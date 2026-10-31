//! Shared V2 causal CNN. INT8 matrices stay quantized; clients share weights.
//! Context is BOS + the last 32 canonical tokens, exactly as in V2 evaluation.
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Deserialize)]
struct Description { dtype: String, shape: Vec<usize>, offset: usize, bytes: usize, scale: f32 }
#[derive(Deserialize)]
struct Metadata {
    format: String, vocab_size: usize, embed_dim: usize, hidden_dim: usize, ffn_dim: usize,
    dilations: Vec<usize>, tensors: HashMap<String, Description>, tokens: Vec<String>,
    tokenizer: String, english_word_ids: Vec<usize>,
}
struct Tensor { shape: Vec<usize>, q: Vec<i8>, values: Vec<f32>, scale: f32 }
impl Tensor {
    fn row(&self, row: usize) -> Vec<f32> {
        let n=self.shape[1];
        self.q[row*n..(row+1)*n].iter().map(|&v|v as f32*self.scale).collect()
    }
    fn linear(&self, x: &[f32], bias: &Tensor) -> Vec<f32> {
        debug_assert_eq!(self.shape[1], x.len());
        let mut values = crate::dot_matrix(&self.q, x);
        for (score, &bias) in values.iter_mut().zip(&bias.values) { *score = *score*self.scale+bias; }
        values
    }
}
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnglishSuggestion {
    pub word: String, pub score: f32, pub edits: u8, pub completed_chars: usize,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApostropheEvidence {
    pub word: String, pub features: [f64; 3], pub score: f32,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnglishSuggestions {
    pub candidates: Vec<EnglishSuggestion>, pub correction: Option<String>, pub known: bool,
    pub literal_score: Option<f32>,
    pub apostrophe_evidence: Vec<ApostropheEvidence>,
}
/// Eligible spellings and split forms, enumerated without inference.
pub struct EnglishCandidatePool {
    input: String,
    original: String,
    pool: Vec<(String, Vec<usize>, u8, usize)>,
    known: bool,
    literal_ids: Vec<usize>,
    merge_splits: Vec<(usize,usize)>,
    apostrophes: Vec<&'static crate::cnn_spelling::ApostropheForm>,
}

struct Weights {
    raw: Box<[u8]>, tokens: Vec<String>, ids: HashMap<String,usize>, english: Regex,
    spellings: Vec<(String, Vec<usize>)>, spelling_index: crate::spelling_index::SpellingIndex, words: HashSet<usize>, embedding: usize, ffn: usize, dilations: Vec<usize>,
    tensors: HashMap<String,Tensor>,
    channel: crate::english_channel::Decoder,
}
static SHARED: OnceLock<Mutex<Weak<Weights>>> = OnceLock::new();

// Exact causal state at a scored prefix. Eight checkpoints cost at most 1.46 MiB
// per inference instance with this model; weights remain shared across workers.
const PREFIX_CHECKPOINTS: usize = 8;
struct PrefixCheckpoint {
    prefix: Vec<usize>,
    buffers: Vec<Vec<f32>>,
    projected: Vec<f32>,
}

pub struct CnnLanguageModel {
    weights: Arc<Weights>, buffers: Vec<Vec<f32>>, prefix: Vec<usize>, projected: Vec<f32>, include_bos: bool,
    checkpoints: VecDeque<PrefixCheckpoint>,
    normalized_projections: VecDeque<(Vec<usize>, Vec<f32>, f32)>,
}
impl CnnLanguageModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self,String> {
        let mut cache=SHARED.get_or_init(||Mutex::new(Weak::new())).lock().map_err(|_|"CNN cache poisoned")?;
        let weights=match cache.upgrade().filter(|w| w.raw.as_ref()==bytes) {
            Some(w)=>w,
            None=>{let w=Arc::new(Self::parse(bytes)?);*cache=Arc::downgrade(&w);w}
        };
        let buffers=weights.dilations.iter().map(|&d|vec![0.;2*d*weights.ffn]).collect();
        Ok(Self {weights,buffers,prefix:Vec::new(),projected:Vec::new(),include_bos:true,checkpoints:VecDeque::new(),normalized_projections:VecDeque::new()})
    }
    fn parse(bytes:&[u8])->Result<Weights,String> {
        if bytes.get(..8)!=Some(b"KBCNN01\0") || bytes.len()<12 {return Err("Invalid CNN header".into());}
        let length=u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let end=12usize.checked_add(length).filter(|&n|n<=bytes.len()).ok_or("Truncated CNN metadata")?;
        let meta:Metadata=serde_json::from_slice(&bytes[12..end]).map_err(|e|e.to_string())?;
        let base=(end+63)/64*64;
        if meta.format!="KBCNN_INT8_V1" || meta.tokenizer!="hybrid-word-first-utf8-v2" ||
           meta.vocab_size!=meta.tokens.len() || !(264..=65536).contains(&meta.vocab_size) ||
           meta.embed_dim!=128 || meta.hidden_dim!=384 || meta.ffn_dim!=768 || meta.dilations!=[1,2,4,8,16] ||
           meta.tokens[..8]!=["[PAD]","[BOS]","[EOS]","[UNK]","[SEP]","[SUGGESTION]","[HOLE]","[INFILL]"] {
            return Err("Unsupported CNN configuration".into());
        }
        let ids:HashMap<_,_>=meta.tokens.iter().enumerate().map(|(i,t)|(t.clone(),i)).collect();
        if ids.len()!=meta.tokens.len() || (0..256).any(|b|!ids.contains_key(&format!("[byte:{b:02x}]"))) {
            return Err("Invalid CNN vocabulary".into());
        }
        let mut shapes:HashMap<String,Vec<usize>>=HashMap::from([
            ("embedding.weight".into(),vec![meta.vocab_size,128]),("output_bias".into(),vec![meta.vocab_size]),
            ("in_proj.weight".into(),vec![384,128]),("in_proj.bias".into(),vec![384]),
            ("out_proj.weight".into(),vec![128,384]),("out_proj.bias".into(),vec![128]),
            ("final_norm.weight".into(),vec![384]),("final_norm.bias".into(),vec![384]),
        ]);
        for i in 0..5 {
            for (name,shape) in [("norm.weight",vec![384]),("norm.bias",vec![384]),
                ("pw_in.weight",vec![768,384]),("pw_in.bias",vec![768]),
                ("pw_out.weight",vec![384,768]),("pw_out.bias",vec![384]),
                ("dw_conv.weight",vec![768,1,3]),("dw_conv.bias",vec![768])] {
                shapes.insert(format!("blocks.{i}.{name}"),shape);
            }
        }
        if shapes.len()!=meta.tensors.len() {return Err("Incorrect CNN tensor count".into());}
        let mut tensors=HashMap::new();let mut ranges=Vec::new();
        for (name,shape) in shapes {
            let desc=meta.tensors.get(&name).ok_or_else(||format!("Missing {name}"))?;
            let quantized=shape.len()==2;
            if desc.shape!=shape || desc.dtype!=if quantized {"int8"} else {"float32"} ||
               !desc.scale.is_finite() || desc.scale<=0. || desc.offset%16!=0 {
                return Err(format!("Invalid CNN tensor: {name}"));
            }
            let elements:usize=shape.iter().product();
            if desc.bytes!=elements*if quantized {1} else {4} {return Err("Invalid CNN tensor length".into());}
            let start=base.checked_add(desc.offset).ok_or("CNN offset overflow")?;
            let stop=start.checked_add(desc.bytes).ok_or("CNN size overflow")?;
            let raw=bytes.get(start..stop).ok_or("Truncated CNN weights")?;
            ranges.push((start,stop));
            tensors.insert(name,Tensor {shape,scale:desc.scale,
                q:if quantized {raw.iter().map(|&v|v as i8).collect()}else{Vec::new()},
                values:if quantized {Vec::new()}else{crate::floats(raw)?}});
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|w|w[0].1>w[1].0) || ranges.last().map(|r|r.1)!=Some(bytes.len()) {
            return Err("Overlapping or trailing CNN weights".into());
        }
        if meta.english_word_ids.iter().any(|&id|id>=meta.vocab_size) {return Err("Invalid English word IDs".into());}
        let mut spellings: std::collections::BTreeMap<String,Vec<usize>>=std::collections::BTreeMap::new();
        for &id in &meta.english_word_ids {
            if let Some(word)=crate::cnn_spelling::canonical(&meta.tokens[id]) {
                spellings.entry(word).or_default().push(id);
            }
        }
        if let Some(&id) = ids.get("[byte:61]") { spellings.entry("a".into()).or_default().push(id); }
        if let Some(&id) = ids.get("[byte:41]") { spellings.entry("a".into()).or_default().push(id); }
        if let Some(&id) = ids.get("[byte:69]") { spellings.entry("i".into()).or_default().push(id); }
        if let Some(&id) = ids.get("[byte:49]") { spellings.entry("i".into()).or_default().push(id); }
        let spellings: Vec<_> = spellings.into_iter().collect();
        let spelling_index = crate::spelling_index::SpellingIndex::new(spellings.iter().map(|(w,_)|w.as_str()));
        let channel = crate::english_channel::Decoder::new(&spellings, &meta.english_word_ids);
        Ok(Weights {raw:bytes.into(),tokens:meta.tokens,ids,spellings,spelling_index,words:meta.english_word_ids.into_iter().collect(),
            english:Regex::new(r"^[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*").unwrap(),
            embedding:meta.embed_dim,ffn:meta.ffn_dim,dilations:meta.dilations,tensors,channel})
    }
    pub fn token(&self,id:usize)->Option<&str>{self.weights.tokens.get(id).map(String::as_str)}
    pub fn token_id(&self,text:&str)->Option<usize>{self.weights.ids.get(text).copied()}
    pub fn vocabulary_size(&self)->usize{self.weights.tokens.len()}
    pub fn tokenize_with_units(&self,text:&str)->(Vec<usize>,usize) {
        let mut output=Vec::new();let mut units=0;let mut offset=0;
        while offset<text.len() {
            let c=text[offset..].chars().next().unwrap();
            if c.is_whitespace() {offset+=c.len_utf8();continue;}
            let end=self.weights.english.find(&text[offset..]).map(|m|offset+m.end()).unwrap_or(offset+c.len_utf8());
            let unit=&text[offset..end];units+=1;
            if let Some(&id)=self.weights.ids.get(unit) {output.push(id)} else {
                output.extend(unit.as_bytes().iter().map(|b|self.weights.ids[&format!("[byte:{b:02x}]")]));
            }
            offset=end;
        }
        (output,units)
    }
    pub fn context(&self,left:&str)->Vec<usize> {
        let ids=self.tokenize_with_units(left).0;
        let n=ids.len().min(32);let mut result=vec![0;32-n];result.extend_from_slice(&ids[ids.len()-n..]);result
    }
    fn step(&mut self,id:usize) {
        let w=&self.weights;let t=&w.tensors;
        let mut h=t["in_proj.weight"].linear(&t["embedding.weight"].row(id),&t["in_proj.bias"]);
        for (i,&d) in w.dilations.iter().enumerate() {
            let p=format!("blocks.{i}");
            let n=crate::norm_values(&h,&t[&format!("{p}.norm.weight")].values,&t[&format!("{p}.norm.bias")].values);
            let mut current=t[&format!("{p}.pw_in.weight")].linear(&n,&t[&format!("{p}.pw_in.bias")]);
            current.iter_mut().for_each(|v|*v=v.max(0.));
            let buffer=&mut self.buffers[i];let conv=&t[&format!("{p}.dw_conv.weight")].values;
            let bias=&t[&format!("{p}.dw_conv.bias")].values;
            let z:Vec<_>=(0..w.ffn).map(|c|(buffer[c]*conv[c*3]+buffer[d*w.ffn+c]*conv[c*3+1]+
                current[c]*conv[c*3+2]+bias[c]).max(0.)).collect();
            buffer.copy_within(w.ffn..,0);let len=buffer.len();buffer[len-w.ffn..].copy_from_slice(&current);
            let residual=t[&format!("{p}.pw_out.weight")].linear(&z,&t[&format!("{p}.pw_out.bias")]);
            for (v,r) in h.iter_mut().zip(residual) {*v+=r;}
        }
        let n=crate::norm_values(&h,&t["final_norm.weight"].values,&t["final_norm.bias"].values);
        self.projected=t["out_proj.weight"].linear(&n,&t["out_proj.bias"]);
    }
    /// Diagnostic ablation. Production construction retains the trained BOS path.
    pub fn set_include_bos(&mut self, include: bool) {
        if self.include_bos != include {
            self.include_bos = include;
            self.prefix.clear(); self.projected.clear();
            self.checkpoints.clear();
            self.normalized_projections.clear();
            self.buffers.iter_mut().for_each(|b|b.fill(0.));
        }
    }
    fn project(&mut self,ids:&[usize])->Result<(),String> {
        if ids.len()>33 || ids.iter().any(|&i|i>=self.vocabulary_size()) {return Err("Invalid CNN context".into());}
        let mut sequence=if self.include_bos {vec![1]} else {vec![]};sequence.extend(ids.iter().copied().filter(|&i|i!=0 && i!=1));
        if sequence.len()>33 {return Err("CNN context exceeds 32 tokens".into());}
        if self.prefix==sequence && self.projected.len()==self.weights.embedding {return Ok(());}
        if sequence.is_empty() {
            // With no input, the no-BOS ablation uses output biases as its prior.
            self.buffers.iter_mut().for_each(|b|b.fill(0.));
            self.prefix.clear(); self.projected=vec![0.;self.weights.embedding];
            return Ok(());
        }
        // Candidate branches often share most of their causal context. Restore
        // the longest exact checkpoint instead of replaying it from BOS. A sliding
        // 32-token window must still match from its start; overlap alone is unsafe.
        let current = if sequence.starts_with(&self.prefix) { self.prefix.len() } else { 0 };
        let saved = self.checkpoints.iter().enumerate()
            .filter(|(_, state)| state.prefix.len() > current && sequence.starts_with(&state.prefix))
            .max_by_key(|(_, state)| state.prefix.len()).map(|(i, _)| i);
        let from = if let Some(index) = saved {
            let state = self.checkpoints.remove(index).unwrap();
            self.buffers.clone_from(&state.buffers);
            self.projected.clone_from(&state.projected);
            let from = state.prefix.len();
            self.checkpoints.push_back(state);
            from
        } else if current > 0 {
            current
        } else {
            self.buffers.iter_mut().for_each(|b| b.fill(0.)); 0
        };
        for &id in &sequence[from..] {
            if let Err(error)=crate::suggestion_request::check() {
                self.prefix.clear();self.projected.clear();
                self.buffers.iter_mut().for_each(|b|b.fill(0.));
                return Err(error);
            }
            self.step(id);
        }
        self.prefix=sequence;
        if self.projected.iter().any(|v|!v.is_finite()) {return Err("Nonfinite CNN projection".into());}
        if !self.checkpoints.iter().any(|state| state.prefix == self.prefix) {
            if self.checkpoints.len() == PREFIX_CHECKPOINTS { self.checkpoints.pop_front(); }
            self.checkpoints.push_back(PrefixCheckpoint {
                prefix: self.prefix.clone(), buffers: self.buffers.clone(), projected: self.projected.clone(),
            });
        }
        Ok(())
    }
    pub fn scores(&mut self,ids:&[usize],targets:&[usize])->Result<Vec<f32>,String> {
        self.project(ids)?;
        let w=&self.weights;let e=&w.tensors["embedding.weight"];let b=&w.tensors["output_bias"].values;
        targets.iter().map(|&id| {
            if id>=self.vocabulary_size() {return Err("Invalid CNN candidate".into());}
            Ok(crate::dot(&e.q[id*w.embedding..(id+1)*w.embedding],&self.projected)*e.scale+b[id])
        }).collect()
    }
    pub fn log_probabilities_for(&mut self, ids: &[usize], targets: &[usize]) -> Result<Vec<f32>, String> {
        crate::suggestion_request::check()?;
        if targets.iter().any(|&id| id >= self.vocabulary_size()) { return Err("Invalid CNN candidate".into()); }
        if let Some(values)=self.cached_target_scores(ids,targets) { return Ok(values); }
        let values = self.log_probabilities(ids)?;
        Ok(targets.iter().map(|&id| values[id]).collect())
    }
    fn cached_target_scores(&mut self, ids: &[usize], targets: &[usize]) -> Option<Vec<f32>> {
        if let Some(values) = crate::suggestion_request::cached_targets(ids, targets) { return Some(values); }
        let reusable = crate::suggestion_request::projection(ids).or_else(|| {
            // Consecutive keys revisit the same Chinese/byte prefixes. Keep a
            // small exact cache across requests on this independent worker.
            let index = self.normalized_projections.iter().position(|(key, _, _)| key == ids)?;
            let row = self.normalized_projections.remove(index).unwrap();
            let result = (row.1.clone(), row.2);
            self.normalized_projections.push_back(row);
            Some(result)
        });
        if let Some((projected, z)) = reusable {
            let w = &self.weights;
            let e = &w.tensors["embedding.weight"]; let b = &w.tensors["output_bias"].values;
            return Some(targets.iter().map(|&id| {
                (crate::dot(&e.q[id*w.embedding..(id+1)*w.embedding], &projected)*e.scale+b[id])-z
            }).collect());
        }
        None
    }

    pub fn log_probabilities_for_many(&mut self, requests: &[(Vec<usize>,Vec<usize>)]) -> Result<Vec<Vec<f32>>,String> {
        let mut result=vec![vec![];requests.len()];
        // Bound temporary output storage to four full tables, independent of pool size.
        for (block,requests) in requests.chunks(16).enumerate() {
            let mut missing=vec![];
            for (index,(ids,targets)) in requests.iter().enumerate() {
                crate::suggestion_request::check()?;
                if targets.iter().any(|&id|id>=self.vocabulary_size()) { return Err("Invalid CNN candidate".into()); }
                if let Some(values)=self.cached_target_scores(ids,targets) {
                    result[block*16+index]=values;
                } else {
                    self.project(ids)?;
                    missing.push((index,self.projected.clone()));
                }
            }
            for batch in missing.chunks(4) {
                crate::suggestion_request::check()?;
                let projected:Vec<_>=batch.iter().map(|(_,p)|p.clone()).collect();
                let w=&self.weights;let embedding=&w.tensors["embedding.weight"];let bias=&w.tensors["output_bias"];
                let mut values=if batch.len()==4 {
                    let mut values=crate::dot_matrix_batch4(&embedding.q,&projected);
                    for row in &mut values {
                        for (score,&bias) in row.iter_mut().zip(&bias.values) { *score=*score*embedding.scale+bias; }
                    }
                    crate::suggestion_request::record_output_batch(4);
                    values
                } else { projected.iter().map(|p|embedding.linear(p,bias)).collect() };
                for ((index,p),values) in batch.iter().zip(values.drain(..)) {
                    let (ids,targets)=&requests[*index];
                    let values=self.normalize_output(ids,p,values)?;
                    result[block*16+index]=targets.iter().map(|&id|values[id]).collect();
                }
            }
        }
        Ok(result)
    }

    pub fn log_probabilities(&mut self,ids:&[usize])->Result<Vec<f32>,String> {
        crate::suggestion_request::check()?;
        if let Some(values)=crate::suggestion_request::cached(ids) {return Ok(values);}
        self.project(ids)?;
        // This path always needs the complete vocabulary, in row order. Avoid
        // materializing/validating 65,536 target IDs and use the tiled kernel.
        let values=self.weights.tensors["embedding.weight"].linear(
            &self.projected, &self.weights.tensors["output_bias"]);
        let projected=self.projected.clone();
        self.normalize_output(ids,&projected,values)
    }

    fn normalize_output(&mut self,ids:&[usize],projected:&[f32],mut values:Vec<f32>)->Result<Vec<f32>,String> {
        let max=values.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let z=max+values.iter().map(|v|(v-max).exp()).sum::<f32>().ln();
        crate::suggestion_request::remember_projection(ids, projected, z);
        if let Some(index) = self.normalized_projections.iter().position(|(key, _, _)| key == ids) {
            self.normalized_projections.remove(index);
        }
        if self.normalized_projections.len() == 128 { self.normalized_projections.pop_front(); }
        self.normalized_projections.push_back((ids.to_vec(), projected.to_vec(), z));
        values.iter_mut().for_each(|v|*v-=z);
        crate::suggestion_request::check()?;
        crate::suggestion_request::remember(ids,&values);
        Ok(values)
    }
    pub fn rank(&mut self,left:&str,candidates:&[String])->Result<Vec<usize>,String> {
        let mut scores=Vec::new();
        for text in candidates {
            let mut context=self.context(left);let ids=self.tokenize_with_units(text).0;
            if ids.is_empty() || ids.len()>64 {scores.push(f32::NEG_INFINITY);continue;}
            let mut score=0.;
            for id in ids {score+=self.log_probabilities(&context)?[id];context.rotate_left(1);context[31]=id;}
            scores.push(score);
        }
        let mut order:Vec<_>=(0..candidates.len()).collect();
        order.sort_by(|&a,&b|scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));Ok(order)
    }
    pub fn predict(&mut self,left:&str,prefix:&str,limit:usize)->Result<Vec<crate::TokenPrediction>,String> {
        self.suggest(left,prefix,limit,false)
    }
    pub fn completions(&mut self,left:&str,prefix:&str,limit:usize)->Result<Vec<crate::TokenPrediction>,String> {
        self.suggest(left,prefix,limit,true)
    }
    /// Enumerate the spelling neighborhood before LM ranking, so lexical order or
    /// corpus frequency cannot prune a plausible correction out of the pool.
    pub fn english_suggestions(&mut self,left:&str,typed:&str,limit:usize)->Result<EnglishSuggestions,String> {
        self.english_suggestions_with_touch(left,typed,limit,None)
    }
    pub fn english_suggestions_with_touch(&mut self,left:&str,typed:&str,limit:usize,touch:Option<&crate::touch::TouchScorer>)->Result<EnglishSuggestions,String> {
        let pool = self.generate_english_candidates(typed)?;
        self.rank_english_candidates(left,pool,limit,touch)
    }

    pub fn generate_english_candidates(&self, typed: &str) -> Result<Option<EnglishCandidatePool>,String> {
        let Some(input)=crate::cnn_spelling::canonical(typed) else {
            return Ok(None);
        };
        let radius=if input.len()>=6 {3}else if input.len()>=4 {2}else{1};
        let mut pool=Vec::new();
        let mut known=false;
        let mut literal_ids=Vec::new();
        for (index, edits) in self.weights.spelling_index.matches(&input, radius, true)? {
            crate::suggestion_request::check()?;
            let (word,ids) = &self.weights.spellings[index];
            if word==&input {
                known=true;
                literal_ids=ids.clone();
                let has_canonical_casing = ids.iter().any(|&id| {
                    let t = &self.weights.tokens[id];
                    t != &input && (t.chars().next().is_some_and(|c| c.is_ascii_uppercase()) || t == "I")
                });
                if !has_canonical_casing {
                    continue;
                }
            }
            let completion=word.starts_with(&input);
            pool.push((word.clone(),ids.clone(),edits,if completion {word.len()-input.len()}else{0}));
        }
        let mut merge_splits = Vec::new();
        if !known && input.len() >= 3 {
            let is_short_word = |w: &str| matches!(w, "a" | "i" | "am" | "an" | "as" | "at" | "be" | "by" | "do" | "go" | "he" | "if" | "in" | "is" | "it" | "me" | "my" | "no" | "of" | "on" | "or" | "so" | "to" | "up" | "us" | "we");
            let is_word = |w: &str| -> Option<usize> {
                self.weights.spellings.binary_search_by(|(cand, _)| cand.as_str().cmp(w)).ok()
            };
            let is_head_word = |w: &str| matches!(w,
                "a" | "i" | "am" | "an" | "as" | "at" | "be" | "by" | "do" | "go" |
                "he" | "if" | "in" | "is" | "it" | "me" | "my" | "no" | "of" | "on" |
                "or" | "so" | "to" | "up" | "us" | "we" | "the" | "you" | "and" |
                "for" | "how" | "now" | "not" | "all" | "are" | "can" | "see" |
                "had" | "has" | "was" | "one" | "who" | "did" | "with" | "this" | "that"
            );
            // 1. Direct split (omitted space), e.g. yousupport -> you support, alot -> a lot
            for k in 1..input.len() {
                let w1 = &input[..k];
                let w2 = &input[k..];
                let cap2 = typed.chars().nth(k).is_some_and(|c| c.is_ascii_uppercase());
                let eligible = cap2 || (w1.len() >= 3 && w2.len() >= 3) || is_head_word(w1);
                let valid_w1 = w1.len() >= 3 || is_short_word(w1);
                let valid_w2 = w2.len() >= 3 || (w2.len() == 2 && is_short_word(w2));
                if eligible && valid_w1 && valid_w2 {
                    if let (Some(idx1), Some(idx2)) = (is_word(w1), is_word(w2)) {
                        merge_splits.push((idx1, idx2));
                    }
                }
            }
            // 2. Space-adjacent key mistap (b, n, v, c, m), e.g. youbsupport -> you support
            if input.len() >= 5 {
                for k in 1..input.len() - 1 {
                    let ch = input.as_bytes()[k];
                    if matches!(ch, b'b' | b'n' | b'v' | b'c' | b'm') {
                        let w1 = &input[..k];
                        let w2 = &input[k + 1..];
                        let cap2 = typed.chars().nth(k + 1).is_some_and(|c| c.is_ascii_uppercase());
                        let eligible = cap2 || (w1.len() >= 3 && w2.len() >= 3) || is_head_word(w1);
                        let valid_w1 = w1.len() >= 3 || is_short_word(w1);
                        let valid_w2 = w2.len() >= 3 || (w2.len() == 2 && is_short_word(w2));
                        if eligible && valid_w1 && valid_w2 {
                            if let (Some(idx1), Some(idx2)) = (is_word(w1), is_word(w2)) {
                                merge_splits.push((idx1, idx2));
                            }
                        }
                    }
                }
            }
            merge_splits.sort_unstable();
            merge_splits.dedup();
        }
        let apostrophes = crate::cnn_spelling::apostrophe_forms(&input).collect();
        Ok(Some(EnglishCandidatePool { input, original:typed.to_owned(), pool, known, literal_ids, merge_splits, apostrophes }))
    }

    pub fn rank_english_candidates(&mut self, left: &str, pool: Option<EnglishCandidatePool>, limit: usize,
        touch: Option<&crate::touch::TouchScorer>) -> Result<EnglishSuggestions,String> {
        let Some(EnglishCandidatePool { input, original, pool, known, literal_ids, merge_splits, apostrophes }) = pool else {
            return Ok(EnglishSuggestions {candidates:vec![],correction:None,known:false,literal_score:None,apostrophe_evidence:vec![]});
        };
        let context=self.context(left);
        let logp=self.log_probabilities(&context)?;
        // Reuse the same distribution as candidate ranking; no extra CNN pass.
        let literal_score=literal_ids.into_iter().map(|id|logp[id]).max_by(f32::total_cmp);
        let input_is_lower = input.bytes().all(|b| b.is_ascii_lowercase());
        let cap1 = original.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        let is_sentence_start = {
            let t = left.trim_end();
            t.ends_with('.') || t.ends_with('?') || t.ends_with('!') || t.ends_with('\n')
        };
        let should_cap = cap1 || (is_sentence_start && !input.is_empty());
        let mut ranked:Vec<_>=pool.into_iter().map(|(word,ids,edits,completed_chars)| {
            let has_lower_token = ids.iter().any(|&i| {
                self.weights.tokens[i].chars().next().is_some_and(|c| c.is_ascii_lowercase())
            });
            let id = ids.into_iter().max_by(|&a,&b| logp[a].total_cmp(&logp[b]).then(b.cmp(&a))).unwrap();
            let penalty = if completed_chars > 0 {
                // A partial word is not an insertion error. Let the contextual
                // LM rank completions without an extra length preference.
                0.0
            } else {
                crate::cnn_spelling::edit_penalty(input.as_bytes(), word.as_bytes(), edits)
            };
            let token_str = &self.weights.tokens[id];
            let word_str = if should_cap {
                let mut s = token_str.clone();
                if let Some(first) = s.chars().next() {
                    if first.is_ascii_lowercase() {
                        s[..1].make_ascii_uppercase();
                    }
                }
                s
            } else if (token_str.len() > 1 && token_str[1..].chars().any(|c| c.is_ascii_uppercase())) || token_str == "I" {
                token_str.clone()
            } else if !has_lower_token && token_str.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && !token_str.starts_with('[') && token_str.is_ascii() {
                token_str.clone()
            } else if input_is_lower {
                word
            } else {
                token_str.clone()
            };
            EnglishSuggestion {word:word_str, score:logp[id]-penalty,
                edits,completed_chars}
        }).collect();
        for (idx1, idx2) in merge_splits {
            let (w1, ids1) = &self.weights.spellings[idx1];
            let (w2, ids2) = &self.weights.spellings[idx2];
            let id1 = ids1.iter().copied().max_by(|&a,&b| logp[a].total_cmp(&logp[b]).then(b.cmp(&a))).unwrap();
            let id2 = ids2.iter().copied().max_by(|&a,&b| logp[a].total_cmp(&logp[b]).then(b.cmp(&a))).unwrap();
            let cap2 = original.chars().nth(w1.len()).is_some_and(|c| c.is_ascii_uppercase());
            let format_tok = |w: &str, token: &str, cap: bool| -> String {
                if w == "i" {
                    "I".to_string()
                } else if !cap {
                    w.to_string()
                } else if token.starts_with('[') || !token.is_ascii() {
                    let mut s = w.to_string();
                    s[..1].make_ascii_uppercase();
                    s
                } else if token.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
                    let mut s = token.to_string();
                    s[..1].make_ascii_uppercase();
                    s
                } else {
                    token.to_string()
                }
            };
            let combined_word = format!("{} {}", format_tok(w1, &self.weights.tokens[id1], cap1), format_tok(w2, &self.weights.tokens[id2], cap2));
            let is_exact = (w1.len() + w2.len() == input.len()) && input.starts_with(w1) && input.ends_with(w2);
            let combined_raw = format!("{w1} {w2}");
            let edits = if is_exact { 1 } else { crate::cnn_spelling::distance(input.as_bytes(), combined_raw.as_bytes(), 3).unwrap_or(2) };
            let score = if is_exact {
                logp[id1] + logp[id2] * 0.5 - 1.5
            } else {
                let pen = crate::cnn_spelling::edit_penalty(input.as_bytes(), combined_raw.as_bytes(), edits);
                logp[id1] + logp[id2] * 0.5 - pen
            };
            ranked.push(EnglishSuggestion {
                word: combined_word,
                score,
                edits,
                completed_chars: 0,
            });
        }
        // Restore punctuation using the same CNN's byte fallback. Word-token and
        // byte-sequence scores have different units, so restoration is calibrated
        // separately on TRAIN instead of treating a byte average as a word score.
        // Punctuation restoration has a separately calibrated policy. Leave it
        // intact in this coordinate-only letter-substitution experiment.
        if apostrophes.is_empty() {
            if let Some(touch) = touch {
                for row in &mut ranked {
                    row.score += touch.adjustment(&row.word, row.edits, row.completed_chars);
                }
            }
        }
        let original_best = ranked.iter().map(|r|r.score).max_by(f32::total_cmp);
        let mut apostrophe_evidence = Vec::new();
        let mut restored = None;
        for form in apostrophes {
            let ids = self.tokenize_with_units(&form.word).0;
            let mut context = context.clone();
            let mut score = 0.;
            for &id in &ids {
                score += self.log_probabilities(&context)?[id];
                context.rotate_left(1); context[31] = id;
            }
            let features = [(score-literal_score.unwrap_or(score)) as f64,
                (score-original_best.unwrap_or(score)) as f64,
                (form.count as f64).ln_1p()-(form.bare_count as f64).ln_1p()];
            if crate::cnn_spelling::accept_apostrophe(features) { restored = Some(form.word.clone()); }
            apostrophe_evidence.push(ApostropheEvidence { word: form.word.clone(), features, score });
            // Keep the word decision intact; restore surfaces only when the fitted
            // punctuation decision passes, otherwise retain the word decision.
        }
        ranked.sort_by(|a,b|b.score.total_cmp(&a.score).then(a.word.cmp(&b.word)));
        let correction=ranked.first().filter(|best| {
            if cap1 && best.edits >= 2 && !known {
                return false;
            }
            crate::cnn_spelling::accept(&input, known, &best.word, best.completed_chars,
                best.score, best.edits, ranked.get(1).map(|r|(r.score, r.edits, r.completed_chars, r.word.as_str())), literal_score)
        }).map(|best|best.word.clone());
        for evidence in &apostrophe_evidence {
            ranked.insert(0, EnglishSuggestion { word: evidence.word.clone(),
                score: evidence.score, edits: 1, completed_chars: 0 });
        }
        ranked.sort_by(|a,b|b.score.total_cmp(&a.score).then(a.word.cmp(&b.word)));
        if let Some(word) = &restored {
            if let Some(index) = ranked.iter().position(|r| &r.word == word) { ranked[..=index].rotate_right(1); }
        }
        let case_restoration=correction.as_ref().is_some_and(|w|w.eq_ignore_ascii_case(&original));
        let channel=if restored.is_none() && !case_restoration && original.replace('’',"'").eq_ignore_ascii_case(&input) {
            self.weights.channel.correction(left,&input,&logp)?
        } else {None};
        let channel = channel.map(|mut w| {
            if should_cap {
                if let Some(first) = w.chars().next() {
                    if first.is_ascii_lowercase() {
                        w[..1].make_ascii_uppercase();
                    }
                }
            }
            w
        });
        if let Some(word)=&channel {
            // The Kotlin boundary accepts only corrections present in the visible
            // candidate list. Keep a new dictionary proposal through truncation.
            let score=ranked.first().map(|r|r.score).unwrap_or(0.);
            if let Some(index)=ranked.iter().position(|r|&r.word==word) {ranked[..=index].rotate_right(1);} else {
                ranked.insert(0,EnglishSuggestion {word:word.clone(),score,
                    edits:crate::cnn_spelling::distance(input.as_bytes(),word.as_bytes(),4).unwrap_or(4),completed_chars:0});
            }
        }
        let correction = restored.or(channel).or(correction);
        ranked.truncate(limit);
        Ok(EnglishSuggestions {candidates:ranked,correction,known,literal_score,apostrophe_evidence})
    }
    fn suggest(&mut self,left:&str,prefix:&str,limit:usize,english_only:bool)->Result<Vec<crate::TokenPrediction>,String> {
        let ids=self.context(left);
        let targets:Vec<_>=(8..self.vocabulary_size()).filter(|&id| {
            let text=&self.weights.tokens[id];
            if text.starts_with('[') || !crate::displayable(text) {return false;}
            if prefix.is_empty() {return self.weights.words.contains(&id) || (!english_only && !text.is_ascii());}
            self.weights.words.contains(&id) && text.to_lowercase().starts_with(&prefix.to_lowercase())
        }).collect();
        let scores=self.scores(&ids,&targets)?;
        crate::top_predictions(targets.into_iter().zip(scores).map(|(id,score)|crate::TokenPrediction {
            token:self.weights.tokens[id].clone(),token_id:Some(id),score }).collect(),limit)
    }
    pub fn shares_weights(&self,other:&Self)->bool {Arc::ptr_eq(&self.weights,&other.weights)}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Set CANTONESE_CNN_TEST_MODEL to the packaged shared model"]
    fn prefix_checkpoints_match_cold_inference_bit_for_bit() {
        let bytes = std::fs::read(std::env::var("CANTONESE_CNN_TEST_MODEL").expect("Model fixture path")).unwrap();
        let mut cached = CnnLanguageModel::from_bytes(&bytes).unwrap();
        let mut cold = CnnLanguageModel::from_bytes(&bytes).unwrap();
        let mut contexts = vec!["", "我想", "我想去", "我想食", "我想去食", "我想去飲", "I have a ", "I have a problem "]
            .into_iter().map(String::from).collect::<Vec<_>>();
        // Exceed cache capacity, revisit branches, then shift a full context window.
        for i in 1..=12 { contexts.push("我".repeat(i)); }
        contexts.extend(["我想去", "I have a ", "", "我們"].into_iter().map(String::from));
        contexts.extend(["我".repeat(32), "我".repeat(31) + "們", "我".repeat(32) + "們"]);
        for include_bos in [true, false, true] {
            cached.set_include_bos(include_bos);
            cold.set_include_bos(include_bos);
            for left in &contexts {
                let ids = cached.context(left);
                cold.prefix.clear(); cold.projected.clear(); cold.checkpoints.clear();
                cold.buffers.iter_mut().for_each(|b| b.fill(0.));
                let actual = cached.log_probabilities(&ids).unwrap();
                let expected = cold.log_probabilities(&ids).unwrap();
                assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "BOS={include_bos} context={left}");
                assert!(cached.checkpoints.len() <= PREFIX_CHECKPOINTS);
                assert!(cached.normalized_projections.len() <= 128);
            }
        }
        let token = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ids = cached.context("取消之後仍然正確");
        assert!(crate::suggestion_request::run(token.clone(), || {
            token.store(true, std::sync::atomic::Ordering::Release);
            cached.scores(&ids, &[8])
        }).is_err());
        cold.prefix.clear(); cold.projected.clear(); cold.checkpoints.clear();
        cold.buffers.iter_mut().for_each(|b| b.fill(0.));
        assert_eq!(cached.log_probabilities(&ids).unwrap(), cold.log_probabilities(&ids).unwrap());
    }

    #[test]
    #[ignore = "Set CANTONESE_CNN_TEST_MODEL to the packaged shared model"]
    fn compact_projection_scores_survive_distribution_eviction_exactly() {
        let bytes = std::fs::read(std::env::var("CANTONESE_CNN_TEST_MODEL").expect("Model fixture path")).unwrap();
        let mut model = CnnLanguageModel::from_bytes(&bytes).unwrap();
        let ids = model.context("我想去食飯");
        let expected = model.log_probabilities(&ids).unwrap();
        let targets = [8, 12, 256, model.vocabulary_size() - 1, 8];
        crate::suggestion_request::run(Arc::new(std::sync::atomic::AtomicBool::new(false)), || {
            let filler = vec![0.; model.vocabulary_size()];
            crate::suggestion_request::remember(&[usize::MAX], &filler); // Pin a different table.
            model.log_probabilities(&ids)?;
            let want = targets.iter().map(|&id| expected[id]).collect::<Vec<_>>();
            assert_eq!(model.log_probabilities_for(&ids, &targets)?, want);
            for i in 0..40 { crate::suggestion_request::remember(&[usize::MAX, i], &filler); }
            assert!(crate::suggestion_request::cached_targets(&ids, &targets).is_none());
            assert_eq!(model.log_probabilities_for(&ids, &targets)?, want);
            assert_eq!(crate::suggestion_request::stats().projection_hits, 1);
            // Compact-cache eviction falls back to ordinary exact inference.
            for i in 0..1025 {
                crate::suggestion_request::remember_projection(&[usize::MAX, i], &[0.; 128], 0.);
            }
            assert!(crate::suggestion_request::projection(&ids).is_none());
            model.normalized_projections.clear();
            assert_eq!(model.log_probabilities_for(&ids, &targets)?, want);
            assert!(model.log_probabilities_for(&ids, &[model.vocabulary_size()]).is_err());
            Ok(())
        }).unwrap();
        assert!(crate::suggestion_request::projection(&ids).is_none());
        // Reuse across independent key requests keeps the exact normalizer.
        crate::suggestion_request::run(Arc::new(std::sync::atomic::AtomicBool::new(false)), || {
            assert_eq!(model.log_probabilities_for(&ids, &targets)?, targets.iter().map(|&id| expected[id]).collect::<Vec<_>>());
            assert_eq!(crate::suggestion_request::stats().forwards, 0);
            Ok(())
        }).unwrap();
        model.set_include_bos(false);
        assert!(model.normalized_projections.is_empty());
        assert!(model.checkpoints.is_empty());
    }

    #[test]
    #[ignore="Set CANTONESE_CNN_TEST_MODEL to the packaged shared model"]
    fn candidate_enumeration_runs_no_inference_and_batch_ranking_is_exact() {
        let bytes=std::fs::read(std::env::var("CANTONESE_CNN_TEST_MODEL").unwrap()).unwrap();
        let mut model=CnnLanguageModel::from_bytes(&bytes).unwrap();
        crate::suggestion_request::run(Arc::new(std::sync::atomic::AtomicBool::new(false)),|| {
            for typed in ["teh","continue","youbsupport","dont","problem"] {
                model.generate_english_candidates(typed)?;
                let radius=if typed.len()>=6 {3}else if typed.len()>=4 {2}else{1};
                for (distance, completions) in [(radius,true),(1,false)] {
                    let expected: Vec<_> = model.weights.spellings.iter().enumerate().filter_map(|(id,(word,_))| {
                        if completions && word.starts_with(typed) { Some((id,0)) } else {
                            crate::cnn_spelling::distance(typed.as_bytes(),word.as_bytes(),distance).map(|d| (id,d))
                        }
                    }).collect();
                    assert_eq!(model.weights.spelling_index.matches(typed,distance,completions)?,expected);
                }
            }
            assert_eq!(crate::suggestion_request::stats().forwards,0);
            assert!(model.prefix.is_empty());
            Ok(())
        }).unwrap();
        let contexts=["","我","我想","我想去","我想食","I have a ","I have a problem "];
        let requests=(0..23).map(|i|(model.context(contexts[i%contexts.len()]),vec![8,12,256,model.vocabulary_size()-1])).collect::<Vec<_>>();
        let mut scalar=CnnLanguageModel::from_bytes(&bytes).unwrap();
        let expected=requests.iter().map(|(ids,targets)|scalar.log_probabilities_for(ids,targets).unwrap()).collect::<Vec<_>>();
        crate::suggestion_request::run(Arc::new(std::sync::atomic::AtomicBool::new(false)),|| {
            let actual=model.log_probabilities_for_many(&requests)?;
            assert_eq!(actual,expected);
            assert!(crate::suggestion_request::stats().output_batches>0);
            assert_eq!(model.log_probabilities_for_many(&requests)?,expected);
            Ok(())
        }).unwrap();
        let token=Arc::new(std::sync::atomic::AtomicBool::new(false));
        assert!(crate::suggestion_request::run(token.clone(),|| {
            token.store(true,std::sync::atomic::Ordering::Release);
            model.log_probabilities_for_many(&requests)
        }).is_err());
        assert_eq!(model.log_probabilities_for_many(&requests).unwrap(),expected);
    }

    #[test] fn rejects_incomplete_bundles() {
        for bytes in [b"".as_slice(),b"KBCNN01\0".as_slice(),b"KBCNN01\0\xff\xff\xff\xff".as_slice()] {
            assert!(CnnLanguageModel::from_bytes(bytes).is_err());
        }
    }
}

