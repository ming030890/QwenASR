//! Shared V2 causal CNN. INT8 matrices stay quantized; clients share weights.
//! Context is BOS + the last 32 canonical tokens, exactly as in V2 evaluation.
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
        let n=self.shape[1];
        (0..self.shape[0]).map(|i|crate::dot(&self.q[i*n..(i+1)*n],x)*self.scale+bias.values[i]).collect()
    }
}
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnglishSuggestion {
    pub word: String, pub score: f32, pub edits: u8, pub completed_chars: usize,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnglishSuggestions {
    pub candidates: Vec<EnglishSuggestion>, pub correction: Option<String>, pub known: bool,
}
struct Weights {
    raw: Box<[u8]>, tokens: Vec<String>, ids: HashMap<String,usize>, english: Regex,
    spellings: Vec<(String, Vec<usize>)>, words: HashSet<usize>, embedding: usize, ffn: usize, dilations: Vec<usize>,
    tensors: HashMap<String,Tensor>,
}
static SHARED: OnceLock<Mutex<Weak<Weights>>> = OnceLock::new();

pub struct CnnLanguageModel {
    weights: Arc<Weights>, buffers: Vec<Vec<f32>>, prefix: Vec<usize>, projected: Vec<f32>,
}
impl CnnLanguageModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self,String> {
        let mut cache=SHARED.get_or_init(||Mutex::new(Weak::new())).lock().map_err(|_|"CNN cache poisoned")?;
        let weights=match cache.upgrade().filter(|w| w.raw.as_ref()==bytes) {
            Some(w)=>w,
            None=>{let w=Arc::new(Self::parse(bytes)?);*cache=Arc::downgrade(&w);w}
        };
        let buffers=weights.dilations.iter().map(|&d|vec![0.;2*d*weights.ffn]).collect();
        Ok(Self {weights,buffers,prefix:Vec::new(),projected:Vec::new()})
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
        Ok(Weights {raw:bytes.into(),tokens:meta.tokens,ids,spellings:spellings.into_iter().collect(),words:meta.english_word_ids.into_iter().collect(),
            english:Regex::new(r"^[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*").unwrap(),
            embedding:meta.embed_dim,ffn:meta.ffn_dim,dilations:meta.dilations,tensors})
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
    fn project(&mut self,ids:&[usize])->Result<(),String> {
        if ids.len()>33 || ids.iter().any(|&i|i>=self.vocabulary_size()) {return Err("Invalid CNN context".into());}
        let mut sequence=vec![1];sequence.extend(ids.iter().copied().filter(|&i|i!=0 && i!=1));
        if sequence.len()>33 {return Err("CNN context exceeds 32 tokens".into());}
        if self.prefix==sequence {return Ok(());}
        let from=if !self.prefix.is_empty() && sequence.starts_with(&self.prefix) {self.prefix.len()} else {
            self.buffers.iter_mut().for_each(|b|b.fill(0.));0
        };
        for &id in &sequence[from..] {self.step(id);}
        self.prefix=sequence;
        if self.projected.iter().any(|v|!v.is_finite()) {return Err("Nonfinite CNN projection".into());}
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
    pub fn log_probabilities(&mut self,ids:&[usize])->Result<Vec<f32>,String> {
        let targets:Vec<_>=(0..self.vocabulary_size()).collect();
        let mut values=self.scores(ids,&targets)?;
        let max=values.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let z=max+values.iter().map(|v|(v-max).exp()).sum::<f32>().ln();
        values.iter_mut().for_each(|v|*v-=z);Ok(values)
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
        let Some(input)=crate::cnn_spelling::canonical(typed) else {
            return Ok(EnglishSuggestions {candidates:vec![],correction:None,known:false});
        };
        let radius=if input.len()>=6 {3}else if input.len()>=4 {2}else{1};
        let mut pool=Vec::new();
        let mut known=false;
        for (word,ids) in &self.weights.spellings {
            if word==&input {known=true;continue;}
            let completion=word.starts_with(&input);
            let edits=if completion {0}else {
                match crate::cnn_spelling::distance(input.as_bytes(),word.as_bytes(),radius) {
                    Some(n)=>n,None=>continue
                }
            };
            pool.push((word.clone(),ids.clone(),edits,if completion {word.len()-input.len()}else{0}));
        }
        let context=self.context(left);
        let logp=self.log_probabilities(&context)?;
        let mut ranked:Vec<_>=pool.into_iter().map(|(word,ids,edits,completed_chars)| {
            let id=ids.into_iter().max_by(|&a,&b|logp[a].total_cmp(&logp[b]).then(b.cmp(&a))).unwrap();
            let penalty = if completed_chars > 0 {
                completed_chars as f32 * 0.5
            } else {
                crate::cnn_spelling::edit_penalty(input.as_bytes(), word.as_bytes(), edits)
            };
            EnglishSuggestion {word:self.weights.tokens[id].clone(),score:logp[id]-penalty,
                edits,completed_chars}
        }).collect();
        ranked.sort_by(|a,b|b.score.total_cmp(&a.score).then(a.word.cmp(&b.word)));
        let correction=ranked.first().filter(|best| {
            if known || input.len() < 3 || best.completed_chars > 0 || best.word.len() < 3 {
                return false;
            }
            let len_diff = input.len().abs_diff(best.word.len());
            if best.edits >= 2 && len_diff > 0 {
                return false;
            }
            if input.len() <= 3 && best.word.len() < input.len() {
                return false;
            }
            let max_allowed_edits = if input.len() >= 6 { 3 } else if input.len() >= 5 { 2 } else { 1 };
            if best.edits > max_allowed_edits {
                return false;
            }
            if best.score < -24.0 {
                return false;
            }
            let base_margin = if left.trim().is_empty() {
                if best.edits == 1 { 0.8 } else { 1.2 }
            } else {
                if best.edits == 1 { 0.5 } else { 0.8 }
            };
            ranked.get(1).map_or(true, |runner| {
                let required = if runner.edits > best.edits {
                    0.0
                } else if runner.edits == best.edits && runner.completed_chars == 0 && input.len().abs_diff(runner.word.len()) > len_diff {
                    base_margin * 0.5
                } else {
                    base_margin
                };
                best.score - runner.score >= required
            })
        }).map(|best|best.word.clone());
        ranked.truncate(limit);
        Ok(EnglishSuggestions {candidates:ranked,correction,known})
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
    #[test] fn rejects_incomplete_bundles() {
        for bytes in [b"".as_slice(),b"KBCNN01\0".as_slice(),b"KBCNN01\0\xff\xff\xff\xff".as_slice()] {
            assert!(CnnLanguageModel::from_bytes(bytes).is_err());
        }
    }
}
