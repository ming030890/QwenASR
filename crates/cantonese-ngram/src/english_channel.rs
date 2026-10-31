//! A conservative learned typing channel, complementing the shared CNN.
//! Parameters and threshold use TRAIN only. Runtime reuses the current CNN pass.
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use serde::Deserialize;

#[derive(Deserialize)]
struct Channel {
    substitution: Vec<Vec<f64>>, deletion: Vec<Vec<f64>>,
    insertion: Vec<Vec<f64>>, transposition: Vec<Vec<f64>>,
}
#[derive(Deserialize)]
struct Metadata { english_log_mass: f64, total_frequency: f64 }
#[derive(Deserialize)]
struct Policy { beta:f64, tau:f64, unknown_floor:f64, cutoff:f64 }
struct Word { text:String, log_frequency:f64, ids:Vec<usize>, prior:Option<f64> }
type Proposals = Arc<Vec<(usize,f64)>>;
pub(crate) struct Decoder {
    words:Vec<Word>, index:crate::spelling_index::SpellingIndex,
    neural:HashMap<String,(Vec<usize>,f64)>, english_ids:Vec<usize>,
    channel:Channel, metadata:Metadata, policy:Policy,
    cache:Mutex<VecDeque<(String,Proposals)>>,
}

fn char_id(c:u8)->usize { match c { b'a'..=b'z'=>(c-b'a'+1) as usize,b'\''=>27,b' '=>28,_=>0 } }
impl Channel {
    fn cost(&self, observed:&[u8], intended:&[u8])->f64 {
        let mut x=[[0.;33];33];
        for i in 0..=observed.len() { for j in 0..=intended.len() {
            if i==0 && j==0 {continue;}
            let mut v=f64::INFINITY;
            if i>0 {let p=if i>1 {char_id(observed[i-2])}else{0};
                v=v.min(x[i-1][j]+self.insertion[p][char_id(observed[i-1])]);}
            if j>0 {let c=char_id(intended[j-1]);
                let repeat=(j>1 && intended[j-2]==intended[j-1]) || (j<intended.len() && intended[j]==intended[j-1]);
                v=v.min(x[i][j-1]+self.deletion[c][usize::from(repeat)]);}
            if i>0 && j>0 {v=v.min(x[i-1][j-1]+self.substitution[char_id(intended[j-1])][char_id(observed[i-1])]);}
            if i>1 && j>1 && observed[i-1]==intended[j-2] && observed[i-2]==intended[j-1] {
                v=v.min(x[i-2][j-2]+self.transposition[char_id(intended[j-2])][char_id(intended[j-1])]);}
            x[i][j]=v;
        }}
        // The fitted feature cache uses float32 edit costs.
        x[observed.len()][intended.len()] as f32 as f64
    }
}
fn logsum(values:impl Iterator<Item=f64>)->f64 {
    let values:Vec<_>=values.collect();
    let max=values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    max+values.into_iter().map(|x|(x-max).exp()).sum::<f64>().ln()
}
impl Decoder {
    pub fn new(spellings:&[(String,Vec<usize>)], english_ids:&[usize])->Self {
        let priors:HashMap<_,_>=include_str!("english_channel_data/priors.tsv").lines().map(|l| {
            let (w,p)=l.split_once('\t').unwrap();(w.to_owned(),p.parse::<f64>().unwrap())
        }).collect();
        let neural:HashMap<_,_>=spellings.iter().filter_map(|(w,ids)|priors.get(w).map(|p|(w.clone(),(ids.clone(),*p)))).collect();
        let words:Vec<_>=include_str!("english_channel_data/lexicon.tsv").lines().map(|l| {
            let (text,f)=l.split_once('\t').unwrap();
            Word {text:text.to_owned(),log_frequency:f.parse::<f64>().unwrap().ln(),
                ids:neural.get(text).map(|v|v.0.clone()).unwrap_or_default(),prior:priors.get(text).copied()}
        }).collect();
        let index=crate::spelling_index::SpellingIndex::new(words.iter().map(|w|w.text.as_str()));
        Self {words,index,neural,english_ids:english_ids.to_vec(),
            channel:serde_json::from_str(include_str!("english_channel_data/channel.json")).unwrap(),
            metadata:serde_json::from_str(include_str!("english_channel_data/metadata.json")).unwrap(),
            policy:serde_json::from_str(include_str!("english_channel_data/policy.json")).unwrap(),
            cache:Mutex::new(VecDeque::new())}
    }
    fn proposals(&self,input:&str)->Result<Proposals,String> {
        if let Some(hit)=self.cache.lock().map_err(|_|"Channel cache poisoned")?.iter().find(|(k,_)|k==input) {return Ok(hit.1.clone());}
        let radius=match input.len() {0..=2=>1,3=>2,4..=5=>3,_=>4};
        let mut pairs=Vec::new();
        for (id,_) in self.index.matches(input,radius,false)? {
            crate::suggestion_request::check()?;
            let word=&self.words[id];
            if word.text!=input {pairs.push((id,self.channel.cost(input.as_bytes(),word.text.as_bytes())));}
        }
        let mut selected=Vec::new();
        for tau in [0.25,0.5,1.,2.] {
            let mut order:Vec<_>=(0..pairs.len()).collect();
            order.sort_unstable_by(|&a,&b| {
                let score=|j:usize|self.words[pairs[j].0].log_frequency-tau*pairs[j].1;
                score(b).total_cmp(&score(a)).then(pairs[a].0.cmp(&pairs[b].0))
            });
            selected.extend(order.into_iter().take(64));
        }
        selected.sort_unstable();selected.dedup();
        let result=Arc::new(selected.into_iter().map(|i|pairs[i]).collect());
        let mut cache=self.cache.lock().map_err(|_|"Channel cache poisoned")?;
        cache.push_back((input.to_owned(),Arc::clone(&result)));
        while cache.len()>256 {cache.pop_front();}
        Ok(result)
    }
    pub fn correction(&self,left:&str,input:&str,logp:&[f32])->Result<Option<String>,String> {
        // Mixed-language contexts retain their separately calibrated decoder.
        if left.trim().is_empty() || left.chars().any(|c|matches!(c,'\u{3400}'..='\u{9fff}'|'\u{20000}'..='\u{323af}')) ||
            !(2..=32).contains(&input.len()) || input.split('\'').any(|s|s.is_empty() || !s.bytes().all(|c|c.is_ascii_lowercase())) {return Ok(None);}
        let pairs=self.proposals(input)?;
        if pairs.is_empty() {return Ok(None);}
        let mass=logsum(self.english_ids.iter().map(|&i|logp[i] as f64));
        let gain=|ids:&[usize],prior:f64| ((logsum(ids.iter().map(|&i|logp[i] as f64))-mass-prior+self.metadata.english_log_mass) as f32) as f64;
        let mut literal_frequency=self.words.binary_search_by(|w|w.text.as_str().cmp(input)).ok().map(|i|self.words[i].log_frequency.exp());
        let literal_gain=if let Some((ids,prior))=self.neural.get(input) {
            if literal_frequency.is_none() {literal_frequency=Some((prior-self.metadata.english_log_mass).exp()*self.metadata.total_frequency);}
            gain(ids,*prior)
        } else {0.};
        let keep=literal_frequency.unwrap_or(self.policy.unknown_floor).ln()+self.policy.beta*literal_gain;
        let mut scores=Vec::with_capacity(pairs.len());
        for &(id,cost) in pairs.iter() {
            let w=&self.words[id];let g=w.prior.filter(|_|!w.ids.is_empty()).map(|p|gain(&w.ids,p)).unwrap_or(0.);
            scores.push((id,(w.log_frequency as f32) as f64+self.policy.beta*g-self.policy.tau*cost));
        }
        let &(id,best)=scores.iter().max_by(|a,b|a.1.total_cmp(&b.1).then(b.0.cmp(&a.0))).unwrap();
        let other=logsum(std::iter::once(keep).chain(scores.iter().filter(|v|v.0!=id).map(|v|v.1)));
        Ok((best-other>=self.policy.cutoff).then(||self.words[id].text.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learned_channel_distinguishes_actual_letters_and_repetition() {
        let c:Channel=serde_json::from_str(include_str!("english_channel_data/channel.json")).unwrap();
        assert_eq!(c.cost(b"game",b"game"),0.);
        assert!((c.cost(b"gwme",b"game")-c.substitution[1][23]).abs()<1e-5);
        assert!(c.cost(b"aoife",b"alice")>c.cost(b"gwme",b"game"));
        assert!(c.cost(b"tommorow",b"tomorrow").is_finite());
    }
}
