//! Corpus-derived prefix candidates, independent of contextual LM scores.
//! Counts select the bounded search pool; only the LM ranks the final hypotheses.
use std::collections::{BTreeMap,HashMap};
pub struct PhraseLexicon { completions:HashMap<String,Vec<(String,u32)>> }
impl PhraseLexicon {
    pub fn from_bytes(bytes:&[u8])->Result<Self,String> {
        if bytes.len()>8_000_000{return Err("Phrase vocabulary exceeds limit".into());}
        let phrases:BTreeMap<String,u32>=serde_json::from_slice(bytes).map_err(|e|e.to_string())?;
        if phrases.len()>200000{return Err("Too many phrases".into());}
        let mut completions=HashMap::<String,Vec<(String,u32)>>::new();
        for (text,count) in phrases {
            let chars:Vec<_>=text.chars().collect();
            if !(2..=16).contains(&chars.len()) || !chars.iter().all(|&c|crate::tokenizer::is_han(c)) || count==0 {continue;}
            for i in 1..chars.len() {
                let prefix:String=chars[..i].iter().collect();let suffix:String=chars[i..].iter().collect();
                completions.entry(prefix).or_default().push((suffix,count));
            }
        }
        for entries in completions.values_mut() {
            entries.sort_by(|a,b|b.1.cmp(&a.1).then(a.0.cmp(&b.0)));entries.dedup_by(|a,b|a.0==b.0);entries.truncate(128);
        }
        Ok(Self{completions})
    }
    pub fn candidates(&self,left:&str,limit:usize)->Vec<String> {
        let tail:String=left.chars().rev().take_while(|&c|crate::tokenizer::is_han(c)).take(15).collect::<Vec<_>>().into_iter().rev().collect();
        let chars:Vec<_>=tail.chars().collect();let mut result=vec![];
        for i in 0..chars.len() {
            let suffix:String=chars[i..].iter().collect();
            if let Some(candidates)=self.completions.get(&suffix) {
                for (text,_) in candidates {if !result.contains(text) {result.push(text.clone());}}
            }
            if result.len()>=limit.min(128) {break;}
        }
        result.truncate(limit.min(128));result
    }
}
#[cfg(test)]mod tests {
    use super::*;
    #[test]fn prefix_generation_does_not_duplicate_committed_text() {
        let lex=PhraseLexicon::from_bytes("{\"譚仔米線\":5,\"譚仔三哥\":3,\"米線\":100}".as_bytes()).unwrap();
        assert!(lex.candidates("今晚食譚仔",32).contains(&"米線".into()));
        assert!(lex.candidates("譚仔米",32).contains(&"線".into()));
        assert!(lex.candidates("book a meeting ",32).is_empty());
    }
}
