//! Complete bounded spelling neighborhoods before frequency-based truncation.
//! Corpus frequency must not hide a rare one-edit word behind distant words.
use super::*;
use super::decoder::{DecodeResult,Hypothesis};
use std::time::{Duration,Instant};

const LIMIT:usize=128_000;
const DEADLINE:Duration=Duration::from_millis(8);
#[derive(Clone,Copy)]
struct Cell { edits:u8,cost:f64 }
impl Cell {
    fn add(self,edits:u8,cost:f64)->Self {Self{edits:self.edits.saturating_add(edits),cost:self.cost+cost}}
    fn best(self,other:Self)->Self {
        if (other.edits,other.cost)<(self.edits,self.cost) {other}else{self}
    }
}

pub fn decode(dict:&Dictionary,typed:&str,touch:&TouchModel,previous:&str,session:i64,revision:i64)->DecodeResult {
    decode_impl(dict,typed,touch,previous,session,revision,false)
}
pub fn decode_robust(dict:&Dictionary,typed:&str,touch:&TouchModel,previous:&str,session:i64,revision:i64)->DecodeResult {
    decode_impl(dict,typed,touch,previous,session,revision,true)
}
fn decode_impl(dict:&Dictionary,typed:&str,touch:&TouchModel,previous:&str,session:i64,revision:i64,robust:bool)->DecodeResult {
    let started=Instant::now();let mut visited=0;let mut exhausted=false;let mut found=Vec::new();
    let valid=!typed.is_empty() && typed.len()<=64 && typed.bytes().all(|c|c.is_ascii_lowercase() || c==b'\'');
    if valid {
        let input=typed.as_bytes();let radius=if input.len()>=6 {3}else{2};
        let mut initial=[Cell{edits:0,cost:0.};65];
        for j in 1..=input.len() {initial[j]=initial[j-1].add(1,1.2);}
        // Touch probabilities are invariant across trie prefixes.
        let matches:Vec<_>=input.iter().enumerate().map(|(i,&c)|touch.match_cost(i,c)).collect();
        let substitutions:Vec<[f64;26]>=input.iter().enumerate().map(|(i,&c)|
            std::array::from_fn(|j|if robust {robust_substitution(touch,i,c,b'a'+j as u8)} else {0.45+touch.substitution(i,c,b'a'+j as u8)*0.5})).collect();
        let mut search=Search{dict,input,radius,matches:&matches,substitutions:&substitutions,started,
            visited:&mut visited,exhausted:&mut exhausted,found:&mut found};
        for edge in dict.children(0) {search.walk(edge.child,edge.label,0,&initial,None);if *search.exhausted {break;}}
        // Exact prefix completions are suggestions with explicit completion length.
        if let Some(prefix)=dict.find(input) {
            let mut heap=BinaryHeap::new();heap.push(HeapEntry{bound:prior(dict.nodes[prefix as usize].maximum),id:prefix as usize});
            let mut pops=0;
            while let Some(entry)=heap.pop() {
                if pops>=128 {break;}pops+=1;let node=entry.id as u32;
                let word=dict.word(node);let tail=word.len()-typed.len();
                if tail>0 && dict.nodes[node as usize].frequency>0 {
                    found.push(Hypothesis{text:word,spatial_score:-(tail as f64)*0.9,lexical_score:prior(dict.nodes[node as usize].frequency),
                        edits:0,completion:tail,replace_before:typed.len(),literal:false});
                }
                if tail<8 {for child in dict.children(node) {heap.push(HeapEntry{bound:prior(dict.nodes[child.child as usize].maximum)-tail as f64*0.9,id:child.child as usize});}}
            }
        }
        for i in 1..typed.len() {
            let (a,b)=typed.split_at(i);let (fa,fb)=(frequency(dict,a),frequency(dict,b));
            if fa>0 && fb>0 {found.push(Hypothesis{text:format!("{a} {b}"),spatial_score:-1.8,lexical_score:(prior(fa)+prior(fb))/2.,
                edits:1,completion:0,replace_before:typed.len(),literal:false});}
        }
        if !previous.is_empty() && previous.len()<=32 && previous.bytes().all(|c|c.is_ascii_lowercase()) {
            let joined=format!("{previous}{typed}");let f=frequency(dict,&joined);
            if f>0 {found.push(Hypothesis{text:joined,spatial_score:-1.8,lexical_score:prior(f),edits:1,completion:0,
                replace_before:previous.len()+1+typed.len(),literal:false});}
        }
    }
    found.retain(|h|h.text!=typed);
    // A correction and an exact-prefix completion can produce the same spelling.
    // Retain the complete edit path for decisions, not an uncharged completion.
    found.sort_by(|a,b|a.text.cmp(&b.text).then(a.replace_before.cmp(&b.replace_before)).then(a.completion.cmp(&b.completion)));
    found.dedup_by(|a,b|a.text==b.text && a.replace_before==b.replace_before);
    let mut closest:Vec<_>=found.iter().filter(|h|h.completion==0 && !h.text.contains(' ') && h.replace_before==typed.len()).cloned().collect();
    closest.sort_by(|a,b|a.edits.cmp(&b.edits).then(b.spatial_score.total_cmp(&a.spatial_score)).then(b.lexical_score.total_cmp(&a.lexical_score)).then(a.text.cmp(&b.text)));
    closest.truncate(16);
    found.sort_by(|a,b|b.score().total_cmp(&a.score()).then(a.text.cmp(&b.text)));
    for h in found {if !closest.iter().any(|a|a.text==h.text && a.replace_before==h.replace_before) {closest.push(h);}if closest.len()>=127 {break;}}
    closest.push(Hypothesis{text:typed.into(),spatial_score:0.,lexical_score:prior(frequency(dict,typed)),edits:0,completion:0,replace_before:typed.len(),literal:true});
    DecodeResult{schema_version:2,session,revision,hypotheses:closest,visited,budget_exhausted:exhausted,elapsed_micros:started.elapsed().as_micros() as u64}
}

fn prior(f:u32)->f64 {(1.+f as f64).ln()*0.55}
fn frequency(dict:&Dictionary,text:&str)->u32 {dict.find(text.as_bytes()).map_or(0,|i|dict.nodes[i as usize].frequency)}
fn robust_substitution(touch:&TouchModel,pos:usize,actual:u8,label:u8)->f64 {
    // A discrete neighboring-key mistake remains possible at the center of the
    // observed key. Touch offsets refine this channel instead of double-charging
    // the full key separation, which saturated all alternatives equally.
    let base=0.45+0.5*if adjacent(actual,label) {CHEAP}else{SUB_FAR};
    if let (Some(point),Some(a),Some(b))=(touch.points.get(pos).copied().flatten(),touch.center(actual),touch.center(label)) {
        let actual_likelihood=-((point.x-a.x).powi(2)+(point.y-a.y).powi(2))/0.5;
        let target_likelihood=-((point.x-b.x).powi(2)+(point.y-b.y).powi(2))/0.5;
        let nominal=((a.x-b.x).powi(2)+(a.y-b.y).powi(2))/0.5;
        (base+0.5*(actual_likelihood-target_likelihood-nominal)).clamp(0.,base+0.5)
    } else {base}
}
struct Search<'a> {
    dict:&'a Dictionary,input:&'a [u8],radius:u8,matches:&'a [f64],substitutions:&'a [[f64;26]],started:Instant,
    visited:&'a mut usize,exhausted:&'a mut bool,found:&'a mut Vec<Hypothesis>,
}
impl Search<'_> {
    fn walk(&mut self,node:u32,letter:u8,previous_letter:u8,previous:&[Cell;65],previous2:Option<&[Cell;65]>) {
        *self.visited+=1;
        if *self.visited>=LIMIT || (*self.visited%256==0 && self.started.elapsed()>=DEADLINE) {*self.exhausted=true;return;}
        let n=self.input.len();let mut row=[Cell{edits:u8::MAX,cost:f64::INFINITY};65];row[0]=previous[0].add(1,1.4);
        let mut minimum=row[0].edits;
        for j in 1..=n {
            let equal=self.input[j-1]==letter;
            let cost=if equal {self.matches[j-1]}else if letter.is_ascii_lowercase() {self.substitutions[j-1][(letter-b'a') as usize]}else{1.4};
            row[j]=previous[j-1].add(u8::from(!equal),cost).best(previous[j].add(1,1.4)).best(row[j-1].add(1,1.2));
            if j>=2 && letter==self.input[j-2] && previous_letter==self.input[j-1] {
                if let Some(old)=previous2 {row[j]=row[j].best(old[j-2].add(1,1.1));}
            }
            minimum=minimum.min(row[j].edits);
        }
        if minimum>self.radius {return;}
        let f=self.dict.nodes[node as usize].frequency;
        if f>0 && row[n].edits<=self.radius {
            self.found.push(Hypothesis{text:self.dict.word(node),spatial_score:-row[n].cost,lexical_score:prior(f),
                edits:row[n].edits,completion:0,replace_before:n,literal:false});
        }
        for edge in self.dict.children(node) {self.walk(edge.child,edge.label,letter,&row,Some(previous));if *self.exhausted {return;}}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn low_frequency_nearest_word_is_not_hidden_by_distant_frequent_words() {
        let d=Dictionary::from_bytes(&pack("iterate 1\nintegrate 10000\ninterested 100000\nsentence 223\nserene 50\nexamples 124\nfixed 379\ntypo 12\n").unwrap()).unwrap();
        for (typed,target,edits) in [("itertate","iterate",1),("snerenxe","sentence",3),("examplwa","examples",2),("fized","fixed",1)] {
            let r=decode(&d,typed,&TouchModel::default(),"",1,1);
            assert!(!r.budget_exhausted);assert!(r.hypotheses.iter().any(|h|h.text==target && h.edits==edits && h.completion==0),"{typed}: {r:?}");
            assert!(r.hypotheses.iter().any(|h|h.literal && h.text==typed));
        }
    }
    #[test] fn center_taps_preserve_neighbor_prior_and_offsets_favor_the_touched_neighbor() {
        let mut touch=TouchModel::default();
        touch.centers[(b'z'-b'a') as usize]=Some(Point{x:0.,y:0.});
        touch.centers[(b'x'-b'a') as usize]=Some(Point{x:1.,y:0.});
        touch.centers[(b'r'-b'a') as usize]=Some(Point{x:3.,y:-2.});
        let near=robust_substitution(&touch,0,b'z',b'x');let far=robust_substitution(&touch,0,b'z',b'r');
        touch.points=vec![Some(Point{x:0.,y:0.})];
        assert!((robust_substitution(&touch,0,b'z',b'x')-near).abs()<1e-9);
        assert!(near<far);
        touch.points[0]=Some(Point{x:0.48,y:0.});
        assert!(robust_substitution(&touch,0,b'z',b'x')<near);
    }
}
