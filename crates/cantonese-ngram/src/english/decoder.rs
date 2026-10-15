//! Experimental joint decoder. Build-49 search remains available for rollback.
//! Scores are evidence, never probabilities. Replacement requires a calibrated policy.
use super::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const SEARCH_WIDTH: usize = 128;
pub const SCORE_WIDTH: usize = 32;
const MAX_STATES: usize = 16_384;
const MAX_SEARCH: Duration = Duration::from_millis(8);

/// Expand the licensed seed without mutating the build-49 dictionary. Generated
/// forms have a discounted prior; the contextual scorer still has to justify them.
pub fn expanded_dictionary(source: &str) -> Result<Vec<u8>, String> {
    let mut rows = BTreeMap::<String,u32>::new();
    for line in source.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let mut f=line.split_whitespace();
        let word=f.next().ok_or("Missing word")?;
        let frequency:u32=f.next().ok_or("Missing frequency")?.parse().map_err(|_|"Invalid frequency")?;
        if f.next().is_some() {return Err("Unexpected dictionary field".into());}
        rows.insert(word.into(),frequency);
    }
    let originals=rows.clone();
    for (word,f) in &originals {
        if word.len()<3 || !word.bytes().all(|c|c.is_ascii_lowercase()) {continue;}
        let mut forms=vec![];
        if word.ends_with('y') && !b"aeiou".contains(&word.as_bytes()[word.len()-2]) {
            forms.push(format!("{}ies",&word[..word.len()-1]));
            forms.push(format!("{}ied",&word[..word.len()-1]));
        } else if word.ends_with("ch") || word.ends_with("sh") || word.ends_with('x') || word.ends_with('z') {
            forms.push(format!("{word}es"));
        } else if !word.ends_with('s') {forms.push(format!("{word}s"));}
        if word.ends_with('e') && !word.ends_with("ee") {
            forms.push(format!("{word}d")); forms.push(format!("{}ing",&word[..word.len()-1]));
        } else {forms.push(format!("{word}ed"));forms.push(format!("{word}ing"));}
        for form in forms { if form.len()<=64 {rows.entry(form).or_insert((f/8).max(1));} }
    }
    pack(&rows.iter().map(|(w,f)|format!("{w} {f}\n")).collect::<String>())
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Hypothesis {
    pub text: String,
    pub spatial_score: f64,
    pub lexical_score: f64,
    pub edits: u8,
    pub completion: usize,
    pub replace_before: usize,
    pub literal: bool,
}
impl Hypothesis { pub fn score(&self) -> f64 { self.lexical_score + self.spatial_score } }

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DecodeResult {
    pub schema_version: u8,
    pub session: i64,
    pub revision: i64,
    pub hypotheses: Vec<Hypothesis>,
    pub visited: usize,
    pub budget_exhausted: bool,
    pub elapsed_micros: u64,
}

#[derive(Clone)]
struct Path { node: u32, pos: usize, cost: f64, edits: u8, tail: usize }
#[derive(Default)]
pub struct Decoder {
    // Only reuse exactly identical snapshots. Different touch samples, context,
    // resize or revision must never inherit evidence from a previous request.
    cached_key: Option<Vec<u64>>,
    cached: Option<DecodeResult>,
}
impl Decoder {
    pub fn decode(&mut self, dict: &Dictionary, typed: &str, touch: &TouchModel,
                  previous: &str, session: i64, revision: i64) -> DecodeResult {
        let key = snapshot(typed, touch, previous, session);
        if self.cached_key.as_ref() == Some(&key) {
            let mut result = self.cached.as_ref().unwrap().clone();
            result.revision = revision;
            return result;
        }
        let started = Instant::now();
        let valid = !typed.is_empty() && typed.len() <= 64 && typed.bytes().all(|c| c.is_ascii_lowercase() || c == b'\'');
        let mut visited = 0;
        let mut exhausted = false;
        let mut found = if valid { walk(dict, typed.as_bytes(), touch, started, &mut visited, &mut exhausted) } else { vec![] };
        if valid {
            let literal_freq = frequency(dict, typed);
            found.retain(|c| c.text != typed);
            found.push(Hypothesis { text: typed.into(), spatial_score: 0., lexical_score: prior(literal_freq),
                edits: 0, completion: 0, replace_before: typed.len(), literal: true });
            // Exact split hypotheses: the LM must judge phrase plausibility.
            for i in 1..typed.len() {
                let (a, b) = typed.split_at(i);
                let (fa, fb) = (frequency(dict, a), frequency(dict, b));
                if fa > 0 && fb > 0 {
                    found.push(Hypothesis { text: format!("{a} {b}"), spatial_score: -1.8,
                        lexical_score: (prior(fa) + prior(fb)) / 2., edits: 1, completion: 0,
                        replace_before: typed.len(), literal: false });
                }
            }
            // Remove exactly one immediately preceding space. No arbitrary
            // rewrite of older committed words, punctuation or line boundaries.
            if !previous.is_empty() && previous.len() <= 32 && previous.bytes().all(|c| c.is_ascii_lowercase()) {
                let joined = format!("{previous}{typed}");
                let f = frequency(dict, &joined);
                if f > 0 { found.push(Hypothesis { text: joined, spatial_score: -1.8,
                    lexical_score: prior(f), edits: 1, completion: 0,
                    replace_before: previous.len() + 1 + typed.len(), literal: false }); }
            }
        }
        found.sort_by(|a, b| b.score().total_cmp(&a.score()).then(a.text.cmp(&b.text)));
        // The literal is reserved even if 128 alternatives outrank it.
        let literal = found.iter().find(|h| h.literal).cloned();
        found.truncate(SEARCH_WIDTH);
        if let Some(literal) = literal { if !found.iter().any(|h| h.literal) { found.pop(); found.push(literal); } }
        let result = DecodeResult { schema_version: 2, session, revision, hypotheses: found, visited,
            budget_exhausted: exhausted, elapsed_micros: started.elapsed().as_micros() as u64 };
        self.cached_key = Some(key); self.cached = Some(result.clone()); result
    }
}
fn snapshot(typed: &str, touch: &TouchModel, previous: &str, session: i64) -> Vec<u64> {
    let mut key = vec![session as u64, typed.len() as u64, previous.len() as u64];
    key.extend(typed.bytes().chain(previous.bytes()).map(u64::from));
    for p in touch.centers.iter().chain(touch.points.iter()) {
        key.push(p.map_or(u64::MAX, |p| p.x.to_bits())); key.push(p.map_or(u64::MAX, |p| p.y.to_bits()));
    }
    key
}
fn frequency(dict: &Dictionary, text: &str) -> u32 { dict.find(text.as_bytes()).map_or(0, |i| dict.nodes[i as usize].frequency) }
fn prior(f: u32) -> f64 { (1. + f as f64).ln() * 0.55 }
fn walk(dict: &Dictionary, typed: &[u8], touch: &TouchModel, started: Instant,
        visited: &mut usize, exhausted: &mut bool) -> Vec<Hypothesis> {
    // Spatial evidence depends only on the event snapshot, not trie state.
    // Precompute it once; recomputing 26 Gaussians at each node wasted the budget.
    let matches:Vec<_>=typed.iter().enumerate().map(|(i,&c)|touch.match_cost(i,c)).collect();
    let substitutions:Vec<[f64;26]>=typed.iter().enumerate().map(|(i,&c)|
        std::array::from_fn(|j|0.45+touch.substitution(i,c,b'a'+j as u8)*0.5)).collect();
    let mut paths = vec![];
    let mut heap = BinaryHeap::new();
    let mut best = HashMap::<(u32, usize), f64>::new();
    let push = |p: Path, paths: &mut Vec<Path>, heap: &mut BinaryHeap<HeapEntry>, best: &mut HashMap<(u32, usize), f64>| {
        if p.cost > 12. || best.get(&(p.node, p.pos)).is_some_and(|&old| old <= p.cost) { return; }
        best.insert((p.node, p.pos), p.cost);
        let bound = prior(dict.nodes[p.node as usize].maximum) - p.cost;
        let id = paths.len(); paths.push(p); heap.push(HeapEntry { bound, id });
    };
    push(Path { node: 0, pos: 0, cost: 0., edits: 0, tail: 0 }, &mut paths, &mut heap, &mut best);
    let mut found: Vec<Hypothesis> = vec![];
    let mut floor = f64::NEG_INFINITY;
    while let Some(entry) = heap.pop() {
        if *visited >= MAX_STATES || (*visited % 64 == 0 && started.elapsed() >= MAX_SEARCH) { *exhausted = true; break; }
        if entry.bound < floor { break; }
        let p = paths[entry.id].clone();
        if best.get(&(p.node, p.pos)).is_some_and(|&cost| cost < p.cost) { continue; }
        *visited += 1;
        if p.pos == typed.len() {
            let f = dict.nodes[p.node as usize].frequency;
            if f > 0 { found.push(Hypothesis { text: dict.word(p.node), spatial_score: -p.cost,
                lexical_score: prior(f), edits: p.edits, completion: p.tail,
                replace_before: typed.len(), literal: false }); }
            if found.len() >= SEARCH_WIDTH {
                found.sort_by(|a,b| b.score().total_cmp(&a.score()).then(a.text.cmp(&b.text)));
                found.truncate(SEARCH_WIDTH); floor = found.last().unwrap().score();
            }
            if p.tail < 8 {
                for e in dict.children(p.node) {
                    push(Path { node: e.child, cost: p.cost + 0.9, tail: p.tail + 1, ..p }, &mut paths, &mut heap, &mut best);
                }
            }
            continue;
        }
        let actual = typed[p.pos];
        // No edit-count gate: every hypothesis pays accumulated spatial costs.
        push(Path { pos: p.pos + 1, cost: p.cost + 1.2, edits: p.edits + 1, ..p }, &mut paths, &mut heap, &mut best);
        for e in dict.children(p.node) {
            let substitution = if e.label == actual { matches[p.pos] }
                else if e.label.is_ascii_lowercase() { substitutions[p.pos][(e.label-b'a') as usize] }
                else {1.4};
            push(Path { node: e.child, pos: p.pos + 1, cost: p.cost + substitution,
                edits: p.edits + u8::from(e.label != actual), ..p }, &mut paths, &mut heap, &mut best);
            push(Path { node: e.child, cost: p.cost + 1.4, edits: p.edits + 1, ..p }, &mut paths, &mut heap, &mut best);
        }
        if p.pos + 1 < typed.len() && actual != typed[p.pos+1] {
            if let Some(a) = dict.child(p.node, typed[p.pos+1]) {
                if let Some(b) = dict.child(a, actual) {
                    push(Path { node: b, pos: p.pos+2, cost: p.cost+1.1, edits: p.edits+1, ..p }, &mut paths, &mut heap, &mut best);
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dict() -> Dictionary { Dictionary::from_bytes(&pack("hello 1000\nworld 900\nkeyboard 800\nin 600\nthe 4000\nedits 500\n").unwrap()).unwrap() }
    #[test] fn multiple_errors_and_spacing_keep_literal_and_original_span() {
        let d = dict(); let mut decoder = Decoder::default();
        for (input, want, previous, span) in [("gwllo","hello","",5), ("eduts","edits","",5),
            ("inthe","in the","",5), ("board","keyboard","key",9)] {
            let result = decoder.decode(&d,input,&TouchModel::default(),previous,1,1);
            assert!(result.hypotheses.iter().any(|h| h.literal && h.text == input));
            assert!(result.hypotheses.iter().any(|h| h.text == want && h.replace_before == span), "{result:?}");
            assert!(result.hypotheses.len() <= SEARCH_WIDTH);
        }
    }
    #[test] fn cache_is_scoped_by_geometry_input_and_session() {
        let d = dict(); let mut decoder=Decoder::default();
        let a=decoder.decode(&d,"gwllo",&TouchModel::default(),"",1,1);
        let b=decoder.decode(&d,"gwllo",&TouchModel::default(),"",1,2);
        assert_eq!(a.hypotheses[0].text,b.hypotheses[0].text); assert_eq!(b.revision,2);
        let c=decoder.decode(&d,"hello",&TouchModel::default(),"",2,1);
        assert_eq!(c.session,2); assert!(c.hypotheses.iter().any(|h| h.literal && h.text=="hello"));
        assert!(decoder.decode(&d,"https://x",&TouchModel::default(),"",2,2).hypotheses.is_empty());
    }
}
