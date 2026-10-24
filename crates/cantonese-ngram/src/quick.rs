//! Quick (速成)3/5 dictionary lookup and shared LM ranking; no Android dependency.
use crate::hybrid::{HybridLanguageModel, Mode};
use std::collections::HashSet;

/// Maximum relative log-probability deficit allowed for a candidate compared to the best
/// candidate of the same character count before being pruned as very low probability.
/// 8.0 log points corresponds to relative probability ~ e^-8 ≈ 0.033%.
const CHAR_COUNT_PRUNE_DELTA: f32 = 8.0;

/// Maximum relative log-probability deficit allowed for any candidate compared to the
/// global best candidate across all character counts at this position.
/// 12.0 log points corresponds to relative probability ~ e^-12 ≈ 0.0006%.
const GLOBAL_PRUNE_DELTA: f32 = 12.0;

/// Prune prior prefixes before expansion if their score is more than this delta behind
/// the best prefix at the start position. Prevents running expensive LM evaluations on dead beams.
const PREFIX_PRUNE_DELTA: f32 = 8.0;

/// Minimum step transition score. Tokens with log-probability below this threshold
/// (such as unsupported byte fallback or sentinel values <= -20.0) are discarded.
const MIN_STEP_TRANSITION_SCORE: f32 = -15.0;

/// Maximum beam caps per character count to bound worst-case memory and branching.
const MAX_UNIGRAM_BEAM: usize = 20;
const MAX_PHRASE_BEAM: usize = 12;
const MAX_SEGMENT_CANDIDATES: usize = 32;

/// Relative LM evidence for the two input interpretations, not a trained language classifier.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuickLanguageEvidence {
    pub english_probability: f32,
    pub chinese_probability: f32,
    pub english_support: f32,
    pub chinese_support: f32,
}
impl QuickLanguageEvidence {
    pub fn should_commit_chinese(&self) -> bool {
        self.chinese_probability >= 0.95 && self.chinese_support >= 0.02
    }
}

#[derive(Clone)]
struct Phrase {
    text: String,
    score: f32,
    joint: f32,
    characters: usize,
}

struct PhraseCache {
    version: u32,
    left: String,
    right: String,
    code: String,
    joined: bool,
    positions: Vec<Vec<Phrase>>,
}

#[derive(Clone)]
pub struct QuickDictionary {
    pub version: u32,
    codes: std::sync::Arc<Vec<Vec<String>>>,
}
impl QuickDictionary {
    pub fn from_bytes(bytes: &[u8], version: u32) -> Result<Self, String> {
        if ![3, 5].contains(&version) || bytes.len() < 703 * 4 {
            return Err("Invalid Quick dictionary".into());
        }
        let offsets: Vec<_> = bytes[..703 * 4]
            .chunks_exact(4)
            .map(|s| u32::from_be_bytes(s.try_into().unwrap()) as usize)
            .collect();
        let pool = &bytes[703 * 4..];
        if offsets[0] != 0 || offsets[702] != pool.len() || offsets.windows(2).any(|p| p[0] > p[1])
        {
            return Err("Invalid Quick offsets".into());
        }
        let mut codes = Vec::with_capacity(702);
        for pair in offsets.windows(2) {
            let text = std::str::from_utf8(&pool[pair[0]..pair[1]]).map_err(|e| e.to_string())?;
            codes.push(text.chars().map(|c| c.to_string()).collect());
        }
        Ok(Self { version, codes: std::sync::Arc::new(codes) })
    }
    pub fn code_index(code: &str) -> Option<usize> {
        let code = code.to_ascii_lowercase();
        let b = code.as_bytes();
        if !b.iter().all(u8::is_ascii_lowercase) {
            return None;
        }
        match b {
            [a] => Some((a - b'a') as usize),
            [a, b] => Some(26 + (a - b'a') as usize * 26 + (b - b'a') as usize),
            _ => None,
        }
    }
    pub fn candidates(&self, code: &str) -> &[String] {
        Self::code_index(code)
            .map(|i| self.codes[i].as_slice())
            .unwrap_or(&[])
    }
}
/// A compact representation of every dictionary-compatible phrase. Edges keep
/// the shared dictionary entries, so generation needs no LM or Cartesian expansion.
pub struct QuickCandidateLattice {
    version: u32,
    code: String,
    dictionary: QuickDictionary,
    transitions: Vec<Vec<(usize, usize)>>, // end position -> (start, dictionary index)
}
impl QuickCandidateLattice {
    fn source(&self, start: usize, end: usize) -> &[String] {
        self.transitions[end].iter().find(|(s, _)| *s == start)
            .map(|(_, index)| self.dictionary.codes[*index].as_slice()).unwrap_or(&[])
    }
}

pub struct QuickEngine {
    three: QuickDictionary,
    five: QuickDictionary,
    pub lm: HybridLanguageModel,
    cache: Option<PhraseCache>,
    distribution_cache: Option<(Vec<usize>, Vec<f32>)>,
}
impl QuickEngine {
    pub fn generate_lattice(&self, version: u32, code: &str) -> Result<QuickCandidateLattice, String> {
        let dictionary = match version { 3 => self.three.clone(), 5 => self.five.clone(), _ => return Err("Unknown Quick version".into()) };
        let code = code.to_ascii_lowercase();
        let valid = code.bytes().all(|b| b.is_ascii_lowercase());
        let mut transitions = vec![vec![]; code.len()+1];
        if valid {
            for end in 1..=code.len() {
                crate::suggestion_request::check()?;
                for size in 1..=2.min(end) {
                    if end == 2 && size == 1 && code.len() <= 2 { continue; }
                    let start = end-size;
                    let index = QuickDictionary::code_index(&code[start..end]).unwrap();
                    if !dictionary.codes[index].is_empty() { transitions[end].push((start,index)); }
                }
            }
        }
        Ok(QuickCandidateLattice { version, code, dictionary, transitions })
    }

    pub fn rank_preview_lattice(&mut self, lattice: &QuickCandidateLattice, left: &str, right: &str)
        -> Result<Vec<String>, String> {
        let code = &lattice.code;
        if code.len() <= 2 { return self.rank_characters(lattice.source(0,code.len()),left,right); }
        if !code.bytes().all(|c| c.is_ascii_alphabetic()) { return Ok(vec![]); }
        let mut prefix = String::new();
        for start in (0..code.len()).step_by(2) {
            crate::suggestion_request::check()?;
            let end = (start+2).min(code.len());
            let context = format!("{left}{prefix}");
            let choices = self.rank_characters(lattice.source(start,end), &context, right)?;
            if end == code.len() { return Ok(choices.into_iter().take(10).map(|s|format!("{prefix}{s}")).collect()); }
            let Some(first) = choices.first() else { return Ok(vec![]); };
            prefix.push_str(first);
        }
        Ok(vec![])
    }

    /// Fast Chinese pool: one greedy character path over the complete input,
    /// with alternatives at its last segment. Both stages then rank bilingually.
    pub fn preview_composition(&mut self, version: u32, code: &str, left: &str, right: &str)
        -> Result<Vec<String>, String> {
        let lattice = self.generate_lattice(version, code)?;
        self.rank_preview_lattice(&lattice, left, right)
    }

    /// First paint for short codes: rank modeled characters with one distribution,
    /// retaining every byte-fallback character in dictionary order at the end.
    /// Full sequence ranking follows asynchronously and remains authoritative.
    pub fn preview_candidates(&mut self, version: u32, code: &str, left: &str, right: &str)
        -> Result<Vec<String>, String> {
        let lattice = self.generate_lattice(version, code)?;
        if code.len() > 2 { return Ok(vec![]); }
        self.rank_characters(lattice.source(0,code.len()),left,right)
    }

    fn rank_characters(&mut self, source: &[String], left: &str, right: &str) -> Result<Vec<String>,String> {
        if source.is_empty() { return Ok(Vec::new()); }
        if self.lm.mode(left, right) != Mode::Neural {
            let order = self.lm.rank(left, right, source)?;
            return Ok(order.into_iter().map(|i| source[i].clone()).collect());
        }
        let neural = self.lm.neural.as_mut().unwrap();
        let context = neural.context(left, right);
        let values = neural.log_probabilities(&context)?;
        let mut scored = Vec::new();
        let mut tail = Vec::new();
        for (i, text) in source.iter().enumerate() {
            let tokens = neural.tokenize(text);
            if tokens.len() == 1 && neural.supports_candidate(&context, tokens[0]) {
                scored.push((i, values[tokens[0]]));
            } else {
                tail.push(text.clone());
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut result: Vec<_> = scored.into_iter().map(|(i, _)| source[i].clone()).collect();
        result.extend(tail);
        Ok(result)
    }

    pub fn new(three: QuickDictionary, five: QuickDictionary, lm: HybridLanguageModel) -> Self {
        Self {
            three,
            five,
            lm,
            cache: None,
            distribution_cache: None,
        }
    }
    /// Start from the full next-token distribution, filter by the dictionary
    /// code, then retain the best matching transitions. Never truncate the
    /// global vocabulary or dictionary before applying the code constraint.
    pub fn collect_candidates(&mut self, version:u32, code:&str, left:&str, right:&str)->Result<Vec<String>,String> {
        let lattice = self.generate_lattice(version,code)?;
        self.rank_lattice(&lattice,left,right,true)
    }

    pub fn candidates(
        &mut self,
        version: u32,
        code: &str,
        left: &str,
        right: &str,
    ) -> Result<Vec<String>, String> {
        self.candidates_impl(version,code,left,right,false)
    }
    fn candidates_impl(&mut self,version:u32,code:&str,left:&str,right:&str,joined:bool)->Result<Vec<String>,String> {
        let lattice = self.generate_lattice(version,code)?;
        self.rank_lattice(&lattice,left,right,joined)
    }

    pub fn rank_lattice(&mut self, lattice: &QuickCandidateLattice, left: &str, right: &str, joined: bool) -> Result<Vec<String>,String> {
        let version = lattice.version;
        let dictionary = &lattice.dictionary;
        let code = &lattice.code;
        if code.is_empty() || !code.bytes().all(|b| b.is_ascii_lowercase()) { return Ok(vec![]); }
        if joined && code.len() <= 2 { return self.rank_characters(lattice.source(0,code.len()),left,right); }
        if self
            .cache
            .as_ref()
            .map(|c| c.version != version || c.left != left || c.right != right || c.joined != joined)
            .unwrap_or(true)
        {
            self.cache = Some(PhraseCache {
                version,
                left: left.into(),
                right: right.into(),
                code: String::new(),
                joined,
                positions: vec![vec![Phrase {
                    text: String::new(),
                    score: 0.,
                    joint: 0.,
                    characters: 0,
                }]],
            });
        }
        // One background distribution lets final phrase ranking measure internal
        // association without another pass for each candidate. Short previews and
        // single-code ordering keep their existing path.
        let background = if code.len()>2 && !joined {
            if let Some(model)=self.lm.neural.as_mut().filter(|m|m.has_byte_fallback()) {
                let ids=model.context(left,right); Some(model.log_probabilities(&ids)?)
            } else {None}
        } else {None};
        let cache = self.cache.as_mut().unwrap();
        let common = cache
            .code
            .bytes()
            .zip(code.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        cache.positions.truncate(common + 1);
        // Each position considers both a root code and a two-letter Quick code.
        // Keep separate beams by character count so root-only and two-key
        // segmentations remain available for the complete composition.
        for end in cache.positions.len()..=code.len() {
            crate::suggestion_request::check()?;
            let mut paths = Vec::new();
            for &(start, index) in &lattice.transitions[end] {
                let source = lattice.dictionary.codes[index].as_slice();
                // ARM mobile optimization: dictionary candidates are ordered by frequency.
                let source_slice = if !joined && source.len() > MAX_SEGMENT_CANDIDATES { &source[..MAX_SEGMENT_CANDIDATES] } else { source };
                let best_prefix_score = cache.positions[start].first().map(|p| p.score).unwrap_or(0.0);
                let prefixes:Vec<_>=cache.positions[start].iter().filter(|prefix|
                    prefix.characters==0 || prefix.score>=best_prefix_score-PREFIX_PRUNE_DELTA).collect();
                let contexts:Vec<_>=prefixes.iter().map(|p|format!("{left}{}",p.text)).collect();
                let all_scores=sequence_scores_many(&mut self.lm,&mut self.distribution_cache,&contexts,right,source_slice)?;
                for (prefix,scores) in prefixes.into_iter().zip(all_scores) {
                    crate::suggestion_request::check()?;
                    let mut matching:Vec<_>=(0..source_slice.len()).collect();
                    if joined {
                        matching.sort_by(|&a,&b|scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                        matching.truncate(MAX_SEGMENT_CANDIDATES);
                    }
                    for index in matching {
                        let text=&source_slice[index];let score=scores[index];
                        // Skip impossible / unsupported transitions with negligible probability
                        if !score.is_finite() || (prefix.characters > 0 && score < MIN_STEP_TRANSITION_SCORE) {
                            continue;
                        }
                        let boost = if prefix.characters > 0 && score > -5.0 {
                            (score + 5.0) * 0.5
                        } else {
                            0.0
                        };
                        paths.push(Phrase {
                            text: format!("{}{text}", prefix.text),
                            score: prefix.score + score + boost,
                            joint: prefix.joint + score,
                            characters: prefix.characters + 1,
                        });
                    }
                }
            }
            paths.sort_by(|a, b| b.score.total_cmp(&a.score));

            // Intelligent probability pruning:
            // 1. Find best score overall and per character-count.
            let best_score = paths.first().map(|p| p.score).unwrap_or(0.0);
            let mut best_by_chars = vec![f32::NEG_INFINITY; end + 1];
            for p in &paths {
                if p.score > best_by_chars[p.characters] {
                    best_by_chars[p.characters] = p.score;
                }
            }

            let mut counts = vec![0; end + 1];
            let mut seen = HashSet::new();
            paths.retain(|p| {
                // Prune beams when they become very low probability:
                // - worse than best score for this character count by more than CHAR_COUNT_PRUNE_DELTA
                // - or worse than overall best score by more than GLOBAL_PRUNE_DELTA
                let char_best = best_by_chars[p.characters];
                if p.score < char_best - CHAR_COUNT_PRUNE_DELTA || p.score < best_score - GLOBAL_PRUNE_DELTA {
                    return false;
                }
                let cap = if p.characters <= 1 { MAX_UNIGRAM_BEAM } else { MAX_PHRASE_BEAM };
                if counts[p.characters] >= cap || !seen.insert(p.text.clone()) {
                    return false;
                }
                counts[p.characters] += 1;
                true
            });
            cache.positions.push(paths);
        }
        cache.code = code.clone();
        let phrases = &cache.positions[code.len()];
        if code.len() <= 2 {
            // Preserve every single-character candidate and its established order.
            let source = dictionary.candidates(&code);
            let order = self.lm.rank(left, right, source)?;
            let mut result: Vec<_> = order.into_iter().map(|i| source[i].clone()).collect();
            result.extend(
                phrases
                    .iter()
                    .filter(|p| p.characters > 1)
                    .map(|p| p.text.clone()),
            );
            Ok(result)
        } else {
            // This is the generated pool. The joined request ranks it once with English.
            if joined {return Ok(phrases.iter().map(|p|p.text.clone()).collect());}
            let mut order: Vec<_> = (0..phrases.len()).collect();
            if let Some(background) = &background {
                let model = self.lm.neural.as_ref().unwrap();
                let scores: Vec<_> = phrases.iter().map(|p| {
                    let independent: Option<Vec<_>> = p.text.chars().map(|c| model.token_id(&c.to_string())
                        .map(|id|background[id])).collect();
                    let cohesion = independent.map(|v|p.joint-v.iter().sum::<f32>()).unwrap_or(0.);
                    mixed_score([p.joint as f64,(p.joint/p.characters as f32) as f64,cohesion as f64,0.])
                }).collect();
                order.sort_by(|&a,&b|scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            }
            Ok(order.into_iter().take(100).map(|i|phrases[i].text.clone()).collect())
        }
    }

    pub fn language_evidence(&mut self, code: &str, left: &str, right: &str, chinese: &str)
        -> Result<Option<QuickLanguageEvidence>, String> {
        let Some(neural) = self.lm.neural.as_mut() else { return Ok(None); };
        let ids = neural.context(left, right);
        if !neural.supports_context(&ids) || code.len() < 6 || chinese.is_empty() { return Ok(None); }
        let values = neural_log_probabilities(neural, left, right)?;
        let code = code.to_ascii_lowercase();
        let mut english_mass = 0.;
        let mut matched_english = 0.;
        for (id, &logp) in values.iter().enumerate().skip(3) {
            let token = neural.token(id).unwrap_or("");
            if !token.is_empty() && token.bytes().all(|c| c.is_ascii_alphabetic() || c == b'\'') {
                let probability = logp.exp();
                english_mass += probability;
                if let Some(cost) = english_prefix_cost(&code, &token.to_ascii_lowercase()) {
                    matched_english += probability * (-cost).exp();
                }
            }
        }
        // Names and out-of-vocabulary English still have support. Absence from a
        // dictionary/model vocabulary must never mean zero probability of English.
        let english_support = matched_english.max(english_mass * 0.1).max(1e-6);
        let mut context = left.to_owned();
        let mut total = 0.;
        let mut count = 0;
        for character in chinese.chars() {
            if !crate::tokenizer::is_han(character) { return Ok(None); }
            let probabilities = if count == 0 { values.clone() }
                else { neural_log_probabilities(neural, &context, right)? };
            let token = character.to_string();
            let Some(id) = neural.token_id(&token) else { return Ok(None); };
            if !probabilities[id].is_finite() { return Ok(None); }
            total += probabilities[id]; count += 1;
            context.push(character);
        }
        let chinese_support = (total / count as f32).exp();
        let sum = chinese_support + english_support;
        Ok(Some(QuickLanguageEvidence {
            english_probability: english_support / sum, chinese_probability: chinese_support / sum,
            english_support, chinese_support,
        }))
    }
}

fn neural_log_probabilities(neural: &mut crate::NeuralLanguageModel, left: &str, right: &str) -> Result<Vec<f32>, String> {
    let ids = neural.context(left, right);
    neural.log_probabilities(&ids)
}

// Banded edit distance against English LM-token prefixes. The LM supplies their
// probability; spelling only relates the observed keys to possible token spellings.
fn english_prefix_cost(typed: &str, token: &str) -> Option<f32> {
    let typed = typed.as_bytes();
    let token = token.as_bytes();
    if token.len() + 2 < typed.len() { return None; }
    let limit = token.len().min(typed.len() + 2);
    let mut previous: Vec<_> = (0..=limit).map(|n| n.min(3)).collect();
    for (i, &letter) in typed.iter().enumerate() {
        let row = i + 1;
        let mut current = vec![3; limit + 1];
        current[0] = row.min(3);
        for column in row.saturating_sub(2).max(1)..=(row + 2).min(limit) {
            current[column] = (previous[column] + 1).min(current[column - 1] + 1)
                .min(previous[column - 1] + usize::from(letter != token[column - 1]));
        }
        if current.iter().all(|&n| n > 2) { return None; }
        previous = current;
    }
    (typed.len().saturating_sub(2)..=limit).filter(|&end| previous[end] <= 2)
        .map(|end| previous[end] as f32 * 2. + (token.len() - end) as f32 * 0.15)
        .min_by(f32::total_cmp)
}

/// Key likelihood is exact for generated Quick candidates, and edit distance for
/// English alternatives. It is evidence for calibration, not a language priority.
pub fn mixed_features(e: &crate::neural::SequenceEvidence, text: &str, observed: &str) -> [f64; 4] {
    let cost = if observed.is_empty() || text.chars().any(crate::tokenizer::is_han) { 0. } else {
        let a = observed.to_ascii_lowercase(); let b = text.to_ascii_lowercase();
        let a = a.as_bytes(); let b = b.as_bytes();
        let mut before = vec![0; b.len()+1];
        let mut previous: Vec<usize> = (0..=b.len()).collect();
        for i in 1..=a.len() {
            let mut current = vec![i; b.len()+1];
            for j in 1..=b.len() {
                current[j] = (previous[j]+1).min(current[j-1]+1).min(previous[j-1]+usize::from(a[i-1]!=b[j-1]));
                if i>1 && j>1 && a[i-1]==b[j-2] && a[i-2]==b[j-1] {current[j]=current[j].min(before[j-2]+1);}
            }
            before=previous; previous=current;
        }
        previous[b.len()] as f64
    };
    [e.log_probability as f64,e.unit_log_probability as f64,e.cohesion_log_probability as f64,cost]
}

pub fn mixed_score(features: [f64;4]) -> f64 {
    #[derive(serde::Deserialize)]
    struct Policy { weights: [f64;4] }
    static POLICY: std::sync::OnceLock<Policy> = std::sync::OnceLock::new();
    let p=POLICY.get_or_init(||serde_json::from_str(include_str!("mixed_ranking.json")).expect("Invalid mixed ranking policy"));
    (0..4).map(|i|p.weights[i]*features[i]).sum()
}

/// Calibrated sequence, per-unit, association and key evidence on one shared LM
/// scale. Candidate scripts do not impose a priority on the final ordering.
pub fn rank_mixed(lm: &mut HybridLanguageModel, left: &str, right: &str, candidates: &[String], observed: &str) -> Result<Vec<usize>, String> {
    if let Some(model)=lm.neural.as_mut().filter(|m|m.has_byte_fallback()) {
        let scores=crate::neural::score_sequences(model,left,right,candidates)?;
        let values: Vec<_> = scores.iter().zip(candidates).map(|(e,text)| mixed_score(mixed_features(e,text,observed))).collect();
        let mut order:Vec<_>=(0..scores.len()).collect();
        order.sort_by(|&a,&b|values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        return Ok(order);
    }
    lm.rank(left, right, candidates)
}

// Normalize over the full neural vocabulary. Normalizing only over one code's
// matches would give a one-letter root with one match an unfair probability of 1.
fn sequence_scores_many(lm:&mut HybridLanguageModel, distribution_cache:&mut Option<(Vec<usize>,Vec<f32>)>,
    contexts:&[String],right:&str,source:&[String])->Result<Vec<Vec<f32>>,String> {
    if matches!(lm.neural,Some(crate::NeuralLanguageModel::Cnn(_))) {
        let neural=lm.neural.as_mut().unwrap();
        let ids_by_candidate:Vec<_>=source.iter().map(|text|neural.token_id(text)).collect();
        let targets:Vec<_>=ids_by_candidate.iter().filter_map(|&id|id).collect();
        let requests:Vec<_>=contexts.iter().map(|left|(neural.context(left,right),targets.clone())).collect();
        return Ok(neural.log_probabilities_for_many(&requests)?.into_iter().map(|scores| {
            let mut scores=scores.into_iter();
            ids_by_candidate.iter().enumerate().map(|(i,id)|id.and_then(|_|scores.next())
                .filter(|v|v.is_finite()).unwrap_or(-20.-(i as f32+1.).ln())).collect()
        }).collect());
    }
    contexts.iter().map(|left|sequence_scores(lm,distribution_cache,left,right,source)).collect()
}

fn sequence_scores(
    lm: &mut HybridLanguageModel,
    distribution_cache: &mut Option<(Vec<usize>, Vec<f32>)>,
    left: &str,
    right: &str,
    source: &[String],
) -> Result<Vec<f32>, String> {
    if lm.mode(left, right) == Mode::Neural {
        let neural = lm.neural.as_mut().unwrap();
        let ids = neural.context(left, right);
        if matches!(neural, crate::NeuralLanguageModel::Cnn(_)) {
            let ids_by_candidate: Vec<_> = source.iter().map(|text| neural.token_id(text)).collect();
            let targets: Vec<_> = ids_by_candidate.iter().filter_map(|&id| id).collect();
            let scores = neural.log_probabilities_for(&ids, &targets)?;
            let mut scores = scores.into_iter();
            return Ok(ids_by_candidate.into_iter().enumerate().map(|(i, id)| {
                id.and_then(|_| scores.next()).filter(|v| v.is_finite())
                    .unwrap_or(-20. - (i as f32 + 1.).ln())
            }).collect());
        }
        let hit = distribution_cache.as_ref().map(|(cached_ids, _)| cached_ids == &ids).unwrap_or(false);
        if !hit {
            let values = neural.log_probabilities(&ids)?;
            *distribution_cache = Some((ids.clone(), values));
        }
        let values = &distribution_cache.as_ref().unwrap().1;
        let neural = lm.neural.as_ref().unwrap();
        return Ok(source
            .iter()
            .enumerate()
            .map(|(i, token)| {
                neural
                    .token_id(token)
                    .map(|id| values[id])
                    .filter(|v| v.is_finite())
                    .unwrap_or(-20. - (i as f32 + 1.).ln())
            })
            .collect());
    }
    let order = lm.rank(left, right, source)?;
    let mut scores = vec![0.; source.len()];
    for (rank, index) in order.into_iter().enumerate() {
        scores[index] = -3. - (rank as f32 + 1.).ln();
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidate_lattice_retains_all_dictionary_edges_without_inference() {
        let engine=engine();
        for version in [3,5] {
            for code in ["hi","hion","qhion","hionhionhionhion"] {
                crate::suggestion_request::run(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),|| {
                    let graph=engine.generate_lattice(version,code)?;
                    assert_eq!(graph.transitions.len(),code.len()+1);
                    for end in 1..=code.len() {
                        for &(start,index) in &graph.transitions[end] {
                            assert_eq!(graph.source(start,end),graph.dictionary.codes[index]);
                            assert_eq!(Some(index),QuickDictionary::code_index(&code[start..end]));
                        }
                    }
                    assert_eq!(crate::suggestion_request::stats().forwards,0);
                    Ok(())
                }).unwrap();
            }
            let graph=engine.generate_lattice(version,&"hion".repeat(70)).unwrap();
            assert_eq!(graph.transitions.len(),281);
        }
    }

    #[test]
    fn language_evidence_requires_a_model_and_strong_chinese_support() {
        assert!(engine().language_evidence("qqqqqq", "", "", "手手手手手手").unwrap().is_none());
        let strong = QuickLanguageEvidence { english_probability: 0.01, chinese_probability: 0.99,
            english_support: 0.001, chinese_support: 0.1 };
        assert!(strong.should_commit_chinese());
        assert!(!QuickLanguageEvidence { chinese_probability: 0.8, ..strong.clone() }.should_commit_chinese());
        assert!(!QuickLanguageEvidence { chinese_support: 0.001, ..strong }.should_commit_chinese());
        assert_eq!(english_prefix_cost("keyboa", "keyboard"), Some(0.3));
        assert_eq!(english_prefix_cost("engkish", "english"), Some(2.));
        assert!(english_prefix_cost("tonysunusualname", "keyboard").is_none());
    }
    #[test]
    fn code_validation_and_truncation() {
        assert_eq!(QuickDictionary::code_index("OF"), Some(26 + 14 * 26 + 5));
        for code in ["", "abc", "ø", "a1"] {
            assert_eq!(QuickDictionary::code_index(code), None);
        }
        assert!(QuickDictionary::from_bytes(&[0; 4], 3).is_err());
    }
    fn engine() -> QuickEngine {
        fn dictionary(version: u32) -> QuickDictionary {
            let mut codes = vec![Vec::new(); 702];
            for (code, text) in [
                ("q", "手"),
                ("h", "竹"),
                ("i", "戈"),
                ("o", "人"),
                ("n", "弓"),
                ("hi", "我得"),
                ("on", "們今"),
                ("qh", "毒"),
            ] {
                codes[QuickDictionary::code_index(code).unwrap()] =
                    text.chars().map(|c| c.to_string()).collect();
            }
            QuickDictionary { version, codes: codes.into() }
        }
        QuickEngine::new(
            dictionary(3),
            dictionary(5),
            HybridLanguageModel::new(None, None, &[]).unwrap(),
        )
    }

    #[test]
    fn short_previews_keep_all_matches_and_reject_phrase_codes() {
        let mut engine = engine();
        for version in [3, 5] {
            for code in ["q", "hi", "HI", "zz"] {
                assert_eq!(engine.preview_candidates(version, code, "", "").unwrap(),
                    engine.candidates(version, code, "", "").unwrap());
            }
            for code in ["", "hion", "a1", "ø"] {
                assert!(engine.preview_candidates(version, code, "", "").unwrap().is_empty());
            }
        }
        assert!(engine.preview_candidates(4, "hi", "", "").is_err());
    }

    #[test]
    fn composition_preview_matches_complete_code_and_preserves_final_segment_choices() {
        let mut engine = engine();
        for version in [3, 5] {
            assert_eq!(engine.preview_composition(version, "HIon", "", "").unwrap(), ["我們", "我今"]);
            assert_eq!(engine.preview_composition(version, "hio", "", "").unwrap(), ["我人"]);
            for code in ["hi", "hion", "hio"] {
                let preview = engine.preview_composition(version, code, "", "").unwrap();
                let full = engine.candidates(version, code, "", "").unwrap();
                assert!(preview.iter().all(|s| full.contains(s)));
            }
            assert!(engine.preview_composition(version, "hi!", "", "").unwrap().is_empty());
        }
    }

    #[test]
    fn continuous_codes_include_phrases_and_root_segmentations() {
        let mut engine = engine();
        for version in [3, 5] {
            assert_eq!(engine.candidates(version, "q", "", "").unwrap(), ["手"]);
            let pairs = engine.candidates(version, "HION", "", "").unwrap();
            assert_eq!(pairs[0], "我們");
            assert!(pairs.contains(&"我今".into()));
            let mixed = engine.candidates(version, "qhion", "", "").unwrap();
            assert!(mixed.contains(&"手我們".into()));
            assert!(mixed.contains(&"毒戈們".into()));
            assert_eq!(
                engine.candidates(version, "hi", "", "").unwrap()[..2],
                ["我", "得"]
            );
        }
    }

    #[test]
    fn unlimited_composition_and_cache_handle_edits_and_context_changes() {
        let mut engine = engine();
        for version in [3, 5] {
            for count in [6, 7, 13, 24] {
                let code = "q".repeat(count);
                let expected = "手".repeat(count);
                assert_eq!(engine.candidates(version, &code, "", "").unwrap(), [expected.clone()]);
            }
            let code = "hion".repeat(4);
            assert!(!engine.collect_candidates(version, &code, "", "").unwrap().is_empty());
            let preview = engine.preview_composition(version, &code, "", "").unwrap();
            assert!(!preview.is_empty());
            assert!(preview.iter().all(|p| p.chars().count() == 8));
        }
        assert!(engine.candidates(3, "hi!", "", "").unwrap().is_empty());
        for (code, left, right) in [
            ("hion", "", ""),
            ("hio", "", ""),
            ("hiq", "", ""),
            ("hion", "我", "呀"),
            ("hion", "你", ""),
            ("hion", "", ""),
        ] {
            assert_eq!(
                engine.candidates(3, code, left, right).unwrap(),
                self::engine().candidates(3, code, left, right).unwrap()
            );
        }
    }
}
