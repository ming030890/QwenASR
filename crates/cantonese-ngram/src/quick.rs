//! Quick (速成)3/5 dictionary lookup and shared LM ranking; no Android dependency.
use crate::hybrid::{HybridLanguageModel, Mode};
use std::collections::HashSet;

pub const MAX_QUICK_CHARACTERS: usize = 6;
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
    characters: usize,
}

struct PhraseCache {
    version: u32,
    left: String,
    right: String,
    code: String,
    positions: Vec<Vec<Phrase>>,
}

pub struct QuickDictionary {
    pub version: u32,
    codes: Vec<Vec<String>>,
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
        Ok(Self { version, codes })
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
pub struct QuickEngine {
    three: QuickDictionary,
    five: QuickDictionary,
    pub lm: HybridLanguageModel,
    cache: Option<PhraseCache>,
    distribution_cache: Option<(Vec<usize>, Vec<f32>)>,
}
impl QuickEngine {
    pub fn new(three: QuickDictionary, five: QuickDictionary, lm: HybridLanguageModel) -> Self {
        Self {
            three,
            five,
            lm,
            cache: None,
            distribution_cache: None,
        }
    }
    pub fn candidates(
        &mut self,
        version: u32,
        code: &str,
        left: &str,
        right: &str,
    ) -> Result<Vec<String>, String> {
        let dictionary = match version {
            3 => &self.three,
            5 => &self.five,
            _ => return Err("Unknown Quick version".into()),
        };
        let code = code.to_ascii_lowercase();
        if code.is_empty()
            || code.len() > MAX_QUICK_CHARACTERS * 2
            || !code.bytes().all(|b| b.is_ascii_lowercase())
        {
            return Ok(Vec::new());
        }
        if self
            .cache
            .as_ref()
            .map(|c| c.version != version || c.left != left || c.right != right)
            .unwrap_or(true)
        {
            self.cache = Some(PhraseCache {
                version,
                left: left.into(),
                right: right.into(),
                code: String::new(),
                positions: vec![vec![Phrase {
                    text: String::new(),
                    score: 0.,
                    characters: 0,
                }]],
            });
        }
        let cache = self.cache.as_mut().unwrap();
        let common = cache
            .code
            .bytes()
            .zip(code.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        cache.positions.truncate(common + 1);
        // Each position considers both a root code and a two-letter Quick code.
        // Keep separate beams by character count so one-letter codes cannot crowd
        // out longer codes that are needed to stay within the six-character cap.
        for end in cache.positions.len()..=code.len() {
            let mut paths = Vec::new();
            for size in 1..=2.min(end) {
                // In Quick, when the user has typed only 2 letters, it represents a single
                // 2-letter character code. Expanding size=1 (two 1-letter root keys) wastes
                // ~20 neural forward passes to find 2-character root phrases that are never
                // displayed ahead of the 2-letter character candidates.
                if end == 2 && size == 1 && code.len() <= 2 {
                    continue;
                }
                let start = end - size;
                let source = dictionary.candidates(&code[start..end]);
                if source.is_empty() {
                    continue;
                }
                // ARM mobile optimization: dictionary candidates are ordered by frequency.
                let source_slice = if source.len() > MAX_SEGMENT_CANDIDATES { &source[..MAX_SEGMENT_CANDIDATES] } else { source };
                let best_prefix_score = cache.positions[start].first().map(|p| p.score).unwrap_or(0.0);
                for prefix in &cache.positions[start] {
                    if prefix.characters == MAX_QUICK_CHARACTERS {
                        continue;
                    }
                    // Prune prefixes that have very low probability compared to best at this position
                    if prefix.characters > 0 && prefix.score < best_prefix_score - PREFIX_PRUNE_DELTA {
                        continue;
                    }
                    let context = format!("{left}{}", prefix.text);
                    let scores = sequence_scores(&mut self.lm, &mut self.distribution_cache, &context, right, source_slice)?;
                    for (text, score) in source_slice.iter().zip(scores) {
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
                            characters: prefix.characters + 1,
                        });
                    }
                }
            }
            paths.sort_by(|a, b| b.score.total_cmp(&a.score));

            // Intelligent probability pruning:
            // 1. Find best score overall and per character-count.
            let best_score = paths.first().map(|p| p.score).unwrap_or(0.0);
            let mut best_by_chars = [f32::NEG_INFINITY; MAX_QUICK_CHARACTERS + 1];
            for p in &paths {
                if p.score > best_by_chars[p.characters] {
                    best_by_chars[p.characters] = p.score;
                }
            }

            let mut counts = [0; MAX_QUICK_CHARACTERS + 1];
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
            Ok(phrases.iter().take(100).map(|p| p.text.clone()).collect())
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

/// One LM scale for Chinese phrases and English words. Average token log probability
/// avoids treating a multi-character Chinese phrase as several competing candidates.
pub fn rank_mixed(lm: &mut HybridLanguageModel, left: &str, right: &str, candidates: &[String]) -> Result<Vec<usize>, String> {
    if let Some(model)=lm.neural.as_mut().filter(|m|m.has_byte_fallback()) {
        let scores=crate::neural::score_sequences(model,left,right,candidates)?;
        let mut order:Vec<_>=(0..scores.len()).collect();
        order.sort_by(|&a,&b| {
            let a_han = candidates[a].chars().any(crate::tokenizer::is_han);
            let b_han = candidates[b].chars().any(crate::tokenizer::is_han);
            if a_han != b_han {
                return b_han.cmp(&a_han);
            }
            scores[b].unit_log_probability.total_cmp(&scores[a].unit_log_probability).then(a.cmp(&b))
        });
        return Ok(order);
    }
    if lm.mode(left, right) != Mode::Neural {
        let mut order = lm.rank(left, right, candidates)?;
        order.sort_by(|&a, &b| {
            let a_han = candidates[a].chars().any(crate::tokenizer::is_han);
            let b_han = candidates[b].chars().any(crate::tokenizer::is_han);
            if a_han != b_han {
                b_han.cmp(&a_han)
            } else {
                std::cmp::Ordering::Equal
            }
        });
        return Ok(order);
    }
    let mut cache: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let tokens: Vec<String> = if candidate.is_ascii() {
            vec![candidate.to_ascii_lowercase()]
        } else {
            candidate.chars().map(|c| c.to_string()).collect()
        };
        let mut context = left.to_owned();
        let mut total = 0.;
        for token in &tokens {
            if !cache.contains_key(&context) {
                let neural = lm.neural.as_mut().unwrap();
                let ids = neural.context(&context, right);
                let targets: Vec<_> = (3..neural.vocabulary_size()).filter(|&id| neural.supports_candidate(&ids, id)).collect();
                let logits = neural.scores(&ids, &targets)?;
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let logsum = max + logits.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
                let mut values = vec![-20.; neural.vocabulary_size()];
                for (id, score) in targets.into_iter().zip(logits) { values[id] = score - logsum; }
                cache.insert(context.clone(), values);
            }
            let neural = lm.neural.as_ref().unwrap();
            total += neural.token_id(token).map(|id| cache[&context][id]).unwrap_or(-20.);
            context.push_str(token);
        }
        scores.push(if tokens.is_empty() { f32::NEG_INFINITY } else { total / tokens.len() as f32 });
    }
    let mut order: Vec<_> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| {
        let a_han = candidates[a].chars().any(crate::tokenizer::is_han);
        let b_han = candidates[b].chars().any(crate::tokenizer::is_han);
        if a_han != b_han {
            return b_han.cmp(&a_han);
        }
        scores[b].total_cmp(&scores[a]).then(a.cmp(&b))
    });
    Ok(order)
}

// Normalize over the full neural vocabulary. Normalizing only over one code's
// matches would give a one-letter root with one match an unfair probability of 1.
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
            QuickDictionary { version, codes }
        }
        QuickEngine::new(
            dictionary(3),
            dictionary(5),
            HybridLanguageModel::new(None, None, &[]).unwrap(),
        )
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
    fn cap_counts_characters_and_cache_handles_edits_and_context_changes() {
        let mut engine = engine();
        for code in ["qqqqqq", "hionhionhion", "qhionqhion"] {
            let phrases = engine.candidates(3, code, "", "").unwrap();
            assert!(!phrases.is_empty());
            assert!(phrases.iter().all(|p| p.chars().count() <= 6));
        }
        assert!(engine.candidates(3, "qqqqqqq", "", "").unwrap().is_empty());
        assert!(engine
            .candidates(3, "hionhionhionhi", "", "")
            .unwrap()
            .is_empty());
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
