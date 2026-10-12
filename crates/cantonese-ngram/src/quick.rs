//! Quick (速成)3/5 dictionary lookup and shared LM ranking; no Android dependency.
use crate::hybrid::{HybridLanguageModel, Mode};
use std::collections::HashSet;

pub const MAX_QUICK_CHARACTERS: usize = 6;
const BEAM_WIDTH: usize = 16;

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
}
impl QuickEngine {
    pub fn new(three: QuickDictionary, five: QuickDictionary, lm: HybridLanguageModel) -> Self {
        Self {
            three,
            five,
            lm,
            cache: None,
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
                let start = end - size;
                let source = dictionary.candidates(&code[start..end]);
                if source.is_empty() {
                    continue;
                }
                for prefix in &cache.positions[start] {
                    if prefix.characters == MAX_QUICK_CHARACTERS {
                        continue;
                    }
                    let context = format!("{left}{}", prefix.text);
                    let scores = sequence_scores(&mut self.lm, &context, right, source)?;
                    for (text, score) in source.iter().zip(scores) {
                        paths.push(Phrase {
                            text: format!("{}{text}", prefix.text),
                            score: prefix.score + score,
                            characters: prefix.characters + 1,
                        });
                    }
                }
            }
            paths.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut counts = [0; MAX_QUICK_CHARACTERS + 1];
            let mut seen = HashSet::new();
            paths.retain(|p| {
                if counts[p.characters] >= BEAM_WIDTH || !seen.insert(p.text.clone()) {
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
}

// Normalize over the full neural vocabulary. Normalizing only over one code's
// matches would give a one-letter root with one match an unfair probability of 1.
fn sequence_scores(
    lm: &mut HybridLanguageModel,
    left: &str,
    right: &str,
    source: &[String],
) -> Result<Vec<f32>, String> {
    if lm.mode(left, right) == Mode::Neural {
        let neural = lm.neural.as_mut().unwrap();
        let ids = neural.context(left, right);
        let targets: Vec<_> = (3..neural.vocabulary_size())
            .filter(|&id| neural.supports_candidate(&ids, id))
            .collect();
        let logits = neural.scores(&ids, &targets)?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let logsum = max + logits.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
        let mut values = vec![f32::NEG_INFINITY; neural.vocabulary_size()];
        for (id, score) in targets.into_iter().zip(logits) {
            values[id] = score - logsum;
        }
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
