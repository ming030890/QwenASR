//! Joined keyboard request: dictionary generation followed by shared model ranking.
use crate::quick::QuickEngine;

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JoinedSuggestions {
    pub chinese: Vec<String>,
    pub display: Vec<String>,
    pub english: crate::cnn::EnglishSuggestions,
    pub evidence: Option<crate::quick::QuickLanguageEvidence>,
    pub code_length: usize,
}

/// Current-input suggestions with a greedy Chinese search and shared bilingual ranking.
/// The authoritative composing request still runs unchanged on separate state.
pub fn preview(engine: &mut QuickEngine, version: u32, code: &str, left: &str, right: &str,
    english_enabled: bool) -> Result<JoinedSuggestions, String> {
    crate::suggestion_request::check()?;
    let lattice = if version == 0 || code.is_empty() { None } else { Some(engine.generate_lattice(version,code)?) };
    let english_pool = if english_enabled && !code.is_empty() {
        engine.lm.neural.as_ref().ok_or("Shared model missing")?.generate_english_candidates(code)?
    } else { None };
    crate::suggestion_request::generation_complete();
    let chinese = if let Some(lattice) = &lattice { engine.rank_preview_lattice(lattice,left,right)? } else { vec![] };
    let english = if english_enabled && !code.is_empty() {
        engine.lm.neural.as_mut().ok_or("Shared model missing")?.rank_english_candidates(left,english_pool,8)?
    } else { empty_english() };
    finish(engine,version,code,left,right,english_enabled,chinese,english)
}

fn empty_english() -> crate::cnn::EnglishSuggestions {
    crate::cnn::EnglishSuggestions { candidates:vec![],correction:None,known:false,literal_score:None,apostrophe_evidence:vec![] }
}

pub fn compose(
    engine: &mut QuickEngine,
    version: u32,
    code: &str,
    left: &str,
    right: &str,
    english_enabled: bool,
) -> Result<JoinedSuggestions, String> {
    crate::suggestion_request::check()?;
    if code.is_empty() {
        let display = if version == 0 {
            engine
                .lm
                .neural
                .as_mut()
                .ok_or("Shared model missing")?
                .completions(left, "", 8)?
        } else {
            engine.lm.predict(left, right, 100)?.1
        };
        return Ok(JoinedSuggestions {
            chinese: vec![],
            display: display.into_iter().map(|r| r.token).collect(),
            english: crate::cnn::EnglishSuggestions {
                candidates: vec![],
                correction: None,
                known: false,
                literal_score: None,
                apostrophe_evidence: vec![],
            },
            evidence: None,
            code_length: 0,
        });
    }
    let lattice = if version == 0 { None } else { Some(engine.generate_lattice(version,code)?) };
    let english_pool = if english_enabled {
        engine.lm.neural.as_ref().ok_or("Shared model missing")?.generate_english_candidates(code)?
    } else { None };
    crate::suggestion_request::generation_complete();
    let chinese = if let Some(lattice) = &lattice { engine.rank_lattice(lattice,left,right,true)? } else { vec![] };
    crate::suggestion_request::check()?;
    let english = if english_enabled {
        engine.lm.neural.as_mut().ok_or("Shared model missing")?.rank_english_candidates(left,english_pool,8)?
    } else { empty_english() };
    finish(engine, version, code, left, right, english_enabled, chinese, english)
}

// Both publications use the same pool assembly, casing and bilingual scorer.
fn finish(engine: &mut QuickEngine, version: u32, code: &str, left: &str, right: &str,
    english_enabled: bool, mut chinese: Vec<String>, english: crate::cnn::EnglishSuggestions)
    -> Result<JoinedSuggestions, String> {
    let case = |text: &str| -> String {
        if code == code.to_uppercase() {
            text.to_uppercase()
        } else if code.chars().next().is_some_and(|c| c.is_uppercase()) {
            let mut chars = text.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        } else {
            text.into()
        }
    };
    let mut words = vec![];
    if english_enabled {
        words.push(code.to_owned());
        words.extend(english.candidates.iter().map(|r| case(&r.word)));
    }
    let mut pool = chinese.clone();
    pool.extend(words.clone());
    let mut seen = std::collections::HashSet::new();
    pool.retain(|s| seen.insert(s.clone()));
    let display = if version == 0 || code.len() == 2 || !english_enabled {
        // Complete two-letter codes and Chinese-only requests retain rank_lattice order directly.
        pool
    } else if chinese.is_empty() {
        let order = crate::quick::rank_mixed(&mut engine.lm, left, right, &pool, code)?;
        order.into_iter().map(|i| pool[i].clone()).collect()
    } else {
        let head_len = if code.len() > 2 {
            chinese.iter().take_while(|s| s.chars().count() > 1).count().clamp(1, 16)
        } else {
            chinese.len().min(16)
        };
        let mut mixed_pool = chinese[..head_len].to_vec();
        let mut mixed_seen: std::collections::HashSet<_> = chinese.iter().cloned().collect();
        for w in words {
            if mixed_seen.insert(w.clone()) {
                mixed_pool.push(w);
            }
        }
        let order = crate::quick::rank_mixed(&mut engine.lm, left, right, &mixed_pool, code)?;
        let mut ranked = Vec::with_capacity(pool.len());
        let mut next_chinese = 0;
        for i in order {
            if i < head_len {
                if next_chinese < chinese.len() {
                    ranked.push(chinese[next_chinese].clone());
                    next_chinese += 1;
                }
            } else {
                ranked.push(mixed_pool[i].clone());
            }
        }
        while next_chinese < chinese.len() {
            ranked.push(chinese[next_chinese].clone());
            next_chinese += 1;
        }
        ranked
    };
    let mut displayed_chinese = 0;
    let display = display
        .into_iter()
        .filter(|word| {
            if chinese.contains(word) {
                displayed_chinese += 1;
                displayed_chinese <= 100 || code.len() <= 2
            } else {
                true
            }
        })
        .collect::<Vec<_>>();
    chinese.truncate(100);
    Ok(JoinedSuggestions {
        chinese,
        display,
        english,
        evidence: None,
        code_length: code.len(),
    })
}
