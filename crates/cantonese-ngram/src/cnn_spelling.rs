//! Bounded spelling candidates from the shared LM vocabulary, without another word list.

#[derive(serde::Deserialize)]
struct AcceptancePolicy {
    means: [f64; 6], scales: [f64; 6], weights: [f64; 7], threshold: f64,
    margin_cap: f64, minimum_observed_length: usize, minimum_candidate_length: usize,
    preserve_known_words: bool,
}

/// Coefficients and cutoff fitted on separate mobile prompt groups. The score is
/// a learned decision value, not a per-user correctness guarantee.
pub fn accept(typed: &str, known: bool, word: &str, completed: usize, score: f32, runner: Option<f32>, literal: Option<f32>) -> bool {
    static POLICY: std::sync::OnceLock<AcceptancePolicy> = std::sync::OnceLock::new();
    let policy = POLICY.get_or_init(|| serde_json::from_str(include_str!("english_acceptance.json"))
        .expect("Invalid compiled English acceptance policy"));
    if (known && (policy.preserve_known_words || literal.is_none())) || completed > 0 ||
        typed.len() < policy.minimum_observed_length || word.len() < policy.minimum_candidate_length {
        return false;
    }
    let margin = runner.map(|r| score as f64-r as f64).unwrap_or(policy.margin_cap).min(policy.margin_cap);
    let values = [score as f64, margin, if known {1.} else {0.}, literal.unwrap_or(0.) as f64,
        literal.map(|s|score as f64-s as f64).unwrap_or(0.), if typed.contains('\'') {1.} else {0.}];
    let logit = policy.weights[0] + (0..6).map(|i|
        policy.weights[i+1]*(values[i]-policy.means[i])/policy.scales[i]).sum::<f64>();
    1. / (1. + (-logit).exp()) >= policy.threshold
}

pub fn canonical(word: &str) -> Option<String> {
    let word = word.replace('’', "'").to_ascii_lowercase();
    (!word.is_empty() && word.len() <= 32 && word.bytes().all(|c| c.is_ascii_lowercase() || c == b'\''))
        .then_some(word)
}

#[derive(serde::Deserialize)]
pub struct ApostropheForm { pub word: String, pub count: usize, pub bare_count: usize }

/// Forms and ambiguity evidence attested only on TRAIN prompts.
pub fn apostrophe_forms(input: &str) -> impl Iterator<Item = &'static ApostropheForm> + '_ {
    static FORMS: std::sync::OnceLock<Vec<ApostropheForm>> = std::sync::OnceLock::new();
    FORMS.get_or_init(|| serde_json::from_str(include_str!("english_apostrophe_forms.json"))
        .expect("Invalid apostrophe surface forms")).iter()
        .filter(move |form| form.word.bytes().filter(|&b| b != b'\'').eq(input.bytes()))
}

pub fn accept_apostrophe(features: [f64; 3]) -> bool {
    #[derive(serde::Deserialize)]
    struct Policy { means: [f64; 3], scales: [f64; 3], weights: [f64; 4], threshold: f64 }
    static POLICY: std::sync::OnceLock<Policy> = std::sync::OnceLock::new();
    let p = POLICY.get_or_init(|| serde_json::from_str(include_str!("english_apostrophe_acceptance.json"))
        .expect("Invalid apostrophe acceptance policy"));
    let z = p.weights[0] + (0..3).map(|i| p.weights[i+1]*(features[i]-p.means[i])/p.scales[i]).sum::<f64>();
    1. / (1. + (-z).exp()) >= p.threshold
}

/// Optimal string alignment: insertion, deletion, substitution, adjacent swap.
pub fn distance(input: &[u8], word: &[u8], radius: usize) -> Option<u8> {
    let len_diff = input.len().abs_diff(word.len());
    if input.is_empty() || input.len() > 32 || word.len() > 32 || len_diff > 1 || len_diff > radius {
        return None;
    }
    let mut previous = [0u8; 33];
    let mut before_previous = [0u8; 33];
    for j in 0..=input.len() { previous[j] = j as u8; }
    for (i, &letter) in word.iter().enumerate() {
        let mut current = [0u8; 33];
        current[0] = (i + 1) as u8;
        for j in 1..=input.len() {
            current[j] = (previous[j] + 1).min(current[j - 1] + 1)
                .min(previous[j - 1] + u8::from(letter != input[j - 1]));
            if i > 0 && j > 1 && letter == input[j - 2] && word[i - 1] == input[j - 1] {
                current[j] = current[j].min(before_previous[j - 2] + 1);
            }
        }
        if *current[..=input.len()].iter().min().unwrap() as usize > radius { return None; }
        before_previous = previous;
        previous = current;
    }
    (previous[input.len()] as usize <= radius).then_some(previous[input.len()])
}

pub fn qwerty_adjacent(a: u8, b: u8) -> bool {
    const ROWS: [&[u8]; 3] = [b"qwertyuiop", b"asdfghjkl", b"zxcvbnm"];
    let pos = |c: u8| -> Option<(usize, usize)> {
        for (r, row) in ROWS.iter().enumerate() {
            if let Some(col) = row.iter().position(|&k| k == c) {
                return Some((r, col));
            }
        }
        None
    };
    match (pos(a), pos(b)) {
        (Some((r1, c1)), Some((r2, c2))) => a != b && r1.abs_diff(r2) <= 1 && c1.abs_diff(c2) <= 1,
        _ => false,
    }
}

pub fn edit_penalty(input: &[u8], word: &[u8], edits: u8) -> f32 {
    if edits == 0 {
        return 0.0;
    }
    // Five existing operation costs fitted on recorded TRAIN typos; fixed on
    // selection before evaluation. See keyboard EDIT_POSITION_COSTS.md.
    const NEIGHBOR: f32 = 0.574436466482213;
    const OTHER: f32 = 3.7162379435897606;
    const SWAP: f32 = 0.9146548439690982;
    const EXTRA: f32 = 4.541078382355206;
    const MISSING: f32 = 1.9126156502899054;
    let len_diff = input.len().abs_diff(word.len());
    let mut cost;
    if len_diff == 0 {
        let mut adj = 0;
        let mut non_adj = 0;
        for (&a, &b) in input.iter().zip(word.iter()) {
            if a != b {
                if qwerty_adjacent(a, b) {
                    adj += 1;
                } else {
                    non_adj += 1;
                }
            }
        }
        if (adj + non_adj) as u8 == edits {
            cost = adj as f32 * NEIGHBOR + non_adj as f32 * OTHER;
        } else {
            // Transposition or alternation error (e.g. teh -> the, thses -> these)
            let foreign_count = word.iter().filter(|b| !input.contains(b)).count();
            cost = edits as f32 * SWAP + foreign_count as f32 * OTHER;
        }
    } else {
        // Candidate generation bounds length difference to one. Keep the
        // original 4.5 surcharge for each further edit in this branch.
        let base = if input.len() > word.len() { EXTRA } else { MISSING };
        return base + edits.saturating_sub(1) as f32 * 4.5;
    }
    if edits >= 2 {
        cost += (edits - 1) as f32 * 3.0;
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restores_only_attested_missing_apostrophes_with_calibrated_evidence() {
        let input = canonical("DONT").unwrap();
        let forms: Vec<_> = apostrophe_forms(&input).collect();
        assert_eq!(forms.len(), 1);
        assert_eq!(forms[0].word, "don't");
        assert!(forms[0].count > 0);
        assert!(apostrophe_forms("don't").next().is_none());
        assert!(apostrophe_forms("dnot").next().is_none());
        assert!(accept_apostrophe([1., 3., 2.]));
        assert!(!accept_apostrophe([-12., -10., -2.]));
    }

    #[test]
    fn learned_acceptance_compares_the_literal_and_never_automatically_completes() {
        assert!(accept("teh", false, "the", 0, -5., Some(-15.),None));
        assert!(!accept("name", true, "word", 0, -5., Some(-15.),Some(-1.)));
        assert!(accept("hou", true, "you", 0, -3., Some(-15.),Some(-15.)));
        assert!(!accept("hou", true, "you", 0, -3., Some(-15.),None));
        assert!(!accept("hel", false, "hello", 2, -5., Some(-15.),None));
        assert!(!accept("name", false, "word", 0, -20., Some(-20.1),None));
        assert!(!accept("a", false, "and", 0, -1., Some(-20.),None));
    }
    #[test]
    fn fitted_costs_preserve_operation_order_without_a_position_prior() {
        assert_eq!(edit_penalty(b"game", b"game", 0), 0.);
        let neighbor = edit_penalty(b"gwme", b"game", 1);
        assert_eq!(neighbor, edit_penalty(b"fame", b"game", 1));
        assert!(neighbor < edit_penalty(b"gpme", b"game", 1));
        assert!(edit_penalty(b"keybord", b"keyboard", 1) <
            edit_penalty(b"keyboarrd", b"keyboard", 1));
        assert!(edit_penalty(b"wroldd", b"world", 2) >
            edit_penalty(b"worldd", b"world", 1));
    }

    #[test]
    fn generates_real_typo_operations_with_a_bounded_radius() {
        for (typed, word, edits) in [("yiu", "you", 1), ("teh", "the", 1),
            ("helllo", "hello", 1), ("keybord", "keyboard", 1), ("briken", "broken", 1),
            ("recieve", "receive", 1), ("wroldd", "world", 2)] {
            assert_eq!(distance(typed.as_bytes(), word.as_bytes(), 2), Some(edits), "{typed}");
        }
        assert_eq!(distance(b"yiu", b"keyboard", 2), None);
        assert_eq!(distance(b"wroldd", b"world", 1), None);
        assert_eq!(canonical("DON’T"), Some("don't".into()));
        assert_eq!(canonical("name@example.com"), None);
        assert_eq!(canonical("中文"), None);
    }
}
