//! Bounded spelling candidates from the shared LM vocabulary, without another word list.

#[derive(serde::Deserialize)]
struct AcceptancePolicy {
    means: [f64; 20],
    scales: [f64; 20],
    w1: [[f64; 20]; 24],
    b1: [f64; 24],
    w2: [f64; 24],
    b2: f64,
    threshold: f64,
    margin_cap: f64,
    minimum_observed_length: usize,
    minimum_candidate_length: usize,
    preserve_known_words: bool,
}

/// 2-layer FFN acceptance policy trained on real and synthetic typo data with
/// learned edit costs and length interactions.
pub fn accept(
    typed: &str,
    known: bool,
    word: &str,
    completed: usize,
    score: f32,
    edits: u8,
    runner: Option<(f32, u8, usize, &str)>,
    literal: Option<f32>,
) -> bool {
    static POLICY: std::sync::OnceLock<AcceptancePolicy> = std::sync::OnceLock::new();
    let policy = POLICY.get_or_init(|| serde_json::from_str(include_str!("english_acceptance.json"))
        .expect("Invalid compiled English acceptance policy"));
    if (known && (policy.preserve_known_words || literal.is_none())) ||
        typed.len() < policy.minimum_observed_length || word.len() < policy.minimum_candidate_length {
        return false;
    }

    let is_split = word.contains(' ');
    let penalty = if is_split { 1.5 } else if completed > 0 { 0.0 } else { edit_penalty(typed.as_bytes(), word.as_bytes(), edits) };
    let raw_score = (score + penalty) as f64;

    let (raw_margin, pen_margin) = match runner {
        Some((r_score, r_edits, r_completed, r_word)) => {
            let r_split = r_word.contains(' ');
            let r_penalty = if r_split { 1.5 } else if r_completed > 0 { 0.0 } else { edit_penalty(typed.as_bytes(), r_word.as_bytes(), r_edits) };
            let r_raw = (r_score + r_penalty) as f64;
            (
                (raw_score - r_raw).min(policy.margin_cap),
                (score as f64 - r_score as f64).min(policy.margin_cap),
            )
        }
        None => (policy.margin_cap, policy.margin_cap),
    };

    let lit_score = literal.unwrap_or(0.0) as f64;
    let score_over_lit = literal.map(|s| raw_score - s as f64).unwrap_or(0.0);
    let len_typed = typed.len() as f64;
    let len_word = word.len() as f64;
    let len_diff = (typed.len().abs_diff(word.len())) as f64;
    let edit_ratio = (edits as f64) / len_typed.max(1.0);

    let mut adj = 0.0;
    let mut non_adj = 0.0;
    let mut swap = 0.0;
    let mut extra = 0.0;
    let mut missing = 0.0;

    let typed_b = typed.as_bytes();
    let word_b = word.as_bytes();
    if typed_b.len() == word_b.len() {
        let mut diff_count = 0;
        let mut a_adj = 0.0;
        let mut a_non_adj = 0.0;
        for (&a, &b) in typed_b.iter().zip(word_b.iter()) {
            if a != b {
                diff_count += 1;
                if qwerty_adjacent(a, b) {
                    a_adj += 1.0;
                } else {
                    a_non_adj += 1.0;
                }
            }
        }
        if diff_count == edits as usize {
            adj = a_adj;
            non_adj = a_non_adj;
        } else {
            swap = edits as f64;
            non_adj = word_b.iter().filter(|b| !typed_b.contains(b)).count() as f64;
        }
    } else {
        if typed_b.len() > word_b.len() {
            extra = 1.0;
        } else {
            missing = 1.0;
        }
    }

    let first_diff = if !typed_b.is_empty() && !word_b.is_empty() && typed_b[0] != word_b[0] { 1.0 } else { 0.0 };
    let is_space_split = if is_split { 1.0 } else { 0.0 };
    let apostrophe_feat = if typed.contains('\'') || typed.contains('’') { 1.0 } else { 0.0 };
    let completed_feat = completed as f64;

    let values = [
        raw_score,
        lit_score,
        score_over_lit,
        raw_margin,
        pen_margin,
        if known { 1.0 } else { 0.0 },
        len_typed,
        len_word,
        len_diff,
        edits as f64,
        edit_ratio,
        adj,
        non_adj,
        swap,
        extra,
        missing,
        first_diff,
        is_space_split,
        apostrophe_feat,
        completed_feat,
    ];

    let mut hidden = policy.b1;
    for i in 0..24 {
        for j in 0..20 {
            hidden[i] += policy.w1[i][j] * (values[j] - policy.means[j]) / policy.scales[j];
        }
        if hidden[i] < 0.0 {
            hidden[i] = 0.0;
        }
    }
    let logit = policy.b2 + (0..24).map(|i| hidden[i] * policy.w2[i]).sum::<f64>();
    1.0 / (1.0 + (-logit).exp()) >= policy.threshold
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
        assert!(accept("teh", false, "the", 0, -5., 1, Some((-15., 1, 0, "tea")), None));
        assert!(!accept("name", true, "word", 0, -5., 1, Some((-15., 1, 0, "ward")), Some(-1.)));
        assert!(accept("hou", true, "you", 0, -3., 1, Some((-15., 1, 0, "how")), Some(-15.)));
        assert!(!accept("hou", true, "you", 0, -3., 1, Some((-15., 1, 0, "how")), None));
        assert!(!accept("hel", false, "hello", 2, -5., 2, Some((-15., 0, 0, "help")), None));
        assert!(!accept("name", false, "word", 0, -20., 1, Some((-20.1, 1, 0, "ward")), None));
        assert!(!accept("a", false, "and", 0, -1., 0, Some((-20., 0, 0, "any")), None));
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
