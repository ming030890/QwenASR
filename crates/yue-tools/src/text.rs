//! Text normalization, Traditional->Simplified, alignment, and CER.

use zhconv::{zhconv, Variant};

/// Convert Traditional Cantonese to Simplified (the model's default output script).
pub fn to_simplified(text: &str) -> String {
    zhconv(text, Variant::ZhHans)
}

/// Fold fullwidth ASCII (U+FF01..U+FF5E) and ideographic space to halfwidth.
fn fold_char(c: char) -> char {
    let u = c as u32;
    if (0xFF01..=0xFF5E).contains(&u) {
        char::from_u32(u - 0xFEE0).unwrap_or(c)
    } else if u == 0x3000 {
        ' '
    } else {
        c
    }
}

fn is_punct_or_space(c: char) -> bool {
    if c.is_whitespace() {
        return true;
    }
    // ASCII punctuation
    if c.is_ascii_punctuation() {
        return true;
    }
    // Common CJK punctuation
    matches!(
        c,
        '。' | '，'
            | '、'
            | '？'
            | '！'
            | '；'
            | '：'
            | '「'
            | '」'
            | '『'
            | '』'
            | '（'
            | '）'
            | '《'
            | '》'
            | '〈'
            | '〉'
            | '…'
            | '—'
            | '·'
            | '．'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '〜'
            | '～'
            | '﹐'
            | '､'
    )
}

/// Normalize text into a Vec<char> for scoring. Folds fullwidth, T->S, removes
/// whitespace; if `strip_punct`, also removes punctuation.
pub fn normalize(text: &str, strip_punct: bool) -> Vec<char> {
    let folded: String = text.chars().map(fold_char).collect();
    let simp = to_simplified(&folded);
    simp.chars()
        .filter(|&c| {
            if c.is_whitespace() {
                return false;
            }
            if strip_punct && is_punct_or_space(c) {
                return false;
            }
            true
        })
        .collect()
}

pub fn levenshtein(a: &[char], b: &[char]) -> usize {
    let n = b.len();
    let mut dp: Vec<usize> = (0..=n).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut prev = dp[0];
        dp[0] = i + 1;
        for j in 1..=n {
            let tmp = dp[j];
            dp[j] = if ca == b[j - 1] {
                prev
            } else {
                1 + prev.min(dp[j]).min(dp[j - 1])
            };
            prev = tmp;
        }
    }
    dp[n]
}

/// One aligned column transforming hyp -> ref.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Col {
    Match(char),
    /// substitution / deletion / insertion: hyp char and/or ref char
    Diff(Option<char>, Option<char>), // (hyp, ref)
}

/// Align hyp -> ref via edit distance with backtrace.
pub fn align(hyp: &[char], rf: &[char]) -> Vec<Col> {
    let m = hyp.len();
    let n = rf.len();
    // cost matrix
    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i as u32;
    }
    for j in 0..=n {
        dp[0][j] = j as u32;
    }
    for i in 1..=m {
        for j in 1..=n {
            let sub = dp[i - 1][j - 1] + if hyp[i - 1] == rf[j - 1] { 0 } else { 1 };
            let del = dp[i - 1][j] + 1; // drop hyp char
            let ins = dp[i][j - 1] + 1; // add ref char
            dp[i][j] = sub.min(del).min(ins);
        }
    }
    // backtrace
    let mut cols = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        if i > 0
            && j > 0
            && dp[i][j] == dp[i - 1][j - 1] + if hyp[i - 1] == rf[j - 1] { 0 } else { 1 }
        {
            if hyp[i - 1] == rf[j - 1] {
                cols.push(Col::Match(hyp[i - 1]));
            } else {
                cols.push(Col::Diff(Some(hyp[i - 1]), Some(rf[j - 1])));
            }
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            cols.push(Col::Diff(Some(hyp[i - 1]), None));
            i -= 1;
        } else {
            cols.push(Col::Diff(None, Some(rf[j - 1])));
            j -= 1;
        }
    }
    cols.reverse();
    cols
}

/// Corpus + macro CER over (hyp, ref) string pairs.
pub struct CerStats {
    pub corpus_cer: f64,
    pub macro_cer: f64,
    pub edits: usize,
    pub ref_chars: usize,
    pub n: usize,
}

pub fn cer<'a, I>(pairs: I, strip_punct: bool) -> CerStats
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut edits = 0usize;
    let mut chars = 0usize;
    let mut macro_sum = 0.0f64;
    let mut n = 0usize;
    for (hyp, rf) in pairs {
        let h = normalize(hyp, strip_punct);
        let r = normalize(rf, strip_punct);
        if r.is_empty() {
            continue;
        }
        let d = levenshtein(&h, &r);
        edits += d;
        chars += r.len();
        macro_sum += d as f64 / r.len() as f64;
        n += 1;
    }
    CerStats {
        corpus_cer: edits as f64 / chars.max(1) as f64,
        macro_cer: macro_sum / n.max(1) as f64,
        edits,
        ref_chars: chars,
        n,
    }
}
