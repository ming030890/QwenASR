//! Rule mining (merge/split via minimal sufficient context) and application.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::text::{align, Col};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub from: String,
    pub to: String,
    pub count: u32,     // positives (times this correction was observed)
    pub occ: u32,       // total substring occurrences of `from` in hyp corpus
    pub precision: f64, // count / occ
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleSet {
    pub params: MineParams,
    pub rules: Vec<Rule>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MineParams {
    pub p_high: f64,
    pub n_min: u32,
    pub max_ctx: usize,
    /// single-char `from` rules need a stricter precision bar (overcorrection risk)
    pub p_high_single: f64,
    pub n_min_single: u32,
}

impl Default for MineParams {
    fn default() -> Self {
        MineParams {
            p_high: 0.90,
            n_min: 20,
            max_ctx: 2,
            p_high_single: 0.97,
            n_min_single: 40,
        }
    }
}

/// Split (from,to) into common prefix / core / common suffix (by chars).
/// Returns (prefix_len, core_from, core_to, suffix_len).
fn core_of(from: &[char], to: &[char]) -> (usize, String, String, usize) {
    let mut p = 0;
    while p < from.len() && p < to.len() && from[p] == to[p] {
        p += 1;
    }
    let mut s = 0;
    while s < from.len() - p && s < to.len() - p && from[from.len() - 1 - s] == to[to.len() - 1 - s]
    {
        s += 1;
    }
    let cf: String = from[p..from.len() - s].iter().collect();
    let ct: String = to[p..to.len() - s].iter().collect();
    (p, cf, ct, s)
}

fn count_overlapping(hay: &[char], needle: &[char]) -> u32 {
    if needle.is_empty() || needle.len() > hay.len() {
        return 0;
    }
    let mut c = 0u32;
    for w in hay.windows(needle.len()) {
        if w == needle {
            c += 1;
        }
    }
    c
}

/// Generate context-expanded candidate corrections for one aligned utterance.
/// Pushes (from, to) keys into `positives`.
fn collect_candidates(cols: &[Col], k: usize, positives: &mut HashMap<(String, String), u32>) {
    // Indices of chunks (maximal runs of Diff) and per-column hyp/ref chars.
    let n = cols.len();
    let mut i = 0;
    while i < n {
        match cols[i] {
            Col::Match(_) => i += 1,
            Col::Diff(_, _) => {
                let start = i;
                let mut hyp_str = String::new();
                let mut ref_str = String::new();
                while i < n {
                    if let Col::Diff(h, r) = cols[i] {
                        if let Some(c) = h {
                            hyp_str.push(c);
                        }
                        if let Some(c) = r {
                            ref_str.push(c);
                        }
                        i += 1;
                    } else {
                        break;
                    }
                }
                let end = i; // exclusive

                // left/right anchor chars (Match cols only, bounded by neighbor chunks)
                let mut left: Vec<char> = Vec::new();
                let mut j = start;
                while j > 0 && left.len() < k {
                    if let Col::Match(c) = cols[j - 1] {
                        left.push(c);
                        j -= 1;
                    } else {
                        break;
                    }
                }
                left.reverse();
                let mut right: Vec<char> = Vec::new();
                let mut j = end;
                while j < n && right.len() < k {
                    if let Col::Match(c) = cols[j] {
                        right.push(c);
                        j += 1;
                    } else {
                        break;
                    }
                }

                for li in 0..=left.len() {
                    for ri in 0..=right.len() {
                        let lc = &left[left.len() - li..];
                        let rc = &right[..ri];
                        let mut from = String::new();
                        from.extend(lc);
                        from.push_str(&hyp_str);
                        from.extend(rc);
                        let mut to = String::new();
                        to.extend(lc);
                        to.push_str(&ref_str);
                        to.extend(rc);
                        if from.is_empty() || from == to {
                            continue;
                        }
                        *positives.entry((from, to)).or_insert(0) += 1;
                    }
                }
            }
        }
    }
}

/// Mine rules from pre-normalized (hyp, ref) char-vector pairs.
pub fn mine(pairs: &[(Vec<char>, Vec<char>)], params: MineParams) -> Vec<Rule> {
    // 1) positives over context families
    let mut positives: HashMap<(String, String), u32> = HashMap::new();
    for (h, r) in pairs {
        let cols = align(h, r);
        collect_candidates(&cols, params.max_ctx, &mut positives);
    }

    // 2) support filter first (keeps occ counting cheap)
    let survivors: Vec<((String, String), u32)> = positives
        .into_iter()
        .filter(|((from, _), cnt)| {
            let len = from.chars().count();
            let nmin = if len == 1 { params.n_min_single } else { params.n_min };
            *cnt >= nmin
        })
        .collect();

    // 3) occ counts for the distinct `from` strings among survivors
    let hyps: Vec<Vec<char>> = pairs.iter().map(|(h, _)| h.clone()).collect();
    let mut occ_cache: HashMap<String, u32> = HashMap::new();
    for ((from, _), _) in &survivors {
        if occ_cache.contains_key(from) {
            continue;
        }
        let needle: Vec<char> = from.chars().collect();
        let mut total = 0u32;
        for h in &hyps {
            total += count_overlapping(h, &needle);
        }
        occ_cache.insert(from.clone(), total.max(1));
    }

    // 4) precision filter
    let mut kept: Vec<Rule> = Vec::new();
    for ((from, to), cnt) in survivors {
        let occ = *occ_cache.get(&from).unwrap_or(&1);
        let precision = cnt as f64 / occ as f64;
        let len = from.chars().count();
        let pbar = if len == 1 { params.p_high_single } else { params.p_high };
        if precision >= pbar {
            kept.push(Rule { from, to, count: cnt, occ, precision });
        }
    }

    // 5) conflict by `from`: keep best precision (then support) per from
    kept.sort_by(|a, b| {
        a.from
            .cmp(&b.from)
            .then(b.precision.partial_cmp(&a.precision).unwrap())
            .then(b.count.cmp(&a.count))
    });
    let mut dedup_from: Vec<Rule> = Vec::new();
    for r in kept {
        if dedup_from.last().map(|p| p.from != r.from).unwrap_or(true) {
            dedup_from.push(r);
        }
    }

    // 6) merge/split: per core, keep minimal-context rules (drop dominated longer)
    // group by core
    let mut by_core: HashMap<(String, String), Vec<(usize, usize, usize)>> = HashMap::new();
    // value: (index into dedup_from, prefix_len, suffix_len)
    for (idx, r) in dedup_from.iter().enumerate() {
        let f: Vec<char> = r.from.chars().collect();
        let t: Vec<char> = r.to.chars().collect();
        let (p, cf, ct, s) = core_of(&f, &t);
        by_core.entry((cf, ct)).or_default().push((idx, p, s));
    }
    let mut keep_flag = vec![false; dedup_from.len()];
    for (_core, mut members) in by_core {
        // smallest total context first
        members.sort_by_key(|(_, p, s)| (*p + *s, *p, *s));
        let mut accepted: Vec<(usize, usize)> = Vec::new(); // (prefix,suffix)
        for (idx, p, s) in members {
            let dominated = accepted.iter().any(|&(ap, asuf)| ap <= p && asuf <= s);
            if !dominated {
                keep_flag[idx] = true;
                accepted.push((p, s));
            }
        }
    }
    let mut out: Vec<Rule> = dedup_from
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep_flag[*i])
        .map(|(_, r)| r)
        .collect();

    sort_for_apply(&mut out);
    out
}

/// Sort rules for application: longest `from` first, then highest precision.
pub fn sort_for_apply(rules: &mut [Rule]) {
    rules.sort_by(|a, b| {
        b.from
            .chars()
            .count()
            .cmp(&a.from.chars().count())
            .then(b.precision.partial_cmp(&a.precision).unwrap())
    });
}

/// Apply rules to a normalized char sequence: single left-to-right longest-match pass.
#[allow(dead_code)]
pub fn apply(text: &[char], rules: &[Rule]) -> Vec<char> {
    // Precompute rule char vectors once for speed.
    let compiled: Vec<(Vec<char>, Vec<char>)> = rules
        .iter()
        .map(|r| (r.from.chars().collect(), r.to.chars().collect()))
        .collect();
    apply_compiled(text, &compiled)
}

pub fn apply_compiled(text: &[char], compiled: &[(Vec<char>, Vec<char>)]) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(text.len());
    let mut i = 0;
    let n = text.len();
    while i < n {
        let mut matched = false;
        for (from, to) in compiled {
            let fl = from.len();
            if fl > 0 && i + fl <= n && &text[i..i + fl] == from.as_slice() {
                out.extend_from_slice(to);
                i += fl;
                matched = true;
                break;
            }
        }
        if !matched {
            out.push(text[i]);
            i += 1;
        }
    }
    out
}
