//! Bounded spelling candidates from the shared LM vocabulary, without another word list.

pub fn canonical(word: &str) -> Option<String> {
    let word = word.replace('’', "'").to_ascii_lowercase();
    (!word.is_empty() && word.len() <= 32 && word.bytes().all(|c| c.is_ascii_lowercase() || c == b'\''))
        .then_some(word)
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
            cost = adj as f32 * 1.0 + non_adj as f32 * 2.0;
        } else {
            // Transposition or alternation error (e.g. teh -> the, thses -> these)
            let has_foreign = word.iter().any(|b| !input.contains(b));
            if !has_foreign {
                cost = edits as f32 * 1.0;
            } else {
                cost = edits as f32 * 1.5;
            }
        }
    } else {
        // Omission or extra character: length difference costs extra
        cost = edits as f32 * 1.5 + len_diff as f32 * 2.0;
    }
    if edits >= 2 {
        cost += (edits - 1) as f32 * 2.0;
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
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
