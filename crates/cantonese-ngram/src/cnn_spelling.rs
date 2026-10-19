//! Bounded spelling candidates from the shared LM vocabulary, without another word list.

pub fn canonical(word: &str) -> Option<String> {
    let word = word.replace('’', "'").to_ascii_lowercase();
    (!word.is_empty() && word.len() <= 32 && word.bytes().all(|c| c.is_ascii_lowercase() || c == b'\''))
        .then_some(word)
}

/// Optimal string alignment: insertion, deletion, substitution, adjacent swap.
pub fn distance(input: &[u8], word: &[u8], radius: usize) -> Option<u8> {
    if input.is_empty() || input.len() > 32 || word.len() > 32 || input.len().abs_diff(word.len()) > radius {
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
