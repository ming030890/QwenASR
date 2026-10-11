use regex::Regex;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

pub struct Tokenizer {
    ids: HashMap<String, usize>,
    english: Regex,
    url: Regex,
}

pub fn normalize(text: &str) -> String {
    text.nfc()
        .map(|c| match c {
            '\u{ff01}'..='\u{ff5e}' => char::from_u32(c as u32 - 0xfee0).unwrap(),
            '’' | '‘' => '\'',
            '“' | '”' => '"',
            other => other,
        })
        .collect()
}

pub fn is_han(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x323af)
}

impl Tokenizer {
    pub fn new(tokens: &[String]) -> Self {
        Self {
            ids: tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i))
                .collect(),
            english: Regex::new(r"^[0-9]+\.[0-9]+|^[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*").unwrap(),
            url: Regex::new(r"(?i)https?://[^\s<>]+|www\.[^\s<>]+").unwrap(),
        }
    }
    pub fn id(&self, word: &str) -> Option<usize> {
        self.ids.get(&normalize(word).to_lowercase()).copied()
    }
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let s = normalize(text);
        let ends: Vec<usize> = s.grapheme_indices(true).map(|(a, g)| a + g.len()).collect();
        let mut ids = Vec::new();
        let mut a = 0;
        while a < s.len() {
            let c = s[a..].chars().next().unwrap();
            if c.is_whitespace() {
                a += c.len_utf8();
                continue;
            }
            let tail = &s[a..];
            let raw_end = if is_han(c) {
                a + c.len_utf8()
            } else {
                self.url
                    .find(tail)
                    .filter(|m| m.start() == 0)
                    .or_else(|| self.english.find(tail))
                    .map(|m| a + m.end())
                    .unwrap_or_else(|| a + tail.graphemes(true).next().unwrap().len())
            };
            // Preserve whole graphemes, including variation selectors/ZWJ emoji.
            let end = ends[ends.partition_point(|&e| e < raw_end)];
            ids.push(self.id(&s[a..end]).unwrap_or(2));
            a = end;
        }
        ids
    }
    fn boundaries(&self, text: &str) -> Vec<usize> {
        let protected: Vec<_> = self.url.find_iter(text).map(|m| m.range()).collect();
        let chars: Vec<_> = text.char_indices().collect();
        let mut ends = Vec::new();
        for (i, &(a, c)) in chars.iter().enumerate() {
            if protected.iter().any(|r| r.contains(&a)) {
                continue;
            }
            let previous = i.checked_sub(1).map(|p| chars[p].1);
            let next = chars.get(i + 1).map(|p| p.1);
            let hard = matches!(c, '。' | '！' | '？' | '!' | '?' | '\n' | '\r')
                || (c == '.'
                    && previous != Some('.')
                    && next != Some('.')
                    && !(previous.map(|c| c.is_numeric()).unwrap_or(false)
                        && next.map(|c| c.is_numeric()).unwrap_or(false)));
            if hard {
                ends.push(a + c.len_utf8());
            }
        }
        ends
    }
    /// Left is visible committed prefix; right is text after the replacement.
    /// No BOS is synthesized. Boundaries match the training tokenizer.
    pub fn context(&self, left: &str, right: &str, nleft: usize, nright: usize) -> Vec<usize> {
        let left = normalize(left);
        let right = normalize(right);
        let start = self.boundaries(&left).last().copied().unwrap_or(0);
        let end = self
            .boundaries(&right)
            .first()
            .copied()
            .unwrap_or(right.len());
        let l = self.tokenize(&left[start..]);
        let r = self.tokenize(&right[..end]);
        let mut result = vec![0; nleft + nright];
        let count = l.len().min(nleft);
        result[nleft - count..nleft].copy_from_slice(&l[l.len() - count..]);
        let count = r.len().min(nright);
        result[nleft..nleft + count].copy_from_slice(&r[..count]);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tok() -> Tokenizer {
        Tokenizer::new(
            &[
                "[PAD]", "[BOS]", "[UNK]", "我", "唔", "iphone", "didn't", ",", "...", "123.4",
            ]
            .map(String::from),
        )
    }
    #[test]
    fn words_unicode_and_boundaries() {
        let t = tok();
        assert_eq!(t.tokenize("我 ＩＰｈｏｎｅ didn’t 👨‍👩‍👧‍👦"), vec![3, 5, 6, 2]);
        assert_eq!(t.context("我。", "", 2, 1), vec![0, 0, 0]);
        assert_eq!(t.context("我，唔", "", 2, 1), vec![7, 4, 0]);
        assert_eq!(t.context("我 https://a.b/x", "", 2, 1), vec![3, 2, 0]);
        assert_eq!(t.context("123.4", "", 2, 1), vec![0, 9, 0]);
        assert_eq!(t.tokenize("我\u{fe00}"), vec![2]);
    }
}
