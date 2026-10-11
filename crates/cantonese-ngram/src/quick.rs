//! Quick (速成)3/5 dictionary lookup and shared LM ranking; no Android dependency.
use crate::hybrid::HybridLanguageModel;

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
}
impl QuickEngine {
    pub fn new(three: QuickDictionary, five: QuickDictionary, lm: HybridLanguageModel) -> Self {
        Self { three, five, lm }
    }
    pub fn candidates(
        &mut self,
        version: u32,
        code: &str,
        left: &str,
        right: &str,
    ) -> Result<Vec<String>, String> {
        let source = match version {
            3 => self.three.candidates(code),
            5 => self.five.candidates(code),
            _ => return Err("Unknown Quick version".into()),
        };
        let order = self.lm.rank(left, right, source)?;
        Ok(order.into_iter().map(|i| source[i].clone()).collect())
    }
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
}
