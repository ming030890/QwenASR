//! Compressed Sparse Row (CSR) Cantonese Language Model.
//!
//! Stores character unigram log-priors and character-follower bigram bonuses
//! in a contiguous, zero-heap-overhead memory layout.
//! Unigram lookup is O(1) array access. Bigram lookup is direct row slice
//! with fast binary search over sorted target class IDs.

#[derive(Clone)]
pub struct CsrLanguageModel {
    pub num_classes: usize,
    pub num_edges: usize,
    pub lambda_unigram: f32,
    pub lambda_bigram: f32,
    pub unigrams: Vec<f32>,
    pub offsets: Vec<u32>,
    pub targets: Vec<u16>,
    pub log_probs: Vec<f32>,
}

impl CsrLanguageModel {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 28 {
            return Err("LM binary too small".to_string());
        }
        if &bytes[0..4] != b"CLMC" {
            return Err("Invalid LM magic".to_string());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 {
            return Err(format!("Unsupported LM version: {}", version));
        }
        let num_classes = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let num_edges = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let _reserved = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let lambda_unigram = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let lambda_bigram = f32::from_le_bytes(bytes[24..28].try_into().unwrap());

        if num_classes > 65536 || num_edges > bytes.len() / 6 {
            return Err("Invalid count dimensions".into());
        }
        let expected_size =
            28 + num_classes * 4 + (num_classes + 1) * 4 + num_edges * 2 + num_edges * 4;
        if bytes.len() < expected_size {
            return Err(format!(
                "LM file truncated: expected {} bytes, got {}",
                expected_size,
                bytes.len()
            ));
        }

        let mut offset = 28;

        // 1. Unigrams
        let mut unigrams = Vec::with_capacity(num_classes);
        for _ in 0..num_classes {
            unigrams.push(f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
            offset += 4;
        }

        // 2. Row offsets
        let mut offsets = Vec::with_capacity(num_classes + 1);
        for _ in 0..=num_classes {
            offsets.push(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
            offset += 4;
        }

        // 3. Edge targets
        let mut targets = Vec::with_capacity(num_edges);
        for _ in 0..num_edges {
            targets.push(u16::from_le_bytes(
                bytes[offset..offset + 2].try_into().unwrap(),
            ));
            offset += 2;
        }

        // 4. Log probs
        let mut log_probs = Vec::with_capacity(num_edges);
        for _ in 0..num_edges {
            log_probs.push(f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
            offset += 4;
        }

        if offsets.first() != Some(&0)
            || offsets.last().copied() != Some(num_edges as u32)
            || offsets.windows(2).any(|p| p[0] > p[1])
            || targets.iter().any(|&t| t as usize >= num_classes)
            || unigrams.iter().chain(&log_probs).any(|v| !v.is_finite())
            || !lambda_unigram.is_finite()
            || !lambda_bigram.is_finite()
        {
            return Err("Invalid count LM arrays".into());
        }
        for pair in offsets.windows(2) {
            if targets[pair[0] as usize..pair[1] as usize]
                .windows(2)
                .any(|p| p[0] >= p[1])
            {
                return Err("Unsorted bigram targets".into());
            }
        }
        Ok(Self {
            num_classes,
            num_edges,
            lambda_unigram,
            lambda_bigram,
            unigrams,
            offsets,
            targets,
            log_probs,
        })
    }

    #[inline(always)]
    pub fn unigram_score(&self, class_id: usize) -> f32 {
        if class_id < self.num_classes {
            self.unigrams[class_id] * self.lambda_unigram
        } else {
            0.0
        }
    }

    #[inline]
    pub fn bigram_bonus(&self, prev_class: usize, curr_class: usize) -> f32 {
        if prev_class >= self.num_classes || curr_class >= self.num_classes {
            return 0.0;
        }
        let start = self.offsets[prev_class] as usize;
        let end = self.offsets[prev_class + 1] as usize;
        if start >= end || end > self.targets.len() {
            return 0.0;
        }
        let target_u16 = curr_class as u16;
        let slice = &self.targets[start..end];
        match slice.binary_search(&target_u16) {
            Ok(idx) => self.log_probs[start + idx] * self.lambda_bigram,
            Err(_) => 0.0,
        }
    }

    pub fn apply_to_scores(&self, scores: &mut [f32], prev_class: Option<usize>) {
        let n = scores.len().min(self.num_classes);
        for i in 0..n {
            scores[i] += self.unigram_score(i);
        }
        if let Some(prev) = prev_class {
            if prev < self.num_classes {
                let start = self.offsets[prev] as usize;
                let end = self.offsets[prev + 1] as usize;
                if start < end && end <= self.targets.len() {
                    for idx in start..end {
                        let target = self.targets[idx] as usize;
                        if target < scores.len() {
                            scores[target] += self.log_probs[idx] * self.lambda_bigram;
                        }
                    }
                }
            }
        }
    }
}
