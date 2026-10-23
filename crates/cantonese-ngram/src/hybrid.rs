//! Shared context policy for Quick and handwriting. Neural fusion strength is
//! configured independently for handwriting; Quick ranking is unchanged.
use crate::{counts::CsrLanguageModel, LanguageModel, NeuralLanguageModel};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Mode {
    Unigram,
    Bigram,
    Neural,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub struct FusionWeights {
    pub unigram: f32,
    pub bigram: f32,
    pub neural: f32,
}
impl FusionWeights {
    pub fn validate(&self) -> Result<(), String> {
        if [self.unigram, self.bigram, self.neural]
            .iter()
            .any(|x| !x.is_finite() || *x < 0. || *x > 16.)
        {
            Err("Fusion weights must be finite and between zero and 16".into())
        } else {
            Ok(())
        }
    }
}

pub struct HybridLanguageModel {
    labels: Vec<String>,
    weights: Option<FusionWeights>,
    pub neural: Option<NeuralLanguageModel>,
    pub counts: Option<CsrLanguageModel>,
    class_ids: HashMap<String, usize>,
    neural_to_class: HashMap<usize, usize>,
    fallback_tokenizer: crate::tokenizer::Tokenizer,
}

fn apply_neural_priors(scores: &mut [f32], supported: &[(usize, usize)], logits: &[f32], weight: f32) {
    if logits.is_empty() { return; }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logsum = max + logits.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    // OOV classes retain their visual evidence, with the weakest supported
    // prior when no finite count fallback is available.
    let floor = logits.iter().copied().fold(f32::INFINITY, f32::min);
    for score in scores.iter_mut() {
        if !score.is_finite() { *score = weight * (floor - logsum); }
    }
    for ((index, _), value) in supported.iter().zip(logits) {
        scores[*index] = weight * (value - logsum);
    }
}
impl HybridLanguageModel {
    pub fn new(
        neural: Option<LanguageModel>,
        counts: Option<CsrLanguageModel>,
        labels: &[String],
    ) -> Result<Self, String> {
        Self::with_neural(neural.map(Into::into), counts, labels)
    }
    pub fn with_neural(
        neural: Option<NeuralLanguageModel>,
        mut counts: Option<CsrLanguageModel>,
        labels: &[String],
    ) -> Result<Self, String> {
        // New dense exports learn zero/one-token distributions in the same NN.
        // Counts supplied by older wrappers must never influence this policy.
        if neural.as_ref().map(|m| m.minimum_context_tokens() == 0).unwrap_or(false) {
            counts = None;
        }
        if counts
            .as_ref()
            .map(|c| c.num_classes != labels.len())
            .unwrap_or(false)
        {
            return Err("Count labels mismatch".into());
        }
        let neural_to_class = labels
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                neural
                    .as_ref()
                    .and_then(|lm| lm.token_id(s))
                    .map(|id| (id, i))
            })
            .collect();
        let mut tokens = vec!["[PAD]".into(), "[BOS]".into(), "[UNK]".into()];
        tokens.extend_from_slice(labels);
        Ok(Self {
            labels: labels.to_vec(),
            weights: None,
            neural,
            counts,
            class_ids: labels
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i))
                .collect(),
            neural_to_class,
            fallback_tokenizer: crate::tokenizer::Tokenizer::new(&tokens),
        })
    }
    /// Next-token predictions with the same short-context fallback as ranking.
    pub fn predict_next(
        &mut self,
        left: &str,
        limit: usize,
    ) -> Result<(Mode, Vec<crate::TokenPrediction>), String> {
        self.predict(left, "", limit)
    }
    /// Optional right context supports insertion/missing-token suggestions.
    pub fn predict(
        &mut self,
        left: &str,
        right: &str,
        limit: usize,
    ) -> Result<(Mode, Vec<crate::TokenPrediction>), String> {
        let mode = self.mode(left, right);
        if limit == 0 {
            return Ok((mode, Vec::new()));
        }
        if mode == Mode::Neural {
            return Ok((
                mode,
                self.neural.as_mut().unwrap().predict(left, right, limit)?,
            ));
        }
        let Some(counts) = &self.counts else {
            return Ok((mode, Vec::new()));
        };
        let previous = if mode == Mode::Bigram {
            self.previous_class(left)
        } else {
            None
        };
        // Count scores are ranking priors, not normalized probabilities.
        let values = self
            .labels
            .iter()
            .enumerate()
            .filter(|(_, token)| crate::displayable(token))
            .map(|(id, token)| crate::TokenPrediction {
                token: token.clone(),
                token_id: self.neural.as_ref().and_then(|lm| lm.token_id(token)),
                score: counts.unigram_score(id)
                    + previous.map(|p| counts.bigram_bonus(p, id)).unwrap_or(0.),
            })
            .collect();
        Ok((mode, crate::top_predictions(values, limit)?))
    }
    pub fn set_fusion_weights(&mut self, weights: FusionWeights) -> Result<(), String> {
        weights.validate()?;
        self.weights = Some(weights);
        Ok(())
    }
    fn fusion_weights(&self) -> FusionWeights {
        self.weights.unwrap_or(FusionWeights {
            unigram: self.counts.as_ref().map(|c| c.lambda_unigram).unwrap_or(0.),
            bigram: self.counts.as_ref().map(|c| c.lambda_bigram).unwrap_or(0.),
            neural: self.counts.as_ref().map(|c| c.lambda_unigram).unwrap_or(1.),
        })
    }
    pub fn mode(&self, left: &str, right: &str) -> Mode {
        let ids = self
            .neural
            .as_ref()
            .map(|lm| lm.context(left, right))
            .unwrap_or_else(|| self.fallback_tokenizer.context(left, right, 8, 4));
        if self.neural.as_ref().map(|lm| lm.supports_context(&ids)).unwrap_or(false) {
            return Mode::Neural;
        }
        // A single English/unknown token cannot supply a character bigram.
        if self.counts.is_some() && self.previous_class(left).is_some() {
            Mode::Bigram
        } else {
            Mode::Unigram
        }
    }
    pub fn previous_class(&self, left: &str) -> Option<usize> {
        if let Some(lm) = &self.neural {
            let ids = lm.context(left, "");
            let id = ids[..lm.context_lengths().0]
                .iter()
                .rev()
                .find(|&&id| id != 0)?;
            // Map via last complete token rather than the last code unit.
            self.neural_to_class.get(id).copied()
        } else {
            let ids = self.fallback_tokenizer.context(left, "", 8, 0);
            ids.iter()
                .rev()
                .find(|&&id| id != 0)
                .copied()
                .filter(|&id| id >= 3)
                .map(|id| id - 3)
        }
    }
    /// Returns class-aligned additive handwriting priors or Quick sorting scores.
    /// Unsupported LM characters retain their count-model priors.
    pub fn scores(
        &mut self,
        left: &str,
        right: &str,
        candidates: &[String],
    ) -> Result<(Mode, Vec<f32>), String> {
        let mode = self.mode(left, right);
        let weights = self.fusion_weights();
        let prev = if mode != Mode::Unigram {
            self.previous_class(left)
        } else {
            None
        };
        let mut scores: Vec<_> = candidates
            .iter()
            .map(|s| {
                self.class_ids
                    .get(s)
                    .and_then(|&id| {
                        self.counts.as_ref().map(|lm| {
                            weights.unigram * lm.unigrams[id]
                                + prev
                                    .map(|p| {
                                        let start = lm.offsets[p] as usize;
                                        let end = lm.offsets[p + 1] as usize;
                                        lm.targets[start..end]
                                            .binary_search(&(id as u16))
                                            .ok()
                                            .map(|j| weights.bigram * lm.log_probs[start + j])
                                            .unwrap_or(0.)
                                    })
                                    .unwrap_or(0.)
                        })
                    })
                    .unwrap_or(f32::NEG_INFINITY)
            })
            .collect();
        if mode == Mode::Neural {
            let neural = self.neural.as_mut().unwrap();
            let ids = neural.context(left, right);
            let supported: Vec<_> = candidates
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    neural
                        .token_id(s)
                        .filter(|&id| neural.supports_candidate(&ids,id))
                        .map(|id| (i, id))
                })
                .collect();
            if !supported.is_empty() {
                let logits =
                    neural.scores(&ids, &supported.iter().map(|p| p.1).collect::<Vec<_>>())?;
                apply_neural_priors(&mut scores, &supported, &logits, weights.neural);
            }
        }
        Ok((mode, scores))
    }
    pub fn rank(
        &mut self,
        left: &str,
        right: &str,
        candidates: &[String],
    ) -> Result<Vec<usize>, String> {
        let mode = self.mode(left, right);
        let mut order: Vec<_> = (0..candidates.len()).collect();
        // Dictionary bytes already carry the authoritative unigram ordering.
        if mode == Mode::Unigram {
            return Ok(order);
        }
        if mode == Mode::Neural {
            return self.neural.as_mut().unwrap().rank(left, right, candidates);
        }
        let (_, scores) = self.scores(left, right, candidates)?;
        // Keep characters missing from the count vocabulary in original slots.
        let slots: Vec<_> = order
            .iter()
            .copied()
            .filter(|&i| scores[i].is_finite())
            .collect();
        let mut sorted = slots.clone();
        sorted.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        for (slot, index) in slots.into_iter().zip(sorted) {
            order[slot] = index;
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unmodeled_handwriting_alternatives_cannot_escape_the_language_prior() {
        // First two classes have equal contextual evidence; the third is OOV.
        // It should not outrank a stronger visual alternative just because its
        // missing -infinity prior used to be skipped by recognition fusion.
        let raw = [-0.01, -1.0, -1.2];
        let mut priors = [f32::NEG_INFINITY; 3];
        apply_neural_priors(&mut priors, &[(0, 3), (1, 4)], &[-8.0, -8.0], 1.0);
        let fused: Vec<_> = raw.iter().zip(priors).map(|(v,p)| v+p).collect();
        assert!(fused[0] > fused[1] && fused[1] > fused[2]);
        // Rare characters remain selectable when their strokes are stronger.
        assert!(0.0 + priors[2] > -1.0 + priors[1]);
        let mut counts = [f32::NEG_INFINITY, f32::NEG_INFINITY, -3.0];
        apply_neural_priors(&mut counts, &[(0, 3), (1, 4)], &[-8.0, -8.0], 0.0);
        assert_eq!(counts, [0.0, 0.0, -3.0]);
    }
    #[test]
    fn rejects_invalid_calibration_and_accepts_zero_weight() {
        assert!(FusionWeights {
            unigram: 0.,
            bigram: 0.,
            neural: 0.
        }
        .validate()
        .is_ok());
        for neural in [-1., f32::NAN, f32::INFINITY, 17.] {
            assert!(FusionWeights {
                unigram: 1.,
                bigram: 1.,
                neural
            }
            .validate()
            .is_err());
        }
    }
    #[test]
    fn count_fallback_preserves_empty_order_and_uses_previous_character() {
        let labels = ["我", "你", "係"].map(String::from);
        let counts = CsrLanguageModel {
            num_classes: 3,
            num_edges: 1,
            lambda_unigram: 1.,
            lambda_bigram: 1.,
            unigrams: vec![-1., -2., -3.],
            offsets: vec![0, 1, 1, 1],
            targets: vec![2],
            log_probs: vec![4.],
        };
        let mut lm = HybridLanguageModel::new(None, Some(counts), &labels).unwrap();
        let candidates = ["你", "𰻞", "係"].map(String::from);
        assert_eq!(
            lm.predict_next("", 2)
                .unwrap()
                .1
                .iter()
                .map(|p| p.token.as_str())
                .collect::<Vec<_>>(),
            ["我", "你"]
        );
        assert_eq!(lm.predict_next("我", 1).unwrap().1[0].token, "係");
        assert!(lm.predict_next("我", 0).unwrap().1.is_empty());
        assert_eq!(lm.mode("", "你"), Mode::Unigram);
        assert_eq!(lm.rank("", "", &candidates).unwrap(), vec![0, 1, 2]);
        assert_eq!(lm.mode("我", ""), Mode::Bigram);
        assert_eq!(lm.rank("我", "", &candidates).unwrap(), vec![2, 1, 0]);
        assert_eq!(lm.scores("", "", &labels).unwrap().1, vec![-1., -2., -3.]);
        // Unknown immediately before cursor must not create a false bigram.
        assert_eq!(lm.mode("我👽", ""), Mode::Unigram);
        assert_eq!(
            lm.scores("我👽", "", &labels).unwrap().1,
            vec![-1., -2., -3.]
        );
        assert_eq!(lm.mode("我。", ""), Mode::Unigram);
    }
}
