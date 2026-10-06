//! In-decoder Cantonese Chinese conversion adapter.
//!
//! Provides synchronous token-commitment-time conversion:
//! 1. O(1) static lookup for unambiguous single/multi-character tokens (12,264 mappings).
//! 2. Dynamic linear classifier heads (W_eff * h_t + b_eff) for 59 polysemous clusters
//!    (87 trigger tokens, e.g. 系/係/繫, 发/發/髮, 面/麵).

use crate::kernels;
use crate::safetensors::SafetensorsFile;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ClusterHead {
    pub name: String,
    pub candidates: Vec<String>,
    pub weights: Vec<f32>, // [num_candidates * dim]
    pub biases: Vec<f32>,  // [num_candidates]
    pub num_candidates: usize,
    pub dim: usize,
}

#[derive(Clone, Debug)]
pub struct CantoneseZhconvAdapter {
    /// O(1) static token ID -> converted string mapping.
    pub static_map: HashMap<i32, String>,
    /// Maps trigger token ID -> index in `clusters`.
    pub trigger_to_cluster: HashMap<i32, usize>,
    /// Curated polysemous clusters with linear classifier heads.
    pub clusters: Vec<ClusterHead>,
}

impl CantoneseZhconvAdapter {
    /// Try loading the adapter from `model_dir`.
    ///
    /// Expects:
    /// - `cantonese_zhconv_adapter.json`
    /// - `cantonese_zhconv_adapter.safetensors`
    ///
    /// Returns `None` if either file is missing.
    pub fn load(model_dir: &str) -> Option<Self> {
        let dir = Path::new(model_dir);
        let json_path = dir.join("cantonese_zhconv_adapter.json");
        let sf_path = dir.join("cantonese_zhconv_adapter.safetensors");

        if !json_path.is_file() || !sf_path.is_file() {
            return None;
        }

        let json_str = match fs::read_to_string(&json_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("adapter: failed to read {}: {}", json_path.display(), e);
                return None;
            }
        };

        let json_val: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("adapter: failed to parse {}: {}", json_path.display(), e);
                return None;
            }
        };

        let sf_path_str = sf_path.to_str()?;
        let sf = match SafetensorsFile::open(sf_path_str) {
            Some(f) => f,
            None => {
                eprintln!("adapter: failed to open safetensors {}", sf_path.display());
                return None;
            }
        };

        let clusters_obj = json_val.get("clusters")?.as_object()?;
        let mut clusters: Vec<ClusterHead> = Vec::with_capacity(clusters_obj.len());
        let mut cluster_name_to_idx: HashMap<String, usize> = HashMap::with_capacity(clusters_obj.len());

        for (c_name, cands_val) in clusters_obj {
            let cands_arr = match cands_val.as_array() {
                Some(arr) => arr,
                None => continue,
            };
            let candidates: Vec<String> = cands_arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            let num_candidates = candidates.len();
            if num_candidates == 0 {
                continue;
            }

            let w_name = format!("{}.weight", c_name);
            let b_name = format!("{}.bias", c_name);

            let w_meta = match sf.find(&w_name) {
                Some(m) => m,
                None => {
                    eprintln!("adapter: missing weight tensor {} in {}", w_name, sf_path.display());
                    continue;
                }
            };
            let b_meta = match sf.find(&b_name) {
                Some(m) => m,
                None => {
                    eprintln!("adapter: missing bias tensor {} in {}", b_name, sf_path.display());
                    continue;
                }
            };

            let weights = match sf.get_f32(w_meta) {
                Some(w) => w,
                None => {
                    eprintln!("adapter: failed to load weights for {}", w_name);
                    continue;
                }
            };
            let biases = match sf.get_f32(b_meta) {
                Some(b) => b,
                None => {
                    eprintln!("adapter: failed to load bias for {}", b_name);
                    continue;
                }
            };

            let dim = if num_candidates > 0 {
                weights.len() / num_candidates
            } else {
                0
            };

            if weights.len() != num_candidates * dim || biases.len() != num_candidates {
                eprintln!(
                    "adapter: shape mismatch for cluster {}: weights len {}, biases len {}, num_cands {}",
                    c_name, weights.len(), biases.len(), num_candidates
                );
                continue;
            }

            let idx = clusters.len();
            clusters.push(ClusterHead {
                name: c_name.clone(),
                candidates,
                weights,
                biases,
                num_candidates,
                dim,
            });
            cluster_name_to_idx.insert(c_name.clone(), idx);
        }

        let ambig_obj = json_val.get("ambiguous_tokens")?.as_object()?;
        let mut trigger_to_cluster = HashMap::with_capacity(ambig_obj.len());

        for (tok_str, meta_val) in ambig_obj {
            let tok_id: i32 = match tok_str.parse() {
                Ok(id) => id,
                Err(_) => continue,
            };
            let c_name = match meta_val.get("cluster").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => continue,
            };
            if let Some(&c_idx) = cluster_name_to_idx.get(c_name) {
                trigger_to_cluster.insert(tok_id, c_idx);
            }
        }

        let static_obj = json_val.get("static_map")?.as_object()?;
        let mut static_map = HashMap::with_capacity(static_obj.len());

        for (tok_str, target_val) in static_obj {
            let tok_id: i32 = match tok_str.parse() {
                Ok(id) => id,
                Err(_) => continue,
            };
            if let Some(target_str) = target_val.as_str() {
                static_map.insert(tok_id, target_str.to_string());
            }
        }

        if kernels::verbose() >= 1 {
            eprintln!(
                "adapter: loaded Cantonese in-decoder converter ({} static mappings, {} ambiguous trigger tokens across {} clusters)",
                static_map.len(),
                trigger_to_cluster.len(),
                clusters.len(),
            );
        }

        Some(CantoneseZhconvAdapter {
            static_map,
            trigger_to_cluster,
            clusters,
        })
    }

    /// Converts a predicted token using either dynamic hidden-state linear classification
    /// or static mapping. Returns `None` if the token requires no conversion.
    #[inline]
    pub fn convert_token<'a>(&'a self, token: i32, hidden: &[f32]) -> Option<&'a str> {
        if let Some(&cluster_idx) = self.trigger_to_cluster.get(&token) {
            let cluster = &self.clusters[cluster_idx];
            if hidden.len() >= cluster.dim && cluster.num_candidates > 0 {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_cand_idx = 0;
                for k in 0..cluster.num_candidates {
                    let w_slice = &cluster.weights[k * cluster.dim..(k + 1) * cluster.dim];
                    let mut dot = 0.0f32;
                    for j in 0..cluster.dim {
                        dot += w_slice[j] * hidden[j];
                    }
                    let score = dot + cluster.biases[k];
                    if score > best_score {
                        best_score = score;
                        best_cand_idx = k;
                    }
                }
                return Some(&cluster.candidates[best_cand_idx]);
            }
        }

        if let Some(static_str) = self.static_map.get(&token) {
            return Some(static_str.as_str());
        }

        None
    }

    /// Static conversion without hidden states (fallback for streaming buffers).
    #[inline]
    pub fn convert_token_static<'a>(&'a self, token: i32) -> Option<&'a str> {
        if let Some(&cluster_idx) = self.trigger_to_cluster.get(&token) {
            let cluster = &self.clusters[cluster_idx];
            if let Some(first_cand) = cluster.candidates.first() {
                return Some(first_cand.as_str());
            }
        }

        if let Some(static_str) = self.static_map.get(&token) {
            return Some(static_str.as_str());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_adapter_and_convert() {
        let adapter_path = "/Users/tony/Documents/keyboard/dataset/cantonese-zhconv/results";
        if !std::path::Path::new(adapter_path).join("cantonese_zhconv_adapter.json").exists() {
            return;
        }

        let adapter = CantoneseZhconvAdapter::load(adapter_path)
            .expect("cantonese_zhconv_adapter should load");

        assert!(adapter.static_map.len() >= 12000);
        assert_eq!(adapter.clusters.len(), 59);

        // Test static mapping (e.g. 100673 -> 導致)
        let static_tok = 100673;
        let converted = adapter.convert_token(static_tok, &[]);
        assert_eq!(converted, Some("導致"));

        // Test ambiguous cluster trigger (38176 -> 系 cluster)
        let trigger_tok = 38176;
        assert!(adapter.trigger_to_cluster.contains_key(&trigger_tok));

        // When hidden state is all zeros, bias decides the class (for 系, candidate 0 '係' has bias 5.039)
        let zero_hidden = vec![0.0f32; 2048];
        let pred = adapter.convert_token(trigger_tok, &zero_hidden);
        assert_eq!(pred, Some("係"));

        // Unmapped token returns None
        assert_eq!(adapter.convert_token(999999, &zero_hidden), None);
    }
}

