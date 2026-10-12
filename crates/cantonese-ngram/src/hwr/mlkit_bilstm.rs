use anyhow::{Context, Result, bail};

use super::safetensors::SafeTensors;

pub const HIDDEN_DIM: usize = 192;
pub const MERGED_DIM: usize = 384;
pub const NUM_LAYERS: usize = 4;
pub const NUM_CLASSES: usize = 12362;
pub const BLANK_CLASS: u32 = 12361;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub class_id: u32,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct LstmLayer {
    // Forward weights: [4, 192, in_dim]
    pub fwd_wx: Vec<f32>,
    // Forward recurrent: [4, 192, 192]
    pub fwd_wh: Vec<f32>,
    // Forward bias: [4, 192]
    pub fwd_b: Vec<f32>,

    // Backward weights: [4, 192, in_dim]
    pub bwd_wx: Vec<f32>,
    // Backward recurrent: [4, 192, 192]
    pub bwd_wh: Vec<f32>,
    // Backward bias: [4, 192]
    pub bwd_b: Vec<f32>,

    pub in_dim: usize,
}

impl LstmLayer {
    fn load(weights: &SafeTensors<'_>, layer_idx: usize, in_dim: usize) -> Result<Self> {
        let load_dequant = |name: &str, scale_name: &str| -> Result<Vec<f32>> {
            let view = weights.get(name)?;
            let scale_view = weights.get(scale_name)?;
            let scales = scale_view.f32_vec(scale_name)?;
            if scales.len() != 4 {
                bail!("'{scale_name}' must have 4 gate scales");
            }
            let raw_i8 = view.i8_vec(name)?;
            let gate_len = raw_i8.len() / 4;
            let mut out = Vec::with_capacity(raw_i8.len());
            for (g, &scale) in scales.iter().enumerate() {
                let start = g * gate_len;
                let end = start + gate_len;
                for &val in &raw_i8[start..end] {
                    out.push(val as f32 * scale);
                }
            }
            Ok(out)
        };

        let fwd_wx = load_dequant(
            &format!("lstm{layer_idx}.fwd.w_x"),
            &format!("lstm{layer_idx}.fwd.w_x_scale"),
        )?;
        let fwd_wh = load_dequant(
            &format!("lstm{layer_idx}.fwd.w_h"),
            &format!("lstm{layer_idx}.fwd.w_h_scale"),
        )?;
        let fwd_b = weights
            .get(&format!("lstm{layer_idx}.fwd.bias"))?
            .f32_vec(&format!("lstm{layer_idx}.fwd.bias"))?;

        let bwd_wx = load_dequant(
            &format!("lstm{layer_idx}.bwd.w_x"),
            &format!("lstm{layer_idx}.bwd.w_x_scale"),
        )?;
        let bwd_wh = load_dequant(
            &format!("lstm{layer_idx}.bwd.w_h"),
            &format!("lstm{layer_idx}.bwd.w_h_scale"),
        )?;
        let bwd_b = weights
            .get(&format!("lstm{layer_idx}.bwd.bias"))?
            .f32_vec(&format!("lstm{layer_idx}.bwd.bias"))?;

        if fwd_wx.len() != 4 * HIDDEN_DIM * in_dim || bwd_wx.len() != 4 * HIDDEN_DIM * in_dim {
            bail!("layer {layer_idx} input weights size mismatch");
        }
        if fwd_wh.len() != 4 * HIDDEN_DIM * HIDDEN_DIM || bwd_wh.len() != 4 * HIDDEN_DIM * HIDDEN_DIM {
            bail!("layer {layer_idx} recurrent weights size mismatch");
        }
        if fwd_b.len() != 4 * HIDDEN_DIM || bwd_b.len() != 4 * HIDDEN_DIM {
            bail!("layer {layer_idx} bias size mismatch");
        }

        Ok(Self {
            fwd_wx,
            fwd_wh,
            fwd_b,
            bwd_wx,
            bwd_wh,
            bwd_b,
            in_dim,
        })
    }

    pub fn forward_sequence(&self, input: &[f32], time_steps: usize, output: &mut [f32]) {
        assert_eq!(input.len(), time_steps * self.in_dim);
        assert_eq!(output.len(), time_steps * MERGED_DIM);

        // Forward direction: t = 0 .. time_steps - 1
        let mut hf = [0.0f32; HIDDEN_DIM];
        let mut cf = [0.0f32; HIDDEN_DIM];
        let mut gates = [0.0f32; 4 * HIDDEN_DIM];

        for t in 0..time_steps {
            let x = &input[t * self.in_dim..(t + 1) * self.in_dim];
            self.compute_gates(x, &hf, &self.fwd_wx, &self.fwd_wh, &self.fwd_b, &mut gates);
            Self::update_cell(&gates, &mut cf, &mut hf);

            let out_slice = &mut output[t * MERGED_DIM..t * MERGED_DIM + HIDDEN_DIM];
            out_slice.copy_from_slice(&hf);
        }

        // Backward direction: t = time_steps - 1 down to 0
        let mut hb = [0.0f32; HIDDEN_DIM];
        let mut cb = [0.0f32; HIDDEN_DIM];

        for t in (0..time_steps).rev() {
            let x = &input[t * self.in_dim..(t + 1) * self.in_dim];
            self.compute_gates(x, &hb, &self.bwd_wx, &self.bwd_wh, &self.bwd_b, &mut gates);
            Self::update_cell(&gates, &mut cb, &mut hb);

            let out_slice = &mut output[t * MERGED_DIM + HIDDEN_DIM..(t + 1) * MERGED_DIM];
            out_slice.copy_from_slice(&hb);
        }
    }

    #[inline(always)]
    fn compute_gates(
        &self,
        x: &[f32],
        h: &[f32; HIDDEN_DIM],
        wx: &[f32],
        wh: &[f32],
        bias: &[f32],
        gates: &mut [f32; 4 * HIDDEN_DIM],
    ) {
        for g in 0..4 {
            let g_offset = g * HIDDEN_DIM;
            for i in 0..HIDDEN_DIM {
                let wx_row = &wx[(g_offset + i) * self.in_dim..(g_offset + i + 1) * self.in_dim];
                let wh_row = &wh[(g_offset + i) * HIDDEN_DIM..(g_offset + i + 1) * HIDDEN_DIM];
                let dot_x = dot_product(wx_row, x);
                let dot_h = dot_product(wh_row, h);
                gates[g_offset + i] = dot_x + dot_h + bias[g_offset + i];
            }
        }
    }

    #[inline(always)]
    fn update_cell(gates: &[f32; 4 * HIDDEN_DIM], c: &mut [f32; HIDDEN_DIM], h: &mut [f32; HIDDEN_DIM]) {
        for i in 0..HIDDEN_DIM {
            let in_gate = sigmoid(gates[0 * HIDDEN_DIM + i]);
            let forget_gate = sigmoid(gates[1 * HIDDEN_DIM + i]);
            let cell_cand = tanh(gates[2 * HIDDEN_DIM + i]);
            let out_gate = sigmoid(gates[3 * HIDDEN_DIM + i]);

            let new_c = (forget_gate * c[i] + in_gate * cell_cand).clamp(-50.0, 50.0);
            c[i] = new_c;
            h[i] = out_gate * tanh(new_c);
        }
    }
}

pub struct MlkitBiLstmRecognizer {
    layers: Vec<LstmLayer>,
    fc_w: Vec<f32>, // [12362, 384]
    fc_b: Vec<f32>, // [12362]
    labels: Vec<String>,
    pub lm: Option<crate::counts::CsrLanguageModel>,
    raw_candidate_scores: Vec<f32>,
    neural_lm_scores: Option<(Vec<f32>, f32)>,
    context_lm: Option<crate::hybrid::HybridLanguageModel>,
    language_context: (String, String),

    // Reusable buffers to avoid allocations during inference
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    logits_buf: Vec<f32>,
}

impl MlkitBiLstmRecognizer {
    pub fn from_bytes(model_bytes: &[u8], labels_str: &str) -> Result<Self> {
        let weights = SafeTensors::parse(model_bytes)?;
        let mut layers = Vec::with_capacity(NUM_LAYERS);

        for l in 0..NUM_LAYERS {
            let in_dim = if l == 0 { 10 } else { MERGED_DIM };
            let layer = LstmLayer::load(&weights, l, in_dim)
                .with_context(|| format!("loading LSTM layer {l}"))?;
            layers.push(layer);
        }

        let fc_view = weights.get("fc.weight")?;
        let fc_scale_view = weights.get("fc.weight_scale")?;
        let fc_scales = fc_scale_view.f32_vec("fc.weight_scale")?;
        let fc_scale = fc_scales[0];
        let fc_w_i8 = fc_view.i8_vec("fc.weight")?;
        let fc_w: Vec<f32> = fc_w_i8.iter().map(|&v| v as f32 * fc_scale).collect();

        let fc_b = weights.get("fc.bias")?.f32_vec("fc.bias")?;

        if fc_w.len() != NUM_CLASSES * MERGED_DIM || fc_b.len() != NUM_CLASSES {
            bail!("fully connected layer dimensions mismatch");
        }

        let labels: Vec<String> = labels_str
            .lines()
            .map(|s| s.trim_end().to_string())
            .collect();

        Ok(Self {
            layers,
            fc_w,
            fc_b,
            labels,
            lm: None,
            raw_candidate_scores: Vec::new(),
            neural_lm_scores: None,
            context_lm: None, language_context: (String::new(), String::new()),
            buf_a: Vec::new(),
            buf_b: Vec::new(),
            logits_buf: vec![0.0f32; NUM_CLASSES],
        })
    }

    pub fn set_lm(&mut self, lm: crate::counts::CsrLanguageModel) {
        self.lm = Some(lm);
    }

    pub fn set_context_lm(&mut self, neural: Option<crate::LanguageModel>) -> Result<()> {
        self.set_neural_context_lm(neural.map(Into::into))
    }

    pub fn set_neural_context_lm(&mut self, neural: Option<crate::NeuralLanguageModel>) -> Result<()> {
        self.context_lm = Some(crate::hybrid::HybridLanguageModel::with_neural(neural, self.lm.clone(), &self.labels).map_err(anyhow::Error::msg)?);
        Ok(())
    }

    pub fn set_fusion_weights(&mut self, weights: crate::hybrid::FusionWeights) -> Result<()> {
        self.context_lm.as_mut().ok_or_else(||anyhow::anyhow!("Context LM not initialized"))?
            .set_fusion_weights(weights).map_err(anyhow::Error::msg)
    }

    pub fn set_language_context(&mut self, left: String, right: String) {
        self.language_context = (left, right);
    }

    /// Raw peak-frame log probabilities from the last prediction, before LM fusion.
    pub fn raw_candidate_scores(&self) -> &[f32] {
        &self.raw_candidate_scores
    }

    /// Replace the count LM with class-aligned neural log priors for this context.
    /// The caller computes new priors whenever the cursor context changes.
    /// None restores the existing count LM. All recognition classes remain available.
    pub fn set_neural_lm_scores(&mut self, scores: Option<Vec<f32>>, weight: f32) -> Result<()> {
        if !weight.is_finite() || weight < 0.0 {
            bail!("neural LM weight must be finite and nonnegative");
        }
        if let Some(ref values) = scores {
            if values.len() != NUM_CLASSES || values.iter().any(|v| !v.is_finite()) {
                bail!("neural LM requires finite scores for all recognition classes");
            }
        }
        self.neural_lm_scores = scores.map(|values| (values, weight));
        Ok(())
    }

    fn fuse_language_model(&mut self, scores: &mut [f32], previous: Option<usize>) {
        self.raw_candidate_scores.clear();
        self.raw_candidate_scores.extend_from_slice(scores);
        if let Some((ref priors, weight)) = self.neural_lm_scores {
            for (score, prior) in scores.iter_mut().zip(priors) {
                *score += weight * prior;
            }
        } else if let Some(ref mut shared) = self.context_lm {
            let (left, right) = &self.language_context;
            let left = if left.is_empty() && right.is_empty() { previous.and_then(|i|self.labels.get(i)).map(String::as_str).unwrap_or("") } else { left.as_str() };
            match shared.scores(left, right, &self.labels) {
                Ok((_, priors)) => for (score, prior) in scores.iter_mut().zip(priors) { if prior.is_finite() { *score += prior; } },
                Err(_) => if let Some(ref counts) = self.lm { counts.apply_to_scores(scores, previous); },
            }
        } else if let Some(ref lm) = self.lm {
            lm.apply_to_scores(scores, previous);
        }
    }

    /// Run inference for input features of shape `[time_steps, 10]`.
    /// Returns decoded CTC text and top-K predictions for the final or dominant time steps.
    pub fn predict_features(&mut self, input_features: &[f32], time_steps: usize) -> Result<InferenceResult> {
        if time_steps == 0 {
            bail!("cannot run inference on empty trajectory");
        }
        if input_features.len() != time_steps * 10 {
            bail!(
                "expected input feature length {} ({} steps * 10), got {}",
                time_steps * 10,
                time_steps,
                input_features.len()
            );
        }

        let needed_buf_size = time_steps * MERGED_DIM;
        if self.buf_a.len() < needed_buf_size {
            self.buf_a.resize(needed_buf_size, 0.0);
            self.buf_b.resize(needed_buf_size, 0.0);
        }

        // Layer 0: input [T, 10] -> buf_a [T, 384]
        self.layers[0].forward_sequence(input_features, time_steps, &mut self.buf_a[..needed_buf_size]);

        // Layer 1: buf_a [T, 384] -> buf_b [T, 384]
        let (src_slice, dst_slice) = (&self.buf_a[..needed_buf_size], &mut self.buf_b[..needed_buf_size]);
        self.layers[1].forward_sequence(src_slice, time_steps, dst_slice);

        // Layer 2: buf_b [T, 384] -> buf_a [T, 384]
        let (src_slice, dst_slice) = (&self.buf_b[..needed_buf_size], &mut self.buf_a[..needed_buf_size]);
        self.layers[2].forward_sequence(src_slice, time_steps, dst_slice);

        // Layer 3: buf_a [T, 384] -> buf_b [T, 384]
        let (src_slice, dst_slice) = (&self.buf_a[..needed_buf_size], &mut self.buf_b[..needed_buf_size]);
        self.layers[3].forward_sequence(src_slice, time_steps, dst_slice);

        // FC layer: compute logits for all time steps
        // CTC greedy decoding + per-class peak log-probability ranking
        let mut frame_argmax = Vec::with_capacity(time_steps);
        let mut best_logprob = vec![f32::NEG_INFINITY; NUM_CLASSES];

        for t in 0..time_steps {
            let h_t = &self.buf_b[t * MERGED_DIM..(t + 1) * MERGED_DIM];
            Self::compute_fc_into(&self.fc_w, &self.fc_b, h_t, &mut self.logits_buf);

            let mut best_c = 0usize;
            let mut best_logit = self.logits_buf[0];
            for c in 1..NUM_CLASSES {
                let logit = self.logits_buf[c];
                if logit > best_logit {
                    best_logit = logit;
                    best_c = c;
                }
            }
            frame_argmax.push(best_c as u32);

            let mut sum_exp = 0.0f32;
            for &l in self.logits_buf.iter() {
                sum_exp += (l - best_logit).exp();
            }
            let log_z = best_logit + sum_exp.ln();
            for c in 0..NUM_CLASSES {
                let lp = self.logits_buf[c] - log_z;
                if lp > best_logprob[c] {
                    best_logprob[c] = lp;
                }
            }
        }

        // CTC greedy collapse: remove consecutive duplicates and drop blank (12361)
        let mut ctc_tokens = Vec::new();
        let mut prev_token = None;
        for &tok in &frame_argmax {
            if Some(tok) != prev_token {
                if tok != BLANK_CLASS {
                    ctc_tokens.push(tok);
                }
                prev_token = Some(tok);
            }
        }

        // Format text from ctc_tokens
        let mut text = String::new();
        for &tok in &ctc_tokens {
            if let Some(lbl) = self.labels.get(tok as usize) {
                text.push_str(lbl);
            }
        }

        // If a language model is attached, apply unigram and bigram scores
        self.fuse_language_model(&mut best_logprob, None);

        // Rank top candidates by peak per-frame log-probability (excluding blank)
        let mut indexed_scores: Vec<(u32, f32)> = best_logprob
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as u32 != BLANK_CLASS)
            .map(|(i, &score)| (i as u32, score))
            .collect();
        indexed_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_candidates: Vec<Prediction> = indexed_scores
            .iter()
            .take(10)
            .map(|&(class_id, score)| Prediction { class_id, score })
            .collect();

        Ok(InferenceResult {
            text,
            ctc_tokens,
            top_candidates,
            time_steps,
        })
    }

    pub fn predict_features_with_context(
        &mut self,
        input_features: &[f32],
        time_steps: usize,
        prev_class: Option<usize>,
    ) -> Result<InferenceResult> {
        if time_steps == 0 {
            bail!("cannot run inference on empty trajectory");
        }
        if input_features.len() != time_steps * 10 {
            bail!(
                "expected input feature length {} ({} steps * 10), got {}",
                time_steps * 10,
                time_steps,
                input_features.len()
            );
        }

        let needed_buf_size = time_steps * MERGED_DIM;
        if self.buf_a.len() < needed_buf_size {
            self.buf_a.resize(needed_buf_size, 0.0);
            self.buf_b.resize(needed_buf_size, 0.0);
        }

        self.layers[0].forward_sequence(input_features, time_steps, &mut self.buf_a[..needed_buf_size]);
        let (src_slice, dst_slice) = (&self.buf_a[..needed_buf_size], &mut self.buf_b[..needed_buf_size]);
        self.layers[1].forward_sequence(src_slice, time_steps, dst_slice);
        let (src_slice, dst_slice) = (&self.buf_b[..needed_buf_size], &mut self.buf_a[..needed_buf_size]);
        self.layers[2].forward_sequence(src_slice, time_steps, dst_slice);
        let (src_slice, dst_slice) = (&self.buf_a[..needed_buf_size], &mut self.buf_b[..needed_buf_size]);
        self.layers[3].forward_sequence(src_slice, time_steps, dst_slice);

        let mut frame_argmax = Vec::with_capacity(time_steps);
        let mut best_logprob = vec![f32::NEG_INFINITY; NUM_CLASSES];

        for t in 0..time_steps {
            let h_t = &self.buf_b[t * MERGED_DIM..(t + 1) * MERGED_DIM];
            Self::compute_fc_into(&self.fc_w, &self.fc_b, h_t, &mut self.logits_buf);

            let mut best_c = 0usize;
            let mut best_logit = self.logits_buf[0];
            for c in 1..NUM_CLASSES {
                let logit = self.logits_buf[c];
                if logit > best_logit {
                    best_logit = logit;
                    best_c = c;
                }
            }
            frame_argmax.push(best_c as u32);

            let mut sum_exp = 0.0f32;
            for &l in self.logits_buf.iter() {
                sum_exp += (l - best_logit).exp();
            }
            let log_z = best_logit + sum_exp.ln();
            for c in 0..NUM_CLASSES {
                let lp = self.logits_buf[c] - log_z;
                if lp > best_logprob[c] {
                    best_logprob[c] = lp;
                }
            }
        }

        let mut ctc_tokens = Vec::new();
        let mut prev_token = None;
        for &tok in &frame_argmax {
            if Some(tok) != prev_token {
                if tok != BLANK_CLASS {
                    ctc_tokens.push(tok);
                }
                prev_token = Some(tok);
            }
        }

        let mut text = String::new();
        for &tok in &ctc_tokens {
            if let Some(lbl) = self.labels.get(tok as usize) {
                text.push_str(lbl);
            }
        }

        // Apply unigram and bigram scores
        self.fuse_language_model(&mut best_logprob, prev_class);

        let mut indexed_scores: Vec<(u32, f32)> = best_logprob
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as u32 != BLANK_CLASS)
            .map(|(i, &score)| (i as u32, score))
            .collect();
        indexed_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_candidates: Vec<Prediction> = indexed_scores
            .iter()
            .take(10)
            .map(|&(class_id, score)| Prediction { class_id, score })
            .collect();

        Ok(InferenceResult {
            text,
            ctc_tokens,
            top_candidates,
            time_steps,
        })
    }

    #[inline(always)]
    fn compute_fc_into(fc_w: &[f32], fc_b: &[f32], h: &[f32], logits: &mut [f32]) {
        for c in 0..NUM_CLASSES {
            let w_row = &fc_w[c * MERGED_DIM..(c + 1) * MERGED_DIM];
            logits[c] = dot_product(w_row, h) + fc_b[c];
        }
    }

    pub fn label_for_class(&self, class_id: u32) -> Option<&str> {
        self.labels.get(class_id as usize).map(|s| s.as_str())
    }

    pub fn class_for_label(&self, label: &str) -> Option<u32> {
        self.labels
            .iter()
            .position(|l| l == label)
            .map(|idx| idx as u32)
    }

    pub fn predict_strokes(
        &mut self,
        strokes: &[Vec<StrokePoint>],
        canvas_width: f32,
        canvas_height: f32,
        top_k: usize,
    ) -> Result<Vec<String>> {
        self.predict_strokes_with_context(strokes, canvas_width, canvas_height, top_k, None)
    }

    pub fn predict_strokes_with_context(
        &mut self,
        strokes: &[Vec<StrokePoint>],
        _canvas_width: f32,
        _canvas_height: f32,
        top_k: usize,
        prev_char: Option<&str>,
    ) -> Result<Vec<String>> {
        let total_points: usize = strokes.iter().map(|s| s.len()).sum();
        if total_points < 2 {
            return Ok(Vec::new());
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_t = f32::INFINITY;

        for stroke in strokes {
            for pt in stroke {
                min_x = min_x.min(pt.x);
                max_x = max_x.max(pt.x);
                min_y = min_y.min(pt.y);
                max_y = max_y.max(pt.y);
                min_t = min_t.min(pt.t_ms);
            }
        }

        let box_w = (max_x - min_x).max(1e-6);
        let box_h = (max_y - min_y).max(1e-6);
        let box_size = box_w.max(box_h);
        let sc = 1.0 / box_size;
        let t_scale = 0.002f32;

        let mut norm_strokes: Vec<Vec<[f32; 3]>> = Vec::with_capacity(strokes.len());
        for stroke in strokes {
            let mut pts: Vec<[f32; 3]> = Vec::with_capacity(stroke.len());
            for pt in stroke {
                let p = [
                    (pt.x - min_x) * sc,
                    (pt.y - min_y) * sc,
                    (pt.t_ms - min_t) * t_scale,
                ];
                if let Some(last) = pts.last() {
                    let dx = p[0] - last[0];
                    let dy = p[1] - last[1];
                    if (dx * dx + dy * dy) <= 1e-9 {
                        continue;
                    }
                }
                pts.push(p);
            }
            if !pts.is_empty() {
                norm_strokes.push(pts);
            }
        }

        let mut features = Vec::new();
        let mut prev_pt: Option<[f32; 3]> = None;

        for stroke in &norm_strokes {
            if let Some(prev) = prev_pt {
                let curr0 = stroke[0];
                let p0 = prev;
                let p1 = [
                    prev[0] + (curr0[0] - prev[0]) / 3.0,
                    prev[1] + (curr0[1] - prev[1]) / 3.0,
                    prev[2] + (curr0[2] - prev[2]) / 3.0,
                ];
                let p2 = [
                    prev[0] + 2.0 * (curr0[0] - prev[0]) / 3.0,
                    prev[1] + 2.0 * (curr0[1] - prev[1]) / 3.0,
                    prev[2] + 2.0 * (curr0[2] - prev[2]) / 3.0,
                ];
                let p3 = curr0;
                features.extend_from_slice(&compute_curve_features([p0, p1, p2, p3], 0.0));
            }

            let mut seg_stroke = stroke.clone();
            if seg_stroke.len() == 1 {
                seg_stroke.push([seg_stroke[0][0] + 1e-3, seg_stroke[0][1], seg_stroke[0][2]]);
            }

            let segs = segment_stroke_greedy(&seg_stroke, 0.01);
            for seg in segs {
                let (ctrl_pts, _) = fit_cubic_bezier(seg);
                features.extend_from_slice(&compute_curve_features(ctrl_pts, 1.0));
            }

            prev_pt = Some(*stroke.last().unwrap());
        }

        let time_steps = features.len() / 10;
        if time_steps == 0 {
            return Ok(Vec::new());
        }

        let prev_class = prev_char
            .and_then(|ch| self.class_for_label(ch))
            .map(|id| id as usize);

        let res = self.predict_features_with_context(&features, time_steps, prev_class)?;

        let mut candidates = Vec::with_capacity(top_k);
        for pred in &res.top_candidates {
            if let Some(lbl) = self.label_for_class(pred.class_id) {
                if !lbl.is_empty() && !lbl.starts_with('<') && !candidates.contains(&lbl.to_string()) {
                    candidates.push(lbl.to_string());
                }
            }
            if candidates.len() >= top_k {
                break;
            }
        }

        Ok(candidates)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokePoint {
    pub x: f32,
    pub y: f32,
    pub t_ms: f32,
}

#[inline]
fn angle_between_vectors(a: [f32; 2], b: [f32; 2]) -> f32 {
    let det = a[0] * b[1] - a[1] * b[0];
    let dot = a[0] * b[0] + a[1] * b[1];
    det.atan2(dot)
}

fn compute_curve_features(p: [[f32; 3]; 4], pen_down: f32) -> [f32; 10] {
    let d = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    let v1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let v2 = [p[2][0] - p[3][0], p[2][1] - p[3][1], p[2][2] - p[3][2]];

    let dn = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let (a1, r1, a2, r2) = if dn > 1e-9 {
        let a1 = angle_between_vectors([d[0], d[1]], [v1[0], v1[1]]);
        let r1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt() / dn;
        let a2 = angle_between_vectors([d[0], d[1]], [-v2[0], -v2[1]]);
        let r2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt() / dn;
        (a1, r1, a2, r2)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    [
        pen_down,
        d[0],
        d[1],
        a1,
        r1,
        a2,
        r2,
        d[2],
        v1[2],
        v2[2],
    ]
}

fn fit_cubic_bezier(pts: &[[f32; 3]]) -> ([[f32; 3]; 4], f32) {
    let n = pts.len();
    let p0 = pts[0];
    let p3 = pts[n - 1];

    if n < 4 {
        let p1 = [
            p0[0] + (p3[0] - p0[0]) / 3.0,
            p0[1] + (p3[1] - p0[1]) / 3.0,
            p0[2] + (p3[2] - p0[2]) / 3.0,
        ];
        let p2 = [
            p0[0] + 2.0 * (p3[0] - p0[0]) / 3.0,
            p0[1] + 2.0 * (p3[1] - p0[1]) / 3.0,
            p0[2] + 2.0 * (p3[2] - p0[2]) / 3.0,
        ];
        return ([p0, p1, p2, p3], 0.0);
    }

    let mut dists = Vec::with_capacity(n);
    dists.push(0.0f32);
    let mut accum = 0.0f32;
    for i in 1..n {
        let dx = pts[i][0] - pts[i - 1][0];
        let dy = pts[i][1] - pts[i - 1][1];
        accum += (dx * dx + dy * dy).sqrt();
        dists.push(accum);
    }
    let total_chord = accum;

    let mut c11 = 0.0f32;
    let mut c12 = 0.0f32;
    let mut c22 = 0.0f32;
    let mut r1 = [0.0f32; 3];
    let mut r2 = [0.0f32; 3];

    let mut b_vals = Vec::with_capacity(n);

    for (k, pt) in pts.iter().enumerate() {
        let s = if total_chord > 1e-9 {
            dists[k] / total_chord
        } else {
            k as f32 / (n - 1) as f32
        };
        let one_minus_s = 1.0 - s;
        let b0 = one_minus_s * one_minus_s * one_minus_s;
        let b1 = 3.0 * s * one_minus_s * one_minus_s;
        let b2 = 3.0 * s * s * one_minus_s;
        let b3 = s * s * s;

        b_vals.push((b0, b1, b2, b3));

        c11 += b1 * b1;
        c12 += b1 * b2;
        c22 += b2 * b2;

        for dim in 0..3 {
            let rhs = pt[dim] - b0 * p0[dim] - b3 * p3[dim];
            r1[dim] += b1 * rhs;
            r2[dim] += b2 * rhs;
        }
    }

    let det = c11 * c22 - c12 * c12;
    let mut p1 = [0.0f32; 3];
    let mut p2 = [0.0f32; 3];

    if det.abs() > 1e-9 {
        let inv_det = 1.0 / det;
        for dim in 0..3 {
            p1[dim] = (c22 * r1[dim] - c12 * r2[dim]) * inv_det;
            p2[dim] = (-c12 * r1[dim] + c11 * r2[dim]) * inv_det;
        }
    } else {
        for dim in 0..3 {
            p1[dim] = p0[dim] + (p3[dim] - p0[dim]) / 3.0;
            p2[dim] = p0[dim] + 2.0 * (p3[dim] - p0[dim]) / 3.0;
        }
    }

    let mut max_err = 0.0f32;
    for (k, pt) in pts.iter().enumerate() {
        let (b0, b1, b2, b3) = b_vals[k];
        let fit_x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0];
        let fit_y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1];
        let err_sq = (fit_x - pt[0]) * (fit_x - pt[0]) + (fit_y - pt[1]) * (fit_y - pt[1]);
        if err_sq > max_err {
            max_err = err_sq;
        }
    }

    ([p0, p1, p2, p3], max_err.sqrt())
}

fn segment_stroke_greedy<'a>(pts: &'a [[f32; 3]], max_err: f32) -> Vec<&'a [[f32; 3]]> {
    let mut segs = Vec::new();
    let n = pts.len();
    let mut i = 0usize;

    while i < n - 1 {
        let mut best = i + 1;
        let mut j = i + 1;
        while j < n - 1 {
            let (_, err) = fit_cubic_bezier(&pts[i..=j + 1]);
            if err > max_err {
                break;
            }
            j += 1;
            best = j;
        }
        segs.push(&pts[i..=best]);
        i = best;
    }

    segs
}


#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub text: String,
    pub ctc_tokens: Vec<u32>,
    pub top_candidates: Vec<Prediction>,
    pub time_steps: usize,
}

#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;

    for (ca, cb) in chunks_a.zip(chunks_b) {
        acc0 += ca[0] * cb[0];
        acc1 += ca[1] * cb[1];
        acc2 += ca[2] * cb[2];
        acc3 += ca[3] * cb[3];
        acc4 += ca[4] * cb[4];
        acc5 += ca[5] * cb[5];
        acc6 += ca[6] * cb[6];
        acc7 += ca[7] * cb[7];
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }
    sum
}

#[inline(always)]

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn shared_context_fallback_does_not_double_apply_count_priors() -> Result<()> {
        let mut recognizer = MlkitBiLstmRecognizer {
            layers: Vec::new(), fc_w: Vec::new(), fc_b: Vec::new(),
            labels: ["我", "你", "係"].map(String::from).to_vec(),
            lm: Some(crate::counts::CsrLanguageModel {
                num_classes: 3, num_edges: 1, lambda_unigram: 1.0, lambda_bigram: 1.0,
                unigrams: vec![-1.0,-2.0,-3.0], offsets: vec![0,1,1,1], targets: vec![2], log_probs: vec![4.0],
            }),
            raw_candidate_scores: Vec::new(), neural_lm_scores: None,
            context_lm: None, language_context: (String::new(),String::new()),
            buf_a: Vec::new(), buf_b: Vec::new(), logits_buf: Vec::new(),
        };
        recognizer.set_context_lm(None)?;
        let mut scores = vec![-1.0;3];
        recognizer.fuse_language_model(&mut scores,None);
        assert_eq!(scores,vec![-2.0,-3.0,-4.0]);
        recognizer.set_language_context("我".into(),String::new());
        let mut scores = vec![-1.0;3];
        recognizer.fuse_language_model(&mut scores,None);
        assert_eq!(scores,vec![-2.0,-3.0,0.0]);
        assert_eq!(recognizer.raw_candidate_scores(),&[-1.0;3]);
        recognizer.set_language_context("我。".into(),String::new());
        let mut scores = vec![-1.0;3];
        recognizer.fuse_language_model(&mut scores,Some(0));
        assert_eq!(scores,vec![-2.0,-3.0,-4.0]);
        recognizer.set_fusion_weights(crate::hybrid::FusionWeights {unigram:0.0,bigram:2.0,neural:0.0})?;
        recognizer.set_language_context("我".into(),String::new());
        let mut scores = vec![-1.0;3];
        recognizer.fuse_language_model(&mut scores,None);
        assert_eq!(scores,vec![-1.0,-1.0,7.0]);
        Ok(())
    }

    #[test]
    fn neural_fusion_replaces_count_lm_and_preserves_raw_scores() -> Result<()> {
        let mut recognizer = MlkitBiLstmRecognizer {
            layers: Vec::new(), fc_w: Vec::new(), fc_b: Vec::new(), labels: Vec::new(),
            lm: Some(crate::counts::CsrLanguageModel {
                num_classes: NUM_CLASSES, num_edges: 0,
                lambda_unigram: 0.5, lambda_bigram: 0.0,
                unigrams: vec![-4.0; NUM_CLASSES], offsets: vec![0; NUM_CLASSES + 1],
                targets: Vec::new(), log_probs: Vec::new(),
            }),
            raw_candidate_scores: Vec::new(), neural_lm_scores: None,
            context_lm: None, language_context: (String::new(), String::new()),
            buf_a: Vec::new(), buf_b: Vec::new(), logits_buf: Vec::new(),
        };
        let raw = vec![-1.0; NUM_CLASSES];
        let mut scores = raw.clone();
        recognizer.fuse_language_model(&mut scores, None);
        assert_eq!(scores[0], -3.0);
        assert_eq!(recognizer.raw_candidate_scores(), &raw);
        recognizer.set_neural_lm_scores(Some(vec![-6.0; NUM_CLASSES]), 0.25)?;
        scores.clone_from(&raw);
        recognizer.fuse_language_model(&mut scores, None);
        assert_eq!(scores[0], -2.5); // Neural only: count prior is not added twice.
        assert_eq!(recognizer.raw_candidate_scores(), &raw);
        assert!(recognizer.set_neural_lm_scores(Some(vec![0.0; 2]), 1.0).is_err());
        assert!(recognizer.set_neural_lm_scores(None, f32::NAN).is_err());
        recognizer.set_neural_lm_scores(None, 0.0)?;
        scores.clone_from(&raw);
        recognizer.fuse_language_model(&mut scores, None);
        assert_eq!(scores[0], -3.0); // Restore the original count LM.
        Ok(())
    }

    #[test]
    fn test_mlkit_reference_parity() -> Result<()> {
        let model_path = "/Users/tony/Documents/keyboard/mlkit_model/chinese_lstm_4x192.safetensors";
        let labels_path = "/Users/tony/Documents/keyboard/mlkit_model/labels.txt";
        let ref_path = "/Users/tony/Documents/keyboard/mlkit_model/test_reference.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return Ok(());
        }

        let model_bytes = fs::read(model_path)?;
        let labels_str = fs::read_to_string(labels_path)?;
        let ref_str = fs::read_to_string(ref_path)?;
        let ref_val: serde_json::Value = serde_json::from_str(&ref_str)?;

        let mut recognizer = MlkitBiLstmRecognizer::from_bytes(&model_bytes, &labels_str)?;
        let time_steps = ref_val["time_steps"].as_u64().unwrap() as usize;
        let input: Vec<f32> = ref_val["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected_argmax: Vec<u32> = ref_val["argmax_per_step"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        let res = recognizer.predict_features(&input, time_steps)?;
        assert_eq!(res.time_steps, time_steps);
        assert!(!res.top_candidates.is_empty());
        assert_eq!(res.top_candidates[0].class_id, *expected_argmax.last().unwrap());

        println!("Test passed! Predicted character: {}", res.text);
        println!("Top candidate class ID: {}, label: {:?}", res.top_candidates[0].class_id, recognizer.label_for_class(res.top_candidates[0].class_id));
        Ok(())
    }

    #[test]
    fn test_predict_captured_wo() -> Result<()> {
        let model_path = "/Users/tony/Documents/keyboard/mlkit_model/chinese_lstm_4x192.safetensors";
        let labels_path = "/Users/tony/Documents/keyboard/mlkit_model/labels.txt";
        let strokes_path = "/tmp/last_input_strokes.json";

        if !std::path::Path::new(strokes_path).exists() {
            eprintln!("Strokes file not found, skipping");
            return Ok(());
        }

        let model_bytes = fs::read(model_path)?;
        let labels_str = fs::read_to_string(labels_path)?;
        let mut recognizer = MlkitBiLstmRecognizer::from_bytes(&model_bytes, &labels_str)?;

        let strokes_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(strokes_path)?)?;
        let raw_strokes = strokes_json["strokes"].as_array().unwrap();

        let mut strokes = Vec::new();
        for s in raw_strokes {
            let mut stroke = Vec::new();
            for pt in s.as_array().unwrap() {
                stroke.push(StrokePoint {
                    x: pt["x"].as_f64().unwrap() as f32,
                    y: pt["y"].as_f64().unwrap() as f32,
                    t_ms: pt["t"].as_f64().unwrap() as f32,
                });
            }
            strokes.push(stroke);
        }

        let candidates = recognizer.predict_strokes(&strokes, 1058.0, 607.0, 10)?;
        println!("Rust engine candidates for captured 我: {:?}", candidates);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0], "我");
        Ok(())
    }
}
