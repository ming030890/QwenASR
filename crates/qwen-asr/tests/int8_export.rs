//! Opt-in checks against a real per-row INT8 export, including scalar references.
//! QWEN_AUDIT_MODEL=/path cargo test -p qwen-asr --release --test int8_export -- --ignored
#![cfg(target_arch = "aarch64")]

use qwen_asr::{context::QwenCtx, decoder, kernels};

fn scalar_rows(x: &[i8], x_scale: f32, w: &[i8], scales: &[f32]) -> Vec<f32> {
    w.chunks_exact(x.len())
        .zip(scales)
        .map(|(row, scale)| {
            let dot: i32 = row.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum();
            dot as f32 * x_scale * scale
        })
        .collect()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 2e-5 * b.abs().max(1.0),
            "row {i}: {a} != {b}"
        );
    }
}

#[test]
#[ignore = "requires QWEN_AUDIT_MODEL pointing to the full INT8 export"]
fn full_int8_export_load_and_scalar_decode_reference() {
    let model = std::env::var("QWEN_AUDIT_MODEL").expect("QWEN_AUDIT_MODEL is required");
    kernels::set_threads(1);
    let ctx = QwenCtx::load(&model).expect("INT8 export must load");
    let cfg = &ctx.config;
    let dim = cfg.dec_hidden;
    let dec = &ctx.decoder;
    assert_eq!(ctx.encoder.layers.len(), 24);
    assert_eq!(dec.layers.len(), 28);
    for layer in &ctx.encoder.layers {
        assert!(layer.wq_weight_bf16.is_null());
        assert!(layer.wk_weight_bf16.is_null());
        assert!(layer.wv_weight_bf16.is_null());
        assert!(layer.wo_weight_bf16.is_null());
        assert!(layer.fc1_weight_bf16.is_null());
        assert!(layer.fc2_weight_bf16.is_null());
        for weights in [
            &layer.wq_weight_int8,
            &layer.wk_weight_int8,
            &layer.wv_weight_int8,
            &layer.wo_weight_int8,
            &layer.fc1_weight_int8,
            &layer.fc2_weight_int8,
        ] {
            assert!(weights.as_ref().is_some_and(|w| !w.is_empty()));
        }
    }
    for layer in &dec.layers {
        assert!(layer.wq_weight_bf16.is_null());
        assert!(layer.wk_weight_bf16.is_null());
        assert!(layer.wv_weight_bf16.is_null());
        assert!(layer.wo_weight_bf16.is_null());
        assert!(layer.down_weight_bf16.is_null());
        assert!(layer.gate_up_fused_bf16.is_empty());
        for (weights, scales, input) in [
            (&layer.wq_int8, &layer.wq_int8_scales, dim),
            (&layer.wk_int8, &layer.wk_int8_scales, dim),
            (&layer.wv_int8, &layer.wv_int8_scales, dim),
            (
                &layer.wo_int8,
                &layer.wo_int8_scales,
                layer.wq_int8_scales.len(),
            ),
            (&layer.gate_up_int8, &layer.gate_up_int8_scales, dim),
            (
                &layer.down_int8,
                &layer.down_int8_scales,
                layer.gate_up_int8_scales.len() / 2,
            ),
        ] {
            assert_eq!(weights.len(), scales.len() * input);
            assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
        }
    }
    // The export drops tied embeddings; ensure the row-scaled LM head is used.
    assert!(dec.tok_embeddings_bf16.is_null());
    assert!(dec.tok_embeddings_int8.is_none());
    let head = dec.lm_head_int8.as_ref().unwrap();
    let scales = dec.lm_head_int8_scales.as_ref().unwrap();
    assert_eq!(head.len(), cfg.vocab_size * dim);
    assert_eq!(scales.len(), cfg.vocab_size);
    for token in [0, 1000, cfg.vocab_size - 1] {
        let mut actual = vec![0.0; dim];
        unsafe {
            decoder::tok_embed_to_f32(dec, &mut actual, token as i32, dim);
        }
        let expected: Vec<_> = head[token * dim..(token + 1) * dim]
            .iter()
            .map(|w| *w as f32 * scales[token])
            .collect();
        assert_eq!(actual, expected);
    }
    let x: Vec<f32> = (0..dim)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) / 128.0)
        .collect();
    let (qx, sx) = kernels::quantize_f32_to_int8(&x);
    for layer in [&dec.layers[0], &dec.layers[27]] {
        let qdim = layer.wq_int8_scales.len();
        let kdim = layer.wk_int8_scales.len();
        let mut q = vec![0.0; qdim];
        let mut k = vec![0.0; kdim];
        let mut v = vec![0.0; kdim];
        kernels::linear_nobias_int8_qkv(
            &mut q,
            &mut k,
            &mut v,
            &x,
            &layer.wq_int8,
            &layer.wq_int8_scales,
            &layer.wk_int8,
            &layer.wk_int8_scales,
            &layer.wv_int8,
            &layer.wv_int8_scales,
            dim,
            qdim,
            kdim,
        );
        close(
            &q,
            &scalar_rows(&qx, sx, &layer.wq_int8, &layer.wq_int8_scales),
        );
        close(
            &k,
            &scalar_rows(&qx, sx, &layer.wk_int8, &layer.wk_int8_scales),
        );
        close(
            &v,
            &scalar_rows(&qx, sx, &layer.wv_int8, &layer.wv_int8_scales),
        );
        let intermediate = layer.gate_up_int8_scales.len() / 2;
        let mut ffn = vec![0.0; intermediate];
        let mut gate_buf = vec![0.0; 2 * intermediate];
        kernels::linear_nobias_int8_swiglu(
            &mut ffn,
            &mut gate_buf,
            &x,
            &layer.gate_up_int8,
            &layer.gate_up_int8_scales,
            dim,
            intermediate,
        );
        let gu = scalar_rows(&qx, sx, &layer.gate_up_int8, &layer.gate_up_int8_scales);
        let expected: Vec<_> = gu
            .chunks_exact(2)
            .map(|r| r[0] / (1.0 + (-r[0]).exp()) * r[1])
            .collect();
        close(&ffn, &expected);
    }
    let rows = 256;
    let logits = scalar_rows(&qx, sx, &head[..rows * dim], &scales[..rows]);
    let expected = logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0;
    assert_eq!(
        kernels::argmax_matvec_int8(&x, &head[..rows * dim], &scales[..rows], dim, rows),
        expected
    );
}
