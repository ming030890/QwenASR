//! Real-model differential check for the one-conversion-per-utterance experiment.
use qwen_asr::{context::QwenCtx, kernels};

#[test]
#[ignore = "requires QWEN_AUDIT_MODEL pointing to the INT8 export"]
fn conv_reuse_matches_control_across_chunk_boundaries_and_reused_buffers() {
    let model = std::env::var("QWEN_AUDIT_MODEL").expect("QWEN_AUDIT_MODEL is required");
    kernels::set_threads(4);
    let mut ctx = QwenCtx::load(&model).expect("model must load");
    let chunk = ctx.config.enc_chunk_size;
    // Include a single-token tail, full chunks, and a shorter following utterance.
    for frames in [2 * chunk + 1, 1, 8, 9, chunk, chunk + 8, 3 * chunk] {
        let mel: Vec<f32> = (0..128 * frames)
            .map(|i| ((i * 17 % 251) as f32 / 251.0) * 4.0 - 3.0)
            .collect();
        std::env::set_var("QWEN_EXPERIMENT_CONV_REUSE", "0");
        let control = ctx
            .encoder
            .forward(&ctx.config, &mel, frames, Some(&mut ctx.enc_bufs))
            .unwrap();
        std::env::set_var("QWEN_EXPERIMENT_CONV_REUSE", "1");
        let candidate = ctx
            .encoder
            .forward(&ctx.config, &mel, frames, Some(&mut ctx.enc_bufs))
            .unwrap();
        assert_eq!(control.1, candidate.1);
        assert!(control.0.iter().all(|x| x.is_finite()));
        assert_eq!(
            control.0, candidate.0,
            "encoder outputs differ at {frames} mel frames"
        );
    }
    std::env::remove_var("QWEN_EXPERIMENT_CONV_REUSE");
}
