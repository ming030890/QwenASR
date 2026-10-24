//! Decompose a fresh CNN context projection, full output projection and softmax.
use cantonese_engine::cnn::CnnLanguageModel;
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let owner = CnnLanguageModel::from_bytes(&bytes)?;
    for left in ["", "今日返工之後", "I went to the shops after work and then came home to see "] {
        let mut rows = vec![];
        for _ in 0..8 {
            let mut model = CnnLanguageModel::from_bytes(&bytes)?;
            let ids = model.context(left);
            let _vocabulary = owner.vocabulary_size();
            let t = std::time::Instant::now();
            model.scores(&ids, &[8])?;
            let context_us = t.elapsed().as_micros();
            let t = std::time::Instant::now();
            let values = model.log_probabilities(&ids)?;
            let output_and_softmax_us = t.elapsed().as_micros();
            let t = std::time::Instant::now();
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z = max + values.iter().map(|v| (v-max).exp()).sum::<f32>().ln();
            std::hint::black_box(z);
            let softmax_us = t.elapsed().as_micros();
            rows.push(serde_json::json!({"context_us":context_us,"output_and_softmax_us":output_and_softmax_us,"softmax_us":softmax_us}));
        }
        println!("{}", serde_json::json!({"left":left,"rows":rows}));
    }
    Ok(())
}
