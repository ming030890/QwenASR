use cantonese_engine::{
    counts::CsrLanguageModel,
    hybrid::HybridLanguageModel,
    quick::{QuickDictionary, QuickEngine},
    NeuralLanguageModel,
};
use std::io::{self, BufRead};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mut lm = NeuralLanguageModel::from_bytes(&std::fs::read(&args[1])?).map_err(io::Error::other)?;
    let mut quick = if args.len() >= 6 {
        let labels: Vec<_> = std::fs::read_to_string(&args[3])?
            .lines()
            .map(|s| s.trim_end().to_owned())
            .collect();
        Some(QuickEngine::new(
            QuickDictionary::from_bytes(&std::fs::read(&args[4])?, 3).map_err(io::Error::other)?,
            QuickDictionary::from_bytes(&std::fs::read(&args[5])?, 5).map_err(io::Error::other)?,
            HybridLanguageModel::with_neural(
                Some(
                    NeuralLanguageModel::from_bytes(&std::fs::read(&args[1])?)
                        .map_err(io::Error::other)?,
                ),
                if args[2] == "-" { None } else {
                    Some(CsrLanguageModel::parse(&std::fs::read(&args[2])?).map_err(io::Error::other)?)
                },
                &labels,
            )
            .map_err(io::Error::other)?,
        ))
    } else {
        None
    };
    for line in io::stdin().lock().lines() {
        let x: serde_json::Value = serde_json::from_str(&line?)?;
        let started = std::time::Instant::now();
        let left = x["left"].as_str().unwrap_or("");
        let right = x["right"].as_str().unwrap_or("");
        let ids = if let Some(raw) = x["context_ids"].as_array() {
            raw.iter().map(|v| v.as_u64().unwrap() as usize).collect()
        } else { lm.context(left, right) };
        let mut result = if let Some(code) = x["code"].as_str() {
            let engine = quick.as_mut().ok_or("Quick assets required")?;
            let mode = format!("{:?}", engine.lm.mode(left, right));
            let c = engine
                .candidates(x["version"].as_u64().unwrap_or(3) as u32, code, left, right)
                .map_err(io::Error::other)?;
            serde_json::json!({"context_ids":ids,"mode":mode,"candidates":c})
        } else if let Some(candidates)=x["sequences"].as_array() {
            let candidates:Vec<String>=candidates.iter().map(|s|s.as_str().unwrap().to_owned()).collect();
            let evidence=cantonese_engine::neural::score_sequences(&mut lm,left,right,&candidates).map_err(io::Error::other)?;
            serde_json::json!({"context_ids":ids,"evidence":evidence})
        } else if let Some(limit) = x["limit"].as_u64() {
            let (mode, predictions) = if let Some(engine) = quick.as_mut() {
                let (mode, predictions) = engine
                    .lm
                    .predict(left, right, limit as usize)
                    .map_err(io::Error::other)?;
                (format!("{mode:?}"), predictions)
            } else {
                (
                    "Neural".into(),
                    lm.predict(left, right, limit as usize)
                        .map_err(io::Error::other)?,
                )
            };
            serde_json::json!({"context_ids":ids,"mode":mode,"predictions":predictions})
        } else {
            let targets: Vec<_> = x["candidate_ids"]
                .as_array()
                .ok_or("candidate_ids required")?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            if x["distribution"].as_bool().unwrap_or(false) {
                let all: Vec<_> = (3..lm.vocabulary_size()).collect();
                let scores = lm.scores(&ids, &all).map_err(io::Error::other)?;
                let max = scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
                let logsum = max + scores.iter().map(|v| (v-max).exp()).sum::<f32>().ln();
                let probabilities: Vec<_> = targets.iter().map(|&id| if id>=3 && id<lm.vocabulary_size() { scores[id-3]-logsum } else { -1e4 }).collect();
                let mut order: Vec<_> = (0..scores.len()).collect();
                order.sort_unstable_by(|&a,&b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                let top3: Vec<_> = order.iter().take(3).map(|i| i+3).collect();
                serde_json::json!({"context_ids":ids,"log_probs":probabilities,"top3":top3})
            } else {
                let scores = lm.scores(&ids, &targets).map_err(io::Error::other)?;
                serde_json::json!({"context_ids":ids,"scores":scores})
            }
        };
        if x["timing"].as_bool().unwrap_or(false) {
            result["elapsedMicros"] = serde_json::json!(started.elapsed().as_micros() as u64);
        }
        println!("{result}");
    }
    Ok(())
}
