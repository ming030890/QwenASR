//! Host probe of joined candidate generation, ranking, and inference counters.
use cantonese_engine::{
    hybrid::HybridLanguageModel,
    quick::{QuickDictionary, QuickEngine},
    suggestion_request, suggestions, NeuralLanguageModel,
};
use std::{
    io::{self, BufRead},
    sync::{atomic::AtomicBool, Arc},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let assets = std::path::Path::new(&args[1]);
    let bytes = std::fs::read(assets.join("language_model/model.cng"))?;
    let mut quick = QuickEngine::new(
        QuickDictionary::from_bytes(&std::fs::read(assets.join("quick/quick3.bin"))?, 3)?,
        QuickDictionary::from_bytes(&std::fs::read(assets.join("quick/quick5.bin"))?, 5)?,
        HybridLanguageModel::with_neural(
            Some(NeuralLanguageModel::from_bytes(&bytes)?),
            None,
            &[],
        )?,
    );
    let mut english = HybridLanguageModel::with_neural(
        Some(NeuralLanguageModel::from_bytes(&bytes)?),
        None,
        &[],
    )?;
    for line in io::stdin().lock().lines() {
        let input: serde_json::Value = serde_json::from_str(&line?)?;
        let code = input["code"].as_str().unwrap_or("");
        let left = input["left"].as_str().unwrap_or("");
        let right = input["right"].as_str().unwrap_or("");
        let version = input["version"].as_u64().unwrap_or(3) as u32;
        let legacy = input["legacy"].as_bool().unwrap_or(false);
        let started = std::time::Instant::now();
        let mut output = if input["preview"].as_bool().unwrap_or(false) {
            suggestion_request::run(Arc::new(AtomicBool::new(false)), || {
                let mut result = serde_json::to_value(suggestions::preview(&mut quick, version, code, left, right, true)?)
                    .map_err(|e| e.to_string())?;
                result["scoring"] = serde_json::to_value(suggestion_request::stats()).map_err(|e| e.to_string())?;
                Ok(result)
            })?
        } else if legacy {
            let chinese = if version == 0 {
                vec![]
            } else {
                quick.candidates(
                    version,
                    &code.chars().take(12).collect::<String>(),
                    left,
                    right,
                )?
            };
            let prediction = english
                .neural
                .as_mut()
                .unwrap()
                .english_suggestions(left, code, 8)?;
            let mut words = vec![code.to_owned()];
            words.extend(prediction.candidates.iter().map(|r| {
                if code == code.to_uppercase() {
                    r.word.to_uppercase()
                } else {
                    r.word.clone()
                }
            }));
            let mut pool = if code.len() == 1 {
                chinese.iter().take(8).cloned().collect::<Vec<_>>()
            } else {
                chinese.clone()
            };
            pool.extend(words);
            let mut seen = std::collections::HashSet::new();
            pool.retain(|s| seen.insert(s.clone()));
            let display = if version == 0 || code.len() == 2 {
                pool
            } else {
                let order =
                    cantonese_engine::quick::rank_mixed(&mut english, left, right, &pool, code)?;
                let mut result: Vec<_> = order.into_iter().map(|i| pool[i].clone()).collect();
                for text in &chinese {
                    if seen.insert(text.clone()) {
                        result.push(text.clone());
                    }
                }
                result
            };
            serde_json::json!({"chinese":chinese,"display":display,"english":prediction})
        } else {
            suggestion_request::run(Arc::new(AtomicBool::new(false)), || {
                let result = suggestions::compose(&mut quick, version, code, left, right, true)?;
                let mut result = serde_json::to_value(result).map_err(|e| e.to_string())?;
                result["scoring"] =
                    serde_json::to_value(suggestion_request::stats()).map_err(|e| e.to_string())?;
                Ok(result)
            })?
        };
        output["elapsed_ms"] = serde_json::json!(started.elapsed().as_secs_f64() * 1000.);
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}
