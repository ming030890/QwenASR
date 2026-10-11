use cantonese_ngram::{
    counts::CsrLanguageModel,
    hybrid::HybridLanguageModel,
    quick::{QuickDictionary, QuickEngine},
    LanguageModel,
};
use std::io::{self, BufRead};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mut lm = LanguageModel::from_bytes(&std::fs::read(&args[1])?).map_err(io::Error::other)?;
    let mut quick = if args.len() >= 6 {
        let labels: Vec<_> = std::fs::read_to_string(&args[3])?
            .lines()
            .map(|s| s.trim_end().to_owned())
            .collect();
        Some(QuickEngine::new(
            QuickDictionary::from_bytes(&std::fs::read(&args[4])?, 3).map_err(io::Error::other)?,
            QuickDictionary::from_bytes(&std::fs::read(&args[5])?, 5).map_err(io::Error::other)?,
            HybridLanguageModel::new(
                Some(
                    LanguageModel::from_bytes(&std::fs::read(&args[1])?)
                        .map_err(io::Error::other)?,
                ),
                Some(CsrLanguageModel::parse(&std::fs::read(&args[2])?).map_err(io::Error::other)?),
                &labels,
            )
            .map_err(io::Error::other)?,
        ))
    } else {
        None
    };
    for line in io::stdin().lock().lines() {
        let x: serde_json::Value = serde_json::from_str(&line?)?;
        let left = x["left"].as_str().unwrap_or("");
        let right = x["right"].as_str().unwrap_or("");
        let ids = lm.context(left, right);
        let result = if let Some(code) = x["code"].as_str() {
            let engine = quick.as_mut().ok_or("Quick assets required")?;
            let mode = format!("{:?}", engine.lm.mode(left, right));
            let c = engine
                .candidates(x["version"].as_u64().unwrap_or(3) as u32, code, left, right)
                .map_err(io::Error::other)?;
            serde_json::json!({"context_ids":ids,"mode":mode,"candidates":c})
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
            let scores = lm.scores(&ids, &targets).map_err(io::Error::other)?;
            serde_json::json!({"context_ids":ids,"scores":scores})
        };
        println!("{result}");
    }
    Ok(())
}
