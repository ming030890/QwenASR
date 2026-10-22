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
    let no_bos = args.iter().any(|a| a=="--no-bos");
    let touch_model = args.iter().position(|a| a=="--touch-model").map(|i| {
        let path=args.get(i+1).ok_or("Missing --touch-model path")?;
        cantonese_engine::touch::TouchModel::from_json(&std::fs::read_to_string(path).map_err(|e|e.to_string())?)
    }).transpose().map_err(io::Error::other)?;
    if no_bos {
        if let NeuralLanguageModel::Cnn(model)=&mut lm {model.set_include_bos(false);}
        if let Some(model)=quick.as_mut().and_then(|q|q.lm.neural.as_mut()) {
            if let NeuralLanguageModel::Cnn(model)=model {model.set_include_bos(false);}
        }
    }
    for line in io::stdin().lock().lines() {
        let x: serde_json::Value = serde_json::from_str(&line?)?;
        let started = std::time::Instant::now();
        let left = x["left"].as_str().unwrap_or("");
        let right = x["right"].as_str().unwrap_or("");
        let ids = if let Some(raw) = x["context_ids"].as_array() {
            raw.iter().map(|v| v.as_u64().unwrap() as usize).collect()
        } else { lm.context(left, right) };
        let mut result = if let Some(typed) = x["typed"].as_str() {
            let limit = x["spelling_limit"].as_u64().unwrap_or(8).min(4096) as usize;
            let floats=|key:&str|->Vec<f32> { x[key].as_array().map(|a|a.iter().map(|v|v.as_f64().map(|n|n as f32).unwrap_or(f32::NAN)).collect()).unwrap_or_default() };
            let geometry=floats("touch_geometry"); let points=floats("touch_points");
            let canonical=typed.to_ascii_lowercase();
            let touch=touch_model.as_ref().and_then(|m|m.prepare(&canonical,&geometry,&points));
            serde_json::to_value(lm.english_suggestions_with_touch(left,typed,limit,touch.as_ref()).map_err(io::Error::other)?)?
        } else if let Some(code) = x["code"].as_str() {
            let engine = quick.as_mut().ok_or("Quick assets required")?;
            let mode = format!("{:?}", engine.lm.mode(left, right));
            let version = x["version"].as_u64().unwrap_or(3) as u32;
            let c = if x["preview"].as_bool().unwrap_or(false) {
                engine.preview_candidates(version, code, left, right)
            } else {
                engine.candidates(version, code, left, right)
            }.map_err(io::Error::other)?;
            serde_json::json!({"context_ids":ids,"mode":mode,"candidates":c})
        } else if let Some(candidates)=x["mixed"].as_array() {
            let candidates:Vec<String>=candidates.iter().map(|s|s.as_str().unwrap().to_owned()).collect();
            let observed=x["observed"].as_str().unwrap_or("");
            let evidence=cantonese_engine::neural::score_sequences(&mut lm,left,right,&candidates).map_err(io::Error::other)?;
            let features:Vec<_>=evidence.iter().zip(&candidates).map(|(e,s)|cantonese_engine::quick::mixed_features(e,s,observed)).collect();
            let scores:Vec<_>=features.iter().map(|f|cantonese_engine::quick::mixed_score(*f)).collect();
            let mut order:Vec<_>=(0..scores.len()).collect();
            order.sort_by(|&a,&b|scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            serde_json::json!({"evidence":evidence,"features":features,"scores":scores,"order":order})
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
                let scores = lm.log_probabilities(&ids).map_err(io::Error::other)?;
                let probabilities: Vec<_> = targets.iter().map(|&id| if id<lm.vocabulary_size() { scores[id] } else { -1e4 }).collect();
                let mut order: Vec<_> = (0..scores.len()).collect();
                order.sort_unstable_by(|&a,&b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                let top3: Vec<_> = order.iter().take(3).copied().collect();
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
