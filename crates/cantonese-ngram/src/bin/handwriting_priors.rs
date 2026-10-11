//! JSONL context input -> JSON header + binary class-aligned unweighted priors.
use cantonese_ngram::{
    counts::CsrLanguageModel,
    hybrid::{FusionWeights, HybridLanguageModel},
    LanguageModel,
};
use std::io::{self, BufRead, Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let labels: Vec<_> = std::fs::read_to_string(&args[3])?
        .lines()
        .map(|s| s.trim_end().to_owned())
        .collect();
    let mut lm = HybridLanguageModel::new(
        Some(LanguageModel::from_bytes(&std::fs::read(&args[1])?).map_err(io::Error::other)?),
        Some(CsrLanguageModel::parse(&std::fs::read(&args[2])?).map_err(io::Error::other)?),
        &labels,
    )
    .map_err(io::Error::other)?;
    lm.set_fusion_weights(FusionWeights {
        unigram: 0.,
        bigram: 0.,
        neural: 1.,
    })
    .map_err(io::Error::other)?;
    if let Some(path) = args.get(4) {
        #[derive(serde::Deserialize)]
        struct Calibration {
            weights: FusionWeights,
        }
        let c: Calibration = serde_json::from_slice(&std::fs::read(path)?)?;
        lm.set_fusion_weights(c.weights).map_err(io::Error::other)?;
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let x: serde_json::Value = serde_json::from_str(&line?)?;
        let left = x["left"].as_str().unwrap_or("");
        let right = x["right"].as_str().unwrap_or("");
        let prev = lm.previous_class(left);
        let (mode, priors) = lm.scores(left, right, &labels).map_err(io::Error::other)?;
        writeln!(
            out,
            "{}",
            serde_json::json!({"mode":format!("{mode:?}"),"previous":prev,"classes":labels.len()})
        )?;
        for p in priors {
            out.write_all(&p.to_le_bytes())?;
        }
        out.flush()?;
    }
    Ok(())
}
