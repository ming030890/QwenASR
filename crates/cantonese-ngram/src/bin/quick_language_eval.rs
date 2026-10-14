//! Probe LM language evidence using the exact keyboard assets.
use cantonese_engine::{hybrid::HybridLanguageModel, quick::{QuickDictionary, QuickEngine}, NeuralLanguageModel};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let assets = std::path::PathBuf::from(args.next().ok_or("Usage: quick_language_eval assets-dir left-context codes...")?);
    let left = args.next().unwrap_or_default();
    let read = |file: &str| std::fs::read(assets.join(file));
    let labels: Vec<_> = String::from_utf8(read("handwriting/labels.txt")?)?.lines().map(str::to_owned).collect();
    let neural = NeuralLanguageModel::from_bytes(&read("language_model/model.cng")?)?;
    let mut engine = QuickEngine::new(QuickDictionary::from_bytes(&read("quick/quick3.bin")?, 3)?,
        QuickDictionary::from_bytes(&read("quick/quick5.bin")?, 5)?, HybridLanguageModel::with_neural(Some(neural), None, &labels)?);
    for code in args {
        let start = std::time::Instant::now();
        let candidates = engine.candidates(3, &code[..code.len().min(12)], &left, "")?;
        let top = candidates.first().cloned().unwrap_or_default();
        let evidence = engine.language_evidence(&code, &left, "", &top)?;
        println!("{}", serde_json::json!({"code":code,"top":top,"evidence":evidence,
            "commit":evidence.as_ref().map(|e| e.should_commit_chinese()).unwrap_or(false),"elapsedMs":start.elapsed().as_millis()}));
    }
    Ok(())
}
