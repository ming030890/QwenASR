//! JSONL stdin strokes -> all raw class log probabilities, little-endian f32 stdout.
use anyhow::Result;
use std::io::{BufRead, Write};
use cantonese_engine::hwr::{MlkitBiLstmRecognizer, StrokePoint};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut recognizer = MlkitBiLstmRecognizer::from_bytes(
        &std::fs::read(&args[1])?, &std::fs::read_to_string(&args[2])?)?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let value: Vec<Vec<[f32; 3]>> = serde_json::from_str(&line?)?;
        let strokes: Vec<Vec<StrokePoint>> = value.iter().map(|s| s.iter()
            .map(|p| StrokePoint {x:p[0], y:p[1], t_ms:p[2]}).collect()).collect();
        recognizer.predict_strokes(&strokes, 0.0, 0.0, 10)?;
        anyhow::ensure!(!recognizer.raw_candidate_scores().is_empty(), "empty trajectory");
        for score in recognizer.raw_candidate_scores() { out.write_all(&score.to_le_bytes())?; }
        out.flush()?;
    }
    Ok(())
}
