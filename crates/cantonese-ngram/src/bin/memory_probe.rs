use cantonese_engine::memory::MemoryLanguageModel;
use std::io::{self, BufRead};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let path = args.get(1).ok_or("Model path required")?;
    let loaded = std::time::Instant::now();
    let mut model =
        MemoryLanguageModel::from_bytes(&std::fs::read(path)?).map_err(io::Error::other)?;
    let load_ms = loaded.elapsed().as_secs_f64() * 1000.;
    if args.get(2).map(String::as_str) == Some("--benchmark") {
        let iterations: usize = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(200);
        if !(20..=10000).contains(&iterations) {
            return Err("Benchmark iterations must be20..10000".into());
        }
        let cases = [
            ("", ""),
            ("我", ""),
            ("雲吞", ""),
            ("", "吞麵"),
            ("我係香港人", ""),
            ("今晚食咩", ""),
            ("我", "你好"),
            ("我 👽", ""),
        ];
        for (left, right) in cases {
            model.predict(left, right, 3).map_err(io::Error::other)?;
        }
        let mut times = Vec::with_capacity(iterations);
        for index in 0..iterations {
            let (left, right) = cases[index % cases.len()];
            let start = std::time::Instant::now();
            let result = model.predict(left, right, 3).map_err(io::Error::other)?;
            std::hint::black_box(result);
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        let mean = times.iter().sum::<f64>() / iterations as f64;
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"host_arch":std::env::consts::ARCH,"iterations":iterations,"cases":cases,"file_read_and_parse_ms":load_ms,"prediction_mean_ms":mean,"prediction_median_ms":times[iterations/2],"prediction_p95_ms":times[(iterations*95/100).min(iterations-1)],"scope":"Host CPU release build; alternating contexts; top3 prediction includes tokenization, full-vocabulary neural normalization and ranking. Not an Android-device benchmark."})
        );
        return Ok(());
    }
    for line in io::stdin().lock().lines() {
        let request: serde_json::Value = serde_json::from_str(&line?)?;
        let left = request["left"].as_str().unwrap_or("");
        let right = request["right"].as_str().unwrap_or("");
        let ids = if let Some(ids) = request["context_ids"].as_array() {
            ids.iter()
                .map(|id| id.as_u64().map(|v| v as usize).ok_or("Invalid context ID"))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            model.context(left, right)
        };
        let result = if let Some(targets) = request["candidate_ids"].as_array() {
            let targets = targets
                .iter()
                .map(|id| id.as_u64().map(|v| v as usize).ok_or("Invalid target ID"))
                .collect::<Result<Vec<_>, _>>()?;
            let scores = model.scores(&ids, &targets).map_err(io::Error::other)?;
            serde_json::json!({"context_ids":ids,"scores":scores})
        } else {
            let predictions = model
                .predict(left, right, request["limit"].as_u64().unwrap_or(3) as usize)
                .map_err(io::Error::other)?;
            serde_json::json!({"context_ids":ids,"predictions":predictions})
        };
        println!("{result}");
    }
    Ok(())
}
