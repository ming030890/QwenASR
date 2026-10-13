//! Host smoke/latency probe for the exact engine shipped through JNI.
use cantonese_engine::english::{search, Dictionary, TouchModel, Workspace};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let file = args
        .next()
        .ok_or("Usage: english_eval dictionary.bin [words...]")?;
    let dictionary = Dictionary::from_bytes(&std::fs::read(file)?)?;
    let mut workspace = Workspace::default();
    let touch = TouchModel::default();
    for word in args {
        println!(
            "{}",
            serde_json::to_string(&search(&dictionary, &word, &touch, &mut workspace))?
        );
    }
    let inputs = [
        "teh",
        "helllo",
        "keyboadr",
        "wprld",
        "aple",
        "don't",
        "zzzzzzzzzz",
    ];
    let mut times = Vec::new();
    for i in 0..700 {
        let start = std::time::Instant::now();
        let _ = search(
            &dictionary,
            inputs[i % inputs.len()],
            &touch,
            &mut workspace,
        );
        times.push(start.elapsed().as_micros());
    }
    times.sort_unstable();
    eprintln!(
        "resident={} bytes, warm host p95={} us",
        dictionary.memory_bytes(),
        times[times.len() * 95 / 100]
    );
    Ok(())
}
