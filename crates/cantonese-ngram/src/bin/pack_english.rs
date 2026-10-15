use cantonese_engine::english::pack;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 && !(args.len() == 4 && args[3] == "--expand") {
        return Err("Usage: pack_english source.txt output.bin [--expand]".into());
    }
    let source = std::fs::read_to_string(&args[1])?;
    let bytes = if args.len() == 4 { cantonese_engine::english::decoder::expanded_dictionary(&source)? } else { pack(&source)? };
    std::fs::write(&args[2], &bytes)?;
    println!("Packed English dictionary: {} bytes", bytes.len());
    Ok(())
}
