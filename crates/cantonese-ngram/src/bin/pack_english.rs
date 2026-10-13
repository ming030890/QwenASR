use cantonese_engine::english::pack;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("Usage: pack_english source.txt output.bin".into());
    }
    let bytes = pack(&std::fs::read_to_string(&args[1])?)?;
    std::fs::write(&args[2], &bytes)?;
    println!("Packed English dictionary: {} bytes", bytes.len());
    Ok(())
}
