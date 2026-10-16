//! Exact Android decoder request semantics for internal-trial profile fitting.
use cantonese_engine::english::{decoder::Decoder, Dictionary, Point, TouchModel};
use std::io::{self, BufRead};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dictionary = Dictionary::from_bytes(&std::fs::read(std::env::args().nth(1).ok_or("dictionary")?)?)
        .map_err(io::Error::other)?;
    let mut decoder = Decoder::default();
    for (revision, line) in io::stdin().lock().lines().enumerate() {
        let request: serde_json::Value = serde_json::from_str(&line?)?;
        let mut touch = TouchModel::default();
        if let Some(centers) = request["centers"].as_array() {
            if centers.len() != 26 { return Err("26 centers required".into()); }
            for (i, p) in centers.iter().enumerate() {
                touch.centers[i] = Some(Point { x: p[0].as_f64().ok_or("x")?, y: p[1].as_f64().ok_or("y")? });
            }
        }
        if let Some(points) = request["touch"].as_array() {
            for p in points { touch.points.push(match (p[0].as_f64(),p[1].as_f64()) {
                (Some(x),Some(y)) => Some(Point{x,y}), _ => None,
            }); }
        }
        let left=request["left"].as_str().unwrap_or("");
        let previous=left.strip_suffix(' ').unwrap_or("").chars().rev().take_while(|c|c.is_ascii_lowercase())
            .collect::<String>().chars().rev().collect::<String>();
        let result=decoder.decode(&dictionary,request["typed"].as_str().unwrap_or(""),&touch,&previous,1,revision as i64);
        println!("{}",serde_json::to_string(&result)?);
    }
    Ok(())
}
