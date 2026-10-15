//! Streaming evaluation interface for v2 hypotheses and frozen build-49 comparison.
use cantonese_engine::english::{decoder::Decoder, Dictionary, Point, TouchModel, Workspace, suggestions};
use std::io::{self,BufRead};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=3 {return Err("Usage: typing_eval legacy.bin expanded.bin < requests.jsonl".into());}
    let legacy=Dictionary::from_bytes(&std::fs::read(&args[1])?)?;
    let expanded=Dictionary::from_bytes(&std::fs::read(&args[2])?)?;
    let mut decoder=Decoder::default();let mut workspace=Workspace::default();
    for (revision,line) in io::stdin().lock().lines().enumerate() {
        let request:serde_json::Value=serde_json::from_str(&line?)?;
        let typed=request["typed"].as_str().ok_or("Missing typed")?;
        let previous=request["previous"].as_str().unwrap_or("");
        let mut touch=TouchModel::default();
        if let Some(samples)=request["touch"].as_array() {
            for (row,letters) in ["qwertyuiop","asdfghjkl","zxcvbnm"].iter().enumerate() {
                for (col,c) in letters.bytes().enumerate() { touch.centers[(c-b'a') as usize]=Some(Point{x:col as f64+[0.,0.5,1.5][row],y:row as f64}); }
            }
            for sample in samples {
                touch.points.push(match (sample[0].as_f64(),sample[1].as_f64()) {
                    (Some(x),Some(y))=>Some(Point{x,y}),_=>None});
            }
        }
        let before=suggestions(&legacy,typed,&touch,&mut workspace);
        let after=decoder.decode(&expanded,typed,&touch,previous,1,revision as i64);
        println!("{}",serde_json::json!({"typed":typed,"legacy":before,"v2":after}));
    }
    Ok(())
}
