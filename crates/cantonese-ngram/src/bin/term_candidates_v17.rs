//! Native broad Cantonese term retrieval; no per-example additions or LM lookup.
use cantonese_engine::phrases::PhraseLexicon;
use std::io::{self,BufRead};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let path=std::env::args().nth(1).ok_or("Usage: term_candidates_v17 phrases.json")?;
 let lex=PhraseLexicon::from_bytes(&std::fs::read(path)?).map_err(io::Error::other)?;
 for line in io::stdin().lock().lines(){
  let x:serde_json::Value=serde_json::from_str(&line?)?;
  let candidates=lex.candidates(x["left"].as_str().unwrap_or(""),128);
  println!("{}",serde_json::json!({"candidates":candidates}));
 }
 Ok(())
}
