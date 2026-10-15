//! Freeze actual native generator outputs, independently of the model under test.
use cantonese_engine::{english::{decoder::Decoder,Dictionary,TouchModel,Point},phrases::PhraseLexicon,quick::QuickDictionary};
use std::{io::{self,BufRead},time::Instant};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let args:Vec<_>=std::env::args().collect();
 if args.len()!=5{return Err("Usage: typing_candidates english.bin phrases.json quick3.bin quick5.bin".into());}
 let en=Dictionary::from_bytes(&std::fs::read(&args[1])?).map_err(io::Error::other)?;
 let phrases=PhraseLexicon::from_bytes(&std::fs::read(&args[2])?).map_err(io::Error::other)?;
 let q3=QuickDictionary::from_bytes(&std::fs::read(&args[3])?,3).map_err(io::Error::other)?;
 let q5=QuickDictionary::from_bytes(&std::fs::read(&args[4])?,5).map_err(io::Error::other)?;
 let mut decoder=Decoder::default();
 for (revision,line) in io::stdin().lock().lines().enumerate() {
  let x:serde_json::Value=serde_json::from_str(&line?)?;let start=Instant::now();
  let typed=x["typed"].as_str().unwrap_or("");let left=x["left"].as_str().unwrap_or("");
  let kind=x["kind"].as_str().unwrap_or("");let mut pool=vec![];let mut exhausted=false;
  if kind=="phrase" {
   for text in phrases.candidates(left,128) {pool.push(serde_json::json!({"text":text,"spatial":0.,"lexical":0.,"literal":false}));}
  } else {
   let mut touch=TouchModel::default();
   if let Some(samples)=x["touch"].as_array() {
    for (r,row) in ["qwertyuiop","asdfghjkl","zxcvbnm"].iter().enumerate() {
     for (c,b) in row.bytes().enumerate(){touch.centers[(b-b'a')as usize]=Some(Point{x:c as f64+[0.,0.5,1.5][r],y:r as f64});}
    }
    for p in samples {touch.points.push(match(p[0].as_f64(),p[1].as_f64()){(Some(x),Some(y))=>Some(Point{x,y}),_=>None});}
   }
   let result=decoder.decode(&en,typed,&touch,"",revision as i64,revision as i64);
   exhausted=result.budget_exhausted;
   let mut english:Vec<_>=result.hypotheses.into_iter().map(|h|serde_json::json!({"text":h.text,"spatial":h.spatial_score,"lexical":h.lexical_score,"literal":h.literal,"completion":h.completion,"edits":h.edits})).collect();
   // Literal must remain in the scored pool, including clean OOV identifiers.
   if let Some(i)=english.iter().position(|h|h["literal"]==true){let h=english.remove(i);english.insert(0,h);}
   if kind=="quick" {
    let q=if x["quick_version"].as_u64()==Some(5){&q5}else{&q3};
    let chinese=q.candidates(typed);
    for i in 0..128 {
     if let Some(h)=english.get(i){pool.push(h.clone());}
     if let Some(text)=chinese.get(i){pool.push(serde_json::json!({"text":text,"spatial":0.,"lexical":0.,"literal":false}));}
     if pool.len()>=128{break;}
    }
   } else {pool=english;}
  }
  pool.truncate(128);
  println!("{}",serde_json::json!({"candidates":pool,"budget_exhausted":exhausted,"generator_ms":start.elapsed().as_secs_f64()*1000.}));
 }
 Ok(())
}
