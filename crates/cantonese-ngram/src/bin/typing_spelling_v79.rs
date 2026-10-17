// Bind the current complete spelling search independently of earlier replays.
use cantonese_engine::english::{spelling,Dictionary,Point,TouchModel};
use std::io::{self,BufRead};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let d=Dictionary::from_bytes(&std::fs::read(std::env::args().nth(1).ok_or("dictionary")?)?).map_err(io::Error::other)?;
    for (revision,line) in io::stdin().lock().lines().enumerate() {
        let r:serde_json::Value=serde_json::from_str(&line?)?;let mut touch=TouchModel::default();
        if let Some(centers)=r["centers"].as_array() {if centers.len()!=26 {return Err("26 centers required".into());}for(i,p)in centers.iter().enumerate(){touch.centers[i]=Some(Point{x:p[0].as_f64().ok_or("x")?,y:p[1].as_f64().ok_or("y")?});}}
        if let Some(points)=r["touch"].as_array(){touch.points=points.iter().map(|p|match(p[0].as_f64(),p[1].as_f64()){(Some(x),Some(y))=>Some(Point{x,y}),_=>None}).collect();}
        let left=r["left"].as_str().unwrap_or("");let previous=left.strip_suffix(' ').unwrap_or("").chars().rev().take_while(|c|c.is_ascii_lowercase()).collect::<String>().chars().rev().collect::<String>();
        println!("{}",serde_json::to_string(&spelling::decode_robust(&d,r["typed"].as_str().unwrap_or(""),&touch,&previous,1,revision as i64))?);
    }Ok(())
}
