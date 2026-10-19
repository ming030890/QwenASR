//! Actual-bundle smoke checks for sharing and incremental invalidation.
use cantonese_engine::cnn::CnnLanguageModel;
fn main()->Result<(),Box<dyn std::error::Error>> {
    let raw=std::fs::read(std::env::args().nth(1).ok_or("model path required")?)?;
    let mut cached=CnnLanguageModel::from_bytes(&raw)?;
    let other=CnnLanguageModel::from_bytes(&raw)?;
    assert!(cached.shares_weights(&other));
    let targets:Vec<_>=(8..264).collect();
    let mut max=0f32;
    for text in ["", "可", "可以", "可以 Hello", "可以 Hello world", "我可以", "I will check tomorrow", "I will check today", &"我可以".repeat(30)] {
        let mut fresh=CnnLanguageModel::from_bytes(&raw)?;
        let ctx=cached.context(text);
        let actual=cached.scores(&ctx,&targets)?;
        let expected=fresh.scores(&ctx,&targets)?;
        for (a,b) in actual.iter().zip(expected) {max=max.max((a-b).abs());}
    }
    assert!(max<1e-5,"cached context differs: {max}");
    let prefixes=cached.completions("I said ","hel",8)?;
    assert!(!prefixes.is_empty());
    assert!(prefixes.iter().all(|p|p.token.to_lowercase().starts_with("hel")));
    let next=cached.completions("I said ","",8)?;
    assert!(!next.is_empty());
    assert!(next.iter().all(|p|p.token.is_ascii()));
    println!("{}",serde_json::json!({"passed":true,"weights_shared":true,"cache_edit_max_difference":max,
        "prefix_candidates":prefixes,"english_next_words":next}));
    Ok(())
}
