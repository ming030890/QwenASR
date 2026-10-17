use cantonese_engine::{
    counts::CsrLanguageModel,
    hybrid::{HybridLanguageModel, Mode},
    quick::{QuickDictionary, QuickEngine},
    NeuralLanguageModel,
};
use serde_json::{json, Value};

fn matrix(
    desc: &mut Vec<Value>,
    data: &mut Vec<u8>,
    name: &str,
    rows: usize,
    cols: usize,
    value: Option<(usize, usize, i8)>,
) {
    desc.push(json!({"name":name,"shape":[rows,cols],"kind":"row_i8"}));
    for _ in 0..rows {
        data.extend_from_slice(&1f32.to_le_bytes());
    }
    let mut q = vec![0u8; rows * cols];
    if let Some((r, c, v)) = value {
        q[r * cols + c] = v as u8;
    }
    data.extend(q);
}
fn vector(desc: &mut Vec<Value>, data: &mut Vec<u8>, name: &str, values: &[f32]) {
    desc.push(json!({"name":name,"shape":[values.len()],"kind":"f32"}));
    for v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
}
fn packed(magic: &[u8], meta: Value, data: Vec<u8>) -> Vec<u8> {
    let json = serde_json::to_vec(&meta).unwrap();
    let mut bytes = magic.to_vec();
    bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
    bytes.extend(json);
    bytes.extend(data);
    bytes
}
fn dense() -> Vec<u8> {
    let (mut d, mut b) = (vec![], vec![]);
    matrix(&mut d, &mut b, "embedding.weight", 6, 128, None);
    matrix(&mut d, &mut b, "hidden.weight", 256, 1536, None);
    matrix(&mut d, &mut b, "project.weight", 128, 256, None);
    vector(&mut d, &mut b, "bias", &[0., 0., 0., 0., 3., 0.]);
    vector(&mut d, &mut b, "hidden.bias", &[0.; 256]);
    vector(&mut d, &mut b, "project.bias", &[0.; 128]);
    vector(&mut d, &mut b, "norm.weight", &[1.; 256]);
    vector(&mut d, &mut b, "norm.bias", &[0.; 256]);
    packed(
        b"CNG1",
        json!({"version":1,"tokens":["[PAD]","[BOS]","[UNK]","我","你","係"],"lookup_normalization":"english_lower_width_quotes_v1","remove_bos":true,"config":{"vocab_size":6,"left":8,"right":4,"embedding":128,"hidden":256,"residual_blocks":0,"ffn_hidden":0,"activation":"relu","layer_norm":true},"tensors":d}),
        b,
    )
}
fn memory_variant(width: usize, expanded: bool) -> Vec<u8> {
    let base = dense();
    let (mut d, mut b) = (vec![], vec![]);
    matrix(&mut d, &mut b, "features.weight", 2, 128, Some((1, 0, 1)));
    matrix(&mut d, &mut b, "hidden.weight", width, 128, None);
    vector(&mut d, &mut b, "hidden.bias", &vec![0.; width]);
    matrix(&mut d, &mut b, "project.weight", 128, width, None);
    vector(&mut d, &mut b, "project.bias", &[0.; 128]);
    let vocabulary = if expanded { 7 } else { 6 };
    matrix(
        &mut d,
        &mut b,
        "output.weight",
        vocabulary,
        128,
        Some((5, 0, 5)),
    );
    let mut bias = vec![0.; vocabulary];
    if expanded {
        bias[6] = 8.;
    }
    vector(&mut d, &mut b, "output.bias", &bias);
    let mut tokens = vec!["[PAD]", "[BOS]", "[UNK]", "我", "你", "係"];
    if expanded {
        tokens.push("新");
    }
    let mut data = base.clone();
    data.extend(b);
    packed(
        b"CNG2",
        json!({"version":2,"architecture":"learned_ngram_feature_memory_v1","tokens":tokens,"lookup_normalization":"english_lower_width_quotes_v1","keys":[[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,3,0,0,0,0]],"parent_share":0.05,"base_bytes":base.len(),"tensors":d}),
        data,
    )
}
fn memory(width: usize) -> Vec<u8> {
    memory_variant(width, false)
}
fn hybrid(bytes: &[u8]) -> HybridLanguageModel {
    HybridLanguageModel::with_neural(
        Some(NeuralLanguageModel::from_bytes(bytes).unwrap()),
        Some(CsrLanguageModel {
            num_classes: 3,
            num_edges: 1,
            lambda_unigram: 1.,
            lambda_bigram: 1.,
            unigrams: vec![-1., -2., -3.],
            offsets: vec![0, 1, 1, 1],
            targets: vec![1],
            log_probs: vec![4.],
        }),
        &["我", "你", "係"].map(String::from),
    )
    .unwrap()
}
#[test]
fn cng2_uses_learned_bigram_and_unseen_neural_base_at_both_widths() {
    for width in [128, 256] {
        let mut lm = hybrid(&memory(width));
        assert_eq!(lm.mode("我", ""), Mode::Neural);
        assert_eq!(lm.predict_next("我", 1).unwrap().1[0].token, "係");
        assert_eq!(lm.predict_next("你", 1).unwrap().1[0].token, "你");
        assert_eq!(lm.mode("", "你"), Mode::Neural);
        assert_eq!(lm.mode("我。", ""), Mode::Unigram);
        assert_eq!(lm.predict_next("", 1).unwrap().1[0].token, "我");
        let candidates = ["你", "𰻞", "係"].map(String::from);
        assert_eq!(lm.rank("我", "", &candidates).unwrap(), vec![2, 1, 0]);
        assert_eq!(lm.rank("", "", &candidates).unwrap(), vec![0, 1, 2]);
        let (_, scores) = lm
            .scores("我", "", &["你", "係"].map(String::from))
            .unwrap();
        assert!(scores[1] > scores[0]);
    }
}
#[test]
fn cng1_retains_previous_bigram_policy() {
    let mut lm = hybrid(&dense());
    assert_eq!(lm.mode("我", ""), Mode::Bigram);
    assert_eq!(lm.predict_next("我", 1).unwrap().1[0].token, "你");
    assert_eq!(lm.mode("我你", ""), Mode::Neural);
}
#[test]
fn dense_short_context_model_uses_one_network_without_counts() {
    let bytes = dense();
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut metadata: Value = serde_json::from_slice(&bytes[8..8+n]).unwrap();
    metadata["neural_short_contexts"] = json!(true);
    let bytes = packed(b"CNG1", metadata, bytes[8+n..].to_vec());
    let mut lm = hybrid(&bytes);
    assert!(lm.counts.is_none());
    for left in ["", "我", "我你", "𰻞"] {
        assert_eq!(lm.mode(left, ""), Mode::Neural);
        assert_eq!(lm.predict_next(left, 1).unwrap().1[0].token, "你");
    }
    assert_eq!(lm.rank("", "", &["係", "你"].map(String::from)).unwrap(), vec![1,0]);
    let (_, priors)=lm.scores("", "", &["我", "你", "係"].map(String::from)).unwrap();
    assert!(priors.iter().all(|s| s.is_finite()));
    assert!(priors[1]>priors[0]);
}
#[test]
fn quick_dictionary_uses_cng2_and_preserves_unknown_candidate_slots() {
    let pool = "你𰻞係".as_bytes();
    let mut bytes = vec![];
    for i in 0..703 {
        bytes.extend_from_slice(&(if i == 0 { 0 } else { pool.len() as u32 }).to_be_bytes());
    }
    bytes.extend(pool);
    let mut engine = QuickEngine::new(
        QuickDictionary::from_bytes(&bytes, 3).unwrap(),
        QuickDictionary::from_bytes(&bytes, 5).unwrap(),
        hybrid(&memory(128)),
    );
    assert_eq!(
        engine.candidates(3, "A", "我", "").unwrap(),
        ["係", "𰻞", "你"]
    );
    assert_eq!(
        engine.candidates(5, "a", "", "").unwrap(),
        ["你", "𰻞", "係"]
    );
}

#[test]
fn new_vocabulary_candidates_keep_count_priors_when_context_has_no_memory_key() {
    let neural = NeuralLanguageModel::from_bytes(&memory_variant(128, true)).unwrap();
    let mut lm = HybridLanguageModel::with_neural(
        Some(neural),
        Some(CsrLanguageModel {
            num_classes: 4,
            num_edges: 1,
            lambda_unigram: 1.,
            lambda_bigram: 1.,
            unigrams: vec![-1., -2., -3., -4.],
            offsets: vec![0, 1, 1, 1, 1],
            targets: vec![1],
            log_probs: vec![4.],
        }),
        &["我", "你", "係", "新"].map(String::from),
    )
    .unwrap();
    assert_eq!(lm.predict_next("我", 1).unwrap().1[0].token, "新");
    assert_eq!(lm.predict_next("你", 1).unwrap().1[0].token, "你");
    assert_eq!(lm.mode("新", ""), Mode::Bigram);
    let (mode, scores) = lm
        .scores("你", "", &["你", "新"].map(String::from))
        .unwrap();
    assert_eq!(mode, Mode::Neural);
    assert_eq!(scores[1], -4.);
    assert_eq!(
        lm.rank("你", "", &["係", "新", "你"].map(String::from))
            .unwrap(),
        vec![2, 1, 0]
    );
}

#[test]
fn test_real_assets_tam_jai() {
    let q3_path = "/Users/tony/Documents/keyboard/cantonese-voice-keyboard/app/src/main/assets/quick/quick3.bin";
    let q5_path = "/Users/tony/Documents/keyboard/cantonese-voice-keyboard/app/src/main/assets/quick/quick5.bin";
    let cng_path = "/Users/tony/Documents/keyboard/cantonese-voice-keyboard/app/src/main/assets/typing/model.cng";
    let (Ok(q3_bytes), Ok(q5_bytes), Ok(cng_bytes)) = (
        std::fs::read(q3_path),
        std::fs::read(q5_path),
        std::fs::read(cng_path),
    ) else {
        return;
    };
    let neural = NeuralLanguageModel::from_bytes(&cng_bytes).unwrap();
    let lm = HybridLanguageModel::with_neural(Some(neural), None, &[]).unwrap();
    let mut engine = QuickEngine::new(
        QuickDictionary::from_bytes(&q3_bytes, 3).unwrap(),
        QuickDictionary::from_bytes(&q5_bytes, 5).unwrap(),
        lm,
    );
    let q_candidates = engine.candidates(3, "yjodfdve", "", "").unwrap();
    let tam_jai_pos = q_candidates.iter().position(|c| c == "譚仔米線");
    assert!(tam_jai_pos.is_some() && tam_jai_pos.unwrap() < 4);

    let mut pool = q_candidates.clone();
    pool.push("yjodfdve".to_string());
    let order = cantonese_engine::quick::rank_mixed(&mut engine.lm, "", "", &pool).unwrap();
    let reranked: Vec<_> = order.iter().map(|&i| &pool[i]).collect();
    let reranked_tam_pos = reranked.iter().position(|c| *c == "譚仔米線").unwrap();
    let reranked_en_pos = reranked.iter().position(|c| *c == "yjodfdve").unwrap();
    assert!(reranked_tam_pos < reranked_en_pos);
    assert_eq!(reranked.last().unwrap().as_str(), "yjodfdve");
}

