use super::*;
use ::jni::{
    objects::{JByteArray, JClass, JFloatArray, JString, JObjectArray},
    sys::{jboolean, jlong, jstring},
    JNIEnv,
};
use std::sync::{Arc, Mutex, OnceLock};
static DICTIONARY: OnceLock<Arc<Dictionary>> = OnceLock::new();
static WORKSPACE: OnceLock<Mutex<Workspace>> = OnceLock::new();

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeLoad(
    mut env: JNIEnv,
    _class: JClass,
    bytes: JByteArray,
) -> jboolean {
    if DICTIONARY.get().is_some() {
        return 1;
    }
    let result = env
        .convert_byte_array(bytes)
        .map_err(|e| e.to_string())
        .and_then(|b| Dictionary::from_bytes(&b));
    match result {
        Ok(dictionary) => {
            let _ = DICTIONARY.set(Arc::new(dictionary));
            1
        }
        Err(error) => {
            let _ = env.throw_new("java/lang/IllegalArgumentException", error);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeSearch(
    mut env: JNIEnv,
    _class: JClass,
    word: JString,
    centers: JFloatArray,
    points: JFloatArray,
    context_handle: jlong,
    left: JString,
    right: JString,
) -> jstring {
    let result = (|| -> Result<String, String> {
        let dictionary = DICTIONARY.get().ok_or("English dictionary not loaded")?;
        let word: String = env.get_string(&word).map_err(|e| e.to_string())?.into();
        let read = |env: &JNIEnv, array: &JFloatArray, max: usize| -> Result<Vec<f32>, String> {
            let n = env.get_array_length(array).map_err(|e| e.to_string())? as usize;
            if n > max || n % 2 != 0 {
                return Err("Invalid touch evidence size".into());
            }
            let mut values = vec![0.0; n];
            env.get_float_array_region(array, 0, &mut values)
                .map_err(|e| e.to_string())?;
            Ok(values)
        };
        let c = read(&env, &centers, 52)?;
        let p = read(&env, &points, 128)?;
        let mut touch = TouchModel::default();
        for (i, xy) in c.chunks_exact(2).enumerate() {
            if xy.iter().all(|v| v.is_finite()) {
                touch.centers[i] = Some(Point {
                    x: xy[0] as f64,
                    y: xy[1] as f64,
                });
            }
        }
        touch.points = p
            .chunks_exact(2)
            .map(|xy| {
                if xy.iter().all(|v| v.is_finite()) {
                    Some(Point {
                        x: xy[0] as f64,
                        y: xy[1] as f64,
                    })
                } else {
                    None
                }
            })
            .collect();
        let mut workspace = WORKSPACE
            .get_or_init(|| Mutex::new(Workspace::default()))
            .lock()
            .map_err(|_| "English workspace unavailable")?;
        let mut prediction = suggestions(dictionary, &word, &touch, &mut workspace);
        drop(workspace);
        let left: String = env.get_string(&left).map_err(|e| e.to_string())?.into();
        let right: String = env.get_string(&right).map_err(|e| e.to_string())?.into();
        let words: Vec<_> = prediction
            .candidates
            .iter()
            .map(|c| c.word.clone())
            .collect();
        if let Some(scores) =
            crate::jni_api::english_context_scores(context_handle, &left, &right, &words)
        {
            rerank(&mut prediction, &scores, word.len());
        }
        serde_json::to_string(&prediction).map_err(|e| e.to_string())
    })();
    match result {
        Ok(json) => env
            .new_string(json)
            .map(|s| s.into_raw())
            .unwrap_or(std::ptr::null_mut()),
        Err(error) => {
            let _ = env.throw_new("java/lang/IllegalStateException", error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeRankCandidates(
    mut env: JNIEnv, _class: JClass, context_handle: jlong, left: JString, right: JString, words: JObjectArray,
) -> jstring {
    let result = (|| -> Result<String, String> {
        let left: String = env.get_string(&left).map_err(|e| e.to_string())?.into();
        let right: String = env.get_string(&right).map_err(|e| e.to_string())?.into();
        let n = env.get_array_length(&words).map_err(|e| e.to_string())?;
        if n > 128 { return Err("Too many mixed candidates".into()); }
        let mut candidates = Vec::with_capacity(n as usize);
        for i in 0..n {
            let object = env.get_object_array_element(&words, i).map_err(|e| e.to_string())?;
            let word = JString::from(object);
            candidates.push(env.get_string(&word).map_err(|e| e.to_string())?.into());
            env.delete_local_ref(word).map_err(|e| e.to_string())?;
        }
        let order = crate::jni_api::mixed_candidate_order(context_handle, &left, &right, &candidates)
            .unwrap_or_else(|| (0..candidates.len()).collect());
        serde_json::to_string(&order).map_err(|e| e.to_string())
    })();
    match result {
        Ok(json) => env.new_string(json).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut()),
        Err(error) => { let _ = env.throw_new("java/lang/IllegalStateException", error); std::ptr::null_mut() }
    }
}
