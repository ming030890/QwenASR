use super::*;
use ::jni::{
    objects::{JByteArray, JClass, JFloatArray, JString, JObjectArray},
    sys::{jboolean, jlong, jstring},
    JNIEnv,
};
use std::sync::{Arc, Mutex, OnceLock};
static DICTIONARY: OnceLock<Arc<Dictionary>> = OnceLock::new();
static V2_DICTIONARY: OnceLock<Arc<Dictionary>> = OnceLock::new();
static V3_DICTIONARY: OnceLock<Arc<Dictionary>> = OnceLock::new();
static V4_DICTIONARY: OnceLock<Arc<Dictionary>> = OnceLock::new();
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

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeDecodeV2(
    env: JNIEnv, _class: JClass, word: JString, centers: JFloatArray, points: JFloatArray,
    previous: JString, session: jlong, revision: jlong,
) -> jstring {
    decode_native(env,word,centers,points,previous,session,revision,2)
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeDecodeV3(
    env: JNIEnv, _class: JClass, word: JString, centers: JFloatArray, points: JFloatArray,
    previous: JString, session: jlong, revision: jlong,
) -> jstring {
    decode_native(env,word,centers,points,previous,session,revision,3)
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeDecodeV4(
    env: JNIEnv, _class: JClass, word: JString, centers: JFloatArray, points: JFloatArray,
    previous: JString, session: jlong, revision: jlong,
) -> jstring {
    decode_native(env,word,centers,points,previous,session,revision,4)
}

fn decode_native(
    mut env: JNIEnv, word: JString, centers: JFloatArray, points: JFloatArray,
    previous: JString, session: jlong, revision: jlong, version: u8,
) -> jstring {
    static DECODER: OnceLock<Mutex<decoder::Decoder>> = OnceLock::new();
    let result = (|| -> Result<String,String> {
        let dictionary = (match version {4=>&V4_DICTIONARY,3=>&V3_DICTIONARY,_=>&V2_DICTIONARY}).get().ok_or("English dictionary not loaded")?;
        let word: String = env.get_string(&word).map_err(|e|e.to_string())?.into();
        let previous: String = env.get_string(&previous).map_err(|e|e.to_string())?.into();
        let read = |array: &JFloatArray, max| -> Result<Vec<f32>,String> {
            let n=env.get_array_length(array).map_err(|e|e.to_string())? as usize;
            if n>max || n%2!=0 {return Err("Invalid touch evidence size".into());}
            let mut values=vec![0.;n]; env.get_float_array_region(array,0,&mut values).map_err(|e|e.to_string())?; Ok(values)
        };
        let mut touch=TouchModel::default();
        for (i,xy) in read(&centers,52)?.chunks_exact(2).enumerate() {
            if xy.iter().all(|v|v.is_finite()) {touch.centers[i]=Some(Point{x:xy[0] as f64,y:xy[1] as f64});}
        }
        touch.points=read(&points,128)?.chunks_exact(2).map(|xy|
            xy.iter().all(|v|v.is_finite()).then_some(Point{x:xy[0] as f64,y:xy[1] as f64})).collect();
        let decoded=if version==4 {super::spelling::decode_robust(dictionary,&word,&touch,&previous,session,revision)}
        else if version==3 {super::spelling::decode(dictionary,&word,&touch,&previous,session,revision)} else {
            DECODER.get_or_init(||Mutex::new(decoder::Decoder::default())).lock().map_err(|_|"Decoder poisoned")?
                .decode(dictionary,&word,&touch,&previous,session,revision)
        };
        serde_json::to_string(&decoded).map_err(|e|e.to_string())
    })();
    match result {
        Ok(json)=>env.new_string(json).map(|s|s.into_raw()).unwrap_or(std::ptr::null_mut()),
        Err(e)=>{let _=env.throw_new("java/lang/IllegalStateException",e);std::ptr::null_mut()}
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeLoadV4(
    mut env:JNIEnv,_class:JClass,bytes:JByteArray,
)->jboolean {
    if V4_DICTIONARY.get().is_some() {return 1;}
    match env.convert_byte_array(bytes).map_err(|e|e.to_string()).and_then(|b|Dictionary::from_bytes(&b)) {
        Ok(d)=>{let _=V4_DICTIONARY.set(Arc::new(d));1}
        Err(e)=>{let _=env.throw_new("java/lang/IllegalArgumentException",e);0}
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeLoadV3(
    mut env:JNIEnv,_class:JClass,bytes:JByteArray,
)->jboolean {
    if V3_DICTIONARY.get().is_some() {return 1;}
    match env.convert_byte_array(bytes).map_err(|e|e.to_string()).and_then(|b|Dictionary::from_bytes(&b)) {
        Ok(d)=>{let _=V3_DICTIONARY.set(Arc::new(d));1}
        Err(e)=>{let _=env.throw_new("java/lang/IllegalArgumentException",e);0}
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeLoadV2(
    mut env: JNIEnv, _class: JClass, bytes: JByteArray,
) -> jboolean {
    if V2_DICTIONARY.get().is_some() {return 1;}
    match env.convert_byte_array(bytes).map_err(|e|e.to_string()).and_then(|b|Dictionary::from_bytes(&b)) {
        Ok(d)=>{let _=V2_DICTIONARY.set(Arc::new(d));1}
        Err(e)=>{let _=env.throw_new("java/lang/IllegalArgumentException",e);0}
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeSequenceEvidence(
    mut env:JNIEnv,_class:JClass,handle:jlong,left:JString,right:JString,words:JObjectArray,
)->jstring {
    let result=(||->Result<String,String>{
        let left:String=env.get_string(&left).map_err(|e|e.to_string())?.into();
        let right:String=env.get_string(&right).map_err(|e|e.to_string())?.into();
        let n=env.get_array_length(&words).map_err(|e|e.to_string())?;
        if n>128{return Err("Too many sequence candidates".into());}
        let mut candidates=Vec::new();
        for i in 0..n {
            let word=JString::from(env.get_object_array_element(&words,i).map_err(|e|e.to_string())?);
            candidates.push(env.get_string(&word).map_err(|e|e.to_string())?.into());env.delete_local_ref(word).map_err(|e|e.to_string())?;
        }
        serde_json::to_string(&crate::jni_api::sequence_evidence(handle,&left,&right,&candidates)).map_err(|e|e.to_string())
    })();
    match result {
        Ok(json)=>env.new_string(json).map(|s|s.into_raw()).unwrap_or(std::ptr::null_mut()),
        Err(e)=>{let _=env.throw_new("java/lang/IllegalStateException",e);std::ptr::null_mut()}
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_NativeEnglishEngine_nativeKnownSequences(
    mut env:JNIEnv,_class:JClass,handle:jlong,words:JObjectArray,
)->jstring {
    let result=(||->Result<String,String>{
        let n=env.get_array_length(&words).map_err(|e|e.to_string())?;
        if n>128{return Err("Too many sequence candidates".into());}
        let mut candidates=Vec::new();
        for i in 0..n {
            let word=JString::from(env.get_object_array_element(&words,i).map_err(|e|e.to_string())?);
            candidates.push(env.get_string(&word).map_err(|e|e.to_string())?.into());env.delete_local_ref(word).map_err(|e|e.to_string())?;
        }
        serde_json::to_string(&crate::jni_api::known_sequences(handle,&candidates)).map_err(|e|e.to_string())
    })();
    match result {
        Ok(json)=>env.new_string(json).map(|s|s.into_raw()).unwrap_or(std::ptr::null_mut()),
        Err(e)=>{let _=env.throw_new("java/lang/IllegalStateException",e);std::ptr::null_mut()}
    }
}
