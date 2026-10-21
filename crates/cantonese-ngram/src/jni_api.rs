use crate::{
    counts::CsrLanguageModel,
    hybrid::HybridLanguageModel,
    quick::{QuickDictionary, QuickEngine},
    NeuralLanguageModel,
};
use jni::{
    objects::{JByteArray, JClass, JString},
    sys::{jint, jlong, jobject, jstring},
    JNIEnv,
};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc, Mutex, OnceLock,
};

// Share immutable NN weights, never mutable inference buffers or the Quick lock.
struct KeyboardEngines {
    quick: Mutex<QuickEngine>,
    english: Mutex<HybridLanguageModel>,
}
static ENGINES: OnceLock<Mutex<HashMap<i64, Arc<KeyboardEngines>>>> = OnceLock::new();
static NEXT: AtomicI64 = AtomicI64::new(1);
fn engines() -> &'static Mutex<HashMap<i64, Arc<KeyboardEngines>>> {
    ENGINES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// English completion shares Quick's vocabulary and weights, with independent state.
#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_SharedLmEnglishEngine_nativeComplete(
    mut env:JNIEnv,_class:JClass,handle:jlong,left:JString,prefix:JString,
) -> jobject {
    let result=(||->Result<Vec<String>,String> {
        let left:String=env.get_string(&left).map_err(|e|e.to_string())?.into();
        let prefix:String=env.get_string(&prefix).map_err(|e|e.to_string())?.into();
        if prefix.len()>128 {return Err("English prefix too long".into());}
        let engine=engines().lock().map_err(|_|"Registry poisoned")?.get(&handle).cloned().ok_or("Closed shared engine")?;
        let mut engine=engine.english.lock().map_err(|_|"Shared engine poisoned")?;
        let model=engine.neural.as_mut().ok_or("Shared model missing")?;
        Ok(model.completions(&left,&prefix,8)?.into_iter().map(|p|p.token).collect())
    })();
    match result {
        Ok(values)=>{
            let array=match env.new_object_array(values.len() as i32,"java/lang/String",jni::objects::JObject::null()) {
                Ok(a)=>a,Err(_)=>return std::ptr::null_mut()
            };
            for (i,value) in values.iter().enumerate() {
                let Ok(s)=env.new_string(value) else {return std::ptr::null_mut();};
                if env.set_object_array_element(&array,i as i32,&s).is_err() {return std::ptr::null_mut();}
                let _=env.delete_local_ref(s);
            }
            array.into_raw()
        },
        Err(error)=>{let _=env.throw_new("java/lang/IllegalStateException",error);std::ptr::null_mut()}
    }
}

/// Typo generation and contextual ranking use the same vocabulary/weights as Quick.
#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_SharedLmEnglishEngine_nativeSuggest(
    mut env:JNIEnv,_class:JClass,handle:jlong,left:JString,typed:JString,
) -> jstring {
    let result=(||->Result<String,String> {
        let left:String=env.get_string(&left).map_err(|e|e.to_string())?.into();
        let typed:String=env.get_string(&typed).map_err(|e|e.to_string())?.into();
        let engine=engines().lock().map_err(|_|"Registry poisoned")?.get(&handle).cloned().ok_or("Closed shared engine")?;
        let mut engine=engine.english.lock().map_err(|_|"Shared engine poisoned")?;
        let model=engine.neural.as_mut().ok_or("Shared model missing")?;
        serde_json::to_string(&model.english_suggestions(&left,&typed,8)?).map_err(|e|e.to_string())
    })();
    match result {
        Ok(json)=>env.new_string(json).map(|s|s.into_raw()).unwrap_or(std::ptr::null_mut()),
        Err(error)=>{let _=env.throw_new("java/lang/IllegalStateException",error);std::ptr::null_mut()}
    }
}

pub(crate) fn mixed_candidate_order(handle: i64, left: &str, right: &str, words: &[String]) -> Option<Vec<usize>> {
    let engine = engines().lock().ok()?.get(&handle)?.clone();
    let mut engine = engine.english.lock().ok()?;
    crate::quick::rank_mixed(&mut engine, left, right, words).ok()
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_english_SharedLmEnglishEngine_nativeRank(
    mut env:JNIEnv,_class:JClass,handle:jlong,left:JString,right:JString,words:jni::objects::JObjectArray,
) -> jstring {
    let result=(||->Result<String,String> {
        let left:String=env.get_string(&left).map_err(|e|e.to_string())?.into();
        let right:String=env.get_string(&right).map_err(|e|e.to_string())?.into();
        let n=env.get_array_length(&words).map_err(|e|e.to_string())?;
        if n>128 {return Err("Too many candidates".into());}
        let mut candidates=Vec::new();
        for i in 0..n {
            let value=JString::from(env.get_object_array_element(&words,i).map_err(|e|e.to_string())?);
            candidates.push(env.get_string(&value).map_err(|e|e.to_string())?.into());
            env.delete_local_ref(value).map_err(|e|e.to_string())?;
        }
        let order=mixed_candidate_order(handle,&left,&right,&candidates).ok_or("Shared model unavailable")?;
        serde_json::to_string(&order).map_err(|e|e.to_string())
    })();
    match result {
        Ok(json)=>env.new_string(json).map(|s|s.into_raw()).unwrap_or(std::ptr::null_mut()),
        Err(error)=>{let _=env.throw_new("java/lang/IllegalStateException",error);std::ptr::null_mut()}
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_quick_NativeQuickEngine_nativeLanguageEvidence(
    mut env: JNIEnv, _class: JClass, handle: jlong, code: JString, left: JString, right: JString, chinese: JString,
) -> jstring {
    let result = (|| -> Result<String, String> {
        let code: String = env.get_string(&code).map_err(|e| e.to_string())?.into();
        let left: String = env.get_string(&left).map_err(|e| e.to_string())?.into();
        let right: String = env.get_string(&right).map_err(|e| e.to_string())?.into();
        let chinese: String = env.get_string(&chinese).map_err(|e| e.to_string())?.into();
        let engine = engines().lock().map_err(|_| "Registry poisoned")?.get(&handle).cloned().ok_or("Closed Quick engine")?;
        let result = engine.quick.lock().map_err(|_| "Quick engine poisoned")?.language_evidence(&code, &left, &right, &chinese)?;
        serde_json::to_string(&result).map_err(|e| e.to_string())
    })();
    match result {
        Ok(json) => env.new_string(json).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut()),
        Err(error) => { let _ = env.throw_new("java/lang/IllegalStateException", error); std::ptr::null_mut() }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_quick_NativeQuickEngine_nativeCreate(
    mut env: JNIEnv,
    _class: JClass,
    three: JByteArray,
    five: JByteArray,
    neural: JByteArray,
    counts: JByteArray,
    labels: JByteArray,
) -> jlong {
    let result = (|| -> Result<KeyboardEngines, String> {
        let three = env.convert_byte_array(three).map_err(|e| e.to_string())?;
        let five = env.convert_byte_array(five).map_err(|e| e.to_string())?;
        let neural = env.convert_byte_array(neural).map_err(|e| e.to_string())?;
        let counts = env.convert_byte_array(counts).map_err(|e| e.to_string())?;
        let labels = env.convert_byte_array(labels).map_err(|e| e.to_string())?;
        let labels: Vec<_> = std::str::from_utf8(&labels)
            .map_err(|e| e.to_string())?
            .lines()
            .map(|s| s.trim_end().to_owned())
            .collect();
        let make_lm = || -> Result<HybridLanguageModel, String> {
            // CNN construction reuses Arc<Weights>; only caches/buffers are private.
            let model = if neural.is_empty() { None } else {
                Some(NeuralLanguageModel::from_bytes(&neural)?)
            };
            let counts = if counts.is_empty() || model.as_ref()
                .map(|m| m.minimum_context_tokens() == 0).unwrap_or(false) {
                None
            } else { Some(CsrLanguageModel::parse(&counts)?) };
            HybridLanguageModel::with_neural(model, counts, &labels)
        };
        Ok(KeyboardEngines {
            quick: Mutex::new(QuickEngine::new(
                QuickDictionary::from_bytes(&three, 3)?,
                QuickDictionary::from_bytes(&five, 5)?,
                make_lm()?,
            )),
            english: Mutex::new(make_lm()?),
        })
    })();
    match result {
        Ok(engine) => {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut registry) = engines().lock() {
                registry.insert(id, Arc::new(engine));
                id
            } else {
                0
            }
        }
        Err(error) => {
            let _ = env.throw_new("java/lang/IllegalArgumentException", error);
            0
        }
    }
}
#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_quick_NativeQuickEngine_nativeCandidates(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    version: jint,
    code: JString,
    left: JString,
    right: JString,
    preview: jni::sys::jboolean,
) -> jobject {
    let result = (|| -> Result<Vec<String>, String> {
        let code: String = env.get_string(&code).map_err(|e| e.to_string())?.into();
        let left: String = env.get_string(&left).map_err(|e| e.to_string())?.into();
        let right: String = env.get_string(&right).map_err(|e| e.to_string())?.into();
        let engine = engines()
            .lock()
            .map_err(|_| "Registry poisoned")?
            .get(&handle)
            .cloned()
            .ok_or("Closed Quick engine")?;
        let mut guard = engine.quick.lock().map_err(|_| "Quick engine poisoned")?;
        if preview != 0 {
            guard.preview_candidates(version as u32, &code, &left, &right)
        } else {
            guard.candidates(version as u32, &code, &left, &right)
        }
    })();
    match result {
        Ok(strings) => {
            let array = match env.new_object_array(
                strings.len() as i32,
                "java/lang/String",
                jni::objects::JObject::null(),
            ) {
                Ok(v) => v,
                Err(_) => return std::ptr::null_mut(),
            };
            for (i, s) in strings.iter().enumerate() {
                let s = match env.new_string(s) {
                    Ok(s) => s,
                    Err(_) => return std::ptr::null_mut(),
                };
                if env.set_object_array_element(&array, i as i32, &s).is_err() {
                    return std::ptr::null_mut();
                }
                let _ = env.delete_local_ref(s);
            }
            array.into_raw()
        }
        Err(e) => {
            let _ = env.throw_new("java/lang/IllegalStateException", e);
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_quick_NativeQuickEngine_nativePredictions(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    left: JString,
    right: JString,
    limit: jint,
) -> jobject {
    let result = (|| -> Result<Vec<String>, String> {
        let left: String = env.get_string(&left).map_err(|e| e.to_string())?.into();
        let right: String = env.get_string(&right).map_err(|e| e.to_string())?.into();
        let engine = engines()
            .lock()
            .map_err(|_| "Registry poisoned")?
            .get(&handle)
            .cloned()
            .ok_or("Closed Quick engine")?;
        let mut guard = engine.quick.lock().map_err(|_| "Quick engine poisoned")?;
        let (_, predictions) = guard
            .lm
            .predict(&left, &right, limit.clamp(0, 100) as usize)?;
        Ok(predictions.into_iter().map(|p| p.token).collect())
    })();
    match result {
        Ok(strings) => {
            let Ok(array) = env.new_object_array(
                strings.len() as i32,
                "java/lang/String",
                jni::objects::JObject::null(),
            ) else {
                return std::ptr::null_mut();
            };
            for (i, text) in strings.iter().enumerate() {
                let Ok(s) = env.new_string(text) else {
                    return std::ptr::null_mut();
                };
                if env.set_object_array_element(&array, i as i32, &s).is_err() {
                    return std::ptr::null_mut();
                }
                let _ = env.delete_local_ref(s);
            }
            array.into_raw()
        }
        Err(e) => {
            let _ = env.throw_new("java/lang/IllegalStateException", e);
            std::ptr::null_mut()
        }
    }
}
#[no_mangle]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_quick_NativeQuickEngine_nativeDestroy(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if let Ok(mut registry) = engines().lock() {
        registry.remove(&handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_ranking_does_not_acquire_quick_inference_lock() {
        let dictionary = vec![0; 703 * 4];
        let make_lm = || HybridLanguageModel::with_neural(None, None, &[]).unwrap();
        let engine = Arc::new(KeyboardEngines {
            quick: Mutex::new(QuickEngine::new(
                QuickDictionary::from_bytes(&dictionary, 3).unwrap(),
                QuickDictionary::from_bytes(&dictionary, 5).unwrap(), make_lm(),
            )),
            english: Mutex::new(make_lm()),
        });
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        engines().lock().unwrap().insert(id, engine.clone());
        let occupied = engine.quick.lock().unwrap();
        assert!(engine.english.try_lock().is_ok());
        // Traverse the production nativeRank registry path on another worker.
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            send.send(mixed_candidate_order(id, "", "", &[])).unwrap();
        });
        let result = receive.recv_timeout(std::time::Duration::from_secs(1));
        drop(occupied);
        worker.join().unwrap();
        assert_eq!(result.unwrap(), Some(vec![]));
        engines().lock().unwrap().remove(&id);
        assert_eq!(mixed_candidate_order(id, "", "", &[]), None);
    }
}
