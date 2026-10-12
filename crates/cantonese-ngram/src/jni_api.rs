use crate::{
    counts::CsrLanguageModel,
    hybrid::HybridLanguageModel,
    quick::{QuickDictionary, QuickEngine},
    NeuralLanguageModel,
};
use jni::{
    objects::{JByteArray, JClass, JString},
    sys::{jint, jlong, jobject},
    JNIEnv,
};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc, Mutex, OnceLock,
};

static ENGINES: OnceLock<Mutex<HashMap<i64, Arc<Mutex<QuickEngine>>>>> = OnceLock::new();
static NEXT: AtomicI64 = AtomicI64::new(1);
fn engines() -> &'static Mutex<HashMap<i64, Arc<Mutex<QuickEngine>>>> {
    ENGINES.get_or_init(|| Mutex::new(HashMap::new()))
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
    let result = (|| -> Result<QuickEngine, String> {
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
        let neural = if neural.is_empty() {
            None
        } else {
            Some(NeuralLanguageModel::from_bytes(&neural)?)
        };
        let counts = if counts.is_empty() || neural.as_ref().map(|m| m.minimum_context_tokens()==0).unwrap_or(false) {
            None
        } else {
            Some(CsrLanguageModel::parse(&counts)?)
        };
        Ok(QuickEngine::new(
            QuickDictionary::from_bytes(&three, 3)?,
            QuickDictionary::from_bytes(&five, 5)?,
            HybridLanguageModel::with_neural(neural, counts, &labels)?,
        ))
    })();
    match result {
        Ok(engine) => {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut registry) = engines().lock() {
                registry.insert(id, Arc::new(Mutex::new(engine)));
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
        let mut guard = engine.lock().map_err(|_| "Quick engine poisoned")?;
        guard.candidates(version as u32, &code, &left, &right)
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
        let mut guard = engine.lock().map_err(|_| "Quick engine poisoned")?;
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
