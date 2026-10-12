//! Minimal JNI adapter for the Android keyboard using pure Rust BiLSTM recognizer.

use std::sync::Mutex;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JFloatArray, JIntArray, JObject};
use jni::sys::{jfloat, jint, jlong, jobject};

use super::mlkit_bilstm::{MlkitBiLstmRecognizer, StrokePoint};

struct HandwritingContext {
    recognizer: Mutex<MlkitBiLstmRecognizer>,
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_hwr_HandwritingRecognizer_nativeCreate(
    env: JNIEnv,
    _class: JClass,
    model_bytes: JByteArray,
    label_bytes: JByteArray,
    lm_bytes: JByteArray,
    neural_bytes: JByteArray,
    fusion_bytes: JByteArray,
) -> jlong {
    let model = match env.convert_byte_array(model_bytes) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => return 0,
    };
    let labels_raw = match env.convert_byte_array(label_bytes) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => return 0,
    };
    let labels_str = match std::str::from_utf8(&labels_raw) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let lm = if !lm_bytes.is_null() {
        env.convert_byte_array(lm_bytes)
            .ok()
            .and_then(|bytes| crate::counts::CsrLanguageModel::parse(&bytes).ok())
    } else {
        None
    };

    match MlkitBiLstmRecognizer::from_bytes(&model, labels_str) {
        Ok(mut recognizer) => {
            if let Some(lm_model) = lm {
                recognizer.set_lm(lm_model);
            }
            let neural_raw = match env.convert_byte_array(neural_bytes) { Ok(b) => b, Err(_) => return 0 };
            let neural = if neural_raw.is_empty() { None } else {
                match crate::NeuralLanguageModel::from_bytes(&neural_raw) { Ok(lm) => Some(lm), Err(_) => return 0 }
            };
            if recognizer.set_neural_context_lm(neural).is_err() { return 0; }
            let fusion_raw = match env.convert_byte_array(fusion_bytes) { Ok(b) => b, Err(_) => return 0 };
            if !fusion_raw.is_empty() {
                #[derive(serde::Deserialize)]
                struct Calibration { version: u32, weights: crate::hybrid::FusionWeights }
                let calibration: Calibration = match serde_json::from_slice(&fusion_raw) { Ok(c) => c, Err(_) => return 0 };
                if calibration.version != 1 || recognizer.set_fusion_weights(calibration.weights).is_err() { return 0; }
            }
            Box::into_raw(Box::new(HandwritingContext {
                recognizer: Mutex::new(recognizer),
            })) as jlong
        }
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_hwr_HandwritingRecognizer_nativeDestroy(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut HandwritingContext));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_hwr_HandwritingRecognizer_nativePredictStrokes(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    flat_points: JFloatArray,
    stroke_counts: JIntArray,
    canvas_width: jfloat,
    canvas_height: jfloat,
    top_k: jint,
    prev_char_str: jni::objects::JString,
) -> jobject {
    if handle == 0 || top_k <= 0 {
        return std::ptr::null_mut();
    }
    let prev_char: Option<String> = if !prev_char_str.is_null() {
        env.get_string(&prev_char_str).ok().map(|s| s.into())
    } else {
        None
    };
    let points_vec = match unsafe { env.get_array_elements(&flat_points, jni::objects::ReleaseMode::NoCopyBack) } {
        Ok(pts) => {
            let slice = unsafe { std::slice::from_raw_parts(pts.as_ptr(), pts.len()) };
            slice.to_vec()
        }
        Err(_) => return std::ptr::null_mut(),
    };
    let counts_vec = match unsafe { env.get_array_elements(&stroke_counts, jni::objects::ReleaseMode::NoCopyBack) } {
        Ok(counts) => {
            let slice = unsafe { std::slice::from_raw_parts(counts.as_ptr(), counts.len()) };
            slice.to_vec()
        }
        Err(_) => return std::ptr::null_mut(),
    };

    let mut strokes = Vec::with_capacity(counts_vec.len());
    let mut offset = 0;
    for &count in &counts_vec {
        let count = count as usize;
        let mut stroke = Vec::with_capacity(count);
        for _ in 0..count {
            if offset + 2 < points_vec.len() {
                stroke.push(StrokePoint {
                    x: points_vec[offset],
                    y: points_vec[offset + 1],
                    t_ms: points_vec[offset + 2],
                });
                offset += 3;
            }
        }
        if !stroke.is_empty() {
            strokes.push(stroke);
        }
    }

    let context = unsafe { &*(handle as *const HandwritingContext) };
    let mut recognizer = match context.recognizer.lock() {
        Ok(guard) => guard,
        Err(_) => return std::ptr::null_mut(),
    };

    let predictions = match recognizer.predict_strokes_with_context(
        &strokes,
        canvas_width as f32,
        canvas_height as f32,
        top_k as usize,
        prev_char.as_deref(),
    ) {
        Ok(preds) => preds,
        Err(_) => return std::ptr::null_mut(),
    };

    let output = match env.new_object_array(
        predictions.len() as jint,
        "java/lang/String",
        JObject::null(),
    ) {
        Ok(output) => output,
        Err(_) => return std::ptr::null_mut(),
    };

    for (index, text) in predictions.into_iter().enumerate() {
        let Ok(label) = env.new_string(&text) else {
            return std::ptr::null_mut();
        };
        if env.set_object_array_element(&output, index as jint, label).is_err() {
            return std::ptr::null_mut();
        }
    }

    output.into_raw()
}

/// Fallback predict function for any legacy raster interface callers.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_hwr_HandwritingRecognizer_nativePredict(
    mut env: JNIEnv,
    _class: JClass,
    _handle: jlong,
    _pixels: JByteArray,
    _width: jint,
    _height: jint,
    _top_k: jint,
) -> jobject {
    match env.new_object_array(0, "java/lang/String", JObject::null()) {
        Ok(arr) => arr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Stub nativePreprocess for backward compatibility.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_hwr_HandwritingRecognizer_nativePreprocess(
    _env: JNIEnv,
    _class: JClass,
    _handle: jlong,
    pixels: JByteArray,
    _width: jint,
    _height: jint,
) -> jobject {
    pixels.into_raw()
}

/// Kotlin serializes context updates, predictions and destruction on this handle.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_madeinhk_cantonesevoicekeyboard_hwr_HandwritingRecognizer_nativeSetLanguageContext(
    mut env: JNIEnv, _class: JClass, handle: jlong,
    left: jni::objects::JString, right: jni::objects::JString,
) {
    if handle == 0 { return; }
    let Ok(left) = env.get_string(&left).map(String::from) else { return; };
    let Ok(right) = env.get_string(&right).map(String::from) else { return; };
    let context = unsafe { &*(handle as *const HandwritingContext) };
    if let Ok(mut recognizer) = context.recognizer.lock() { recognizer.set_language_context(left, right); }
}
