/// JNI API for Android integration.
/// Build with: cargo ndk -t arm64-v8a build --release --features android
///
/// Java class: com.qwenasr.QAsrEngine
///

use std::ffi::{CStr, CString};
use jni::JNIEnv;
use jni::objects::{JClass, JString, JByteBuffer};
use jni::sys::{jlong, jint, jfloat, jstring};

/// Load model. Returns native handle as jlong (pointer to QwenAsrEngine).
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeLoadModel(
    mut env: JNIEnv,
    _class: JClass,
    model_dir: JString,
    n_threads: jint,
) -> jlong {
    let dir_str: String = match env.get_string(&model_dir) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };

    let dir_c = match CString::new(dir_str) {
        Ok(c) => c,
        Err(_) => return 0,
    };

    let engine_ptr = crate::c_api::qwen_asr_load_model(dir_c.as_ptr(), n_threads, 0);
    engine_ptr as jlong
}

/// Transcribe direct byte buffer contents without copying memory (zero-copy).
/// The direct buffer is expected to contain 16kHz mono float PCM samples.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeTranscribePcmDirect(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    buffer: JByteBuffer,
    size_in_samples: jint,
) -> jstring {
    if handle == 0 || size_in_samples <= 0 {
        return std::ptr::null_mut();
    }

    let address_ptr = match env.get_direct_buffer_address(&buffer) {
        Ok(ptr) => ptr,
        Err(_) => return std::ptr::null_mut(),
    };

    if address_ptr.is_null() {
        return std::ptr::null_mut();
    }

    // Cast the raw byte buffer pointer to a float PCM sample pointer
    let samples_ptr = address_ptr as *const f32;

    let result_c = crate::c_api::qwen_asr_transcribe_pcm(
        handle as *mut crate::c_api::QwenAsrEngine,
        samples_ptr,
        size_in_samples,
    );

    if result_c.is_null() {
        return std::ptr::null_mut();
    }

    let result_cstr = CStr::from_ptr(result_c);
    let result_str = match result_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            crate::c_api::qwen_asr_free_string(result_c);
            return std::ptr::null_mut();
        }
    };

    let jstr = match env.new_string(result_str) {
        Ok(js) => js.into_raw(),
        Err(_) => std::ptr::null_mut(),
    };

    crate::c_api::qwen_asr_free_string(result_c);
    jstr
}

/// Set transcription language configuration.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeSetLanguage(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    language: JString,
) {
    if handle == 0 {
        return;
    }

    let lang_str: String = match env.get_string(&language) {
        Ok(s) => s.into(),
        Err(_) => return,
    };

    let lang_c = match CString::new(lang_str) {
        Ok(c) => c,
        Err(_) => return,
    };

    crate::c_api::qwen_asr_set_language(
        handle as *mut crate::c_api::QwenAsrEngine,
        lang_c.as_ptr(),
    );
}

/// Set transcription prompt text configuration.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeSetPrompt(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    prompt: JString,
) {
    if handle == 0 {
        return;
    }

    let prompt_str: String = match env.get_string(&prompt) {
        Ok(s) => s.into(),
        Err(_) => return,
    };

    let engine = &mut *(handle as *mut crate::c_api::QwenAsrEngine);
    let _ = engine.ctx.set_prompt(&prompt_str);
}


/// Set transcription segment seconds.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeSetSegmentSec(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    sec: jfloat,
) {
    if handle != 0 {
        crate::c_api::qwen_asr_set_segment_sec(
            handle as *mut crate::c_api::QwenAsrEngine,
            sec,
        );
    }
}

/// Free the native model allocation and release memory.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        crate::c_api::qwen_asr_free(handle as *mut crate::c_api::QwenAsrEngine);
    }
}
