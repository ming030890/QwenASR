use jni::objects::{GlobalRef, JByteBuffer, JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jfloat, jint, jlong, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;
/// JNI API for Android integration.
/// Build with: cargo ndk -t arm64-v8a build --release --features android
///
/// Java class: com.qwenasr.QAsrEngine
///
use crate::vad_stream::{VadStreamConfig, VadStreamEvent, VadStreamSession, SAMPLE_RATE};
#[cfg(test)]
use crate::vad_stream::VAD_FRAME_SIZE;
use std::ffi::{CStr, CString};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

const MAX_QUEUED_SAMPLES: u64 = (SAMPLE_RATE * 90) as u64;
const INFERENCE_BACKLOG_SAMPLES: u64 = (SAMPLE_RATE / 5) as u64;

enum VadMsg {
    Audio(Vec<f32>),
    Stop { finalize: bool },
}

struct VadAsrSession {
    engine_addr: usize,
    task_tx: mpsc::Sender<VadMsg>,
    queued_samples: Arc<AtomicU64>,
    closed: AtomicBool,
    worker: Mutex<Option<JoinHandle<()>>>,
    _save_audio_dir: Option<String>,
}

fn write_wav_file(path: &str, samples: &[i16], sample_rate: u32) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;
    let data_size = (samples.len() * 2) as u32;
    let chunk_size = 36 + data_size;

    // RIFF Header
    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Subchunk1Size
    file.write_all(&1u16.to_le_bytes())?; // AudioFormat (PCM = 1)
    file.write_all(&1u16.to_le_bytes())?; // NumChannels (Mono = 1)
    file.write_all(&sample_rate.to_le_bytes())?; // SampleRate
    let byte_rate = sample_rate * 1 * 2;
    file.write_all(&byte_rate.to_le_bytes())?; // ByteRate
    file.write_all(&2u16.to_le_bytes())?; // BlockAlign
    file.write_all(&16u16.to_le_bytes())?; // BitsPerSample

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // Raw samples
    for &sample in samples {
        file.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}

fn notify_hypothesis(java_vm: &jni::JavaVM, listener: &GlobalRef, text: &str) {
    if let Ok(mut env) = java_vm.attach_current_thread() {
        if let Ok(j_text) = env.new_string(text) {
            let _ = env.call_method(
                listener.as_obj(),
                "onHypothesisChanged",
                "(Ljava/lang/String;)V",
                &[JValue::Object(&JObject::from(j_text))],
            );
        }
    }
}

fn notify_complete(java_vm: &jni::JavaVM, listener: &GlobalRef) {
    if let Ok(mut env) = java_vm.attach_current_thread() {
        let _ = env.call_method(listener.as_obj(), "onTranscriptComplete", "()V", &[]);
    }
}

fn start_asr_worker(
    engine_addr: usize,
    java_vm: Arc<jni::JavaVM>,
    listener: GlobalRef,
    rx: mpsc::Receiver<VadMsg>,
    queued_samples: Arc<AtomicU64>,
    save_audio_dir: Option<String>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut vad_session = VadStreamSession::new(VadStreamConfig::default());
        let mut all_audio_samples: Vec<i16> = Vec::new();
        let mut last_hypothesis = String::new();

        while let Ok(msg) = rx.recv() {
            match msg {
                VadMsg::Audio(samples_f32) => {
                    queued_samples.fetch_sub(samples_f32.len() as u64, Ordering::Relaxed);
                    if save_audio_dir.is_some() {
                        for &s in &samples_f32 {
                            all_audio_samples.push((s * 32767.0).clamp(-32768.0, 32767.0) as i16);
                        }
                    }

                    let engine = unsafe { &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine) };
                    let has_audio_backlog =
                        queued_samples.load(Ordering::Relaxed) >= INFERENCE_BACKLOG_SAMPLES;
                    let vm = java_vm.clone();
                    let lis = listener.clone();

                    let on_delta = |delta: &str| {
                        if !delta.is_empty() {
                            if let Ok(mut env) = vm.attach_current_thread() {
                                if let Ok(j_delta) = env.new_string(delta) {
                                    let _ = env.call_method(
                                        lis.as_obj(),
                                        "onTranscriptDelta",
                                        "(Ljava/lang/String;)V",
                                        &[JValue::Object(&JObject::from(j_delta))],
                                    );
                                }
                            }
                        }
                    };
                    let events = if has_audio_backlog {
                        vad_session.push_audio_backlogged(&mut engine.ctx, &samples_f32, on_delta)
                    } else {
                        vad_session.push_audio(&mut engine.ctx, &samples_f32, on_delta)
                    };

                    let mut utterance_ended = false;
                    for event in events {
                        if let VadStreamEvent::UtteranceEnd { text, .. } = event {
                            if text != last_hypothesis {
                                notify_hypothesis(&java_vm, &listener, &text);
                            }
                            notify_complete(&java_vm, &listener);
                            last_hypothesis.clear();
                            utterance_ended = true;
                        }
                    }
                    if !utterance_ended {
                        let hypothesis = vad_session.current_hypothesis();
                        if hypothesis != last_hypothesis {
                            notify_hypothesis(&java_vm, &listener, hypothesis);
                            last_hypothesis.clear();
                            last_hypothesis.push_str(hypothesis);
                        }
                    }
                }
                VadMsg::Stop { finalize } => {
                    let engine = unsafe { &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine) };
                    if finalize {
                        let vm = java_vm.clone();
                        let lis = listener.clone();
                        let events = vad_session.finish(&mut engine.ctx, |delta| {
                            if !delta.is_empty() {
                                if let Ok(mut env) = vm.attach_current_thread() {
                                    if let Ok(j_delta) = env.new_string(delta) {
                                        let _ = env.call_method(
                                            lis.as_obj(),
                                            "onTranscriptDelta",
                                            "(Ljava/lang/String;)V",
                                            &[JValue::Object(&JObject::from(j_delta))],
                                        );
                                    }
                                }
                            }
                        });
                        for event in events {
                            if let VadStreamEvent::UtteranceEnd { text, .. } = event {
                                if text != last_hypothesis {
                                    notify_hypothesis(&java_vm, &listener, &text);
                                }
                                notify_complete(&java_vm, &listener);
                                last_hypothesis.clear();
                            }
                        }
                    }
                    if let Some(ref dir) = save_audio_dir {
                        if !all_audio_samples.is_empty() {
                            let wav_path = format!("{}/session_audio.wav", dir);
                            if let Err(e) = write_wav_file(&wav_path, &all_audio_samples, 16000) {
                                eprintln!("qwen-asr: failed to save session audio: {}", e);
                            }
                        }
                    }
                    break;
                }
            }
        }
    })
}

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
) -> jboolean {
    if handle == 0 {
        return JNI_FALSE;
    }

    let lang_str: String = match env.get_string(&language) {
        Ok(s) => s.into(),
        Err(_) => return JNI_FALSE,
    };

    let lang_c = match CString::new(lang_str) {
        Ok(c) => c,
        Err(_) => return JNI_FALSE,
    };

    if crate::c_api::qwen_asr_set_language(
        handle as *mut crate::c_api::QwenAsrEngine,
        lang_c.as_ptr(),
    ) == 0
    {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

/// Set the output Chinese variant for Simplified->Traditional conversion.
///
/// Pass e.g. "zh-Hant", "zh-TW", or "zh-HK". Empty string disables conversion.
/// Returns true on success, false on error/unrecognized variant.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeSetChineseVariant(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    variant: JString,
) -> jboolean {
    if handle == 0 {
        return JNI_FALSE;
    }

    let variant_str: String = match env.get_string(&variant) {
        Ok(s) => s.into(),
        Err(_) => return JNI_FALSE,
    };

    let variant_c = match CString::new(variant_str) {
        Ok(c) => c,
        Err(_) => return JNI_FALSE,
    };

    if crate::c_api::qwen_asr_set_chinese_variant(
        handle as *mut crate::c_api::QwenAsrEngine,
        variant_c.as_ptr(),
    ) == 0
    {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
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
        crate::c_api::qwen_asr_set_segment_sec(handle as *mut crate::c_api::QwenAsrEngine, sec);
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

/// Start a native VAD + ASR session. The listener must implement:
/// fun onTranscriptDelta(textDelta: String)
/// fun onHypothesisChanged(text: String)
/// fun onTranscriptComplete()
/// fun onTranscriptError()
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeStartVadSession(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    listener: JObject,
    save_audio_dir: JString,
) -> jlong {
    if handle == 0 || listener.is_null() {
        return 0;
    }

    let save_dir_opt: Option<String> = if !save_audio_dir.is_null() {
        match env.get_string(&save_audio_dir) {
            Ok(s) => {
                let s_str: String = s.into();
                if s_str.trim().is_empty() {
                    None
                } else {
                    Some(s_str)
                }
            }
            Err(_) => None,
        }
    } else {
        None
    };
    if let Some(ref directory) = save_dir_opt {
        if fs::create_dir_all(directory).is_err() {
            return 0;
        }
    }

    let java_vm = match env.get_java_vm() {
        Ok(vm) => vm,
        Err(_) => return 0,
    };
    let listener_ref = match env.new_global_ref(listener) {
        Ok(r) => r,
        Err(_) => return 0,
    };

    let (tx, rx) = mpsc::channel();
    let queued_samples = Arc::new(AtomicU64::new(0));
    let worker = start_asr_worker(
        handle as usize,
        Arc::new(java_vm),
        listener_ref,
        rx,
        queued_samples.clone(),
        save_dir_opt.clone(),
    );

    let session = VadAsrSession {
        engine_addr: handle as usize,
        task_tx: tx,
        queued_samples,
        closed: AtomicBool::new(false),
        worker: Mutex::new(Some(worker)),
        _save_audio_dir: save_dir_opt,
    };
    Box::into_raw(Box::new(session)) as jlong
}

/// Push 16 kHz mono PCM16 from a direct ByteBuffer into the native VAD session.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativePushPcm16(
    env: JNIEnv,
    _class: JClass,
    session_handle: jlong,
    buffer: JByteBuffer,
    byte_count: jint,
) -> jint {
    if session_handle == 0 || byte_count <= 1 {
        return 0;
    }
    let session = &*(session_handle as *mut VadAsrSession);
    if session.closed.load(Ordering::Relaxed) {
        return 0;
    }

    let address_ptr = match env.get_direct_buffer_address(&buffer) {
        Ok(ptr) => ptr,
        Err(_) => return -1,
    };
    if address_ptr.is_null() {
        return -1;
    }

    let sample_count = (byte_count as usize) / 2;
    let samples = std::slice::from_raw_parts(address_ptr as *const i16, sample_count);

    let previous = session.queued_samples.fetch_add(sample_count as u64, Ordering::Relaxed);
    if previous + sample_count as u64 > MAX_QUEUED_SAMPLES {
        session.queued_samples.fetch_sub(sample_count as u64, Ordering::Relaxed);
        return -1;
    }

    let samples_f32: Vec<f32> = samples.iter().map(|&s| (s as f32) / 32768.0).collect();
    if session.task_tx.send(VadMsg::Audio(samples_f32)).is_err() {
        return -1;
    }
    1
}

/// Stop the VAD session. If finalize is true, the current speech segment is flushed.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeStopVadSession(
    _env: JNIEnv,
    _class: JClass,
    session_handle: jlong,
    finalize: jboolean,
) {
    if session_handle == 0 {
        return;
    }

    let session = Box::from_raw(session_handle as *mut VadAsrSession);
    session.closed.store(true, Ordering::Relaxed);
    let _ = session.task_tx.send(VadMsg::Stop {
        finalize: finalize != JNI_FALSE,
    });
    let worker_handle = match session.worker.lock() {
        Ok(mut worker) => worker.take(),
        Err(_) => None,
    };
    if let Some(handle) = worker_handle {
        let _ = handle.join();
    }
    let engine = &mut *(session.engine_addr as *mut crate::c_api::QwenAsrEngine);
    let _ = engine.ctx.set_prompt("");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_stream_constants_match_hill_climbing() {
        let cfg = VadStreamConfig::default();
        assert_eq!(VAD_FRAME_SIZE, 512);
        assert_eq!(VAD_FRAME_SIZE * 1000 / SAMPLE_RATE, 32);
        assert_eq!(INFERENCE_BACKLOG_SAMPLES * 1000 / SAMPLE_RATE as u64, 200);
        assert_eq!(cfg.threshold, 0.50);
        assert_eq!(cfg.neg_threshold, 0.35);
        assert_eq!(cfg.min_silence_ms, 800);
        assert_eq!(cfg.speech_pad_ms, 180);
        assert_eq!(cfg.speech_tail_pad_ms, 100);
        assert_eq!(cfg.stream_chunk_sec, 0.60);
        assert_eq!(cfg.stream_rollback, 2);
        assert!(cfg.adaptive_drain);
    }
}
