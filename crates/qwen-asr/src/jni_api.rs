use jni::objects::{GlobalRef, JByteBuffer, JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jfloat, jint, jlong, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;
use silero_vad_rust::RawDetector;
/// JNI API for Android integration.
/// Build with: cargo ndk -t arm64-v8a build --release --features android
///
/// Java class: com.qwenasr.QAsrEngine
///
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

const VAD_FRAME_SAMPLES: usize = 512;
const SAMPLE_RATE: usize = 16_000;
const VAD_THRESHOLD: f32 = 0.5;
const VAD_NEG_THRESHOLD: f32 = 0.35;
const MIN_SILENCE_DURATION_MS: usize = 100;
const SPEECH_PAD_MS: usize = 30;
const MIN_SILENCE_SAMPLES: usize = SAMPLE_RATE * MIN_SILENCE_DURATION_MS / 1000;
const SPEECH_PAD_SAMPLES: usize = SAMPLE_RATE * SPEECH_PAD_MS / 1000;
const MAX_SEGMENT_SAMPLES: usize = SAMPLE_RATE * 12;
const MAX_QUEUED_SAMPLES: u64 = (SAMPLE_RATE * 90) as u64;

fn silero_stream_start_sample(current_sample: u64) -> u64 {
    current_sample.saturating_sub((SPEECH_PAD_SAMPLES + VAD_FRAME_SAMPLES) as u64)
}

fn silero_stream_end_sample(temp_end_sample: u64) -> u64 {
    temp_end_sample
        .saturating_add(SPEECH_PAD_SAMPLES as u64)
        .saturating_sub(VAD_FRAME_SAMPLES as u64)
}

struct AsrTask {
    segment_id: u64,
    samples: Vec<i16>,
    start_sample: u64,
    end_sample: u64,
    end_reason: &'static str,
    vad_score_min: f32,
    vad_score_max: f32,
    vad_score_mean: f32,
    vad_frame_count: u64,
}

struct VadAsrSession {
    engine_addr: usize,
    vad: Mutex<VadState>,
    task_tx: mpsc::Sender<Option<AsrTask>>,
    queued_samples: Arc<AtomicU64>,
    closed: AtomicBool,
    worker: Mutex<Option<JoinHandle<()>>>,
    save_audio_dir: Option<String>,
}

struct VadState {
    detector: RawDetector,
    frame_buf: Vec<i16>,
    speech_pad_history: VecDeque<i16>,
    active_segment: Vec<i16>,
    triggered: bool,
    temp_end_sample: Option<u64>,
    current_sample: u64,
    speech_start_sample: u64,
    segment_id: u64,
    samples_received: u64,
    samples_vad_processed: u64,
    samples_enqueued_for_asr: u64,
    queue_overflow: u64,
    active_score_min: f32,
    active_score_max: f32,
    active_score_sum: f64,
    active_score_count: u64,
}

impl VadState {
    fn new() -> Self {
        Self {
            detector: RawDetector::default(),
            frame_buf: Vec::with_capacity(VAD_FRAME_SAMPLES),
            speech_pad_history: VecDeque::with_capacity(SPEECH_PAD_SAMPLES),
            active_segment: Vec::with_capacity(SAMPLE_RATE * 4),
            triggered: false,
            temp_end_sample: None,
            current_sample: 0,
            speech_start_sample: 0,
            segment_id: 0,
            samples_received: 0,
            samples_vad_processed: 0,
            samples_enqueued_for_asr: 0,
            queue_overflow: 0,
            active_score_min: 1.0,
            active_score_max: 0.0,
            active_score_sum: 0.0,
            active_score_count: 0,
        }
    }

    fn push_pcm(
        &mut self,
        pcm: &[i16],
        task_tx: &mpsc::Sender<Option<AsrTask>>,
        queued_samples: &AtomicU64,
    ) -> i32 {
        self.samples_received += pcm.len() as u64;
        let mut emitted = 0;

        for &sample in pcm {
            self.frame_buf.push(sample);
            if self.frame_buf.len() == VAD_FRAME_SAMPLES {
                let mut frame = [0i16; VAD_FRAME_SAMPLES];
                frame.copy_from_slice(&self.frame_buf);
                self.frame_buf.clear();
                if self.process_frame(frame, task_tx, queued_samples, false) {
                    emitted += 1;
                }
            }
        }

        emitted
    }

    fn finalize(
        &mut self,
        task_tx: &mpsc::Sender<Option<AsrTask>>,
        queued_samples: &AtomicU64,
    ) -> i32 {
        if !self.frame_buf.is_empty() {
            let mut frame = [0i16; VAD_FRAME_SAMPLES];
            frame[..self.frame_buf.len()].copy_from_slice(&self.frame_buf);
            self.frame_buf.clear();
            self.process_frame(frame, task_tx, queued_samples, false);
        }
        if !self.active_segment.is_empty() {
            self.enqueue_segment(task_tx, queued_samples, "session_finalized");
            return 1;
        }
        0
    }

    fn process_frame(
        &mut self,
        frame: [i16; VAD_FRAME_SAMPLES],
        task_tx: &mpsc::Sender<Option<AsrTask>>,
        queued_samples: &AtomicU64,
        is_final: bool,
    ) -> bool {
        self.samples_vad_processed += VAD_FRAME_SAMPLES as u64;
        let frame_start_sample = self.current_sample;
        self.current_sample += VAD_FRAME_SAMPLES as u64;

        let score = self.detector.predict_i16(&frame).unwrap_or(0.0);

        if score >= VAD_THRESHOLD && self.temp_end_sample.is_some() {
            self.temp_end_sample = None;
        }

        if score >= VAD_THRESHOLD && !self.triggered {
            self.start_speech(&frame, frame_start_sample, score);
            return false;
        }

        if self.triggered {
            self.active_segment.extend_from_slice(&frame);
            self.record_active_score(score);
        } else {
            self.remember_for_speech_pad(&frame);
            return false;
        }

        let reached_max = self.active_segment.len() >= MAX_SEGMENT_SAMPLES;
        if reached_max || is_final {
            self.enqueue_segment(
                task_tx,
                queued_samples,
                if reached_max {
                    "max_duration"
                } else {
                    "session_finalized"
                },
            );
            self.remember_for_speech_pad(&frame);
            return true;
        }

        if score < VAD_NEG_THRESHOLD {
            let temp_end_sample = *self.temp_end_sample.get_or_insert(self.current_sample);
            if self.current_sample.saturating_sub(temp_end_sample) >= MIN_SILENCE_SAMPLES as u64 {
                let speech_end_sample = silero_stream_end_sample(temp_end_sample);
                self.truncate_active_segment_at(speech_end_sample);
                self.enqueue_segment(task_tx, queued_samples, "silence");
                self.remember_for_speech_pad(&frame);
                return true;
            }
        }

        false
    }

    fn remember_for_speech_pad(&mut self, samples: &[i16]) {
        for &sample in samples {
            self.speech_pad_history.push_back(sample);
            while self.speech_pad_history.len() > SPEECH_PAD_SAMPLES {
                self.speech_pad_history.pop_front();
            }
        }
    }

    fn start_speech(
        &mut self,
        frame: &[i16; VAD_FRAME_SAMPLES],
        frame_start_sample: u64,
        score: f32,
    ) {
        let start_sample = silero_stream_start_sample(self.current_sample);
        let history_samples_to_copy = frame_start_sample.saturating_sub(start_sample) as usize;
        let history_skip = self
            .speech_pad_history
            .len()
            .saturating_sub(history_samples_to_copy);

        self.triggered = true;
        self.temp_end_sample = None;
        self.speech_start_sample = start_sample;
        self.active_segment.clear();
        self.active_segment
            .extend(self.speech_pad_history.iter().skip(history_skip).copied());
        self.active_segment.extend_from_slice(frame);
        self.active_score_min = score_sentinel();
        self.active_score_max = 0.0;
        self.active_score_sum = 0.0;
        self.active_score_count = 0;
        self.record_active_score(score);
    }

    fn record_active_score(&mut self, score: f32) {
        self.active_score_min = self.active_score_min.min(score);
        self.active_score_max = self.active_score_max.max(score);
        self.active_score_sum += score as f64;
        self.active_score_count += 1;
    }

    fn truncate_active_segment_at(&mut self, end_sample: u64) {
        let keep = end_sample.saturating_sub(self.speech_start_sample) as usize;
        if keep < self.active_segment.len() {
            self.active_segment.truncate(keep);
        }
    }

    fn enqueue_segment(
        &mut self,
        task_tx: &mpsc::Sender<Option<AsrTask>>,
        queued_samples: &AtomicU64,
        end_reason: &'static str,
    ) {
        if self.active_segment.is_empty() {
            return;
        }

        let segment = std::mem::take(&mut self.active_segment);

        let n = segment.len() as u64;
        let previous = queued_samples.fetch_add(n, Ordering::Relaxed);
        if previous + n > MAX_QUEUED_SAMPLES {
            queued_samples.fetch_sub(n, Ordering::Relaxed);
            self.queue_overflow += 1;
        } else {
            self.segment_id += 1;
            self.samples_enqueued_for_asr += n;
            let _ = task_tx.send(Some(AsrTask {
                segment_id: self.segment_id,
                samples: segment,
                start_sample: self.speech_start_sample,
                end_sample: self.speech_start_sample + n,
                end_reason,
                vad_score_min: if self.active_score_count == 0 {
                    0.0
                } else {
                    self.active_score_min
                },
                vad_score_max: self.active_score_max,
                vad_score_mean: if self.active_score_count == 0 {
                    0.0
                } else {
                    (self.active_score_sum / self.active_score_count as f64) as f32
                },
                vad_frame_count: self.active_score_count,
            }));
        }

        self.reset_segment_state();
    }

    fn reset_segment_state(&mut self) {
        self.triggered = false;
        self.temp_end_sample = None;
        self.speech_start_sample = 0;
        self.active_segment.clear();
        self.active_score_min = score_sentinel();
        self.active_score_max = 0.0;
        self.active_score_sum = 0.0;
        self.active_score_count = 0;
    }
}

fn score_sentinel() -> f32 {
    1.0
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

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn append_vad_segment_metadata(
    directory: &str,
    task: &AsrTask,
    audio_file: &str,
    transcript: Option<&str>,
) -> std::io::Result<()> {
    let path = format!("{}/vad_segments.jsonl", directory);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let transcript_json = transcript
        .map(json_string)
        .unwrap_or_else(|| "null".to_string());
    writeln!(
        file,
        "{{\"segment_id\":{},\"start_sample\":{},\"end_sample\":{},\"start_ms\":{},\"end_ms\":{},\"sample_count\":{},\"audio_file\":{},\"end_reason\":{},\"vad_score_min\":{:.6},\"vad_score_max\":{:.6},\"vad_score_mean\":{:.6},\"vad_frame_count\":{},\"transcript\":{},\"asr_success\":{}}}",
        task.segment_id,
        task.start_sample,
        task.end_sample,
        task.start_sample * 1000 / SAMPLE_RATE as u64,
        task.end_sample * 1000 / SAMPLE_RATE as u64,
        task.samples.len(),
        json_string(audio_file),
        json_string(task.end_reason),
        task.vad_score_min,
        task.vad_score_max,
        task.vad_score_mean,
        task.vad_frame_count,
        transcript_json,
        transcript.is_some(),
    )
}

fn write_vad_session_metadata(directory: &str, vad: &VadState) -> std::io::Result<()> {
    let path = format!("{}/vad_session.json", directory);
    let json = format!(
        "{{\n  \"sample_rate_hz\": {},\n  \"frame_samples\": {},\n  \"positive_threshold\": {},\n  \"negative_threshold\": {},\n  \"minimum_silence_ms\": {},\n  \"speech_padding_ms\": {},\n  \"maximum_segment_samples\": {},\n  \"samples_received\": {},\n  \"samples_vad_processed\": {},\n  \"samples_enqueued_for_asr\": {},\n  \"segment_count\": {},\n  \"queue_overflow_count\": {}\n}}\n",
        SAMPLE_RATE,
        VAD_FRAME_SAMPLES,
        VAD_THRESHOLD,
        VAD_NEG_THRESHOLD,
        MIN_SILENCE_DURATION_MS,
        SPEECH_PAD_MS,
        MAX_SEGMENT_SAMPLES,
        vad.samples_received,
        vad.samples_vad_processed,
        vad.samples_enqueued_for_asr,
        vad.segment_id,
        vad.queue_overflow,
    );
    fs::write(path, json)
}

fn start_asr_worker(
    engine_addr: usize,
    java_vm: Arc<jni::JavaVM>,
    listener: GlobalRef,
    rx: mpsc::Receiver<Option<AsrTask>>,
    queued_samples: Arc<AtomicU64>,
    save_audio_dir: Option<String>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        while let Ok(message) = rx.recv() {
            let Some(task) = message else {
                break;
            };

            // Save the segment audio if save_audio_dir is provided
            if let Some(ref dir) = save_audio_dir {
                let wav_path = format!("{}/model_input_seg{:03}.wav", dir, task.segment_id);
                if let Err(e) = write_wav_file(&wav_path, &task.samples, 16000) {
                    eprintln!(
                        "qwen-asr worker: failed to save post-VAD audio segment: {}",
                        e
                    );
                } else {
                    eprintln!(
                        "qwen-asr worker: saved post-VAD audio segment to {}",
                        wav_path
                    );
                }
            }

            let samples_f32: Vec<f32> =
                task.samples.iter().map(|&s| (s as f32) / 32768.0).collect();
            queued_samples.fetch_sub(task.samples.len() as u64, Ordering::Relaxed);

            let callback_vm = java_vm.clone();
            let callback_listener = listener.clone();
            let text = unsafe {
                let engine = &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine);
                let old_skip_silence = engine.ctx.skip_silence;
                engine.ctx.skip_silence = false;
                engine.ctx.token_cb = Some(Box::new(move |delta| {
                    let Ok(mut env) = callback_vm.attach_current_thread() else {
                        return;
                    };
                    let Ok(j_delta) = env.new_string(delta) else {
                        return;
                    };
                    let _ = env.call_method(
                        callback_listener.as_obj(),
                        "onTranscriptDelta",
                        "(Ljava/lang/String;)V",
                        &[JValue::Object(&JObject::from(j_delta))],
                    );
                }));
                let text = crate::transcribe::transcribe_audio(&mut engine.ctx, &samples_f32);
                engine.ctx.token_cb = None;
                engine.ctx.skip_silence = old_skip_silence;
                text
            };

            if let Some(ref dir) = save_audio_dir {
                let audio_file = format!("model_input_seg{:03}.wav", task.segment_id);
                if let Err(error) =
                    append_vad_segment_metadata(dir, &task, &audio_file, text.as_deref())
                {
                    eprintln!("qwen-asr worker: failed to save VAD metadata: {}", error);
                }
            }

            let Ok(mut env) = java_vm.attach_current_thread() else {
                continue;
            };
            let method = if text.is_some() {
                "onTranscriptComplete"
            } else {
                "onTranscriptError"
            };
            let _ = env.call_method(listener.as_obj(), method, "()V", &[]);
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
        vad: Mutex::new(VadState::new()),
        task_tx: tx,
        queued_samples,
        closed: AtomicBool::new(false),
        worker: Mutex::new(Some(worker)),
        save_audio_dir: save_dir_opt,
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
    match session.vad.lock() {
        Ok(mut vad) => vad.push_pcm(samples, &session.task_tx, &session.queued_samples),
        Err(_) => -1,
    }
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
    let engine = &mut *(session.engine_addr as *mut crate::c_api::QwenAsrEngine);
    let _ = engine.ctx.set_prompt("");
    if finalize != 0 {
        if let Ok(mut vad) = session.vad.lock() {
            vad.finalize(&session.task_tx, &session.queued_samples);
        }
    }
    let _ = session.task_tx.send(None);
    let worker_handle = match session.worker.lock() {
        Ok(mut worker) => worker.take(),
        Err(_) => None,
    };
    if let Some(handle) = worker_handle {
        let _ = handle.join();
    }
    if let Some(ref directory) = session.save_audio_dir {
        if let Ok(vad) = session.vad.lock() {
            if let Err(error) = write_vad_session_metadata(directory, &vad) {
                eprintln!("qwen-asr: failed to save VAD session metadata: {}", error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silero_stream_uses_512_sample_frames() {
        assert_eq!(VAD_FRAME_SAMPLES, 512);
        assert_eq!(VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE, 32);
    }

    #[test]
    fn silero_stream_start_matches_vad_iterator_padding() {
        assert_eq!(SPEECH_PAD_SAMPLES, 480);

        // First frame: upstream VADIterator saturates start at 0.
        assert_eq!(silero_stream_start_sample(512), 0);

        // Later frames: start = current_sample - speech_pad - window_size.
        assert_eq!(silero_stream_start_sample(2048), 1056);
    }

    #[test]
    fn silero_stream_end_matches_vad_iterator_padding() {
        // Upstream VADIterator reports end = temp_end + speech_pad - window_size.
        assert_eq!(silero_stream_end_sample(4096), 4064);
    }
}
