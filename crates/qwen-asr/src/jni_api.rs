use earshot::Detector;
use jni::objects::{GlobalRef, JByteBuffer, JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jfloat, jint, jlong, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;
/// JNI API for Android integration.
/// Build with: cargo ndk -t arm64-v8a build --release --features android
///
/// Java class: com.qwenasr.QAsrEngine
///
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

const VAD_FRAME_SAMPLES: usize = 256;
const SAMPLE_RATE: usize = 16_000;
const VAD_THRESHOLD: f32 = 0.5;
const START_VOICED_FRAMES: usize = 2;
const END_SILENCE_FRAMES: usize = 38; // ~608 ms.
const PRE_ROLL_FRAMES: usize = 19; // ~304 ms.
const POST_ROLL_FRAMES: usize = 13; // ~208 ms.
const MAX_SEGMENT_SAMPLES: usize = SAMPLE_RATE * 12;
const MAX_QUEUED_SAMPLES: u64 = (SAMPLE_RATE * 90) as u64;

struct AsrTask {
    segment_id: u64,
    samples: Vec<i16>,
    is_final: bool,
}

struct VadAsrSession {
    engine_addr: usize,
    vad: Mutex<VadState>,
    task_tx: mpsc::Sender<Option<AsrTask>>,
    queued_samples: Arc<AtomicU64>,
    closed: AtomicBool,
    worker: Mutex<Option<JoinHandle<()>>>,
}

struct VadState {
    detector: Detector,
    frame_buf: Vec<i16>,
    pre_roll: std::collections::VecDeque<[i16; VAD_FRAME_SAMPLES]>,
    active_segment: Vec<i16>,
    triggered: bool,
    voiced_run: usize,
    voiced_frames_in_segment: usize,
    silence_run: usize,
    segment_id: u64,
    samples_received: u64,
    samples_vad_processed: u64,
    samples_enqueued_for_asr: u64,
    queue_overflow: u64,
}

impl VadState {
    fn new() -> Self {
        Self {
            detector: Detector::default(),
            frame_buf: Vec::with_capacity(VAD_FRAME_SAMPLES),
            pre_roll: std::collections::VecDeque::with_capacity(PRE_ROLL_FRAMES),
            active_segment: Vec::with_capacity(SAMPLE_RATE * 4),
            triggered: false,
            voiced_run: 0,
            voiced_frames_in_segment: 0,
            silence_run: 0,
            segment_id: 0,
            samples_received: 0,
            samples_vad_processed: 0,
            samples_enqueued_for_asr: 0,
            queue_overflow: 0,
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
            self.enqueue_segment(task_tx, queued_samples, true);
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
        let score = self.detector.predict_i16(&frame);
        let voiced = score >= VAD_THRESHOLD;

        if !self.triggered {
            self.pre_roll.push_back(frame);
            while self.pre_roll.len() > PRE_ROLL_FRAMES {
                self.pre_roll.pop_front();
            }

            self.voiced_run = if voiced { self.voiced_run + 1 } else { 0 };
            if self.voiced_run >= START_VOICED_FRAMES {
                self.triggered = true;
                self.silence_run = 0;
                self.voiced_frames_in_segment = START_VOICED_FRAMES;
                self.active_segment.clear();
                for old_frame in self.pre_roll.drain(..) {
                    self.active_segment.extend_from_slice(&old_frame);
                }
            }
            return false;
        }

        self.active_segment.extend_from_slice(&frame);
        if voiced {
            self.voiced_frames_in_segment += 1;
            self.silence_run = 0;
        } else {
            self.silence_run += 1;
        }

        let reached_silence = self.silence_run >= END_SILENCE_FRAMES + POST_ROLL_FRAMES;
        let reached_max = self.active_segment.len() >= MAX_SEGMENT_SAMPLES;
        if reached_silence || reached_max || is_final {
            self.enqueue_segment(task_tx, queued_samples, is_final);
            return true;
        }
        false
    }

    fn enqueue_segment(
        &mut self,
        task_tx: &mpsc::Sender<Option<AsrTask>>,
        queued_samples: &AtomicU64,
        is_final: bool,
    ) {
        if self.active_segment.is_empty() {
            return;
        }

        if self.voiced_frames_in_segment < START_VOICED_FRAMES {
            self.reset_segment_state();
            return;
        }

        let mut segment = std::mem::take(&mut self.active_segment);
        let trailing_silence_to_drop = self
            .silence_run
            .saturating_sub(POST_ROLL_FRAMES)
            .saturating_mul(VAD_FRAME_SAMPLES);
        if trailing_silence_to_drop > 0 && trailing_silence_to_drop < segment.len() {
            segment.truncate(segment.len() - trailing_silence_to_drop);
        }

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
                is_final,
            }));
        }

        self.reset_segment_state();
    }

    fn reset_segment_state(&mut self) {
        self.triggered = false;
        self.voiced_run = 0;
        self.voiced_frames_in_segment = 0;
        self.silence_run = 0;
        self.pre_roll.clear();
        self.active_segment.clear();
    }
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
    file.write_all(&1u16.to_le_bytes())?;  // AudioFormat (PCM = 1)
    file.write_all(&1u16.to_le_bytes())?;  // NumChannels (Mono = 1)
    file.write_all(&sample_rate.to_le_bytes())?; // SampleRate
    let byte_rate = sample_rate * 1 * 2;
    file.write_all(&byte_rate.to_le_bytes())?; // ByteRate
    file.write_all(&2u16.to_le_bytes())?;  // BlockAlign
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

fn start_asr_worker(
    engine_addr: usize,
    java_vm: jni::JavaVM,
    listener: GlobalRef,
    rx: mpsc::Receiver<Option<AsrTask>>,
    queued_samples: Arc<AtomicU64>,
    save_audio_dir: Option<String>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let session_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut transcript_prompt = String::new();
        while let Ok(message) = rx.recv() {
            let Some(task) = message else {
                break;
            };

            // Save the segment audio if save_audio_dir is provided
            if let Some(ref dir) = save_audio_dir {
                let wav_path = format!("{}/{}_seg{:03}.wav", dir, session_timestamp, task.segment_id);
                if let Err(e) = write_wav_file(&wav_path, &task.samples, 16000) {
                    eprintln!("qwen-asr worker: failed to save post-VAD audio segment: {}", e);
                } else {
                    eprintln!("qwen-asr worker: saved post-VAD audio segment to {}", wav_path);
                }
            }

            let samples_f32: Vec<f32> =
                task.samples.iter().map(|&s| (s as f32) / 32768.0).collect();
            queued_samples.fetch_sub(task.samples.len() as u64, Ordering::Relaxed);

            let text = unsafe {
                let engine = &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine);
                let old_skip_silence = engine.ctx.skip_silence;
                engine.ctx.skip_silence = false;
                let text = crate::transcribe::transcribe_audio(&mut engine.ctx, &samples_f32);
                engine.ctx.skip_silence = old_skip_silence;
                text
            };

            let Some(text) = text else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }

            transcript_prompt.push_str(text.trim());
            unsafe {
                let engine = &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine);
                let _ = engine.ctx.set_prompt(&transcript_prompt);
            }

            let Ok(mut env) = java_vm.attach_current_thread() else {
                continue;
            };
            let Ok(j_text) = env.new_string(text) else {
                continue;
            };
            let _ = env.call_method(
                listener.as_obj(),
                "onTranscript",
                "(JLjava/lang/String;Z)V",
                &[
                    JValue::Long(task.segment_id as jlong),
                    JValue::Object(&JObject::from(j_text)),
                    JValue::Bool(if task.is_final { JNI_TRUE } else { 0 } as jboolean),
                ],
            );
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
/// fun onTranscript(segmentId: Long, text: String, isFinal: Boolean)
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

    let engine = &mut *(handle as *mut crate::c_api::QwenAsrEngine);
    let _ = engine.ctx.set_prompt("");

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
        java_vm,
        listener_ref,
        rx,
        queued_samples.clone(),
        save_dir_opt,
    );

    let session = VadAsrSession {
        engine_addr: handle as usize,
        vad: Mutex::new(VadState::new()),
        task_tx: tx,
        queued_samples,
        closed: AtomicBool::new(false),
        worker: Mutex::new(Some(worker)),
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
}
