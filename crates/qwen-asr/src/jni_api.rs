#[cfg(test)]
use crate::vad_stream::VAD_FRAME_SIZE;
/// JNI API for Android integration.
/// Build with: cargo ndk -t arm64-v8a build --release --features android
///
/// Java class: com.qwenasr.QAsrEngine
///
use crate::vad_stream::{VadStreamConfig, VadStreamEvent, VadStreamSession, SAMPLE_RATE};
use jni::objects::{GlobalRef, JByteBuffer, JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jfloat, jint, jlong, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;
use std::ffi::{CStr, CString};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

// The Android app supplies its build flag; the shared optimized Rust library
// cannot use debug_assertions to distinguish debug and release APKs.
static DEBUG_APP: AtomicBool = AtomicBool::new(false);

// Opt in on a debug APK with: adb shell setprop debug.qwen_asr.latency 1
#[cfg(target_os = "android")]
fn debug_property(name: &str) -> String {
    unsafe extern "C" {
        fn __system_property_get(
            name: *const std::ffi::c_char,
            value: *mut std::ffi::c_char,
        ) -> i32;
    }
    let name = CString::new(name).expect("static property name");
    let mut value = [0 as std::ffi::c_char; 92];
    unsafe {
        __system_property_get(name.as_ptr(), value.as_mut_ptr());
        CStr::from_ptr(value.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}
#[cfg(not(target_os = "android"))]
fn debug_property(_name: &str) -> String {
    String::new()
}

fn diagnostics_enabled(debug_app: bool, opt_in: &str) -> bool {
    debug_app && opt_in == "1"
}

fn latency_enabled() -> bool {
    let debug_app = DEBUG_APP.load(Ordering::Relaxed);
    debug_app && diagnostics_enabled(debug_app, &debug_property("debug.qwen_asr.latency"))
}

// Thread scopes attribute work to that thread. Process scope includes every app
// thread and overlaps those scopes: never add the two kinds of counters together.
#[derive(Clone, Copy)]
struct Usage { cpu_ms: f64, minor: i64, major: i64 }

fn resource_usage(process: bool) -> Option<Usage> {
    #[cfg(target_os = "android")]
    {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let who = if process { libc::RUSAGE_SELF } else { 1 /* Bionic RUSAGE_THREAD */ };
        if unsafe { libc::getrusage(who, &mut usage) } != 0 { return None; }
        let ms = |t: libc::timeval| t.tv_sec as f64 * 1000.0 + t.tv_usec as f64 / 1000.0;
        Some(Usage { cpu_ms: ms(usage.ru_utime) + ms(usage.ru_stime),
            minor: usage.ru_minflt, major: usage.ru_majflt })
    }
    #[cfg(not(target_os = "android"))]
    { let _ = process; None }
}

fn storage_read_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/io").ok()?.lines()
        .find_map(|line| line.strip_prefix("read_bytes:")?.trim().parse().ok())
}

thread_local! { static TRACE_SESSION: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

pub(crate) struct ResourceProbe {
    stage: &'static str,
    session_id: u64,
    process: bool,
    start: Instant,
    usage: Option<Usage>,
    read_bytes: Option<u64>,
}

impl ResourceProbe {
    pub(crate) fn start(enabled: bool, stage: &'static str) -> Option<Self> {
        if !enabled { return None; }
        Self::with_scope(true, stage, TRACE_SESSION.with(|id| id.get()), false)
    }

    fn with_scope(enabled: bool, stage: &'static str, session_id: u64, process: bool) -> Option<Self> {
        if !enabled { return None; }
        Some(Self { stage, session_id, process, start: Instant::now(),
            usage: resource_usage(process),
            read_bytes: if process { storage_read_bytes() } else { None } })
    }
}

impl Drop for ResourceProbe {
    fn drop(&mut self) {
        let end = resource_usage(self.process);
        let reads = if self.process { storage_read_bytes() } else { None };
        let counters = match (self.usage, end) {
            (Some(a), Some(b)) => format!("valid=true cpu_ms={:.2} minor_faults={} major_faults={}",
                b.cpu_ms-a.cpu_ms, b.minor-a.minor, b.major-a.major),
            _ => "valid=false".to_owned(),
        };
        let io = match (self.read_bytes, reads) {
            (Some(a), Some(b)) => format!(" read_bytes={}", b.saturating_sub(a)),
            _ => String::new(),
        };
        latency_log(&format!("event=resources session_id={} stage={} scope={} wall_ms={:.2} {}{}",
            self.session_id, self.stage, if self.process { "process" } else { "thread" },
            self.start.elapsed().as_secs_f64()*1000.0, counters, io));
    }
}

fn log_prefetch(stage: &str, session_id: u64, stats: crate::safetensors::PrefetchStats) {
    latency_log(&format!("event=prefetch_advice stage={} session_id={} bytes={} populate_ok={} populate_failed={} fallback_ok={} fallback_failed={} first_errno={} pages={} resident_before={} resident_after={} mincore_failed={}",
        stage, session_id, stats.bytes, stats.populate_ok, stats.populate_failed,
        stats.fallback_ok, stats.fallback_failed, stats.first_errno, stats.pages,
        stats.resident_before, stats.resident_after, stats.mincore_failed));
}

fn log_kernel_deltas(before: &[crate::kernels::ProfileCounterSnapshot]) {
    for (before, after) in before.iter().zip(crate::kernels::profile_snapshot()) {
        let elapsed = after.total_ms - before.total_ms;
        if elapsed >= 0.1 {
            latency_log(&format!(
                "event=kernel name={} ms={:.2} calls={}",
                after.name,
                elapsed,
                after.calls - before.calls
            ));
        }
    }
}

pub(crate) fn latency_log(message: &str) {
    if !DEBUG_APP.load(Ordering::Relaxed) { return; }
    #[cfg(target_os = "android")]
    {
        #[link(name = "log")]
        unsafe extern "C" {
            fn __android_log_write(
                priority: i32,
                tag: *const std::ffi::c_char,
                text: *const std::ffi::c_char,
            ) -> i32;
        }
        if let Ok(message) = CString::new(message) {
            unsafe {
                __android_log_write(4, c"VoiceNative".as_ptr(), message.as_ptr());
            }
        }
    }
    #[cfg(not(target_os = "android"))]
    let _ = message;
}

fn phase_times(ctx: &crate::context::QwenCtx) -> [f64; 4] {
    [
        ctx.perf_mel_ms,
        ctx.perf_encoder_ms,
        ctx.perf_prefill_ms,
        ctx.perf_decode_loop_ms,
    ]
}

const MAX_QUEUED_SAMPLES: u64 = (SAMPLE_RATE * 90) as u64;
const INFERENCE_BACKLOG_SAMPLES: u64 = (SAMPLE_RATE as u64 * 800) / 1000;

enum VadMsg {
    Audio {
        samples: Vec<f32>,
        enqueued_at: Instant,
    },
    Stop {
        finalize: bool,
    },
}

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

struct VadAsrSession {
    session_id: u64,
    engine_addr: usize,
    task_tx: mpsc::Sender<VadMsg>,
    queued_samples: Arc<AtomicU64>,
    closed: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    _save_audio_dir: Option<String>,
    _resources: Option<ResourceProbe>,
    _capture_resources: Option<ResourceProbe>,
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
    session_id: u64,
    engine_addr: usize,
    java_vm: Arc<jni::JavaVM>,
    listener: GlobalRef,
    rx: mpsc::Receiver<VadMsg>,
    queued_samples: Arc<AtomicU64>,
    save_audio_dir: Option<String>,
    closed: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let trace = latency_enabled();
        TRACE_SESSION.with(|id| id.set(session_id));
        let _worker_resources = ResourceProbe::start(trace, "worker_total");
        let session_at = Instant::now();
        // set_threads() runs on the model-loading coroutine, not this JNI worker.
        // Single-thread kernels bypass parallel_for(), so explicitly apply the same
        // existing inference CPU policy here as the CLI and pool workers use.
        #[cfg(feature = "kleidiai")]
        crate::kernels::set_kleidiai_matvec_enabled(true);
        crate::affinity::ensure_current_thread_pinned();
        if trace {
            crate::kernels::set_profile(true);
            latency_log(&format!(
                "event=worker_policy session_id={} pin={} threads={} status={}",
                session_id,
                true,
                crate::kernels::get_num_threads(),
                std::fs::read_to_string("/proc/thread-self/status")
                    .unwrap_or_default()
                    .lines()
                    .find(|line| line.starts_with("Cpus_allowed_list:"))
                    .unwrap_or("")
                    .trim()
            ));
        }
        {
            let engine = unsafe { &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine) };
            engine.ctx.stream_latency_trace = trace;
            engine.ctx.stream_stop = Some(closed.clone());
        }
        let mut processed_samples = 0usize;
        let mut max_queue_ms = 0.0f64;
        let vad_config = VadStreamConfig::default();
        let init_resources = ResourceProbe::start(trace, "vad_init");
        let mut vad_session = VadStreamSession::new(vad_config);
        drop(init_resources);
        let mut all_audio_samples: Vec<i16> = Vec::new();
        let weights = unsafe { &*(engine_addr as *const crate::c_api::QwenAsrEngine) }.ctx._safetensors.clone();
        let prefetch_handle = thread::Builder::new()
            .name(format!("qwen-prefetch-{}", session_id))
            .spawn(move || {
                TRACE_SESSION.with(|id| id.set(session_id));
                let _resources = ResourceProbe::start(trace, "decoder_preload");
                let preload_at = Instant::now();
                if trace {
                    log_prefetch("decoder", session_id, weights.preload_decoder_profiled());
                } else {
                    weights.preload_decoder();
                }
                if trace {
                    latency_log(&format!(
                        "event=decoder_preload session_id={} ms={:.2}",
                        session_id,
                        preload_at.elapsed().as_secs_f64() * 1000.0
                    ));
                }
            })
            .ok();
        let mut last_hypothesis = String::new();
        let mut pass_id = 0u32;
        if trace {
            latency_log(&format!(
                "event=worker_ready session_id={} wall_ms={:.2}",
                session_id,
                session_at.elapsed().as_secs_f64() * 1000.0
            ));
        }

        while let Ok(msg) = rx.recv() {
            match msg {
                VadMsg::Audio {
                    samples: samples_f32,
                    enqueued_at,
                } => {
                    let queue_ms = enqueued_at.elapsed().as_secs_f64() * 1000.0;
                    max_queue_ms = max_queue_ms.max(queue_ms);
                    let call_at = Instant::now();
                    let usage_before = if trace { resource_usage(false) } else { None };
                    let kernels_before = if trace {
                        crate::kernels::profile_snapshot()
                    } else {
                        Vec::new()
                    };
                    let engine = unsafe { &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine) };
                    queued_samples.fetch_sub(samples_f32.len() as u64, Ordering::Relaxed);
                    if save_audio_dir.is_some() {
                        for &s in &samples_f32 {
                            all_audio_samples.push((s * 32767.0).clamp(-32768.0, 32767.0) as i16);
                        }
                    }

                    // Once Stop is queued, feed every remaining sample into VAD but
                    // do not run a replaceable preview immediately before finalizing
                    // the same utterance again. finish() owns the one final decode.
                    let is_stopped = closed.load(Ordering::Acquire);
                    let has_audio_backlog = is_stopped
                        || queued_samples.load(Ordering::Relaxed) >= INFERENCE_BACKLOG_SAMPLES;
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
                    let phases_before = phase_times(&engine.ctx);
                    let events = if has_audio_backlog {
                        vad_session.push_audio_backlogged(&mut engine.ctx, &samples_f32, on_delta)
                    } else {
                        vad_session.push_audio(&mut engine.ctx, &samples_f32, on_delta)
                    };

                    processed_samples += samples_f32.len();
                    let phases_after = phase_times(&engine.ctx);
                    let mut audio_start_ms = 0.0f64;
                    let mut audio_end_ms = 0.0f64;
                    let pass_executed = phases_after != phases_before;
                    if pass_executed {
                        pass_id += 1;
                        let range = vad_session.current_speech_range_ms();
                        audio_start_ms = range.0;
                        audio_end_ms = range.1;
                        if is_stopped && trace {
                            latency_log(&format!(
                                "event=preview_end session_id={} pass_id={} wall_ms={:.2}",
                                session_id,
                                pass_id,
                                session_at.elapsed().as_secs_f64() * 1000.0
                            ));
                        }
                    }
                    if trace && (call_at.elapsed().as_millis() >= 10 || pass_executed) {
                        if let (Some(before), Some(after)) = (usage_before, resource_usage(false)) {
                            latency_log(&format!(
                                "event=thread_work session_id={} cpu_ms={:.2} minor_faults={} major_faults={}",
                                session_id, after.cpu_ms-before.cpu_ms,
                                after.minor-before.minor, after.major-before.major
                            ));
                        }
                        log_kernel_deltas(&kernels_before);
                        latency_log(&format!("event=process session_id={} pass_id={} audio_start_ms={:.2} audio_end_ms={:.2} wall_ms={:.2} audio_ms={:.2} queue_wait_ms={:.2} backlog_ms={:.2} deferred={} process_ms={:.2} mel_ms={:.2} encoder_ms={:.2} prefill_ms={:.2} decode_ms={:.2}",
                            session_id, pass_id, audio_start_ms, audio_end_ms,
                            session_at.elapsed().as_secs_f64()*1000.0, processed_samples as f64 / 16.0, queue_ms,
                            queued_samples.load(Ordering::Relaxed) as f64 / 16.0, has_audio_backlog,
                            call_at.elapsed().as_secs_f64()*1000.0,
                            phases_after[0]-phases_before[0], phases_after[1]-phases_before[1],
                            phases_after[2]-phases_before[2], phases_after[3]-phases_before[3]));
                    }
                    if trace {
                        for event in &events {
                            match event {
                                VadStreamEvent::SpeechStart { audio_ms } => latency_log(&format!(
                                    "event=speech_start session_id={} wall_ms={:.2} audio_ms={:.2}",
                                    session_id,
                                    session_at.elapsed().as_secs_f64() * 1000.0,
                                    audio_ms
                                )),
                                VadStreamEvent::UtteranceEnd { end_audio_ms, .. } => {
                                    latency_log(&format!(
                                        "event=endpoint session_id={} wall_ms={:.2} speech_end_audio_ms={:.2}",
                                        session_id,
                                        session_at.elapsed().as_secs_f64() * 1000.0,
                                        end_audio_ms
                                    ))
                                }
                                _ => {}
                            }
                        }
                    }
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
                    let stop_at = Instant::now();
                    let _resources = ResourceProbe::start(trace, "finalize");
                    if trace {
                        latency_log(&format!(
                            "event=queued_audio_drained session_id={} wall_ms={:.2}",
                            session_id,
                            session_at.elapsed().as_secs_f64() * 1000.0
                        ));
                        latency_log(&format!(
                            "event=native_stop_received session_id={} finalize={} wall_ms={:.2}",
                            session_id,
                            finalize,
                            session_at.elapsed().as_secs_f64() * 1000.0
                        ));
                    }
                    let kernels_before = if trace {
                        crate::kernels::profile_snapshot()
                    } else {
                        Vec::new()
                    };
                    let phases_before = phase_times(&engine.ctx);
                    let mut audio_start_ms = 0.0f64;
                    let mut audio_end_ms = 0.0f64;
                    if finalize {
                        pass_id += 1;
                        let range = vad_session.current_speech_range_ms();
                        audio_start_ms = range.0;
                        audio_end_ms = range.1;
                        if trace {
                            latency_log(&format!(
                                "event=final_inference_start session_id={} pass_id={} wall_ms={:.2}",
                                session_id,
                                pass_id,
                                session_at.elapsed().as_secs_f64() * 1000.0
                            ));
                        }
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
                        if trace {
                            latency_log(&format!(
                                "event=final_inference_end session_id={} pass_id={} wall_ms={:.2}",
                                session_id,
                                pass_id,
                                session_at.elapsed().as_secs_f64() * 1000.0
                            ));
                        }
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
                    if trace {
                        log_kernel_deltas(&kernels_before);
                        let after = phase_times(&engine.ctx);
                        latency_log(&format!("event=finish session_id={} pass_id={} audio_start_ms={:.2} audio_end_ms={:.2} wall_ms={:.2} finalize_ms={:.2} max_queue_wait_ms={:.2} mel_ms={:.2} encoder_ms={:.2} prefill_ms={:.2} decode_ms={:.2}",
                            session_id, pass_id, audio_start_ms, audio_end_ms,
                            session_at.elapsed().as_secs_f64()*1000.0, stop_at.elapsed().as_secs_f64()*1000.0, max_queue_ms,
                            after[0]-phases_before[0], after[1]-phases_before[1], after[2]-phases_before[2], after[3]-phases_before[3]));
                    }
                    if let Some(ref dir) = save_audio_dir {
                        if !all_audio_samples.is_empty() {
                            let wav_path = format!("{}/session_audio.wav", dir);
                            if let Err(e) = write_wav_file(&wav_path, &all_audio_samples, 16000) {
                                eprintln!("qwen-asr: failed to save session audio: {}", e);
                            }
                        }
                    }
                    if trace {
                        crate::kernels::set_profile(false);
                    }
                    break;
                }
            }
        }
        unsafe { &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine) }.ctx.stream_stop = None;
        if let Some(handle) = prefetch_handle {
            let _ = handle.join();
        }
        let engine = unsafe { &mut *(engine_addr as *mut crate::c_api::QwenAsrEngine) };
        engine.ctx.stream_latency_trace = false;
    })
}

/// Runtime build gate, independent of the Rust library's optimization profile.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativeConfigureDiagnostics(
    _env: JNIEnv,
    _class: JClass,
    debug_app: jboolean,
) -> jboolean {
    DEBUG_APP.store(debug_app == JNI_TRUE, Ordering::Relaxed);
    if latency_enabled() { JNI_TRUE } else { JNI_FALSE }
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

    // Test-only cache control on this app's model files, before they are mapped.
    // DONTNEED is advisory: residency/read counters must confirm the outcome.
    #[cfg(target_os = "android")]
    if latency_enabled() && debug_property("debug.qwen_asr.cold_cache") == "1" {
        use std::os::fd::AsRawFd;
        if let Ok(entries) = fs::read_dir(&dir_str) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|ext| ext == "safetensors") {
                    if let Ok(file) = fs::File::open(entry.path()) {
                        let result = libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
                        latency_log(&format!("event=cache_control file={} result={}", entry.file_name().to_string_lossy(), result));
                    }
                }
            }
        }
    }
    let _resources = ResourceProbe::with_scope(latency_enabled(), "model_load", 0, true);
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

/// Preload Shard 1 (Audio Encoder: 320 MB) into RAM using MADV_WILLNEED.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativePreloadEncoder(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        let engine = &mut *(handle as *mut crate::c_api::QwenAsrEngine);
        engine.capture_resources = ResourceProbe::with_scope(latency_enabled(), "capture_total", 0, true);
        let trace = latency_enabled();
        let _resources = ResourceProbe::with_scope(trace, "encoder_preload_api", 0, true);
        if trace {
            log_prefetch("encoder_api", 0, engine.ctx._safetensors.preload_encoder_profiled());
        } else { engine.ctx.preload_encoder(); }
    }
}

/// Preload Shards 2..N (LLM Decoder: 1.64 GB) into RAM using MADV_WILLNEED.
#[no_mangle]
pub unsafe extern "system" fn Java_com_qwenasr_QAsrEngine_nativePreloadDecoder(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        let engine = &mut *(handle as *mut crate::c_api::QwenAsrEngine);
        engine.ctx.preload_decoder();
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

    let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
    let resources = ResourceProbe::with_scope(latency_enabled(), "session_total", session_id, true);
    if latency_enabled() {
        latency_log(&format!("event=session_created session_id={}", session_id));
    }
    // Immediately trigger eager readahead of Shard 1 (Encoder: 320 MB)
    let engine = &mut *(handle as *mut crate::c_api::QwenAsrEngine);
    {
        let _resources = ResourceProbe::with_scope(latency_enabled(), "encoder_preload", session_id, false);
        engine.ctx.preload_encoder();
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
    let closed = Arc::new(AtomicBool::new(false));
    let mut capture_resources = engine.capture_resources.take();
    if let Some(resources) = &mut capture_resources { resources.session_id = session_id; }
    let worker = start_asr_worker(
        session_id,
        handle as usize,
        Arc::new(java_vm),
        listener_ref,
        rx,
        queued_samples.clone(),
        save_dir_opt.clone(),
        closed.clone(),
    );

    let session = VadAsrSession {
        session_id,
        engine_addr: handle as usize,
        task_tx: tx,
        queued_samples,
        closed,
        worker: Mutex::new(Some(worker)),
        _save_audio_dir: save_dir_opt,
        _resources: resources,
        _capture_resources: capture_resources,
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

    let previous = session
        .queued_samples
        .fetch_add(sample_count as u64, Ordering::Relaxed);
    if previous + sample_count as u64 > MAX_QUEUED_SAMPLES {
        session
            .queued_samples
            .fetch_sub(sample_count as u64, Ordering::Relaxed);
        return -1;
    }

    let samples_f32: Vec<f32> = samples.iter().map(|&s| (s as f32) / 32768.0).collect();
    if session
        .task_tx
        .send(VadMsg::Audio {
            samples: samples_f32,
            enqueued_at: Instant::now(),
        })
        .is_err()
    {
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
    let trace = latency_enabled();
    if trace {
        latency_log(&format!(
            "event=native_stop_call session_id={}",
            session.session_id
        ));
    }
    session.closed.store(true, Ordering::Release);
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
    if trace {
        latency_log(&format!(
            "event=native_worker_joined session_id={}",
            session.session_id
        ));
    }
    let engine = &mut *(session.engine_addr as *mut crate::c_api::QwenAsrEngine);
    let _ = engine.ctx.set_prompt("");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_require_debug_app_and_explicit_opt_in() {
        for debug_app in [false, true] {
            for flag in ["", "0", "1", "true"] {
                assert_eq!(diagnostics_enabled(debug_app, flag), debug_app && flag == "1");
            }
        }
    }

    #[test]
    fn vad_stream_constants_match_hill_climbing() {
        let cfg = VadStreamConfig::default();
        assert_eq!(VAD_FRAME_SIZE, 512);
        assert_eq!(VAD_FRAME_SIZE * 1000 / SAMPLE_RATE, 32);
        assert_eq!(INFERENCE_BACKLOG_SAMPLES * 1000 / SAMPLE_RATE as u64, 800);
        assert_eq!(cfg.threshold, 0.50);
        assert_eq!(cfg.neg_threshold, 0.35);
        assert_eq!(cfg.min_silence_ms, 800);
        assert_eq!(cfg.speech_pad_ms, 180);
        assert_eq!(cfg.speech_tail_pad_ms, 100);
        assert_eq!(cfg.stream_chunk_sec, 0.40);
        assert_eq!(cfg.stream_rollback, 2);
        assert!(cfg.adaptive_drain);
    }
}
