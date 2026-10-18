//! End-to-end Voice Activity Detection (VAD) + Streaming ASR pipeline.
//!
//! Integrates Silero VAD with Qwen3-ASR streaming decoder for real-time
//! speech recognition with automatic speech endpointing and low latency.

use crate::context::QwenCtx;
use crate::transcribe::{self, StreamState};
use silero_vad_rust::RawDetector;
use std::collections::VecDeque;

pub const VAD_FRAME_SIZE: usize = 512; // 32 ms at 16 kHz
pub const SAMPLE_RATE: usize = 16_000;

/// Configuration parameters for the VAD + Streaming ASR pipeline.
#[derive(Debug, Clone)]
pub struct VadStreamConfig {
    /// Speech onset probability threshold (default 0.50).
    pub threshold: f32,
    /// Speech offset probability threshold (default 0.35).
    pub neg_threshold: f32,
    /// Minimum consecutive silence duration in ms to close an utterance (default 400 ms).
    pub min_silence_ms: u32,
    /// Pre-speech lookback audio prepended to speech onset in ms (default 60 ms).
    pub speech_pad_ms: u32,
    /// Post-speech audio padding retained when closing an utterance in ms (default 100 ms).
    pub speech_tail_pad_ms: u32,
    /// Maximum speech duration before forced flush in seconds (default 25.0 s).
    pub max_speech_sec: f32,
    /// Streaming forward chunk size in seconds (default 0.44 s).
    pub stream_chunk_sec: f32,
    /// Number of tokens to roll back per chunk (default 1).
    pub stream_rollback: i32,
    /// Adaptive buffer drain when backlog builds up.
    pub adaptive_drain: bool,
}

impl Default for VadStreamConfig {
    fn default() -> Self {
        Self {
            threshold: 0.50,
            neg_threshold: 0.35,
            min_silence_ms: 800,
            speech_pad_ms: 180,
            speech_tail_pad_ms: 100,
            max_speech_sec: 25.0,
            stream_chunk_sec: 0.40,
            stream_rollback: 2,
            adaptive_drain: true,
        }
    }
}

/// Events produced during streaming VAD processing.
#[derive(Debug, Clone, PartialEq)]
pub enum VadStreamEvent {
    /// Speech onset detected at `audio_ms`.
    SpeechStart { audio_ms: f64 },
    /// Partial text delta produced during streaming.
    TextDelta { delta: String, audio_ms: f64 },
    /// Utterance finalized at silence endpoint or finish.
    UtteranceEnd {
        text: String,
        start_audio_ms: f64,
        end_audio_ms: f64,
        speech_duration_ms: f64,
    },
}

/// End-to-end streaming VAD session state.
pub struct VadStreamSession {
    detector: RawDetector,
    pub config: VadStreamConfig,
    stream_state: StreamState,

    // Audio buffers
    frame_buf: Vec<f32>,
    pre_speech_pad: VecDeque<f32>,
    utterance_audio: Vec<f32>,

    // Tracking
    speech_active: bool,
    temp_silence_sample: Option<usize>,
    total_samples_pushed: usize,
    speech_start_sample: usize,
    last_speech_frame_sample: usize,
    accumulated_full_text: String,
}

impl VadStreamSession {
    /// Create a new session with given configuration.
    pub fn new(config: VadStreamConfig) -> Self {
        let detector = RawDetector::default();
        Self {
            detector,
            config,
            stream_state: StreamState::new(),
            frame_buf: Vec::with_capacity(VAD_FRAME_SIZE),
            pre_speech_pad: VecDeque::new(),
            utterance_audio: Vec::new(),
            speech_active: false,
            temp_silence_sample: None,
            total_samples_pushed: 0,
            speech_start_sample: 0,
            last_speech_frame_sample: 0,
            accumulated_full_text: String::new(),
        }
    }

    /// Reset session state.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.stream_state.reset();
        self.frame_buf.clear();
        self.pre_speech_pad.clear();
        self.utterance_audio.clear();
        self.speech_active = false;
        self.temp_silence_sample = None;
        self.total_samples_pushed = 0;
        self.speech_start_sample = 0;
        self.last_speech_frame_sample = 0;
        self.accumulated_full_text.clear();
    }

    /// Current active speech audio range in milliseconds: (start_ms, end_ms).
    pub fn current_speech_range_ms(&self) -> (f64, f64) {
        if self.speech_active || !self.utterance_audio.is_empty() {
            let start_ms = (self.speech_start_sample as f64 * 1000.0) / SAMPLE_RATE as f64;
            let end_ms = ((self.speech_start_sample + self.utterance_audio.len()) as f64 * 1000.0)
                / SAMPLE_RATE as f64;
            (start_ms, end_ms)
        } else {
            (0.0, (self.total_samples_pushed as f64 * 1000.0) / SAMPLE_RATE as f64)
        }
    }

    /// Push incoming 16kHz mono audio slice.
    ///
    /// The input can be any length (e.g. 160 samples, 512 samples, 1024 samples).
    /// Emits events as speech starts, stream deltas arrive, and utterances end.
    pub fn push_audio<F: FnMut(&str)>(
        &mut self,
        ctx: &mut QwenCtx,
        pcm: &[f32],
        on_delta: F,
    ) -> Vec<VadStreamEvent> {
        self.push_audio_impl(ctx, pcm, false, on_delta)
    }

    /// Push audio while the ASR worker still has newer audio queued.
    ///
    /// VAD and endpointing continue frame-by-frame, but replaceable intermediate
    /// inference is deferred until the worker reaches the newest queued chunk.
    #[cfg(feature = "android")]
    pub(crate) fn push_audio_backlogged<F: FnMut(&str)>(
        &mut self,
        ctx: &mut QwenCtx,
        pcm: &[f32],
        on_delta: F,
    ) -> Vec<VadStreamEvent> {
        self.push_audio_impl(ctx, pcm, true, on_delta)
    }

    fn push_audio_impl<F: FnMut(&str)>(
        &mut self,
        ctx: &mut QwenCtx,
        pcm: &[f32],
        defer_intermediate: bool,
        mut on_delta: F,
    ) -> Vec<VadStreamEvent> {
        let mut events = Vec::new();
        let pad_limit = (self.config.speech_pad_ms as usize * SAMPLE_RATE as usize) / 1000;
        let min_silence_samples =
            (self.config.min_silence_ms as usize * SAMPLE_RATE as usize) / 1000;
        let max_speech_samples =
            (self.config.max_speech_sec * SAMPLE_RATE as usize as f32) as usize;

        // Apply config to context
        ctx.stream_chunk_sec = self.config.stream_chunk_sec;
        ctx.stream_rollback = self.config.stream_rollback;
        ctx.stream_adaptive_drain = self.config.adaptive_drain;

        for &sample in pcm {
            self.total_samples_pushed += 1;
            self.frame_buf.push(sample);

            if self.frame_buf.len() == VAD_FRAME_SIZE {
                let mut frame = [0.0f32; VAD_FRAME_SIZE];
                frame.copy_from_slice(&self.frame_buf);
                self.frame_buf.clear();

                let score = self.detector.predict_f32(&frame).unwrap_or(0.0);
                let current_audio_ms =
                    (self.total_samples_pushed as f64 * 1000.0) / SAMPLE_RATE as usize as f64;

                if !self.speech_active {
                    if score >= self.config.threshold {
                        // ---- Speech Onset ----
                        self.speech_active = true;
                        self.temp_silence_sample = None;
                        self.last_speech_frame_sample = self.total_samples_pushed;

                        let pad_len = self.pre_speech_pad.len();
                        self.speech_start_sample = self
                            .total_samples_pushed
                            .saturating_sub(VAD_FRAME_SIZE + pad_len);

                        eprintln!(
                            "DEBUG: Speech onset at sample {} ({:.3}s), pad_samples={}, score={:.3}",
                            self.total_samples_pushed,
                            self.total_samples_pushed as f64 / 16000.0,
                            self.pre_speech_pad.len(),
                            score
                        );
                        self.utterance_audio.clear();
                        self.utterance_audio.extend(self.pre_speech_pad.drain(..));
                        self.utterance_audio.extend_from_slice(&frame);

                        self.stream_state.reset();

                        events.push(VadStreamEvent::SpeechStart {
                            audio_ms: (self.speech_start_sample as f64 * 1000.0)
                                / SAMPLE_RATE as usize as f64,
                        });

                        // Check if we already have a full chunk
                        let chunk_samples =
                            (self.config.stream_chunk_sec * SAMPLE_RATE as usize as f32) as usize;
                        if !defer_intermediate && self.utterance_audio.len() >= chunk_samples {
                            if let Some(delta) = transcribe::stream_push_audio(
                                ctx,
                                &self.utterance_audio,
                                &mut self.stream_state,
                                false,
                            ) {
                                if !delta.is_empty() {
                                    let c_delta = ctx.convert_chinese(&delta);
                                    on_delta(&c_delta);
                                    events.push(VadStreamEvent::TextDelta {
                                        delta: c_delta,
                                        audio_ms: current_audio_ms,
                                    });
                                }
                            }
                        }
                    } else {
                        // Inactive: maintain pre-speech pad history
                        for s in frame {
                            self.pre_speech_pad.push_back(s);
                        }
                        while self.pre_speech_pad.len() > pad_limit {
                            self.pre_speech_pad.pop_front();
                        }
                    }
                } else {
                    // ---- Speech Active ----
                    self.utterance_audio.extend_from_slice(&frame);

                    if score >= self.config.threshold {
                        self.temp_silence_sample = None;
                        self.last_speech_frame_sample = self.total_samples_pushed;
                    } else if score < self.config.neg_threshold {
                        if self.temp_silence_sample.is_none() {
                            self.temp_silence_sample = Some(self.utterance_audio.len());
                        }
                    }

                    // Check for silence timeout: use punctuation boundary to distinguish
                    // intra-sentence breath pauses from true sentence boundaries.
                    let is_silence_timeout = if let Some(silence_start) = self.temp_silence_sample {
                        let silence_samples =
                            self.utterance_audio.len().saturating_sub(silence_start);
                        if silence_samples >= min_silence_samples {
                            let current_text = self.stream_state.text();
                            let trimmed = current_text.trim_end();
                            let is_sentence_end = trimmed.ends_with('。')
                                || trimmed.ends_with('？')
                                || trimmed.ends_with('！')
                                || trimmed.ends_with('.')
                                || trimmed.ends_with('?')
                                || trimmed.ends_with('!');
                            is_sentence_end || silence_samples >= min_silence_samples * 3
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    let is_max_duration = self.utterance_audio.len() >= max_speech_samples;

                    if is_silence_timeout || is_max_duration {
                        // Finalize utterance on silence timeout or max duration

                        if let Some(delta) = transcribe::stream_push_audio(
                            ctx,
                            &self.utterance_audio,
                            &mut self.stream_state,
                            true,
                        ) {
                            if !delta.is_empty() {
                                let c_delta = ctx.convert_chinese(&delta);
                                on_delta(&c_delta);
                                events.push(VadStreamEvent::TextDelta {
                                    delta: c_delta,
                                    audio_ms: current_audio_ms,
                                });
                            }
                        }

                        let full_text = ctx.convert_chinese(self.stream_state.text().trim());
                        if !full_text.is_empty() {
                            self.accumulated_full_text.push_str(&full_text);
                        }

                        let start_ms =
                            (self.speech_start_sample as f64 * 1000.0) / SAMPLE_RATE as f64;
                        let end_ms = ((self.speech_start_sample + self.utterance_audio.len())
                            as f64
                            * 1000.0)
                            / SAMPLE_RATE as f64;
                        let duration_ms =
                            (self.utterance_audio.len() as f64 * 1000.0) / SAMPLE_RATE as f64;

                        events.push(VadStreamEvent::UtteranceEnd {
                            text: full_text,
                            start_audio_ms: start_ms,
                            end_audio_ms: end_ms,
                            speech_duration_ms: duration_ms,
                        });

                        self.speech_active = false;
                        self.temp_silence_sample = None;
                        self.stream_state.reset();
                        self.utterance_audio.clear();
                    } else {
                        // Stream step if enough new audio is available
                        let chunk_samples =
                            (self.config.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
                        let available = self
                            .utterance_audio
                            .len()
                            .saturating_sub(self.stream_state.audio_cursor());
                        if !defer_intermediate && available >= chunk_samples {
                            if let Some(delta) = transcribe::stream_push_audio(
                                ctx,
                                &self.utterance_audio,
                                &mut self.stream_state,
                                false,
                            ) {
                                if !delta.is_empty() {
                                    let c_delta = ctx.convert_chinese(&delta);
                                    on_delta(&c_delta);
                                    events.push(VadStreamEvent::TextDelta {
                                        delta: c_delta,
                                        audio_ms: current_audio_ms,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        events
    }

    /// Flush remaining audio and finalize current speech segment if active.
    pub fn finish<F: FnMut(&str)>(
        &mut self,
        ctx: &mut QwenCtx,
        mut on_delta: F,
    ) -> Vec<VadStreamEvent> {
        let mut events = Vec::new();
        if !self.speech_active && self.utterance_audio.is_empty() {
            return events;
        }

        // Process any leftover samples in frame_buf
        if !self.frame_buf.is_empty() {
            let mut padded = [0.0f32; VAD_FRAME_SIZE];
            padded[..self.frame_buf.len()].copy_from_slice(&self.frame_buf);
            self.utterance_audio.extend_from_slice(&self.frame_buf);
            self.frame_buf.clear();
        }

        // Acoustic tail flush: append trailing silence pad (e.g. 100ms)
        // to provide proper acoustic decay context for final phonemes
        let tail_pad_samples = (self.config.speech_tail_pad_ms as usize * SAMPLE_RATE) / 1000;
        if tail_pad_samples > 0 && !self.utterance_audio.is_empty() {
            self.utterance_audio
                .resize(self.utterance_audio.len() + tail_pad_samples, 0.0);
        }

        if !self.utterance_audio.is_empty() {
            let current_audio_ms = (self.total_samples_pushed as f64 * 1000.0) / SAMPLE_RATE as f64;
            if let Some(delta) = transcribe::stream_push_audio(
                ctx,
                &self.utterance_audio,
                &mut self.stream_state,
                true,
            ) {
                if !delta.is_empty() {
                    let c_delta = ctx.convert_chinese(&delta);
                    on_delta(&c_delta);
                    events.push(VadStreamEvent::TextDelta {
                        delta: c_delta,
                        audio_ms: current_audio_ms,
                    });
                }
            }

            let full_text = ctx.convert_chinese(self.stream_state.text().trim());
            if !full_text.is_empty() {
                self.accumulated_full_text.push_str(&full_text);
            }

            let start_ms = (self.speech_start_sample as f64 * 1000.0) / SAMPLE_RATE as f64;
            let end_ms = ((self.speech_start_sample + self.utterance_audio.len()) as f64 * 1000.0)
                / SAMPLE_RATE as f64;
            let duration_ms = (self.utterance_audio.len() as f64 * 1000.0) / SAMPLE_RATE as f64;

            events.push(VadStreamEvent::UtteranceEnd {
                text: full_text,
                start_audio_ms: start_ms,
                end_audio_ms: end_ms,
                speech_duration_ms: duration_ms,
            });

            self.speech_active = false;
            self.temp_silence_sample = None;
            self.stream_state.reset();
            self.utterance_audio.clear();
        }

        events
    }

    /// Complete accumulated text across all utterances in this session.
    pub fn full_text(&self) -> &str {
        &self.accumulated_full_text
    }

    /// Latest replaceable hypothesis for the active utterance.
    pub fn current_hypothesis(&self) -> &str {
        self.stream_state.hypothesis()
    }
}

/// Character latency record during streaming.
#[derive(Debug, Clone)]
pub struct StreamCharLatency {
    pub char_index: usize,
    pub character: char,
    /// Position in audio clip when this character became available (ms)
    pub audio_time_ms: f64,
    /// Wall-clock time elapsed since speech onset (ms)
    pub wall_time_since_speech_ms: f64,
}

/// Metrics collected from simulated streaming execution.
#[derive(Debug, Clone)]
pub struct VadStreamMetrics {
    pub text: String,
    pub utterances_count: usize,
    pub speech_detected: bool,
    pub speech_start_audio_ms: f64,
    pub speech_end_audio_ms: f64,
    /// Time To First Character from audio start (ms)
    pub ttfc_from_audio_start_ms: Option<f64>,
    /// Time To First Character from speech start (ms)
    pub ttfc_from_speech_start_ms: Option<f64>,
    /// Tail latency: time from last speech frame to final utterance completion (ms)
    pub tail_latency_ms: f64,
    /// Per-character latency list
    pub char_latencies: Vec<StreamCharLatency>,
    pub total_wall_ms: f64,
    pub real_time_factor: f64,
}

/// Simulate streaming an audio clip frame-by-frame with VAD and collect precise latency metrics.
pub fn simulate_vad_stream(
    ctx: &mut QwenCtx,
    samples: &[f32],
    config: &VadStreamConfig,
) -> VadStreamMetrics {
    let mut session = VadStreamSession::new(config.clone());
    let mut char_latencies = Vec::new();
    let mut total_chars_emitted = 0;
    let mut first_char_audio_ms: Option<f64> = None;
    let mut first_char_speech_ms: Option<f64> = None;
    let mut speech_start_audio_ms = 0.0;
    let mut speech_detected = false;
    let mut utterances_count = 0;

    let t_start = std::time::Instant::now();
    let mut speech_onset_wall_time: Option<std::time::Instant> = None;
    let mut last_speech_audio_ms = 0.0;
    let chunk_size = VAD_FRAME_SIZE;
    for chunk in samples.chunks(chunk_size) {
        let events = session.push_audio(ctx, chunk, |_| {});
        for event in events {
            match event {
                VadStreamEvent::SpeechStart { audio_ms } => {
                    if !speech_detected {
                        speech_detected = true;
                        speech_start_audio_ms = audio_ms;
                        speech_onset_wall_time = Some(std::time::Instant::now());
                    }
                }
                VadStreamEvent::TextDelta { delta, audio_ms } => {
                    last_speech_audio_ms = audio_ms;
                    for c in delta.chars() {
                        if c.is_whitespace() {
                            continue;
                        }
                        total_chars_emitted += 1;
                        let wall_elapsed = speech_onset_wall_time
                            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                            .unwrap_or(0.0);

                        if first_char_audio_ms.is_none() {
                            first_char_audio_ms = Some(audio_ms);
                            first_char_speech_ms = Some(audio_ms - speech_start_audio_ms);
                        }

                        char_latencies.push(StreamCharLatency {
                            char_index: total_chars_emitted,
                            character: c,
                            audio_time_ms: audio_ms,
                            wall_time_since_speech_ms: wall_elapsed,
                        });
                    }
                }
                VadStreamEvent::UtteranceEnd {
                    text: _,
                    start_audio_ms: _,
                    end_audio_ms,
                    speech_duration_ms: _,
                } => {
                    utterances_count += 1;
                    last_speech_audio_ms = end_audio_ms;
                }
            }
        }
    }

    // Finish remaining
    let finish_events = session.finish(ctx, |_| {});
    for event in finish_events {
        match event {
            VadStreamEvent::TextDelta { delta, audio_ms } => {
                last_speech_audio_ms = audio_ms;
                for c in delta.chars() {
                    if c.is_whitespace() {
                        continue;
                    }
                    total_chars_emitted += 1;
                    let wall_elapsed = speech_onset_wall_time
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);

                    if first_char_audio_ms.is_none() {
                        first_char_audio_ms = Some(audio_ms);
                        first_char_speech_ms = Some(audio_ms - speech_start_audio_ms);
                    }

                    char_latencies.push(StreamCharLatency {
                        char_index: total_chars_emitted,
                        character: c,
                        audio_time_ms: audio_ms,
                        wall_time_since_speech_ms: wall_elapsed,
                    });
                }
            }
            VadStreamEvent::UtteranceEnd {
                text: _,
                start_audio_ms: _,
                end_audio_ms,
                speech_duration_ms: _,
            } => {
                utterances_count += 1;
                last_speech_audio_ms = end_audio_ms;
            }
            _ => {}
        }
    }

    let full_text = session.full_text().to_string();
    ctx.set_prompt("").ok();
    let total_wall_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let audio_duration_ms = (samples.len() as f64 * 1000.0) / SAMPLE_RATE as f64;
    let rtf = if audio_duration_ms > 0.0 {
        total_wall_ms / audio_duration_ms
    } else {
        0.0
    };

    let tail_latency_ms = if speech_detected {
        (audio_duration_ms - last_speech_audio_ms).max(0.0)
    } else {
        0.0
    };

    VadStreamMetrics {
        text: full_text,
        utterances_count,
        speech_detected,
        speech_start_audio_ms,
        speech_end_audio_ms: last_speech_audio_ms,
        ttfc_from_audio_start_ms: first_char_audio_ms,
        ttfc_from_speech_start_ms: first_char_speech_ms,
        tail_latency_ms,
        char_latencies,
        total_wall_ms,
        real_time_factor: rtf,
    }
}
