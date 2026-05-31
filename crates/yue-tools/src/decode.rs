//! mp3 -> 16kHz mono f32 decoding via symphonia (lives in this bin, not the lib).

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const TARGET_RATE: i32 = 16000;

/// Decode an audio file (mp3/m4a/...) to 16kHz mono f32 samples.
pub fn decode_to_16k_mono(path: &str) -> Option<Vec<f32>> {
    let file = std::fs::File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?.clone();
    let track_id = track.id;
    let src_rate = track.codec_params.sample_rate? as i32;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;

    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break, // end of stream
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let ch = spec.channels.count().max(1);
                let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buf.copy_interleaved_ref(decoded);
                let s = buf.samples();
                if ch == 1 {
                    mono.extend_from_slice(s);
                } else {
                    for frame in s.chunks(ch) {
                        mono.push(frame.iter().copied().sum::<f32>() / ch as f32);
                    }
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    if mono.is_empty() {
        return None;
    }
    if src_rate != TARGET_RATE {
        mono = qwen_asr::audio::resample(&mono, src_rate, TARGET_RATE);
    }
    Some(mono)
}
