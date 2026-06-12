use std::path::PathBuf;

use anyhow::{Context, Result};
use motlie_model::TranscriptSegment;
use motlie_model::typed::{AudioBuf, Mono};
use motlie_voice::pipeline::convert::{decode_samples_to_f32, downmix_to_mono};
use motlie_voice::pipeline::resample::{LinearInterpolator, Resampler};
use motlie_voice::wav::decode_streaming_wav_to_f32;

use crate::audio_support::{WavInput, open_wav_reader};

pub const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;

pub struct AsrInput {
    pub source: WavInput,
    pub spec: hound::WavSpec,
    pub samples: Vec<f32>,
}

pub fn open_asr_input(wav_path: Option<PathBuf>) -> Result<AsrInput> {
    match wav_path {
        Some(path) => {
            let (source, reader) = open_wav_reader(Some(path.as_path()))?;
            let (spec, samples) =
                decode_samples_to_f32(reader).context("failed to decode wav samples")?;
            Ok(AsrInput {
                source,
                spec,
                samples,
            })
        }
        None => {
            let (spec, samples) = decode_streaming_wav_to_f32(std::io::stdin())
                .context("failed to parse wav stream from stdin")?;
            Ok(AsrInput {
                source: WavInput::Stdin,
                spec,
                samples,
            })
        }
    }
}

pub fn describe_input(source: &WavInput) -> String {
    match source {
        WavInput::File(path) => path.display().to_string(),
        WavInput::Stdin => "<stdin>".into(),
    }
}

pub fn log_status(quiet: bool, message: &str) {
    if !quiet {
        eprintln!("{message}");
    }
}

pub fn render_plain_transcript(segments: &[TranscriptSegment]) -> Option<String> {
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() { None } else { Some(text) }
}

pub fn print_plain_transcript(segments: &[TranscriptSegment]) {
    if let Some(text) = render_plain_transcript(segments) {
        println!("{text}");
    }
}

/// @codex-tts 2026-04-21 -- The generic example path accepts file/stdin WAV
/// sources with runtime sample-rate metadata, so it normalizes through the
/// runtime-rate `motlie_voice` resampler. `motlie_model::typed::
/// I16MonoResampler` remains useful when the input rate is a compile-time
/// invariant, but that does not hold for the generic `--wav`/stdin example UX.
pub fn decode_f32_to_f32_mono16k(
    spec: hound::WavSpec,
    samples: Vec<f32>,
) -> Result<AudioBuf<f32, TARGET_SAMPLE_RATE_HZ, Mono>> {
    let mono = downmix_to_mono(&samples, spec.channels).context("failed to downmix wav to mono")?;
    let resampled = LinearInterpolator
        .resample_f32(&mono, spec.sample_rate, TARGET_SAMPLE_RATE_HZ)
        .context("failed to resample wav to 16 kHz")?;
    Ok(AudioBuf::new(resampled))
}

#[cfg(test)]
mod tests {
    use motlie_model::TranscriptSegment;

    use super::render_plain_transcript;

    #[test]
    fn render_plain_transcript_joins_trimmed_segments() {
        let rendered = render_plain_transcript(&[
            TranscriptSegment {
                start_ms: 0,
                end_ms: 100,
                text: " Hello ".into(),
                confidence: None,
                final_segment: true,
            },
            TranscriptSegment {
                start_ms: 100,
                end_ms: 200,
                text: "world".into(),
                confidence: Some(0.8),
                final_segment: true,
            },
        ]);
        assert_eq!(rendered.as_deref(), Some("Hello world"));
    }

    #[test]
    fn render_plain_transcript_skips_empty_segments() {
        let rendered = render_plain_transcript(&[
            TranscriptSegment {
                start_ms: 0,
                end_ms: 100,
                text: " ".into(),
                confidence: None,
                final_segment: false,
            },
            TranscriptSegment {
                start_ms: 100,
                end_ms: 200,
                text: "\n".into(),
                confidence: None,
                final_segment: true,
            },
        ]);
        assert_eq!(rendered, None);
    }
}
