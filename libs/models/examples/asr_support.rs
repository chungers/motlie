use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::TranscriptSegment;

use crate::audio_support::{WavInput, decode_wav_to_f32, open_wav_reader};

pub struct AsrInput {
    pub source: WavInput,
    pub spec: hound::WavSpec,
    pub samples: Vec<f32>,
}

pub fn open_asr_input(wav_path: Option<PathBuf>) -> Result<AsrInput> {
    match wav_path {
        Some(path) => {
            let (source, reader) = open_wav_reader(Some(path.as_path()))?;
            let (spec, samples) = decode_wav_to_f32(reader)?;
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

fn decode_streaming_wav_to_f32<R: std::io::Read>(
    mut reader: R,
) -> Result<(hound::WavSpec, Vec<f32>)> {
    let mut riff_header = [0u8; 12];
    reader
        .read_exact(&mut riff_header)
        .context("failed to read wav RIFF header")?;
    if &riff_header[0..4] != b"RIFF" {
        bail!("Ill-formed WAVE file: no RIFF tag found");
    }
    if &riff_header[8..12] != b"WAVE" {
        bail!("Ill-formed WAVE file: no WAVE tag found");
    }

    let mut spec = None;
    let mut data = None;

    loop {
        let mut chunk_header = [0u8; 8];
        match reader.read_exact(&mut chunk_header) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error).context("failed to read wav chunk header"),
        }

        let chunk_len = u32::from_le_bytes([
            chunk_header[4],
            chunk_header[5],
            chunk_header[6],
            chunk_header[7],
        ]) as usize;
        match &chunk_header[0..4] {
            b"fmt " => {
                let mut fmt = vec![0u8; chunk_len];
                reader
                    .read_exact(&mut fmt)
                    .context("failed to read wav fmt chunk")?;
                spec = Some(parse_wav_fmt_chunk(&fmt)?);
            }
            b"data" => {
                let mut bytes = Vec::new();
                reader
                    .read_to_end(&mut bytes)
                    .context("failed to read wav data chunk")?;
                data = Some(bytes);
                break;
            }
            _ => {
                let mut skipped = vec![0u8; chunk_len];
                reader.read_exact(&mut skipped).with_context(|| {
                    format!(
                        "failed to skip wav chunk `{}`",
                        String::from_utf8_lossy(&chunk_header[0..4])
                    )
                })?;
            }
        }

        if chunk_len % 2 == 1 {
            let mut padding = [0u8; 1];
            reader
                .read_exact(&mut padding)
                .context("failed to read wav chunk padding")?;
        }
    }

    let spec = spec.context("wav fmt chunk missing")?;
    let data = data.context("wav data chunk missing")?;
    let samples = decode_wav_data_bytes(&spec, &data)?;
    Ok((spec, samples))
}

fn parse_wav_fmt_chunk(bytes: &[u8]) -> Result<hound::WavSpec> {
    if bytes.len() < 16 {
        bail!(
            "wav fmt chunk too short: expected at least 16 bytes, got {}",
            bytes.len()
        );
    }

    let format_code = u16::from_le_bytes([bytes[0], bytes[1]]);
    let channels = u16::from_le_bytes([bytes[2], bytes[3]]);
    let sample_rate = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let bits_per_sample = u16::from_le_bytes([bytes[14], bytes[15]]);
    let sample_format = match format_code {
        1 => hound::SampleFormat::Int,
        3 => hound::SampleFormat::Float,
        other => bail!("unsupported wav format code {other}; expected PCM (1) or float (3)"),
    };

    Ok(hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        sample_format,
    })
}

fn decode_wav_data_bytes(spec: &hound::WavSpec, data: &[u8]) -> Result<Vec<f32>> {
    let sample_width = usize::from(spec.bits_per_sample / 8);
    if sample_width == 0 {
        bail!("invalid wav bits per sample: {}", spec.bits_per_sample);
    }
    if !data.len().is_multiple_of(sample_width) {
        bail!("Failed to read enough bytes.");
    }

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => Ok(data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect()),
        (hound::SampleFormat::Float, 32) => Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        (format, bits) => bail!(
            "unsupported wav sample format {:?} with {} bits per sample",
            format,
            bits
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use motlie_model::TranscriptSegment;

    use super::{decode_streaming_wav_to_f32, render_plain_transcript};

    #[test]
    fn render_plain_transcript_joins_trimmed_segments() {
        let rendered = render_plain_transcript(&[
            TranscriptSegment {
                start_ms: 0,
                end_ms: 100,
                text: " Hello ".into(),
                final_segment: true,
            },
            TranscriptSegment {
                start_ms: 100,
                end_ms: 200,
                text: "world".into(),
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
                final_segment: false,
            },
            TranscriptSegment {
                start_ms: 100,
                end_ms: 200,
                text: "\n".into(),
                final_segment: true,
            },
        ]);
        assert_eq!(rendered, None);
    }

    #[test]
    fn decode_streaming_wav_accepts_aligned_indefinite_data_size() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(u32::MAX - 1).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&22_050u32.to_le_bytes());
        bytes.extend_from_slice(&44_100u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(u32::MAX - 1).to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());
        bytes.extend_from_slice(&16_384i16.to_le_bytes());

        let (spec, samples) =
            decode_streaming_wav_to_f32(Cursor::new(bytes)).expect("streaming wav should decode");
        assert_eq!(spec.sample_rate, 22_050);
        assert_eq!(spec.channels, 1);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 0.5).abs() < 0.0001);
    }
}
