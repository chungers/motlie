use std::path::PathBuf;

use anyhow::Result;
use motlie_model::TranscriptSegment;

use crate::audio_support::{DynWavReader, WavInput, open_wav_reader};

pub struct AsrInput {
    pub source: WavInput,
    pub reader: DynWavReader,
}

pub fn open_asr_input(wav_path: Option<PathBuf>) -> Result<AsrInput> {
    let (source, reader) = open_wav_reader(wav_path.as_deref())?;
    Ok(AsrInput { source, reader })
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
}
