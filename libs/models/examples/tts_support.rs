use std::io::Read;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_voice::wav::{StreamingWavWriter, WavSample};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TtsOutput {
    WavFile(PathBuf),
    Stdout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TtsIo {
    pub text: String,
    pub output: TtsOutput,
}

pub fn resolve_text_and_output(text: Option<String>, wav_path: Option<PathBuf>) -> Result<TtsIo> {
    let text = match text {
        Some(text) => normalize_text(text)?,
        None => read_text_from_stdin()?,
    };

    let output = match wav_path {
        Some(path) => TtsOutput::WavFile(path),
        None => TtsOutput::Stdout,
    };

    Ok(TtsIo { text, output })
}

fn read_text_from_stdin() -> Result<String> {
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read synthesis text from stdin")?;
    normalize_text(buffer)
}

fn normalize_text(text: String) -> Result<String> {
    let trimmed = text.trim_end_matches(['\r', '\n']);
    if trimmed.is_empty() {
        bail!("synthesis text is empty");
    }
    Ok(trimmed.to_owned())
}

pub fn log_status(quiet: bool, message: &str) {
    if !quiet {
        eprintln!("{message}");
    }
}

pub enum WavSink<S: WavSample> {
    File {
        writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    },
    Stdout {
        writer: StreamingWavWriter<std::io::StdoutLock<'static>, S>,
    },
    _Marker(std::marker::PhantomData<S>),
}

impl<S: WavSample> WavSink<S> {
    pub fn new(output: &TtsOutput, sample_rate_hz: u32) -> Result<Self> {
        match output {
            TtsOutput::WavFile(path) => {
                let writer = hound::WavWriter::create(
                    path,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: sample_rate_hz,
                        bits_per_sample: S::BITS_PER_SAMPLE,
                        sample_format: S::SAMPLE_FORMAT,
                    },
                )
                .with_context(|| format!("failed to create wav file `{}`", path.display()))?;
                Ok(Self::File { writer })
            }
            TtsOutput::Stdout => {
                let stdout = Box::leak(Box::new(std::io::stdout()));
                let writer = StreamingWavWriter::new(stdout.lock(), sample_rate_hz, 1)
                    .context("failed to start stdout wav stream")?;
                Ok(Self::Stdout { writer })
            }
        }
    }

    pub fn write_chunk(&mut self, samples: &[S]) -> Result<()> {
        match self {
            Self::File { writer } => {
                for &sample in samples {
                    S::write_to_hound(writer, sample)?;
                }
            }
            Self::Stdout { writer } => {
                writer
                    .write_chunk(samples)
                    .context("failed to write stdout wav chunk")?;
            }
            Self::_Marker(_) => unreachable!(),
        }
        Ok(())
    }

    pub fn finalize(self) -> Result<()> {
        match self {
            Self::File { writer } => writer.finalize().context("failed to finalize wav file"),
            Self::Stdout { writer } => writer
                .finalize()
                .context("failed to finalize stdout wav stream"),
            Self::_Marker(_) => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_text_uses_stdin_path_only_when_needed() {
        let io = resolve_text_and_output(Some("hello\n".into()), Some(PathBuf::from("out.wav")))
            .expect("args should resolve");
        assert_eq!(io.text, "hello");
        assert_eq!(io.output, TtsOutput::WavFile(PathBuf::from("out.wav")));
    }
}
