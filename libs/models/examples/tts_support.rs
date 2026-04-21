use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

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

pub trait WavSample: Copy {
    const SAMPLE_FORMAT: hound::SampleFormat;
    const BITS_PER_SAMPLE: u16;
    const BYTES_PER_SAMPLE: u32;

    fn write_to_hound<W: Write + std::io::Seek>(
        writer: &mut hound::WavWriter<W>,
        sample: Self,
    ) -> Result<()>;
    fn write_to_stream<W: Write>(writer: &mut W, sample: Self) -> Result<()>;
}

impl WavSample for i16 {
    const SAMPLE_FORMAT: hound::SampleFormat = hound::SampleFormat::Int;
    const BITS_PER_SAMPLE: u16 = 16;
    const BYTES_PER_SAMPLE: u32 = 2;

    fn write_to_hound<W: Write + std::io::Seek>(
        writer: &mut hound::WavWriter<W>,
        sample: Self,
    ) -> Result<()> {
        writer
            .write_sample(sample)
            .context("failed to write wav sample")
    }

    fn write_to_stream<W: Write>(writer: &mut W, sample: Self) -> Result<()> {
        writer
            .write_all(&sample.to_le_bytes())
            .context("failed to write wav sample to stdout")
    }
}

impl WavSample for f32 {
    const SAMPLE_FORMAT: hound::SampleFormat = hound::SampleFormat::Float;
    const BITS_PER_SAMPLE: u16 = 32;
    const BYTES_PER_SAMPLE: u32 = 4;

    fn write_to_hound<W: Write + std::io::Seek>(
        writer: &mut hound::WavWriter<W>,
        sample: Self,
    ) -> Result<()> {
        writer
            .write_sample(sample)
            .context("failed to write wav sample")
    }

    fn write_to_stream<W: Write>(writer: &mut W, sample: Self) -> Result<()> {
        writer
            .write_all(&sample.to_le_bytes())
            .context("failed to write wav sample to stdout")
    }
}

pub fn write_wav<S: WavSample>(
    output: &TtsOutput,
    sample_rate_hz: u32,
    samples: &[S],
) -> Result<()> {
    match output {
        TtsOutput::WavFile(path) => {
            let mut writer = hound::WavWriter::create(
                path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: sample_rate_hz,
                    bits_per_sample: S::BITS_PER_SAMPLE,
                    sample_format: S::SAMPLE_FORMAT,
                },
            )
            .with_context(|| format!("failed to create wav file `{}`", path.display()))?;

            for &sample in samples {
                S::write_to_hound(&mut writer, sample)?;
            }

            writer.finalize().context("failed to finalize wav file")?;
        }
        TtsOutput::Stdout => {
            write_stdout_wav(
                sample_rate_hz,
                S::SAMPLE_FORMAT,
                S::BITS_PER_SAMPLE,
                samples.len() as u32 * S::BYTES_PER_SAMPLE,
                |stdout| {
                    for sample in samples {
                        S::write_to_stream(stdout, *sample)?;
                    }
                    Ok(())
                },
            )?;
        }
    }

    Ok(())
}

fn write_stdout_wav<F>(
    sample_rate_hz: u32,
    sample_format: hound::SampleFormat,
    bits_per_sample: u16,
    data_bytes_len: u32,
    write_samples: F,
) -> Result<()>
where
    F: FnOnce(&mut BufWriter<std::io::StdoutLock<'_>>) -> Result<()>,
{
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    write_wav_header(
        &mut stdout,
        sample_rate_hz,
        1,
        sample_format,
        bits_per_sample,
        data_bytes_len,
    )?;
    write_samples(&mut stdout)?;
    stdout
        .flush()
        .context("failed to flush wav stream to stdout")?;
    Ok(())
}

fn write_wav_header<W: Write>(
    writer: &mut W,
    sample_rate_hz: u32,
    channels: u16,
    sample_format: hound::SampleFormat,
    bits_per_sample: u16,
    data_bytes_len: u32,
) -> Result<()> {
    let format_code: u16 = match sample_format {
        hound::SampleFormat::Int => 1,
        hound::SampleFormat::Float => 3,
    };
    let block_align = channels
        .checked_mul(bits_per_sample / 8)
        .context("wav block align overflow")?;
    let byte_rate = sample_rate_hz
        .checked_mul(block_align as u32)
        .context("wav byte rate overflow")?;
    let riff_size = 36u32
        .checked_add(data_bytes_len)
        .context("wav riff size overflow")?;

    writer
        .write_all(b"RIFF")
        .context("failed to write wav RIFF tag")?;
    writer
        .write_all(&riff_size.to_le_bytes())
        .context("failed to write wav RIFF size")?;
    writer
        .write_all(b"WAVE")
        .context("failed to write wav WAVE tag")?;
    writer
        .write_all(b"fmt ")
        .context("failed to write wav fmt tag")?;
    writer
        .write_all(&16u32.to_le_bytes())
        .context("failed to write wav fmt size")?;
    writer
        .write_all(&format_code.to_le_bytes())
        .context("failed to write wav format code")?;
    writer
        .write_all(&channels.to_le_bytes())
        .context("failed to write wav channel count")?;
    writer
        .write_all(&sample_rate_hz.to_le_bytes())
        .context("failed to write wav sample rate")?;
    writer
        .write_all(&byte_rate.to_le_bytes())
        .context("failed to write wav byte rate")?;
    writer
        .write_all(&block_align.to_le_bytes())
        .context("failed to write wav block align")?;
    writer
        .write_all(&bits_per_sample.to_le_bytes())
        .context("failed to write wav bits per sample")?;
    writer
        .write_all(b"data")
        .context("failed to write wav data tag")?;
    writer
        .write_all(&data_bytes_len.to_le_bytes())
        .context("failed to write wav data size")?;
    Ok(())
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

    #[test]
    fn write_wav_header_for_i16_looks_like_wave() {
        let mut bytes = Vec::new();
        write_wav_header(&mut bytes, 22_050, 1, hound::SampleFormat::Int, 16, 4)
            .expect("header should write");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
    }
}
