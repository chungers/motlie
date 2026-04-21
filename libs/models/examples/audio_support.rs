use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub type DynWavReader = hound::WavReader<BufReader<WavReaderInput>>;

pub enum WavReaderInput {
    File(File),
    Stdin(std::io::Stdin),
}

impl Read for WavReaderInput {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::File(file) => file.read(buf),
            Self::Stdin(stdin) => stdin.read(buf),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WavInput {
    File(PathBuf),
    Stdin,
}

pub fn open_wav_reader(path: Option<&Path>) -> Result<(WavInput, DynWavReader)> {
    match path {
        Some(path) => {
            let file = File::open(path)
                .with_context(|| format!("failed to open wav file `{}`", path.display()))?;
            let input = WavInput::File(path.to_path_buf());
            let reader = hound::WavReader::new(BufReader::new(WavReaderInput::File(file)))
                .with_context(|| format!("failed to parse wav file `{}`", path.display()))?;
            Ok((input, reader))
        }
        None => {
            let reader =
                hound::WavReader::new(BufReader::new(WavReaderInput::Stdin(std::io::stdin())))
                    .context("failed to parse wav stream from stdin")?;
            Ok((WavInput::Stdin, reader))
        }
    }
}

pub fn decode_wav_to_f32<R: Read>(
    reader: hound::WavReader<R>,
) -> Result<(hound::WavSpec, Vec<f32>)> {
    let spec = reader.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to decode integer wav samples")?
            .into_iter()
            .map(|sample| sample as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to decode float wav samples")?,
    };

    Ok((spec, samples))
}

pub fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }

    let channels = channels as usize;
    debug_assert_eq!(
        samples.len() % channels,
        0,
        "well-formed interleaved wav input should contain complete frames"
    );
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

pub fn resample_linear_f32(samples: &[f32], input_rate_hz: u32, output_rate_hz: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate_hz == output_rate_hz {
        return samples.to_vec();
    }

    let ratio = input_rate_hz as f64 / output_rate_hz as f64;
    let out_len = ((samples.len() as f64) * output_rate_hz as f64 / input_rate_hz as f64)
        .ceil()
        .clamp(0.0, usize::MAX as f64) as usize;
    let max_index = samples.len().saturating_sub(1);
    let mut output = Vec::with_capacity(out_len.max(1));

    for out_idx in 0..out_len {
        let src_pos = out_idx as f64 * ratio;
        let left_idx = src_pos.floor() as usize;
        let right_idx = (left_idx + 1).min(max_index);
        let frac = (src_pos - left_idx as f64) as f32;
        let left = samples[left_idx.min(max_index)];
        let right = samples[right_idx];
        output.push(left + (right - left) * frac);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn decode_float_wav_to_f32() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(
                &mut cursor,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .expect("writer should start");
            writer.write_sample(0.25f32).expect("sample should write");
            writer.finalize().expect("writer should finalize");
        }

        cursor.set_position(0);
        let reader = hound::WavReader::new(cursor).expect("reader should open");
        let (spec, samples) = decode_wav_to_f32(reader).expect("decode should succeed");
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(samples, vec![0.25]);
    }

    #[test]
    fn downmix_to_mono_averages_channels() {
        let mono = downmix_to_mono(&[1.0, 3.0, 2.0, 6.0], 2);
        assert_eq!(mono, vec![2.0, 4.0]);
    }
}
