use std::io::Read;

use crate::{Result, VoiceError, VoiceSampleFormat};

pub fn decode_samples_to_f32<R: Read>(
    reader: hound::WavReader<R>,
) -> Result<(hound::WavSpec, Vec<f32>)> {
    let spec = reader.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .into_samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .map(|sample| sample as f32 / 32768.0)
                .collect(),
            bits_per_sample => {
                return Err(VoiceError::UnsupportedWavSampleFormat {
                    sample_format: VoiceSampleFormat::Int,
                    bits_per_sample,
                });
            }
        },
        hound::SampleFormat::Float => match spec.bits_per_sample {
            32 => reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?,
            bits_per_sample => {
                return Err(VoiceError::UnsupportedWavSampleFormat {
                    sample_format: VoiceSampleFormat::Float,
                    bits_per_sample,
                });
            }
        },
    };

    Ok((spec, samples))
}

pub fn downmix_to_mono(samples: &[f32], channels: u16) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(VoiceError::InvalidChannelCount { channels });
    }
    if channels == 1 {
        return Ok(samples.to_vec());
    }

    let channels = channels as usize;
    if !samples.len().is_multiple_of(channels) {
        return Err(VoiceError::IncompleteFrame {
            sample_len: samples.len(),
            channels: channels as u16,
        });
    }

    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect())
}

pub fn decode_wav_data_bytes(spec: &hound::WavSpec, data: &[u8]) -> Result<Vec<f32>> {
    let sample_width = usize::from(spec.bits_per_sample / 8);
    if sample_width == 0 {
        return Err(VoiceError::InvalidBitsPerSample {
            bits_per_sample: spec.bits_per_sample,
        });
    }
    if !data.len().is_multiple_of(sample_width) {
        return Err(VoiceError::MisalignedDataLen {
            data_len: data.len(),
            sample_width,
        });
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
        (sample_format, bits_per_sample) => Err(VoiceError::UnsupportedWavSampleFormat {
            sample_format: sample_format.into(),
            bits_per_sample,
        }),
    }
}

pub fn f32_to_i16_clamped(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

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
        let (spec, samples) = decode_samples_to_f32(reader).expect("decode should succeed");
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(samples, vec![0.25]);
    }

    #[test]
    fn downmix_to_mono_averages_channels() {
        let mono = downmix_to_mono(&[1.0, 3.0, 2.0, 6.0], 2).expect("downmix should work");
        assert_eq!(mono, vec![2.0, 4.0]);
    }
}
