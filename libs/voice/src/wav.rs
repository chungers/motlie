use std::io::{BufWriter, Read, Write};
use std::marker::PhantomData;

use crate::pipeline::convert::decode_wav_data_bytes;
use crate::{Result, VoiceError};

pub trait WavSample: Copy {
    const SAMPLE_FORMAT: hound::SampleFormat;
    const BITS_PER_SAMPLE: u16;

    fn write_to_hound<W: Write + std::io::Seek>(
        writer: &mut hound::WavWriter<W>,
        sample: Self,
    ) -> Result<()>;
    fn write_to_stream<W: Write>(writer: &mut W, sample: Self) -> Result<()>;
}

impl WavSample for i16 {
    const SAMPLE_FORMAT: hound::SampleFormat = hound::SampleFormat::Int;
    const BITS_PER_SAMPLE: u16 = 16;

    fn write_to_hound<W: Write + std::io::Seek>(
        writer: &mut hound::WavWriter<W>,
        sample: Self,
    ) -> Result<()> {
        writer.write_sample(sample)?;
        Ok(())
    }

    fn write_to_stream<W: Write>(writer: &mut W, sample: Self) -> Result<()> {
        writer.write_all(&sample.to_le_bytes())?;
        Ok(())
    }
}

impl WavSample for f32 {
    const SAMPLE_FORMAT: hound::SampleFormat = hound::SampleFormat::Float;
    const BITS_PER_SAMPLE: u16 = 32;

    fn write_to_hound<W: Write + std::io::Seek>(
        writer: &mut hound::WavWriter<W>,
        sample: Self,
    ) -> Result<()> {
        writer.write_sample(sample)?;
        Ok(())
    }

    fn write_to_stream<W: Write>(writer: &mut W, sample: Self) -> Result<()> {
        writer.write_all(&sample.to_le_bytes())?;
        Ok(())
    }
}

pub struct StreamingWavWriter<W: Write, S: WavSample> {
    writer: BufWriter<W>,
    _sample: PhantomData<S>,
}

impl<W: Write, S: WavSample> StreamingWavWriter<W, S> {
    pub fn new(writer: W, sample_rate_hz: u32, channels: u16) -> Result<Self> {
        let mut writer = BufWriter::new(writer);
        write_wav_header(
            &mut writer,
            sample_rate_hz,
            channels,
            S::SAMPLE_FORMAT,
            S::BITS_PER_SAMPLE,
        )?;
        Ok(Self {
            writer,
            _sample: PhantomData,
        })
    }

    pub fn write_chunk(&mut self, samples: &[S]) -> Result<()> {
        for &sample in samples {
            S::write_to_stream(&mut self.writer, sample)?;
        }
        self.writer.flush()?;
        Ok(())
    }

    pub fn finalize(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

pub fn decode_streaming_wav_to_f32<R: Read>(mut reader: R) -> Result<(hound::WavSpec, Vec<f32>)> {
    let mut riff_header = [0u8; 12];
    reader.read_exact(&mut riff_header)?;
    if &riff_header[0..4] != b"RIFF" {
        return Err(VoiceError::Wav(hound::Error::FormatError(
            "Ill-formed WAVE file: no RIFF tag found",
        )));
    }
    if &riff_header[8..12] != b"WAVE" {
        return Err(VoiceError::Wav(hound::Error::FormatError(
            "Ill-formed WAVE file: no WAVE tag found",
        )));
    }

    let mut spec = None;
    let mut data = None;

    loop {
        let mut chunk_header = [0u8; 8];
        match reader.read_exact(&mut chunk_header) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(VoiceError::Io(error)),
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
                reader.read_exact(&mut fmt)?;
                spec = Some(parse_wav_fmt_chunk(&fmt)?);
            }
            b"data" => {
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes)?;
                data = Some(bytes);
                break;
            }
            _ => {
                let mut skipped = vec![0u8; chunk_len];
                reader.read_exact(&mut skipped)?;
            }
        }

        if chunk_len % 2 == 1 {
            let mut padding = [0u8; 1];
            reader.read_exact(&mut padding)?;
        }
    }

    let spec = spec.ok_or(VoiceError::MissingFmtChunk)?;
    let data = data.ok_or(VoiceError::MissingDataChunk)?;
    let samples = decode_wav_data_bytes(&spec, &data)?;
    Ok((spec, samples))
}

fn parse_wav_fmt_chunk(bytes: &[u8]) -> Result<hound::WavSpec> {
    if bytes.len() < 16 {
        return Err(VoiceError::WavFmtTooShort {
            actual: bytes.len(),
        });
    }

    let format_code = u16::from_le_bytes([bytes[0], bytes[1]]);
    let channels = u16::from_le_bytes([bytes[2], bytes[3]]);
    let sample_rate = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let bits_per_sample = u16::from_le_bytes([bytes[14], bytes[15]]);
    let sample_format = match format_code {
        1 => hound::SampleFormat::Int,
        3 => hound::SampleFormat::Float,
        format_code => return Err(VoiceError::UnsupportedWavFormatCode { format_code }),
    };

    Ok(hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        sample_format,
    })
}

fn write_wav_header<W: Write>(
    writer: &mut W,
    sample_rate_hz: u32,
    channels: u16,
    sample_format: hound::SampleFormat,
    bits_per_sample: u16,
) -> Result<()> {
    let format_code: u16 = match sample_format {
        hound::SampleFormat::Int => 1,
        hound::SampleFormat::Float => 3,
    };
    let block_align =
        channels
            .checked_mul(bits_per_sample / 8)
            .ok_or(VoiceError::BlockAlignOverflow {
                channels,
                bits_per_sample,
            })?;
    let byte_rate =
        sample_rate_hz
            .checked_mul(block_align as u32)
            .ok_or(VoiceError::ByteRateOverflow {
                sample_rate_hz,
                block_align,
            })?;
    let streaming_data_bytes_len = u32::MAX - (u32::MAX % u32::from(block_align.max(1)));
    let riff_size = streaming_data_bytes_len;
    let data_bytes_len = streaming_data_bytes_len;

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&format_code.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate_hz.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_bytes_len.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{decode_streaming_wav_to_f32, StreamingWavWriter};

    #[test]
    fn streaming_wav_header_for_i16_looks_like_wave() {
        let mut writer =
            StreamingWavWriter::<_, i16>::new(Vec::new(), 22_050, 1).expect("writer should start");
        writer
            .write_chunk(&[0i16, 1i16])
            .expect("samples should write");
        let bytes = writer.writer.into_inner().expect("writer should unwrap");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(&bytes[4..8], &(u32::MAX - 1).to_le_bytes());
        assert_eq!(&bytes[40..44], &(u32::MAX - 1).to_le_bytes());
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
