use motlie_model::typed::{AudioBuf, Mono};

use crate::codec::{g711, l16};
use crate::pipeline::resample::{resample_i16_mono, WindowedSincResampler};
use crate::{Result, VoiceError};

pub const ASR_SAMPLE_RATE_HZ: u32 = 16_000;
pub const PCMU_SAMPLE_RATE_HZ: u32 = 8_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaTrack {
    Inbound,
    Outbound,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallDirection {
    Inbound,
    Outbound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CallAction {
    Answer,
    Reject,
    Hangup,
    Transfer { destination: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DtmfDigit {
    Digit(u8),
    Star,
    Pound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelnyxAsrAudioSpec {
    L16_16k,
    Pcmu8k,
}

impl TelnyxAsrAudioSpec {
    pub fn label(self) -> &'static str {
        match self {
            Self::L16_16k => "L16-16k",
            Self::Pcmu8k => "PCMU-8k",
        }
    }

    pub fn encoding(self) -> &'static str {
        match self {
            Self::L16_16k => "L16",
            Self::Pcmu8k => "PCMU",
        }
    }

    pub fn media_sample_rate_hz(self) -> u32 {
        match self {
            Self::L16_16k => ASR_SAMPLE_RATE_HZ,
            Self::Pcmu8k => PCMU_SAMPLE_RATE_HZ,
        }
    }
}

pub fn round_trip_telnyx_asr_samples(
    samples: &[i16],
    input_sample_rate_hz: u32,
    spec: TelnyxAsrAudioSpec,
) -> Result<Vec<i16>> {
    let resampler = WindowedSincResampler::default();
    let media_samples = resample_i16_mono(
        &resampler,
        samples,
        input_sample_rate_hz,
        spec.media_sample_rate_hz(),
    )?;

    let decoded = match spec {
        TelnyxAsrAudioSpec::L16_16k => {
            let encoded = l16::encode_l16_le(&media_samples);
            l16::decode_l16_le(&encoded)?
        }
        TelnyxAsrAudioSpec::Pcmu8k => {
            let encoded = g711::encode_pcmu(&media_samples);
            g711::decode_pcmu(&encoded)
        }
    };

    resample_i16_mono(
        &resampler,
        &decoded,
        spec.media_sample_rate_hz(),
        ASR_SAMPLE_RATE_HZ,
    )
}

pub fn round_trip_telnyx_asr_chunks(
    samples: &[i16],
    input_sample_rate_hz: u32,
    spec: TelnyxAsrAudioSpec,
    chunk_ms: u32,
) -> Result<Vec<AudioBuf<i16, ASR_SAMPLE_RATE_HZ, Mono>>> {
    if chunk_ms == 0 {
        return Err(VoiceError::InvalidChunkDuration { chunk_ms });
    }

    let chunk_samples = ((u64::from(ASR_SAMPLE_RATE_HZ) * u64::from(chunk_ms)) / 1_000) as usize;
    if chunk_samples == 0 {
        return Err(VoiceError::InvalidChunkDuration { chunk_ms });
    }

    let asr_samples = round_trip_telnyx_asr_samples(samples, input_sample_rate_hz, spec)?;
    Ok(asr_samples
        .chunks(chunk_samples)
        .map(|chunk| AudioBuf::<i16, ASR_SAMPLE_RATE_HZ, Mono>::new(chunk.to_vec()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l16_round_trip_preserves_16khz_samples() {
        let input = vec![0, 1000, -1000, 16_000, -16_000];
        let output =
            round_trip_telnyx_asr_samples(&input, ASR_SAMPLE_RATE_HZ, TelnyxAsrAudioSpec::L16_16k)
                .expect("L16 round trip should succeed");

        assert_eq!(output, input);
    }

    #[test]
    fn pcmu_round_trip_resamples_back_to_16khz() {
        let input = vec![0_i16; ASR_SAMPLE_RATE_HZ as usize];
        let output =
            round_trip_telnyx_asr_samples(&input, ASR_SAMPLE_RATE_HZ, TelnyxAsrAudioSpec::Pcmu8k)
                .expect("PCMU round trip should succeed");

        assert!((output.len() as isize - ASR_SAMPLE_RATE_HZ as isize).abs() <= 2);
    }

    #[test]
    fn round_trip_chunks_use_requested_duration() {
        let input = vec![0_i16; ASR_SAMPLE_RATE_HZ as usize];
        let chunks = round_trip_telnyx_asr_chunks(
            &input,
            ASR_SAMPLE_RATE_HZ,
            TelnyxAsrAudioSpec::L16_16k,
            20,
        )
        .expect("chunking should succeed");

        assert_eq!(chunks.len(), 50);
        assert!(chunks.iter().all(|chunk| chunk.sample_rate_hz() == 16_000));
    }
}
