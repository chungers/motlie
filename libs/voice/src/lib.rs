//! Shared audio pipeline helpers used by speech examples and adapters.

pub mod app;
pub mod codec;
pub mod frame;
pub mod pipeline;
pub mod runtime;
pub mod telephony;
pub mod wav;

use std::io;

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoiceSampleFormat {
    Int,
    Float,
}

impl From<hound::SampleFormat> for VoiceSampleFormat {
    fn from(value: hound::SampleFormat) -> Self {
        match value {
            hound::SampleFormat::Int => Self::Int,
            hound::SampleFormat::Float => Self::Float,
        }
    }
}

#[derive(Debug, Error)]
pub enum VoiceError {
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),
    #[error("wav error: {0}")]
    Wav(#[from] hound::Error),
    #[error("invalid channel count: {channels}")]
    InvalidChannelCount { channels: u16 },
    #[error("interleaved sample count {sample_len} is not divisible by channel count {channels}")]
    IncompleteFrame { sample_len: usize, channels: u16 },
    #[error("invalid wav bits per sample: {bits_per_sample}")]
    InvalidBitsPerSample { bits_per_sample: u16 },
    #[error("wav fmt chunk too short: expected at least 16 bytes, got {actual}")]
    WavFmtTooShort { actual: usize },
    #[error("unsupported wav format code {format_code}; expected PCM (1) or float (3)")]
    UnsupportedWavFormatCode { format_code: u16 },
    #[error(
        "unsupported wav sample format {sample_format:?} with {bits_per_sample} bits per sample"
    )]
    UnsupportedWavSampleFormat {
        sample_format: VoiceSampleFormat,
        bits_per_sample: u16,
    },
    #[error("wav data payload length {data_len} is not divisible by sample width {sample_width}")]
    MisalignedDataLen {
        data_len: usize,
        sample_width: usize,
    },
    #[error("wav fmt chunk missing")]
    MissingFmtChunk,
    #[error("wav data chunk missing")]
    MissingDataChunk,
    #[error("wav block align overflow: channels={channels} bits_per_sample={bits_per_sample}")]
    BlockAlignOverflow { channels: u16, bits_per_sample: u16 },
    #[error("wav byte rate overflow: sample_rate_hz={sample_rate_hz} block_align={block_align}")]
    ByteRateOverflow {
        sample_rate_hz: u32,
        block_align: u16,
    },
    #[error("invalid sample rate conversion: input={input_rate_hz} output={output_rate_hz}")]
    InvalidSampleRate {
        input_rate_hz: u32,
        output_rate_hz: u32,
    },
    #[error("invalid audio chunk duration: {chunk_ms} ms")]
    InvalidChunkDuration { chunk_ms: u32 },
    #[error("encoded payload length {payload_len} is not aligned to sample width {sample_width}")]
    MisalignedEncodedPayload {
        payload_len: usize,
        sample_width: usize,
    },
    #[error("frame sequence {sequence} is older than the next expected sequence {next_expected}")]
    StaleFrameSequence { sequence: u64, next_expected: u64 },
    #[error(
        "frame sequence {sequence} is too far ahead of {next_expected} for reorder capacity {capacity}"
    )]
    ReorderCapacityExceeded {
        sequence: u64,
        next_expected: u64,
        capacity: usize,
    },
}

pub type Result<T> = std::result::Result<T, VoiceError>;
