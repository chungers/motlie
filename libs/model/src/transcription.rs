//! Shared ASR data types.
//!
//! The typed ASR contracts live in [`crate::typed`]. This module only carries
//! the shared PCM/transcript data structures used across backend and bundle
//! boundaries.

/// PCM sample encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PcmEncoding {
    /// Signed 16-bit little-endian.
    S16Le,
    /// 32-bit float little-endian.
    F32Le,
}

/// How a backend consumes or emits stream data internally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendMode {
    Batch,
    Streaming,
}

/// Audio format specification established once per transcription stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioSpec {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub encoding: PcmEncoding,
    /// Advisory chunk size in bytes. Callers should query this instead of
    /// guessing chunk sizes for stream wiring.
    pub preferred_chunk_bytes: usize,
}

impl AudioSpec {
    pub fn frame_bytes(&self) -> usize {
        let bytes_per_sample = match self.encoding {
            PcmEncoding::S16Le => 2,
            PcmEncoding::F32Le => 4,
        };
        bytes_per_sample * self.channels.max(1) as usize
    }

    pub fn normalized_chunk_size(&self) -> usize {
        let frame_bytes = self.frame_bytes();
        let preferred = self.preferred_chunk_bytes.max(frame_bytes);
        let aligned = preferred - (preferred % frame_bytes);
        aligned.max(frame_bytes)
    }
}

/// One chunk of raw PCM audio pushed into a typed streaming transcription
/// session.
///
/// The audio format is established by the caller's typed `AudioBuf` boundary
/// and applies to all chunks in the session. Chunks must arrive with
/// monotonically increasing `sequence` numbers whenever a backend keeps
/// sequence state internally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PcmChunk {
    pub data: Vec<u8>,
    pub sequence: u64,
    pub end_of_stream: bool,
}

/// Parameters controlling transcription behavior for a stream session.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TranscriptionParams {
    pub language: Option<String>,
    pub emit_partials: bool,
}

/// One time-aligned transcript segment emitted by the backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// `false` for interim results that may still change;
    /// `true` for committed text that will not be rewritten.
    pub final_segment: bool,
}

/// Batch of transcript segments returned from a decode step.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TranscriptionUpdate {
    pub segments: Vec<TranscriptSegment>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_chunk_is_independent_of_audio_spec() {
        let chunk = PcmChunk {
            data: vec![0u8; 320],
            sequence: 0,
            end_of_stream: false,
        };

        assert_eq!(chunk.data.len(), 320);
        assert_eq!(chunk.sequence, 0);
        assert!(!chunk.end_of_stream);
    }

    #[test]
    fn audio_spec_describes_stream_format() {
        let spec = AudioSpec {
            sample_rate_hz: 16_000,
            channels: 1,
            encoding: PcmEncoding::S16Le,
            preferred_chunk_bytes: 6_400,
        };

        assert_eq!(spec.sample_rate_hz, 16_000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.encoding, PcmEncoding::S16Le);
        assert_eq!(spec.preferred_chunk_bytes, 6_400);
    }

    #[test]
    fn audio_spec_normalizes_chunk_size_to_frame_alignment() {
        let spec = AudioSpec {
            sample_rate_hz: 22_050,
            channels: 2,
            encoding: PcmEncoding::S16Le,
            preferred_chunk_bytes: 16_001,
        };

        assert_eq!(spec.frame_bytes(), 4);
        assert_eq!(spec.normalized_chunk_size(), 16_000);
    }

    #[test]
    fn transcription_params_default_to_no_language_no_partials() {
        let params = TranscriptionParams::default();

        assert_eq!(params.language, None);
        assert!(!params.emit_partials);
    }

    #[test]
    fn transcript_segment_distinguishes_partial_from_final() {
        let partial = TranscriptSegment {
            start_ms: 0,
            end_ms: 500,
            text: "hel".into(),
            final_segment: false,
        };
        let committed = TranscriptSegment {
            start_ms: 0,
            end_ms: 1000,
            text: "hello world".into(),
            final_segment: true,
        };

        assert!(!partial.final_segment);
        assert!(committed.final_segment);
    }

    #[test]
    fn transcription_update_defaults_to_empty() {
        let update = TranscriptionUpdate::default();

        assert!(update.segments.is_empty());
    }
}
