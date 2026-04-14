//! Streaming voice-to-text transcription contracts.
//!
//! The transcription capability accepts PCM audio chunks and emits transcript
//! segments. The stream contract standardizes on raw PCM input so both `.wav`
//! files and live websocket audio map to the same API.

use async_trait::async_trait;

use crate::ModelError;

/// PCM sample encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PcmEncoding {
    /// Signed 16-bit little-endian.
    S16Le,
    /// 32-bit float little-endian.
    F32Le,
}

/// Audio format specification established once per transcription stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioSpec {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub encoding: PcmEncoding,
}

/// One chunk of raw PCM audio pushed into a transcription stream.
///
/// The audio format is established at `TranscriptionModel::open_stream()` time
/// and applies to all chunks in the stream. Chunks must arrive with
/// monotonically increasing `sequence` numbers.
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

/// Streaming transcription capability.
///
/// `TranscriptionModel` is `Send + Sync` — it represents the shareable loaded
/// model. Each call to `open_stream` produces an independent, stateful stream
/// session that is `Send` but intentionally not `Sync` because it owns mutable
/// rolling-buffer state and requires ordered, exclusive access.
#[async_trait]
pub trait TranscriptionModel: Send + Sync {
    async fn open_stream(
        &self,
        spec: AudioSpec,
        params: TranscriptionParams,
    ) -> Result<Box<dyn TranscriptionStream>, ModelError>;
}

/// Live transcription stream session.
///
/// `TranscriptionStream` is `Send` but not `Sync`. Callers must hold exclusive
/// access to the stream — sharing across tasks requires caller-side
/// synchronization.
///
/// ## Edge-case semantics
///
/// - `push_chunk` after `end_of_stream = true`: returns `ModelError::InvalidConfiguration`
/// - Non-monotonic `sequence`: returns `ModelError::InvalidConfiguration`
/// - Empty `data` with `end_of_stream = false`: returns `Ok(None)` (no-op)
/// - Calling after `finish()`: impossible by construction (`finish` consumes `Box<Self>`)
#[async_trait]
pub trait TranscriptionStream: Send {
    /// Push a PCM chunk and optionally receive new transcript output.
    ///
    /// Returns `Ok(None)` when the chunk does not cross a decode boundary.
    /// Returns `Ok(Some(update))` when the backend emits new segments.
    async fn push_chunk(
        &mut self,
        chunk: PcmChunk,
    ) -> Result<Option<TranscriptionUpdate>, ModelError>;

    /// Finalize the stream and flush any remaining transcript output.
    async fn finish(self: Box<Self>) -> Result<TranscriptionUpdate, ModelError>;
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
        };

        assert_eq!(spec.sample_rate_hz, 16_000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.encoding, PcmEncoding::S16Le);
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
