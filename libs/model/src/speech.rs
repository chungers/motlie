//! Streaming text-to-speech synthesis contracts.
//!
//! The speech capability accepts text input plus optional voice conditioning
//! and emits PCM chunks that higher layers can write to `.wav`, local audio
//! sinks, or telephony transports.

use async_trait::async_trait;

use crate::{AudioSpec, ModelError, PcmChunk};

/// Stream-scoped speech synthesis parameters.
///
/// Backends may support only a subset of these controls. The v1 Piper backend
/// supports `speaking_rate` and rejects `seed` with `InvalidConfiguration`.
/// Future backends may accept a larger subset without changing the public API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeechParams {
    pub speaking_rate: Option<f32>,
    pub seed: Option<u64>,
}

/// Optional conditioning for speaker choice or reference-audio cloning.
#[derive(Clone, Debug, PartialEq)]
pub enum VoiceConditioning {
    SpeakerId(u32),
    /// Reference-audio voice cloning. `reference_text` is the transcript of the
    /// reference audio — required by models like Qwen3-TTS that use prompted
    /// cloning. Backends that do not need the transcript (e.g., x-vector-only
    /// models) may ignore it with a documented quality caveat.
    ReferenceAudio {
        audio_spec: AudioSpec,
        pcm: Vec<u8>,
        reference_text: Option<String>,
    },
}

/// Speech synthesis request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeechRequest {
    pub text: String,
    pub params: SpeechParams,
    pub conditioning: Option<VoiceConditioning>,
}

/// Streaming speech-synthesis capability.
///
/// `SpeechModel` is `Send + Sync` — it represents the shareable loaded model.
/// Each call to `open_stream` produces an independent, stateful stream session
/// that is `Send` but intentionally not `Sync` because it owns mutable chunk
/// iteration state and requires ordered, exclusive access.
#[async_trait]
pub trait SpeechModel: Send + Sync {
    async fn open_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<Box<dyn SpeechStream>, ModelError>;
}

/// Live speech-synthesis stream session.
///
/// `SpeechStream` is `Send` but not `Sync`. Callers must hold exclusive access
/// to the stream — sharing across tasks requires caller-side synchronization.
///
/// ## Edge-case semantics
///
/// - Empty or whitespace-only `SpeechRequest.text`: returns `ModelError::InvalidConfiguration`
/// - `next_chunk()` returns `Ok(None)` only after the stream is exhausted
/// - After the final chunk, subsequent `next_chunk()` calls must return `Ok(None)` idempotently
/// - Calling after `finish()`: impossible by construction (`finish` consumes `Box<Self>`)
#[async_trait]
pub trait SpeechStream: Send {
    fn audio_spec(&self) -> &AudioSpec;
    async fn next_chunk(&mut self) -> Result<Option<PcmChunk>, ModelError>;
    async fn finish(self: Box<Self>) -> Result<(), ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioSpec, PcmEncoding};

    #[test]
    fn speech_request_defaults_to_empty_text_and_no_conditioning() {
        let request = SpeechRequest::default();

        assert!(request.text.is_empty());
        assert_eq!(request.params, SpeechParams::default());
        assert_eq!(request.conditioning, None);
    }

    #[test]
    fn speech_params_defaults_to_no_optional_controls() {
        let params = SpeechParams::default();

        assert_eq!(params.speaking_rate, None);
        assert_eq!(params.seed, None);
    }

    #[test]
    fn voice_conditioning_can_hold_reference_audio() {
        let conditioning = VoiceConditioning::ReferenceAudio {
            audio_spec: AudioSpec {
                sample_rate_hz: 16_000,
                channels: 1,
                encoding: PcmEncoding::S16Le,
            },
            pcm: vec![0_u8; 320],
            reference_text: Some("test transcript".into()),
        };

        match conditioning {
            VoiceConditioning::ReferenceAudio {
                audio_spec,
                pcm,
                reference_text,
            } => {
                assert_eq!(audio_spec.sample_rate_hz, 16_000);
                assert_eq!(audio_spec.channels, 1);
                assert_eq!(audio_spec.encoding, PcmEncoding::S16Le);
                assert_eq!(pcm.len(), 320);
                assert_eq!(reference_text.as_deref(), Some("test transcript"));
            }
            VoiceConditioning::SpeakerId(_) => panic!("expected reference audio"),
        }
    }
}
