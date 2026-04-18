//! Shared TTS data types.
//!
//! The typed TTS contracts live in [`crate::typed`]. This module only carries
//! the request/conditioning types that are shared across bundles and backends.

use crate::AudioSpec;

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
                preferred_chunk_bytes: 0,
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
