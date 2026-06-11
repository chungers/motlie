//! Shared ASR data types.
//!
//! The typed ASR contracts live in [`crate::typed`]. This module only carries
//! transcript data structures used across backend and bundle boundaries.

/// Parameters controlling transcription behavior for a stream session.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TranscriptionParams {
    pub language: Option<String>,
    pub emit_partials: bool,
}

/// One time-aligned transcript segment emitted by the backend.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// Optional backend-native confidence carried without calibration or
    /// downstream policy. Values are normalized to `0.0..=1.0` only when a
    /// backend emits log probabilities and the conversion is a direct `exp`.
    /// Values are uncalibrated and are not comparable across backends.
    ///
    /// `None` means the backend has no reliable native confidence signal or the
    /// current decode path does not expose one. The scalar is the latest token's
    /// native confidence because the approved contract is segment-shaped; for
    /// final segments this scores the tail token, not the whole segment.
    /// Current mappings are:
    ///
    /// - sherpa-onnx: latest native online token log-probability from `ys_probs`,
    ///   converted with `exp`. `lm_probs` is retained in the vendored wrapper but
    ///   is not used for this field. No token aggregation or stability modeling
    ///   is applied.
    /// - whisper.cpp: latest native decoded token probability from
    ///   `whisper_full_get_token_p`. No token aggregation, avg-logprob synthesis,
    ///   or `no_speech_probability` penalty is applied. Segments may end on
    ///   punctuation or special tokens, so this value can be optimistic for
    ///   segment-level quality.
    /// - moonshine: `None` until decoder confidence is exposed.
    ///
    /// This field is model/backend confidence, not interim stability. Stability
    /// is out of scope because none of the current engines emits it natively.
    pub confidence: Option<f32>,
    /// `false` for interim results that may still change;
    /// `true` for committed text that will not be rewritten.
    pub final_segment: bool,
}

/// Batch of transcript segments returned from a decode step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptionUpdate {
    pub segments: Vec<TranscriptSegment>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
            confidence: None,
            final_segment: false,
        };
        let committed = TranscriptSegment {
            start_ms: 0,
            end_ms: 1000,
            text: "hello world".into(),
            confidence: Some(0.92),
            final_segment: true,
        };

        assert!(!partial.final_segment);
        assert!(committed.final_segment);
        assert_eq!(partial.confidence, None);
        assert_eq!(committed.confidence, Some(0.92));
    }

    #[test]
    fn transcription_update_defaults_to_empty() {
        let update = TranscriptionUpdate::default();

        assert!(update.segments.is_empty());
    }
}
