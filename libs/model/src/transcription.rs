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
