//! Lightweight eval-facing contracts that belong next to the model API.
//!
//! Substantial harness tooling should live in `motlie-model-eval`.

use std::collections::BTreeSet;

use crate::{Capabilities, CapabilityDescriptor, CapabilityKind};

/// High-level evaluation tracks used to group bundle assessment suites.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvalTrack {
    Chat,
    Classification,
    Embeddings,
    Reasoning,
    Summarization,
    Transcription,
}

impl EvalTrack {
    /// Returns the primary evaluation track implied by a capability descriptor, when one exists.
    ///
    /// The current v0.1 contract only treats embedding capabilities as having a stable,
    /// non-ambiguous evaluation track. Text-generation capabilities may participate in
    /// multiple tracks (chat, reasoning, summarization, classification) and remain the
    /// responsibility of higher-level harness configuration.
    pub fn primary_for_descriptor(descriptor: &CapabilityDescriptor) -> Option<Self> {
        match descriptor.kind {
            CapabilityKind::Embeddings => Some(Self::Embeddings),
            CapabilityKind::Transcription => Some(Self::Transcription),
            _ => None,
        }
    }
}

/// Derives the stable evaluation tracks implied directly by a capability set.
pub fn tracks_for_capabilities(capabilities: &Capabilities) -> BTreeSet<EvalTrack> {
    capabilities
        .descriptors()
        .iter()
        .filter_map(EvalTrack::primary_for_descriptor)
        .collect()
}

/// Returns whether a capability set directly implies support for an evaluation track.
pub fn capabilities_support_track(capabilities: &Capabilities, track: EvalTrack) -> bool {
    capabilities
        .descriptors()
        .iter()
        .any(|descriptor| EvalTrack::primary_for_descriptor(descriptor) == Some(track))
}

/// Stable identifier for a single evaluation case.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvalCaseId(String);

impl EvalCaseId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Lightweight metadata for a single evaluation case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalCase {
    pub id: EvalCaseId,
    pub track: EvalTrack,
    pub prompt: String,
}

/// Structured result surface that higher-level harness tooling can aggregate.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalResult {
    pub case_id: EvalCaseId,
    pub score: f32,
    pub passed: bool,
    pub notes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapabilityDescriptor;

    #[test]
    fn eval_case_ids_are_stable_strings() {
        let case_id = EvalCaseId::new("embeddings-basic-001");

        assert_eq!(case_id.as_str(), "embeddings-basic-001");
    }

    #[test]
    fn eval_result_construction_is_straightforward() {
        let case = EvalCase {
            id: EvalCaseId::new("embeddings-basic-001"),
            track: EvalTrack::Embeddings,
            prompt: "motlie model bundle".into(),
        };

        let result = EvalResult {
            case_id: case.id.clone(),
            score: 1.0,
            passed: true,
            notes: vec!["vector count matched input count".into()],
        };

        assert_eq!(result.case_id, case.id);
        assert!(result.passed);
    }

    #[test]
    fn embeddings_capability_maps_to_embeddings_track() {
        let descriptor = CapabilityDescriptor::embeddings();

        assert_eq!(
            EvalTrack::primary_for_descriptor(&descriptor),
            Some(EvalTrack::Embeddings)
        );
    }

    #[test]
    fn transcription_capability_maps_to_transcription_track() {
        let descriptor = CapabilityDescriptor::transcription_stream();

        assert_eq!(
            EvalTrack::primary_for_descriptor(&descriptor),
            Some(EvalTrack::Transcription)
        );
    }

    #[test]
    fn text_generation_capabilities_do_not_claim_primary_eval_tracks_yet() {
        assert_eq!(
            EvalTrack::primary_for_descriptor(&CapabilityDescriptor::chat()),
            None
        );
        assert_eq!(
            EvalTrack::primary_for_descriptor(&CapabilityDescriptor::completion()),
            None
        );
    }

    #[test]
    fn capabilities_project_to_embedding_track_without_duplicates() {
        let capabilities = Capabilities::new(vec![
            CapabilityDescriptor::embeddings(),
            CapabilityDescriptor::embeddings(),
        ]);

        let tracks = tracks_for_capabilities(&capabilities);

        assert_eq!(tracks.len(), 1);
        assert!(tracks.contains(&EvalTrack::Embeddings));
    }

    #[test]
    fn capability_track_boolean_check_does_not_need_allocation() {
        let capabilities = Capabilities::new(vec![CapabilityDescriptor::embeddings()]);

        assert!(capabilities_support_track(
            &capabilities,
            EvalTrack::Embeddings
        ));
        assert!(!capabilities_support_track(
            &capabilities,
            EvalTrack::Reasoning
        ));
    }
}
