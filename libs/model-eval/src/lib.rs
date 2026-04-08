//! Scaffold for substantial evaluation tooling built on top of `motlie-model`.
//!
//! The long-term role of this crate is to host harness runners, suite loading,
//! scoring, reporting, and other executable evaluation machinery. Lightweight
//! eval-facing contracts remain in `motlie_model::eval`.

use motlie_model::eval::{tracks_for_capabilities, EvalTrack};
use motlie_model::Capabilities;

/// Minimal track-selection helper proving `motlie-model-eval` can consume the
/// stable `motlie_model::eval` mapping without bundle-specific branching.
pub fn supports_track(capabilities: &Capabilities, track: EvalTrack) -> bool {
    tracks_for_capabilities(capabilities).contains(&track)
}

/// Placeholder type so the scaffold builds cleanly once wired into the workspace.
pub struct ModelEvalScaffold;

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::BundleId;
    use motlie_models::Catalog;

    #[test]
    fn embedding_capabilities_support_embeddings_track() {
        assert!(supports_track(
            &motlie_model::Capabilities::embeddings_only(),
            EvalTrack::Embeddings
        ));
    }

    #[test]
    fn default_catalog_bundle_is_eligible_for_embeddings_track() {
        let catalog = Catalog::with_defaults();
        let descriptor = catalog
            .bundle(&BundleId::new("embeddinggemma_300m"))
            .expect("default catalog should register embeddinggemma_300m");

        assert!(supports_track(&descriptor.capabilities, EvalTrack::Embeddings));
    }
}
