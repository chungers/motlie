//! Coverage reconciliation (#521): the runtime half that classifies each
//! `(CuratedBundle, QuantizationScheme, CapabilityKind, Profile)` cell into the
//! 4-state taxonomy by joining the compile-time declaration
//! (`BackendKind::accel_support`, reached via `Profile::accelerator()`) with the
//! recorded results.
//!
//! The reconciliation *is* the framework's value: the declaration and the
//! runtime evidence must agree, and any disagreement is a flagged finding
//! (`Finding`) that fails the fail-closed completeness check.

use motlie_model::{AccelSupport, BackendKind, Reason};

use crate::profile::Profile;

/// The state of one coverage cell — exactly one of four.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoverageState {
    /// A passing record ran on the requested accelerator. Carries the metric
    /// (held elsewhere; this enum is the classification only).
    Validated,
    /// The backend fundamentally cannot target this accelerator — compile
    /// permanent (e.g. ORT-on-Metal).
    NotApplicable(Reason),
    /// The backend *can* target it, but this build/SHA did not provide the
    /// native path (recorded native-provider evidence shows it absent). Buildable.
    BuildGap,
    /// The path is available but no passing record exists / never scheduled.
    Gap,
}

/// The runtime evidence distilled from one result record for a cell, reduced to
/// just what reconciliation needs. Construction from a full `ResultRecord` lives
/// at the report boundary; keeping this small keeps the logic unit-testable.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellEvidence {
    /// `terminal_outcome == Passed` AND the run resolved to the requested
    /// accelerator (not a silent CPU fallback).
    pub passed_on_target: bool,
    /// `terminal_outcome` is `Blocked`/`Skipped`/`Failed` (the cell did not pass).
    pub blocked: bool,
    /// Recorded native-provider evidence shows the accelerated path was actually
    /// present in this build (EP probe available / static archive linked /
    /// `resolved_accelerator` reached the target). Per David's refinement,
    /// `BuildGap` keys on THIS, not on cargo-feature strings.
    pub native_path_present: bool,
}

/// A reconciliation disagreement between the declaration and the runtime — these
/// fail the completeness check (fail-closed).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Finding {
    /// A record passed on a cell the declaration says is `Unsupported` — the
    /// declaration is wrong or the accelerator was misreported.
    PassedOnUnsupported { reason: Reason },
}

/// Declared accelerator support for a bundle on a profile: the backend's
/// intrinsic support at the profile's target accelerator. Per-bundle overrides
/// (Q3) layer on top here once enumerated; today bundles inherit their backend.
pub fn applicability(backend: BackendKind, profile: Profile) -> AccelSupport {
    backend.accel_support(profile.accelerator())
}

/// Classify a cell from its declaration + the evidence of all records for the
/// tuple. Total: returns exactly one `CoverageState` for any input.
pub fn classify(support: AccelSupport, evidence: &[CellEvidence]) -> CoverageState {
    match support {
        AccelSupport::Unsupported(reason) => CoverageState::NotApplicable(reason),
        AccelSupport::Targetable { .. } => {
            if evidence.iter().any(|e| e.passed_on_target) {
                CoverageState::Validated
            } else if evidence.iter().any(|e| e.blocked && !e.native_path_present) {
                // Capable backend, but this build did not provide the native path.
                CoverageState::BuildGap
            } else {
                // No passing record and no build-absent evidence: unscheduled, or
                // ran with the path present but did not pass — both are "should
                // run / fill", i.e. a transient Gap.
                CoverageState::Gap
            }
        }
    }
}

/// Reconcile one record's evidence against the declaration, returning a finding
/// when they contradict. A `passed_on_target` record on an `Unsupported` cell is
/// the contradiction the framework must surface.
pub fn reconcile(support: AccelSupport, evidence: CellEvidence) -> Option<Finding> {
    match support {
        AccelSupport::Unsupported(reason) if evidence.passed_on_target => {
            Some(Finding::PassedOnUnsupported { reason })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::Accelerator;

    fn targetable() -> AccelSupport {
        AccelSupport::Targetable { feature: None }
    }

    #[test]
    fn applicability_joins_profile_to_backend_declaration() {
        // llama.cpp targets metal; mistralrs does not; ORT never does.
        assert!(matches!(
            applicability(BackendKind::LlamaCpp, Profile::AppleMetal),
            AccelSupport::Targetable { .. }
        ));
        assert_eq!(
            applicability(BackendKind::MistralRs, Profile::AppleMetal),
            AccelSupport::Unsupported(Reason::UpstreamNoGpuForwarding),
        );
        assert_eq!(
            applicability(BackendKind::Ort, Profile::AppleMetal),
            AccelSupport::Unsupported(Reason::NoExecutionProviderForAccelerator),
        );
        // And a cuda profile maps to the cuda accel: llama.cpp targetable there.
        assert!(matches!(
            applicability(BackendKind::LlamaCpp, Profile::DgxSpark),
            AccelSupport::Targetable { .. }
        ));
        // Sanity: the bridge is the profile's accelerator.
        assert_eq!(Profile::DgxSpark.accelerator(), Accelerator::Cuda);
    }

    #[test]
    fn classify_unsupported_is_not_applicable() {
        let support = AccelSupport::Unsupported(Reason::NoExecutionProviderForAccelerator);
        assert_eq!(
            classify(support, &[]),
            CoverageState::NotApplicable(Reason::NoExecutionProviderForAccelerator),
        );
    }

    #[test]
    fn classify_passed_on_target_is_validated() {
        let ev = CellEvidence {
            passed_on_target: true,
            native_path_present: true,
            ..Default::default()
        };
        assert_eq!(classify(targetable(), &[ev]), CoverageState::Validated);
    }

    #[test]
    fn classify_blocked_without_native_path_is_build_gap() {
        // The #513 class: capable backend, but this build didn't compile/provision
        // the native path (evidence shows it absent).
        let ev = CellEvidence {
            blocked: true,
            native_path_present: false,
            ..Default::default()
        };
        assert_eq!(classify(targetable(), &[ev]), CoverageState::BuildGap);
    }

    #[test]
    fn classify_no_records_is_gap() {
        assert_eq!(classify(targetable(), &[]), CoverageState::Gap);
    }

    #[test]
    fn classify_blocked_with_native_path_present_is_gap_not_build_gap() {
        // Path was present but the cell still didn't pass -> not a BuildGap
        // (nothing to build); it's a Gap/should-run, distinct from #513.
        let ev = CellEvidence {
            blocked: true,
            native_path_present: true,
            ..Default::default()
        };
        assert_eq!(classify(targetable(), &[ev]), CoverageState::Gap);
    }

    #[test]
    fn reconcile_flags_pass_on_unsupported() {
        // The fail-closed contradiction: a pass on a declared-Unsupported cell.
        let support = AccelSupport::Unsupported(Reason::UpstreamNoGpuForwarding);
        let ev = CellEvidence {
            passed_on_target: true,
            ..Default::default()
        };
        assert_eq!(
            reconcile(support, ev),
            Some(Finding::PassedOnUnsupported {
                reason: Reason::UpstreamNoGpuForwarding
            }),
        );
        // A blocked record on the same cell is consistent -> no finding.
        let blocked = CellEvidence {
            blocked: true,
            ..Default::default()
        };
        assert_eq!(reconcile(support, blocked), None);
    }
}
