//! Coverage reconciliation (#521): the runtime half that classifies each
//! `(CuratedBundle, QuantizationScheme, CapabilityKind, Profile)` cell into the
//! 4-state taxonomy by joining the compile-time declaration
//! (`BackendKind::accel_support`, reached via `Profile::accelerator()`) with the
//! recorded results.
//!
//! The reconciliation *is* the framework's value: the declaration and the
//! runtime evidence must agree, and any disagreement is a flagged finding
//! (`Finding`) that fails the fail-closed completeness check.
//!
//! HOOK(#531) — Speech sub-dimension: the cell key will gain a
//! capability-conditional `SpeechGeneration::{Buffered, Streaming}` sub-dimension
//! that is present iff `capability == Speech`, so buffered vs streaming become
//! distinct cells (DESIGN §G). `SpeechGeneration` is owned by motlie-model
//! (#531/#524) and is intentionally NOT defined here; when it lands, add the
//! optional field to the cell-key type (and the index entry) in this module —
//! reconciliation/`applicability` are unaffected (accelerator support is
//! speech-mode-independent), so the change is localized to cell keying/slicing.

use std::collections::BTreeMap;

use motlie_model::{AccelSupport, BackendKind, CapabilityKind, QuantizationScheme, Reason};
use motlie_models::CuratedBundle;

use crate::profile::Profile;
use crate::result::{AcceleratorClass, ResultRecord, TerminalOutcome};

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

/// The enum-keyed coverage cell identity (#521 tuple). `bundle_id` is validated
/// against `CuratedBundle::CANONICAL_IDS` rather than resolved to the (feature-
/// gated) variant, so reconciliation works without the all-curated-features
/// build. The HOOK(#531) Speech sub-dimension will be added here, conditional on
/// `capability == Speech`.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CellKey {
    pub bundle_id: String,
    pub quant: QuantizationScheme,
    pub capability: CapabilityKind,
    pub profile: Profile,
}

/// Why a record failed to parse into the enum-keyed tuple — every variant is a
/// fail-closed condition (an undeclared/uncanonical value the test must reject).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TupleParseError {
    BundleNotCanonical(String),
    UnknownQuant(String),
    UnknownCapability(String),
    UnknownProfile(String),
    UnknownBackend(String),
}

/// The mistralrs-perf scenario maps onto Chat (perf is a scenario over Chat, not
/// a `CapabilityKind`; DESIGN §E). asr→Transcription, tts→Speech.
fn capability_from_eval(token: &str) -> Option<CapabilityKind> {
    match token {
        "chat" | "perf" => Some(CapabilityKind::Chat),
        "tool_use" => Some(CapabilityKind::ToolUse),
        "asr" => Some(CapabilityKind::Transcription),
        "tts" => Some(CapabilityKind::Speech),
        "embeddings" => Some(CapabilityKind::Embeddings),
        _ => None,
    }
}

fn backend_from_token(token: &str) -> Option<BackendKind> {
    match token {
        "mistralrs" => Some(BackendKind::MistralRs),
        "llama_cpp" => Some(BackendKind::LlamaCpp),
        "ort" => Some(BackendKind::Ort),
        "sherpa_onnx" => Some(BackendKind::SherpaOnnx),
        "whisper_cpp" => Some(BackendKind::WhisperCpp),
        "qwen3_tts_cpp" => Some(BackendKind::Qwen3TtsCpp),
        "http" => Some(BackendKind::Http),
        _ => None,
    }
}

/// Parse one record into its enum-keyed cell + backend + runtime evidence —
/// fail-closed on any non-canonical dimension value.
pub fn parse_cell(
    record: &ResultRecord,
) -> Result<(CellKey, BackendKind, CellEvidence), TupleParseError> {
    let coverage = &record.coverage;
    if !CuratedBundle::CANONICAL_IDS.contains(&coverage.bundle_id.as_str()) {
        return Err(TupleParseError::BundleNotCanonical(
            coverage.bundle_id.clone(),
        ));
    }
    let quant = coverage
        .quantization
        .parse::<QuantizationScheme>()
        .map_err(|_| TupleParseError::UnknownQuant(coverage.quantization.clone()))?;
    let capability = capability_from_eval(&coverage.capability)
        .ok_or_else(|| TupleParseError::UnknownCapability(coverage.capability.clone()))?;
    let profile = Profile::from_id(&coverage.profile)
        .ok_or_else(|| TupleParseError::UnknownProfile(coverage.profile.clone()))?;
    let backend = backend_from_token(&coverage.backend)
        .ok_or_else(|| TupleParseError::UnknownBackend(coverage.backend.clone()))?;

    let key = CellKey {
        bundle_id: coverage.bundle_id.clone(),
        quant,
        capability,
        profile,
    };
    Ok((key, backend, evidence_from_record(record)))
}

fn on_physical_target(requested: AcceleratorClass, resolved: AcceleratorClass) -> bool {
    requested == resolved
        && !matches!(
            resolved,
            AcceleratorClass::Any | AcceleratorClass::Unavailable
        )
}

/// Distil the runtime evidence a record carries for reconciliation. `BuildGap`
/// keys on whether the native path was present (reached the target accel, or the
/// backend reported the target execution mode) — recorded evidence, NOT cargo
/// features (David's refinement).
pub fn evidence_from_record(record: &ResultRecord) -> CellEvidence {
    let requested = record.coverage.requested_accelerator;
    let resolved = record.coverage.resolved_accelerator;
    let on_target = on_physical_target(requested, resolved);
    let backend_reached_target = record
        .accelerator
        .backend_mode
        .as_deref()
        .is_some_and(|mode| mode.eq_ignore_ascii_case(requested.as_str()));
    CellEvidence {
        passed_on_target: record.coverage.terminal_outcome == TerminalOutcome::Passed && on_target,
        blocked: matches!(
            record.coverage.terminal_outcome,
            TerminalOutcome::Blocked | TerminalOutcome::Failed | TerminalOutcome::Skipped
        ),
        native_path_present: on_target || backend_reached_target,
    }
}

/// The outcome of reconciling a record set: per-cell state, contradiction
/// findings, and any records that didn't parse into the enum tuple. A clean
/// record set has empty `findings` and `parse_errors` (the fail-closed bar).
#[derive(Clone, Debug, Default)]
pub struct ReconcileReport {
    pub states: BTreeMap<CellKey, CoverageState>,
    pub findings: Vec<(CellKey, Finding)>,
    pub parse_errors: Vec<TupleParseError>,
}

/// Reconcile a record set into per-cell states + findings, fail-closed: a record
/// that doesn't parse into the enum tuple, or a record that contradicts the
/// declaration, is reported (and fails the completeness test).
pub fn reconcile_records<'a, I>(records: I) -> ReconcileReport
where
    I: IntoIterator<Item = &'a ResultRecord>,
{
    let mut cells: BTreeMap<CellKey, (BackendKind, Vec<CellEvidence>)> = BTreeMap::new();
    let mut report = ReconcileReport::default();
    for record in records {
        match parse_cell(record) {
            Ok((key, backend, evidence)) => {
                let support = applicability(backend, key.profile);
                if let Some(finding) = reconcile(support, evidence) {
                    report.findings.push((key.clone(), finding));
                }
                cells
                    .entry(key)
                    .or_insert((backend, Vec::new()))
                    .1
                    .push(evidence);
            }
            Err(err) => report.parse_errors.push(err),
        }
    }
    for (key, (backend, evidence)) in cells {
        let support = applicability(backend, key.profile);
        report.states.insert(key, classify(support, &evidence));
    }
    report
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

    fn collect_results_jsonl(root: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_results_jsonl(&path, out);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("results.jsonl") {
                out.push(path);
            }
        }
    }

    fn committed_records() -> Vec<ResultRecord> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evals/results");
        let mut files = Vec::new();
        collect_results_jsonl(&root, &mut files);
        files.sort();
        let mut records = Vec::new();
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(record) = serde_json::from_str::<ResultRecord>(line) {
                    records.push(record);
                }
            }
        }
        records
    }

    #[test]
    fn committed_records_reconcile_fail_closed() {
        // Fail-closed completeness over the real committed data set: every record
        // must parse into the enum-keyed tuple (canonical bundle/quant/capability/
        // profile/backend), and no record may contradict the declaration.
        let records = committed_records();
        assert!(
            !records.is_empty(),
            "expected committed result records under evals/results",
        );
        let report = reconcile_records(&records);
        eprintln!(
            "reconciled {} records -> {} cells, {} parse_errors, {} findings",
            records.len(),
            report.states.len(),
            report.parse_errors.len(),
            report.findings.len(),
        );
        for err in report.parse_errors.iter().take(10) {
            eprintln!("  parse_error: {err:?}");
        }
        for (key, finding) in report.findings.iter().take(10) {
            eprintln!(
                "  finding: {finding:?} @ {}/{:?}/{:?}",
                key.bundle_id, key.capability, key.profile
            );
        }
        assert!(
            report.parse_errors.is_empty(),
            "{} records failed to parse into the enum tuple (fail-closed)",
            report.parse_errors.len(),
        );
        assert!(
            report.findings.is_empty(),
            "{} declaration/runtime contradictions found (fail-closed)",
            report.findings.len(),
        );
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
