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
//! Speech sub-dimension (#531, WIRED): `CellKey.speech_mode` carries the
//! capability-conditional `SpeechGeneration::{Buffered, Streaming}` (present iff
//! `capability == Speech`; DESIGN §G), so buffered vs streaming are distinct
//! cells. Streaming eval is out of scope this iteration: streaming cells are
//! *synthesized* as declared `Gap(StreamingEvalOutOfScope)` (accounted, never
//! silently absent) for bundles advertising streaming, while no streaming eval
//! runs. `applicability` is unaffected (accelerator support is
//! speech-mode-independent).

use std::collections::BTreeMap;

use motlie_model::{
    AccelSupport, BackendKind, CapabilityKind, QuantizationScheme, Reason, SpeechGeneration,
};
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
    /// Carries an optional eval-level reason (e.g. a dimension value that is
    /// declared but deliberately out of scope this iteration — accounted, never
    /// silently absent).
    Gap(Option<GapReason>),
}

/// Eval-level reason a cell is a `Gap` (distinct from the model-capability
/// `Reason` that drives `NotApplicable`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GapReason {
    /// The cell is a declared dimension value whose eval is out of scope for this
    /// iteration (e.g. streaming TTS, #531) — declared, not run, not silent.
    StreamingEvalOutOfScope,
}

impl GapReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StreamingEvalOutOfScope => "streaming_eval_out_of_scope",
        }
    }
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
                // run / fill", i.e. a transient Gap (no special eval reason).
                CoverageState::Gap(None)
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
    /// Capability-conditional Speech sub-dimension (#531): `Some(mode)` iff
    /// `capability == Speech`, `None` otherwise — keeps the matrix sparse
    /// (DESIGN §G). Buffered and Streaming are distinct cells.
    pub speech_mode: Option<SpeechGeneration>,
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

    // Speech sub-dimension: today only buffered TTS is run, so a recorded Speech
    // cell is Buffered (streaming has no records — it is synthesized as a declared
    // out-of-scope Gap in `reconcile_records`). Non-Speech cells carry no mode.
    let speech_mode = (capability == CapabilityKind::Speech).then_some(SpeechGeneration::Buffered);
    let key = CellKey {
        bundle_id: coverage.bundle_id.clone(),
        quant,
        capability,
        profile,
        speech_mode,
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
    synthesize_streaming_cells(&mut report);
    report
}

/// Curated bundles that advertise `SpeechGeneration::Streaming` in addition to
/// Buffered (mirrors the motlie-model descriptor, #531 —
/// `kokoro_82m` is the one curated bundle advertising both modes). Kept as a
/// declared canonical-id set so the feature-light report can account streaming
/// without compiling the catalog; HOOK(#531) for when more bundles add streaming.
fn bundle_advertises_streaming(bundle_id: &str) -> bool {
    matches!(bundle_id, "kokoro_82m")
}

/// Account for the declared-but-out-of-scope Streaming sub-dimension: for every
/// covered Buffered Speech cell whose bundle advertises streaming, emit the
/// matching Streaming cell as a `Gap(StreamingEvalOutOfScope)`. Streaming is
/// therefore declared and visible (honest coverage), never silently absent —
/// while no streaming eval is executed this iteration (David).
fn synthesize_streaming_cells(report: &mut ReconcileReport) {
    let buffered_keys: Vec<CellKey> = report
        .states
        .keys()
        .filter(|key| {
            key.speech_mode == Some(SpeechGeneration::Buffered)
                && bundle_advertises_streaming(&key.bundle_id)
        })
        .cloned()
        .collect();
    for buffered in buffered_keys {
        let streaming = CellKey {
            speech_mode: Some(SpeechGeneration::Streaming),
            ..buffered
        };
        report
            .states
            .entry(streaming)
            .or_insert(CoverageState::Gap(Some(GapReason::StreamingEvalOutOfScope)));
    }
}

/// One typed, slice-able entry of the coverage index (§F). Every dimension is a
/// canonical string field, so the index is queryable by ANY dimension and any
/// cross-cut (e.g. `profile == "dgx-spark"`, `quant == "gguf_q4_0"`,
/// `accelerator == "cuda"`) without ad-hoc grep. Serialized in a stable field
/// order; the index as a whole is sorted, so regeneration is byte-stable.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize)]
pub struct CoverageIndexEntry {
    pub bundle_id: String,
    pub quant: String,
    pub capability: String,
    /// Speech sub-dimension (`buffered`/`streaming`), present iff capability is
    /// Speech/`tts` (#531); `None` otherwise.
    pub speech_mode: Option<String>,
    pub profile: String,
    pub accelerator: String,
    pub state: String,
    pub reason: Option<String>,
}

fn speech_mode_token(mode: SpeechGeneration) -> &'static str {
    match mode {
        SpeechGeneration::Buffered => "buffered",
        SpeechGeneration::Streaming => "streaming",
    }
}

fn capability_token(capability: CapabilityKind) -> &'static str {
    match capability {
        CapabilityKind::Chat => "chat",
        CapabilityKind::ToolUse => "tool_use",
        CapabilityKind::Transcription => "asr",
        CapabilityKind::Speech => "tts",
        CapabilityKind::Embeddings => "embeddings",
        CapabilityKind::Completion => "completion",
        CapabilityKind::Ocr => "ocr",
        CapabilityKind::Vision => "vision",
        CapabilityKind::VoiceClone => "voice_clone",
    }
}

fn state_fields(state: &CoverageState) -> (&'static str, Option<String>) {
    match state {
        CoverageState::Validated => ("validated", None),
        CoverageState::NotApplicable(reason) => {
            ("not_applicable", Some(reason.as_str().to_owned()))
        }
        CoverageState::BuildGap => ("build_gap", None),
        CoverageState::Gap(reason) => ("gap", reason.map(|r| r.as_str().to_owned())),
    }
}

/// Build the deterministic, enum-keyed coverage index from a record set: one
/// entry per reconciled cell, sorted by `(bundle, quant, capability, profile)`
/// (via the `BTreeMap` key order in the reconcile report), so the output is
/// byte-stable across runs given the same inputs (Q4 / David's regen
/// constraint). The Accounting Matrix (report) is one pivot over this; arbitrary
/// slices are a filter on the typed fields.
pub fn build_index(report: &ReconcileReport) -> Vec<CoverageIndexEntry> {
    let mut entries: Vec<CoverageIndexEntry> = report
        .states
        .iter()
        .map(|(key, state)| {
            let (state, reason) = state_fields(state);
            CoverageIndexEntry {
                bundle_id: key.bundle_id.clone(),
                quant: key.quant.as_str().to_owned(),
                capability: capability_token(key.capability).to_owned(),
                speech_mode: key.speech_mode.map(|m| speech_mode_token(m).to_owned()),
                profile: key.profile.canonical_id().to_owned(),
                accelerator: key.profile.accelerator().as_str().to_owned(),
                state: state.to_owned(),
                reason,
            }
        })
        .collect();
    // Sort by the canonical string fields (Ord derive) so the order is stable
    // and independent of enum discriminant order — byte-stable regen (§9/§F).
    entries.sort();
    entries
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
        assert_eq!(classify(targetable(), &[]), CoverageState::Gap(None));
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
        assert_eq!(classify(targetable(), &[ev]), CoverageState::Gap(None));
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
    fn coverage_index_is_deterministic_and_sliceable() {
        let records = committed_records();
        let report = reconcile_records(&records);

        // Byte-stable regeneration (Q4 / David's constraint): same inputs ->
        // identical, sorted output.
        let index = build_index(&report);
        assert_eq!(index, build_index(&report), "index regen must be stable");
        let mut sorted = index.clone();
        sorted.sort();
        assert_eq!(
            index, sorted,
            "index entries must be in stable sorted order"
        );
        assert!(!index.is_empty());

        // Slice-and-dice by ANY dimension + cross-cut (§F).
        assert!(index.iter().any(|e| e.profile == "dgx-spark"));
        assert!(index
            .iter()
            .filter(|e| e.quant == "gguf_q4_0")
            .all(|e| e.bundle_id == "gemma4_12b_qat_gguf")); // 1:1 quant<->bundle
        assert!(index
            .iter()
            .filter(|e| e.accelerator == "cuda")
            .all(|e| e.profile == "dgx-spark" || e.profile == "cuda-workstation"));
        assert!(index.iter().any(|e| e.capability == "asr"));

        // Serializes with stable typed fields.
        let json = serde_json::to_string(&index).expect("index serializes");
        assert!(json.contains("\"capability\":\"asr\""));
        assert!(json.contains("\"state\":"));
    }

    #[test]
    fn streaming_cells_are_declared_out_of_scope_gaps() {
        // #531: kokoro_82m advertises Buffered + Streaming. Buffered cells stay as
        // recorded; streaming cells are synthesized as declared out-of-scope Gaps
        // (accounted, never silently absent), and they introduce no findings.
        let records = committed_records();
        let report = reconcile_records(&records);

        let streaming: Vec<_> = report
            .states
            .iter()
            .filter(|(key, _)| {
                key.bundle_id == "kokoro_82m"
                    && key.speech_mode == Some(SpeechGeneration::Streaming)
            })
            .collect();
        assert!(!streaming.is_empty(), "streaming cells must be declared");
        assert!(streaming.iter().all(|(_, state)| matches!(
            state,
            CoverageState::Gap(Some(GapReason::StreamingEvalOutOfScope))
        )));

        // A matching Buffered cell exists for each synthesized streaming cell.
        for (streaming_key, _) in &streaming {
            let buffered = CellKey {
                speech_mode: Some(SpeechGeneration::Buffered),
                ..(*streaming_key).clone()
            };
            assert!(report.states.contains_key(&buffered));
        }
        // Streaming declaration does not break the fail-closed bar.
        assert!(report.findings.is_empty() && report.parse_errors.is_empty());

        // Index surfaces the speech_mode + the gap reason (sliceable).
        let index = build_index(&report);
        assert!(index
            .iter()
            .any(|e| e.speech_mode.as_deref() == Some("streaming")
                && e.state == "gap"
                && e.reason.as_deref() == Some("streaming_eval_out_of_scope")));
        assert!(index
            .iter()
            .any(|e| e.speech_mode.as_deref() == Some("buffered")));
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
