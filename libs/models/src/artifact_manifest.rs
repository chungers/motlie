use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use motlie_model::{ArtifactRule, ArtifactSource, BundleId, CheckpointFormat, QuantizationScheme};
use serde_json::Value;
use thiserror::Error;

use crate::{
    download_bundle_artifacts_with_options, ArtifactDownloadOptions, ArtifactDownloadSummary,
    ArtifactProvenance, Catalog, CuratedBundle, DerivedArtifactRecipe, ModelsError,
};

pub const CANONICAL_ARTIFACT_ROOT_DISPLAY: &str = "$HOME/artifacts/hf-cache";

#[derive(Debug, Error)]
pub enum ArtifactManifestError {
    #[error("compiled catalog is missing {missing_count} curated artifact bundle(s): {missing}")]
    IncompleteCatalog {
        missing_count: usize,
        missing: String,
    },
    #[error("canonical bundle `{bundle_id}` is missing from the compiled catalog")]
    MissingBundle { bundle_id: String },
    #[error("bundle `{bundle_id}` does not define curated artifacts")]
    MissingArtifacts { bundle_id: String },
    #[error("unsupported artifact source for bundle `{bundle_id}`")]
    UnsupportedSource { bundle_id: String },
    #[error("failed to inspect `{repo}` on Hugging Face: {message}")]
    HuggingFaceMetadata { repo: String, message: String },
    #[error("failed to read artifact cache `{path}`: {source}")]
    ReadCache {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("artifact download failed: {0}")]
    Download(#[from] ModelsError),
}

pub type Result<T> = std::result::Result<T, ArtifactManifestError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactBundleEntry {
    pub bundle_id: String,
    pub display_name: String,
    pub capabilities: Vec<String>,
    pub format: CheckpointFormat,
    pub quantization: Option<QuantizationScheme>,
    pub sources: Vec<ArtifactSourceEntry>,
    pub derived: Vec<ArtifactDerivedEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSourceEntry {
    pub label: String,
    pub repo: String,
    pub requirements: Vec<ArtifactRequirement>,
    pub provenance: ArtifactProvenance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactDerivedEntry {
    pub output: String,
    pub source: String,
    pub recipe: String,
}

impl ArtifactDerivedEntry {
    pub fn matches(&self, filename: &str) -> bool {
        if self.output.ends_with('/') {
            filename.starts_with(&self.output)
        } else {
            filename == self.output
        }
    }

    pub fn label(&self) -> String {
        if self.output.ends_with('/') {
            format!("{}** <- {} ({})", self.output, self.source, self.recipe)
        } else {
            format!("{} <- {} ({})", self.output, self.source, self.recipe)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactRequirement {
    Exact(String),
    Prefix(String),
    Suffix(String),
}

impl ArtifactRequirement {
    pub fn from_rule(rule: &ArtifactRule) -> Self {
        match rule {
            ArtifactRule::Exact(value) => Self::Exact((*value).to_owned()),
            ArtifactRule::Prefix(value) => Self::Prefix((*value).to_owned()),
            ArtifactRule::Suffix(value) => Self::Suffix((*value).to_owned()),
        }
    }

    pub fn matches(&self, filename: &str) -> bool {
        match self {
            Self::Exact(expected) => filename == expected,
            Self::Prefix(prefix) => filename.starts_with(prefix),
            Self::Suffix(suffix) => filename.ends_with(suffix),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Exact(value) => value.clone(),
            Self::Prefix(value) => format!("{value}**"),
            Self::Suffix(value) => format!("*{value}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfRepoMetadata {
    pub repo: String,
    pub sha: Option<String>,
    pub license: Option<String>,
    pub gated: Option<String>,
    pub siblings: Vec<String>,
}

impl HfRepoMetadata {
    pub fn unknown(repo: &str) -> Self {
        Self {
            repo: repo.to_owned(),
            sha: None,
            license: None,
            gated: None,
            siblings: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactPreflightReport {
    pub artifact_root: PathBuf,
    pub bundles: Vec<ArtifactBundleCheck>,
}

impl ArtifactPreflightReport {
    pub fn complete_bundle_count(&self) -> usize {
        self.bundles
            .iter()
            .filter(|bundle| bundle.is_complete())
            .count()
    }

    pub fn missing_rule_count(&self) -> usize {
        self.bundles
            .iter()
            .map(ArtifactBundleCheck::missing_requirement_count)
            .sum()
    }

    pub fn is_complete(&self) -> bool {
        self.missing_rule_count() == 0 && self.bundles.iter().all(ArtifactBundleCheck::is_complete)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactBundleCheck {
    pub bundle_id: String,
    pub sources: Vec<ArtifactSourceCheck>,
    pub derived: Vec<ArtifactDerivedCheck>,
}

impl ArtifactBundleCheck {
    pub fn is_complete(&self) -> bool {
        self.sources.iter().all(ArtifactSourceCheck::is_complete)
            && self.derived.iter().all(|artifact| artifact.present)
    }

    pub fn missing_requirement_count(&self) -> usize {
        self.sources
            .iter()
            .map(|source| source.missing_requirements().count())
            .sum::<usize>()
            + self
                .derived
                .iter()
                .filter(|artifact| !artifact.present)
                .count()
    }

    pub fn missing_requirements(&self) -> impl Iterator<Item = &ArtifactRequirementCheck> {
        self.sources
            .iter()
            .flat_map(ArtifactSourceCheck::missing_requirements)
    }

    pub fn missing_derived(&self) -> impl Iterator<Item = &ArtifactDerivedCheck> {
        self.derived.iter().filter(|artifact| !artifact.present)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSourceCheck {
    pub label: String,
    pub repo: String,
    pub repo_cache_dir: PathBuf,
    pub snapshot: Option<String>,
    pub remote_snapshot: Option<String>,
    pub metadata: Option<HfRepoMetadata>,
    pub metadata_error: Option<String>,
    pub requirements: Vec<ArtifactRequirementCheck>,
    pub files: Vec<String>,
}

impl ArtifactSourceCheck {
    pub fn is_complete(&self) -> bool {
        self.snapshot.is_some() && self.requirements.iter().all(|rule| rule.present)
    }

    pub fn missing_requirements(&self) -> impl Iterator<Item = &ArtifactRequirementCheck> {
        self.requirements.iter().filter(|rule| !rule.present)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRequirementCheck {
    pub requirement: ArtifactRequirement,
    pub present: bool,
    pub matches: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactDerivedCheck {
    pub artifact: ArtifactDerivedEntry,
    pub present: bool,
    pub matches: Vec<String>,
}

pub fn canonical_artifact_root() -> PathBuf {
    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("artifacts/hf-cache")
}

pub fn curated_artifact_entries(catalog: &Catalog) -> Result<Vec<ArtifactBundleEntry>> {
    ensure_complete_curated_catalog(catalog)?;
    let mut entries = Vec::new();
    for bundle_id in CuratedBundle::CANONICAL_IDS {
        let descriptor = catalog.bundle(&BundleId::new(*bundle_id)).ok_or_else(|| {
            ArtifactManifestError::MissingBundle {
                bundle_id: (*bundle_id).to_owned(),
            }
        })?;
        let artifacts = descriptor.artifacts.as_ref().ok_or_else(|| {
            ArtifactManifestError::MissingArtifacts {
                bundle_id: (*bundle_id).to_owned(),
            }
        })?;
        let mut sources = Vec::with_capacity(1 + artifacts.extra_sources.len());
        sources.push(source_entry(
            "primary",
            &artifacts.source,
            &artifacts.include,
            artifacts.provenance,
            bundle_id,
        )?);
        for source in &artifacts.extra_sources {
            sources.push(source_entry(
                source.label,
                &source.source,
                &source.include,
                source.provenance,
                bundle_id,
            )?);
        }
        entries.push(ArtifactBundleEntry {
            bundle_id: descriptor.id.as_str().to_owned(),
            display_name: descriptor.display_name.clone(),
            capabilities: capability_labels(descriptor),
            format: artifacts.format,
            quantization: artifacts.quantization,
            sources,
            derived: artifacts
                .derived
                .iter()
                .map(|artifact| ArtifactDerivedEntry {
                    output: artifact.output.to_owned(),
                    source: artifact.recipe.source().to_owned(),
                    recipe: recipe_label(&artifact.recipe).to_owned(),
                })
                .collect(),
        });
    }
    Ok(entries)
}

fn source_entry(
    label: &str,
    source: &ArtifactSource,
    include: &[ArtifactRule],
    provenance: ArtifactProvenance,
    bundle_id: &&str,
) -> Result<ArtifactSourceEntry> {
    let ArtifactSource::HuggingFace { repo } = source;
    if include.is_empty() {
        return Err(ArtifactManifestError::MissingArtifacts {
            bundle_id: (**bundle_id).to_owned(),
        });
    }
    Ok(ArtifactSourceEntry {
        label: label.to_owned(),
        repo: (*repo).to_owned(),
        requirements: include.iter().map(ArtifactRequirement::from_rule).collect(),
        provenance,
    })
}

fn recipe_label(recipe: &DerivedArtifactRecipe) -> &'static str {
    recipe.label()
}

fn capability_labels(descriptor: &crate::BundleDescriptor) -> Vec<String> {
    descriptor
        .capability_descriptors()
        .iter()
        .map(|capability| {
            let mut label = format!("{:?}", capability.kind);
            if !capability.speech_generations.is_empty() {
                let generations = capability
                    .speech_generations
                    .iter()
                    .map(|generation| format!("{:?}", generation))
                    .collect::<Vec<_>>()
                    .join(",");
                label.push('[');
                label.push_str(&generations);
                label.push(']');
            }
            label
        })
        .collect()
}

pub fn ensure_complete_curated_catalog(catalog: &Catalog) -> Result<()> {
    let compiled = catalog
        .bundles()
        .map(|descriptor| descriptor.id.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    let missing = CuratedBundle::CANONICAL_IDS
        .iter()
        .copied()
        .filter(|bundle_id| !compiled.contains(*bundle_id))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(ArtifactManifestError::IncompleteCatalog {
            missing_count: missing.len(),
            missing: missing.join(", "),
        })
    }
}

pub fn check_curated_artifacts(
    catalog: &Catalog,
    artifact_root: &Path,
    metadata: &BTreeMap<String, std::result::Result<HfRepoMetadata, String>>,
) -> Result<ArtifactPreflightReport> {
    let entries = curated_artifact_entries(catalog)?;
    check_artifact_entries(&entries, artifact_root, metadata).map_err(|source| {
        ArtifactManifestError::ReadCache {
            path: artifact_root.to_path_buf(),
            source,
        }
    })
}

pub fn check_artifact_entries(
    entries: &[ArtifactBundleEntry],
    artifact_root: &Path,
    metadata: &BTreeMap<String, std::result::Result<HfRepoMetadata, String>>,
) -> std::result::Result<ArtifactPreflightReport, std::io::Error> {
    let bundles = entries
        .iter()
        .map(|entry| check_entry(entry, artifact_root, metadata))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(ArtifactPreflightReport {
        artifact_root: artifact_root.to_path_buf(),
        bundles,
    })
}

pub fn fetch_metadata_for_entries(
    entries: &[ArtifactBundleEntry],
    hf_token: Option<String>,
) -> BTreeMap<String, std::result::Result<HfRepoMetadata, String>> {
    let mut repos = entries
        .iter()
        .flat_map(|entry| entry.sources.iter())
        .map(|source| source.repo.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    repos.sort();

    let mut metadata = BTreeMap::new();
    for repo in repos {
        let result = fetch_hf_repo_metadata(&repo, hf_token.clone()).map_err(|error| match error {
            ArtifactManifestError::HuggingFaceMetadata { message, .. } => message,
            other => other.to_string(),
        });
        metadata.insert(repo, result);
    }
    metadata
}

pub fn fetch_hf_repo_metadata(repo: &str, hf_token: Option<String>) -> Result<HfRepoMetadata> {
    let api = ApiBuilder::new()
        .with_token(hf_token)
        .with_progress(false)
        .build()
        .map_err(|source| ArtifactManifestError::HuggingFaceMetadata {
            repo: repo.to_owned(),
            message: source.to_string(),
        })?;
    let repo_api = api.model(repo.to_owned());
    let mut response = repo_api.info_request().call().map_err(|source| {
        ArtifactManifestError::HuggingFaceMetadata {
            repo: repo.to_owned(),
            message: source.to_string(),
        }
    })?;
    let value: Value = response.body_mut().read_json().map_err(|source| {
        ArtifactManifestError::HuggingFaceMetadata {
            repo: repo.to_owned(),
            message: source.to_string(),
        }
    })?;

    Ok(HfRepoMetadata {
        repo: repo.to_owned(),
        sha: value.get("sha").and_then(Value::as_str).map(str::to_owned),
        license: value
            .get("cardData")
            .and_then(|card| card.get("license"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| license_from_tags(&value)),
        gated: value.get("gated").map(gated_value_to_string),
        siblings: value
            .get("siblings")
            .and_then(Value::as_array)
            .map(|siblings| {
                siblings
                    .iter()
                    .filter_map(|sibling| sibling.get("rfilename"))
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    })
}

pub fn sync_missing_curated_artifacts(
    catalog: &Catalog,
    artifact_root: &Path,
    hf_token: Option<String>,
) -> Result<Vec<ArtifactDownloadSummary>> {
    let entries = curated_artifact_entries(catalog)?;
    let metadata = fetch_metadata_for_entries(&entries, hf_token.clone());
    let report = check_curated_artifacts(catalog, artifact_root, &metadata)?;
    let mut downloaded = Vec::new();
    for bundle in report.bundles.iter().filter(|bundle| !bundle.is_complete()) {
        downloaded.push(download_bundle_artifacts_with_options(
            catalog,
            &BundleId::new(&bundle.bundle_id),
            artifact_root,
            &ArtifactDownloadOptions {
                hf_token: hf_token.clone(),
                quantization_scheme: None,
            },
        )?);
    }
    Ok(downloaded)
}

pub fn render_preflight_report(report: &ArtifactPreflightReport) -> String {
    let mut output = String::new();
    output.push_str("# Artifact Preflight\n");
    output.push_str(&format!(
        "artifact_root: {}\n",
        report.artifact_root.display()
    ));
    output.push_str(&format!(
        "bundles: {}/{} complete\n",
        report.complete_bundle_count(),
        report.bundles.len()
    ));
    output.push_str(&format!(
        "missing_requirements: {}\n\n",
        report.missing_rule_count()
    ));

    for bundle in &report.bundles {
        let status = if bundle.is_complete() {
            "OK"
        } else {
            "MISSING"
        };
        if bundle.sources.len() == 1 && bundle.derived.is_empty() {
            let source = &bundle.sources[0];
            output.push_str(&format!(
                "{status}\t{}\trepo={}\tsnapshot={}\tlicense={}\tgated={}\n",
                bundle.bundle_id,
                source.repo,
                source_snapshot_label(source),
                source_license_label(source),
                source_gated_label(source)
            ));
            render_source_details(&mut output, source);
        } else {
            output.push_str(&format!(
                "{status}\t{}\tsources={}\tderived={}\n",
                bundle.bundle_id,
                bundle.sources.len(),
                bundle.derived.len()
            ));
            for source in &bundle.sources {
                output.push_str(&format!(
                    "  source: {}\trepo={}\tsnapshot={}\tlicense={}\tgated={}\n",
                    source.label,
                    source.repo,
                    source_snapshot_label(source),
                    source_license_label(source),
                    source_gated_label(source)
                ));
                render_source_details(&mut output, source);
            }
            for missing in bundle.missing_derived() {
                output.push_str(&format!(
                    "  missing: derived {}\n",
                    missing.artifact.label()
                ));
            }
        }
    }

    output
}

fn render_source_details(output: &mut String, source: &ArtifactSourceCheck) {
    if let Some(error) = &source.metadata_error {
        output.push_str(&format!("  metadata_warning: {error}\n"));
    }
    if source.snapshot.is_none() {
        output.push_str(&format!("  missing: {} refs/main\n", source.label));
    }
    for missing in source.missing_requirements() {
        output.push_str(&format!(
            "  missing: {} {}\n",
            source.label,
            missing.requirement.label()
        ));
    }
}

fn source_snapshot_label(source: &ArtifactSourceCheck) -> &str {
    source
        .snapshot
        .as_deref()
        .or(source.remote_snapshot.as_deref())
        .unwrap_or("unresolved")
}

fn source_license_label(source: &ArtifactSourceCheck) -> &str {
    source
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.license.as_deref())
        .unwrap_or("unknown")
}

fn source_gated_label(source: &ArtifactSourceCheck) -> &str {
    source
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.gated.as_deref())
        .unwrap_or("unknown")
}

pub fn render_provenance_markdown(entries: &[ArtifactBundleEntry]) -> String {
    let mut output = String::new();
    output.push_str("# Curated Artifact Provenance\n\n");
    output.push_str("Generated from the curated bundle registry. Do not edit by hand; regenerate with `evals artifacts provenance`.\n\n");
    output.push_str(&format!(
        "Canonical cache root: `{}`.\n\n",
        CANONICAL_ARTIFACT_ROOT_DISPLAY
    ));
    output.push_str("Snapshot hashes are resolved by `evals preflight` from the local Hugging Face cache and live HF metadata. Derived artifacts are reproducibly generated or copied by the registry sync path after downloads complete.\n\n");
    output.push_str(
        "| Bundle | Capabilities | HF sources | License/Gating | Downloaded artifact rules | Derived/local artifacts |\n",
    );
    output.push_str("| --- | --- | --- | --- | --- | --- |\n");

    for entry in entries {
        let capabilities = entry.capabilities.join("<br>");
        let sources = entry
            .sources
            .iter()
            .map(|source| format!("{}: `{}`", source.label, source.repo))
            .collect::<Vec<_>>()
            .join("<br>");
        let provenance = entry
            .sources
            .iter()
            .map(|source| {
                format!(
                    "{}: `{}`/`{}`",
                    source.label,
                    source.provenance.license,
                    source.provenance.gating.as_str()
                )
            })
            .collect::<Vec<_>>()
            .join("<br>");
        let rules = entry
            .sources
            .iter()
            .map(|source| {
                let rules = source
                    .requirements
                    .iter()
                    .map(ArtifactRequirement::label)
                    .collect::<Vec<_>>()
                    .join("<br>");
                format!("{}:<br>{rules}", source.label)
            })
            .collect::<Vec<_>>()
            .join("<br>");
        let derived = if entry.derived.is_empty() {
            "none".to_owned()
        } else {
            entry
                .derived
                .iter()
                .map(ArtifactDerivedEntry::label)
                .collect::<Vec<_>>()
                .join("<br>")
        };
        output.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} |\n",
            entry.bundle_id, capabilities, sources, provenance, rules, derived
        ));
    }

    output
}

fn check_entry(
    entry: &ArtifactBundleEntry,
    artifact_root: &Path,
    metadata: &BTreeMap<String, std::result::Result<HfRepoMetadata, String>>,
) -> std::result::Result<ArtifactBundleCheck, std::io::Error> {
    let sources = entry
        .sources
        .iter()
        .map(|source| check_source(source, artifact_root, metadata))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let primary_files = sources
        .first()
        .map(|source| source.files.as_slice())
        .unwrap_or(&[]);
    let derived = entry
        .derived
        .iter()
        .map(|artifact| {
            let matches = primary_files
                .iter()
                .filter(|filename| artifact.matches(filename))
                .cloned()
                .collect::<Vec<_>>();
            ArtifactDerivedCheck {
                artifact: artifact.clone(),
                present: !matches.is_empty(),
                matches,
            }
        })
        .collect();

    Ok(ArtifactBundleCheck {
        bundle_id: entry.bundle_id.clone(),
        sources,
        derived,
    })
}

fn check_source(
    entry: &ArtifactSourceEntry,
    artifact_root: &Path,
    metadata: &BTreeMap<String, std::result::Result<HfRepoMetadata, String>>,
) -> std::result::Result<ArtifactSourceCheck, std::io::Error> {
    let repo_cache_dir = artifact_root.join(hf_cache_repo_folder(&entry.repo));
    let snapshot = read_main_ref(&repo_cache_dir)?;
    let snapshot_dir = snapshot
        .as_ref()
        .map(|commit| repo_cache_dir.join("snapshots").join(commit));
    let files = match snapshot_dir.as_ref() {
        Some(dir) if dir.is_dir() => collect_snapshot_files(dir)?,
        _ => Vec::new(),
    };

    let metadata_result = metadata.get(&entry.repo);
    let metadata_value = metadata_result
        .and_then(|result| result.as_ref().ok())
        .cloned();
    let remote_snapshot = metadata_value
        .as_ref()
        .and_then(|repo_metadata| repo_metadata.sha.clone());
    let metadata_error = metadata_result
        .and_then(|result| result.as_ref().err())
        .cloned();
    let effective_requirements = metadata_value
        .as_ref()
        .map(|repo_metadata| concrete_requirements(&entry.requirements, repo_metadata))
        .filter(|requirements| !requirements.is_empty())
        .unwrap_or_else(|| entry.requirements.clone());
    let requirements = effective_requirements
        .iter()
        .map(|requirement| {
            let matches = files
                .iter()
                .filter(|filename| requirement.matches(filename))
                .cloned()
                .collect::<Vec<_>>();
            ArtifactRequirementCheck {
                requirement: requirement.clone(),
                present: !matches.is_empty(),
                matches,
            }
        })
        .collect::<Vec<_>>();

    Ok(ArtifactSourceCheck {
        label: entry.label.clone(),
        repo: entry.repo.clone(),
        repo_cache_dir,
        snapshot,
        remote_snapshot,
        metadata: metadata_value,
        metadata_error,
        requirements,
        files,
    })
}

fn concrete_requirements(
    requirements_in: &[ArtifactRequirement],
    metadata: &HfRepoMetadata,
) -> Vec<ArtifactRequirement> {
    let mut seen = BTreeSet::new();
    let mut requirements = Vec::new();

    for requirement in requirements_in {
        match requirement {
            ArtifactRequirement::Exact(filename) => {
                if seen.insert(filename.clone()) {
                    requirements.push(requirement.clone());
                }
            }
            ArtifactRequirement::Prefix(_) | ArtifactRequirement::Suffix(_) => {
                let mut matches = metadata
                    .siblings
                    .iter()
                    .filter(|filename| requirement.matches(filename))
                    .cloned()
                    .collect::<Vec<_>>();
                matches.sort();
                if matches.is_empty() {
                    if seen.insert(requirement.label()) {
                        requirements.push(requirement.clone());
                    }
                } else {
                    for filename in matches {
                        if seen.insert(filename.clone()) {
                            requirements.push(ArtifactRequirement::Exact(filename));
                        }
                    }
                }
            }
        }
    }

    requirements
}

fn read_main_ref(repo_cache_dir: &Path) -> std::result::Result<Option<String>, std::io::Error> {
    let ref_path = repo_cache_dir.join("refs/main");
    match fs::read_to_string(&ref_path) {
        Ok(value) => Ok(Some(value.trim().to_owned()).filter(|value| !value.is_empty())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn collect_snapshot_files(snapshot_dir: &Path) -> std::result::Result<Vec<String>, std::io::Error> {
    let mut files = Vec::new();
    collect_snapshot_files_inner(snapshot_dir, snapshot_dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_snapshot_files_inner(
    root: &Path,
    dir: &Path,
    files: &mut Vec<String>,
) -> std::result::Result<(), std::io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_snapshot_files_inner(root, &path, files)?;
        } else if file_type.is_file() || file_type.is_symlink() {
            if let Ok(relative) = path.strip_prefix(root) {
                files.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    Ok(())
}

fn hf_cache_repo_folder(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

fn license_from_tags(value: &Value) -> Option<String> {
    value
        .get("tags")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(Value::as_str)
        .find_map(|tag| tag.strip_prefix("license:").map(str::to_owned))
}

fn gated_value_to_string(value: &Value) -> String {
    match value {
        Value::Bool(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Null => "unknown".to_owned(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn canonical_root_uses_home_artifacts_cache() {
        let root = canonical_artifact_root();
        assert!(root.ends_with("artifacts/hf-cache"));
    }

    #[test]
    fn local_check_fails_closed_when_required_file_is_missing() {
        let root = unique_temp_dir();
        let repo_dir = root.join(hf_cache_repo_folder("owner/model"));
        let snapshot = repo_dir.join("snapshots/test-sha");
        fs::create_dir_all(&snapshot).expect("snapshot should be creatable");
        fs::create_dir_all(repo_dir.join("refs")).expect("refs should be creatable");
        fs::write(repo_dir.join("refs/main"), "test-sha\n").expect("ref should be writable");
        fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");

        let entries = vec![test_entry(vec![
            ArtifactRequirement::Exact("config.json".to_owned()),
            ArtifactRequirement::Exact("model.safetensors".to_owned()),
        ])];
        let report = check_artifact_entries(&entries, &root, &BTreeMap::new()).unwrap();

        assert!(!report.is_complete());
        assert_eq!(report.missing_rule_count(), 1);
        assert_eq!(
            report.bundles[0]
                .missing_requirements()
                .next()
                .unwrap()
                .requirement,
            ArtifactRequirement::Exact("model.safetensors".to_owned())
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn suffix_rule_matches_nested_snapshot_files() {
        let root = unique_temp_dir();
        let repo_dir = root.join(hf_cache_repo_folder("owner/model"));
        let snapshot = repo_dir.join("snapshots/test-sha/subdir");
        fs::create_dir_all(&snapshot).expect("snapshot should be creatable");
        fs::create_dir_all(repo_dir.join("refs")).expect("refs should be creatable");
        fs::write(repo_dir.join("refs/main"), "test-sha\n").expect("ref should be writable");
        fs::write(snapshot.join("weights.safetensors"), "stub").expect("weights");

        let entries = vec![test_entry(vec![ArtifactRequirement::Suffix(
            ".safetensors".to_owned(),
        )])];
        let report = check_artifact_entries(&entries, &root, &BTreeMap::new()).unwrap();

        assert!(report.is_complete());
        assert_eq!(
            report.bundles[0].sources[0].requirements[0].matches,
            vec!["subdir/weights.safetensors"]
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn prefix_rule_matches_nested_snapshot_tree() {
        let root = unique_temp_dir();
        let repo_dir = root.join(hf_cache_repo_folder("owner/model"));
        let snapshot = repo_dir.join("snapshots/test-sha/espeak-ng-data/lang/gmw");
        fs::create_dir_all(&snapshot).expect("snapshot should be creatable");
        fs::create_dir_all(repo_dir.join("refs")).expect("refs should be creatable");
        fs::write(repo_dir.join("refs/main"), "test-sha\n").expect("ref should be writable");
        fs::write(snapshot.join("en-US"), "stub").expect("voice data");

        let entries = vec![test_entry(vec![ArtifactRequirement::Prefix(
            "espeak-ng-data/".to_owned(),
        )])];
        let report = check_artifact_entries(&entries, &root, &BTreeMap::new()).unwrap();

        assert!(report.is_complete());
        assert_eq!(
            report.bundles[0].sources[0].requirements[0].matches,
            vec!["espeak-ng-data/lang/gmw/en-US"]
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn metadata_without_suffix_matches_keeps_suffix_rule_fail_closed() {
        let root = unique_temp_dir();
        let repo_dir = root.join(hf_cache_repo_folder("owner/model"));
        let snapshot = repo_dir.join("snapshots/test-sha");
        fs::create_dir_all(&snapshot).expect("snapshot should be creatable");
        fs::create_dir_all(repo_dir.join("refs")).expect("refs should be creatable");
        fs::write(repo_dir.join("refs/main"), "test-sha\n").expect("ref should be writable");
        fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");

        let entries = vec![test_entry(vec![
            ArtifactRequirement::Exact("config.json".to_owned()),
            ArtifactRequirement::Suffix(".safetensors".to_owned()),
        ])];
        let metadata = BTreeMap::from([(
            "owner/model".to_owned(),
            Ok(HfRepoMetadata {
                repo: "owner/model".to_owned(),
                sha: Some("test-sha".to_owned()),
                license: None,
                gated: None,
                siblings: vec!["config.json".to_owned()],
            }),
        )]);
        let report = check_artifact_entries(&entries, &root, &metadata).unwrap();

        assert!(!report.is_complete());
        assert_eq!(report.missing_rule_count(), 1);
        assert_eq!(
            report.bundles[0]
                .missing_requirements()
                .next()
                .unwrap()
                .requirement,
            ArtifactRequirement::Suffix(".safetensors".to_owned())
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn derived_artifact_is_checked_under_primary_snapshot() {
        let root = unique_temp_dir();
        let repo_dir = root.join(hf_cache_repo_folder("owner/model"));
        let snapshot = repo_dir.join("snapshots/test-sha");
        fs::create_dir_all(&snapshot).expect("snapshot should be creatable");
        fs::create_dir_all(repo_dir.join("refs")).expect("refs should be creatable");
        fs::write(repo_dir.join("refs/main"), "test-sha\n").expect("ref should be writable");
        fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");

        let mut entry = test_entry(vec![ArtifactRequirement::Exact("config.json".to_owned())]);
        entry.derived.push(ArtifactDerivedEntry {
            output: "tokens.txt".to_owned(),
            source: "tokenizer.json".to_owned(),
            recipe: "test recipe".to_owned(),
        });
        let report = check_artifact_entries(&[entry.clone()], &root, &BTreeMap::new()).unwrap();
        assert!(!report.is_complete());
        assert_eq!(report.missing_rule_count(), 1);
        assert_eq!(
            report.bundles[0]
                .missing_derived()
                .next()
                .unwrap()
                .artifact
                .output,
            "tokens.txt"
        );

        fs::write(repo_dir.join("snapshots/test-sha/tokens.txt"), "x")
            .expect("tokens should be writable");
        let report = check_artifact_entries(&[entry], &root, &BTreeMap::new()).unwrap();
        assert!(report.is_complete());

        fs::remove_dir_all(root).ok();
    }

    fn test_entry(requirements: Vec<ArtifactRequirement>) -> ArtifactBundleEntry {
        ArtifactBundleEntry {
            bundle_id: "bundle".to_owned(),
            display_name: "Bundle".to_owned(),
            capabilities: vec!["Chat".to_owned()],
            format: CheckpointFormat::Safetensors,
            quantization: None,
            sources: vec![ArtifactSourceEntry {
                label: "primary".to_owned(),
                repo: "owner/model".to_owned(),
                requirements,
                provenance: ArtifactProvenance::unknown(),
            }],
            derived: Vec::new(),
        }
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        env::temp_dir().join(format!("motlie-artifact-manifest-test-{unique}"))
    }
}
