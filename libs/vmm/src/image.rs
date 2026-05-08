use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::read::GzDecoder;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::AsyncWriteExt;

use crate::backend::BackendKind;

/// First supported external OCI import profile.
pub const UBUNTU_SYSTEMD_PROFILE: &str = "ubuntu-systemd";
pub const UBUNTU_SYSTEMD_SOURCE_REF: &str = "docker.io/library/ubuntu:24.04";
pub const MOTLIE_V15_CONTRACT_VERSION: &str = "v1.5";
pub const MOTLIE_V15_GUEST_MOUNTER_MARKER: &str = "MOTLIE_VMM_GUEST_MOUNTER_V1_5";
pub const MOTLIE_V15_GUEST_BUILD_FEATURES: &str = "--no-default-features --features guest-vfs";
pub const MOTLIE_V15_GUEST_BIN_OPT: &str = "/opt/motlie/v1.5/guest/bin/motlie-vfs-guest";
pub const MOTLIE_V15_GUEST_BIN_COMPAT: &str = "/usr/local/bin/motlie-vfs-guest";
pub const MOTLIE_V15_BACKEND_ENV_PATH: &str = "/etc/motlie/v1.5/backend.env";
pub const MOTLIE_V15_MOUNTS_PATH: &str = "/etc/motlie-vfs/mounts.yaml";
pub const MOTLIE_V15_SSHD_CA_CONFIG_PATH: &str = "/etc/ssh/sshd_config.d/90-motlie-vmm-ca.conf";
pub const MOTLIE_V15_VFS_HOST_CID: u32 = 2;
pub const MOTLIE_V15_VFS_PORT: u16 = 5000;
pub const MOTLIE_V15_SSH_VSOCK_PORT: u16 = 2222;

/// Guest CPU architecture as named by OCI platform descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GuestArchitecture {
    Amd64,
    Arm64,
}

impl GuestArchitecture {
    pub const fn oci_architecture(self) -> &'static str {
        match self {
            Self::Amd64 => "amd64",
            Self::Arm64 => "arm64",
        }
    }

    pub const fn oci_platform(self) -> OciPlatform {
        OciPlatform {
            os: OciOs::Linux,
            architecture: self,
        }
    }
}

impl std::fmt::Display for GuestArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.oci_architecture())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OciOs {
    Linux,
}

impl std::fmt::Display for OciOs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linux => f.write_str("linux"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OciPlatform {
    pub os: OciOs,
    pub architecture: GuestArchitecture,
}

impl OciPlatform {
    pub const fn linux_amd64() -> Self {
        GuestArchitecture::Amd64.oci_platform()
    }

    pub const fn linux_arm64() -> Self {
        GuestArchitecture::Arm64.oci_platform()
    }

    /// Current v1.5 lab defaults, not a backend invariant.
    ///
    /// CH validation runs on native amd64 DGX hosts today; VZ validation runs on
    /// Apple Silicon arm64 hosts. Callers that know their target guest platform
    /// should pass it explicitly instead of deriving it from backend kind.
    pub const fn default_for_v1_5_validation_backend(backend_kind: BackendKind) -> Self {
        match backend_kind {
            BackendKind::Vz => Self::linux_arm64(),
            BackendKind::ChShell | BackendKind::ChForkExec | BackendKind::ChVmmThread => {
                Self::linux_amd64()
            }
        }
    }
}

impl std::fmt::Display for OciPlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.os, self.architecture)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OciImageReferenceKind {
    Tag(String),
    Digest(OciDigest),
}

impl OciImageReferenceKind {
    pub fn registry_reference(&self) -> &str {
        match self {
            Self::Tag(tag) => tag,
            Self::Digest(digest) => digest.as_ref(),
        }
    }
}

impl std::fmt::Display for OciImageReferenceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tag(tag) => f.write_str(tag),
            Self::Digest(digest) => digest.fmt(f),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OciImageReference {
    pub registry: String,
    pub repository: String,
    pub reference: OciImageReferenceKind,
}

impl OciImageReference {
    pub fn registry_api_host(&self) -> &str {
        if self.registry == "docker.io" {
            "registry-1.docker.io"
        } else {
            &self.registry
        }
    }

    pub fn pull_scope(&self) -> String {
        format!("repository:{}:pull", self.repository)
    }

    pub fn normalized(&self) -> String {
        match &self.reference {
            OciImageReferenceKind::Tag(tag) => {
                format!("{}/{}:{tag}", self.registry, self.repository)
            }
            OciImageReferenceKind::Digest(digest) => {
                format!("{}/{}@{digest}", self.registry, self.repository)
            }
        }
    }

    pub fn with_digest(&self, digest: OciDigest) -> Self {
        Self {
            registry: self.registry.clone(),
            repository: self.repository.clone(),
            reference: OciImageReferenceKind::Digest(digest),
        }
    }
}

impl std::fmt::Display for OciImageReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.normalized())
    }
}

impl FromStr for OciImageReference {
    type Err = ImageContractError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_image_reference(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OciDigest(String);

impl OciDigest {
    pub fn new(value: impl Into<String>) -> Result<Self, ImageContractError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(ImageContractError::EmptyDigest);
        }
        let Some((algorithm, encoded)) = trimmed.split_once(':') else {
            return Err(ImageContractError::InvalidDigest {
                value,
                reason: "digest must use algorithm:encoded format".to_string(),
            });
        };
        if algorithm.is_empty() || encoded.is_empty() {
            return Err(ImageContractError::InvalidDigest {
                value,
                reason: "digest algorithm and encoded value are required".to_string(),
            });
        }
        if !algorithm.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'_' | b'+' | b'.' | b'-')
        }) {
            return Err(ImageContractError::InvalidDigest {
                value,
                reason: "digest algorithm must be lowercase alphanumeric, '_', '+', '.', or '-'"
                    .to_string(),
            });
        }
        if !encoded
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'=' | b'_' | b'-'))
        {
            return Err(ImageContractError::InvalidDigest {
                value,
                reason: "digest encoded value contains unsupported characters".to_string(),
            });
        }
        match algorithm {
            "sha256" => validate_hex_digest(&value, encoded, 64, algorithm)?,
            "sha384" => validate_hex_digest(&value, encoded, 96, algorithm)?,
            "sha512" => validate_hex_digest(&value, encoded, 128, algorithm)?,
            _ => {}
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn validate(&self) -> Result<(), ImageContractError> {
        Self::new(self.0.clone()).map(|_| ())
    }

    pub fn algorithm(&self) -> &str {
        self.0
            .split_once(':')
            .map(|(algorithm, _)| algorithm)
            .unwrap_or("")
    }

    pub fn encoded(&self) -> &str {
        self.0
            .split_once(':')
            .map(|(_, encoded)| encoded)
            .unwrap_or("")
    }
}

impl Serialize for OciDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for OciDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for OciDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl AsRef<str> for OciDigest {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalOciSource {
    pub image_ref: String,
    pub image_index_digest: OciDigest,
    pub platform: OciPlatform,
    pub platform_manifest_digest: OciDigest,
}

impl ExternalOciSource {
    pub fn ubuntu_systemd(
        platform: OciPlatform,
        image_index_digest: OciDigest,
        platform_manifest_digest: OciDigest,
    ) -> Self {
        Self {
            image_ref: UBUNTU_SYSTEMD_SOURCE_REF.to_string(),
            image_index_digest,
            platform,
            platform_manifest_digest,
        }
    }

    pub fn validate(&self) -> Result<(), ImageContractError> {
        if self.image_ref.trim().is_empty() {
            return Err(ImageContractError::EmptyImageRef);
        }
        self.image_index_digest.validate()?;
        self.platform_manifest_digest.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InitProfile {
    UbuntuSystemd,
    AlpineOpenRc,
    MotlieInit,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestImageProfile {
    pub name: String,
    pub init: InitProfile,
    pub source: ExternalOciSource,
    pub required_packages: Vec<String>,
    pub required_mount_points: Vec<PathBuf>,
}

impl GuestImageProfile {
    pub fn ubuntu_systemd(source: ExternalOciSource) -> Self {
        Self {
            name: UBUNTU_SYSTEMD_PROFILE.to_string(),
            init: InitProfile::UbuntuSystemd,
            source,
            required_packages: vec![
                "bash".to_string(),
                "bubblewrap".to_string(),
                "ca-certificates".to_string(),
                "cloud-init".to_string(),
                "coreutils".to_string(),
                "curl".to_string(),
                "dbus".to_string(),
                "dnsutils".to_string(),
                "fuse3".to_string(),
                "git".to_string(),
                "iproute2".to_string(),
                "iputils-ping".to_string(),
                "libfuse3-3".to_string(),
                "locales".to_string(),
                "npm".to_string(),
                "openssh-server".to_string(),
                "python3".to_string(),
                "socat".to_string(),
                "strace".to_string(),
                "sudo".to_string(),
                "systemd".to_string(),
                "systemd-sysv".to_string(),
                "tmux".to_string(),
                "vim".to_string(),
                "wget".to_string(),
            ],
            required_mount_points: vec![
                PathBuf::from("/workspace"),
                PathBuf::from("/agent-state"),
                PathBuf::from("/home"),
            ],
        }
    }

    pub fn validate(&self) -> Result<(), ImageContractError> {
        if self.name.trim().is_empty() {
            return Err(ImageContractError::EmptyProfileName);
        }
        self.source.validate()?;
        if self.name == UBUNTU_SYSTEMD_PROFILE {
            if self.init != InitProfile::UbuntuSystemd {
                return Err(ImageContractError::ProfileInitMismatch {
                    profile: self.name.clone(),
                    expected: InitProfile::UbuntuSystemd,
                    actual: self.init,
                });
            }
            if self.source.image_ref != UBUNTU_SYSTEMD_SOURCE_REF {
                return Err(ImageContractError::ProfileSourceMismatch {
                    profile: self.name.clone(),
                    expected_image_ref: UBUNTU_SYSTEMD_SOURCE_REF.to_string(),
                    actual_image_ref: self.source.image_ref.clone(),
                });
            }
        }
        if self
            .required_packages
            .iter()
            .any(|pkg| pkg.trim().is_empty())
        {
            return Err(ImageContractError::EmptyRequiredPackage);
        }
        if self
            .required_mount_points
            .iter()
            .any(|path| path.as_os_str().is_empty())
        {
            return Err(ImageContractError::EmptyRequiredMountPoint);
        }
        if let Some(path) = self
            .required_mount_points
            .iter()
            .find(|path| !path.is_absolute())
        {
            return Err(ImageContractError::RelativeRequiredMountPoint(path.clone()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmittedArtifactDigest {
    pub label: String,
    pub path: PathBuf,
    pub digest: OciDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestImageValidationRecord {
    pub profile: GuestImageProfile,
    pub contract_version: String,
    pub backend_kind: BackendKind,
    pub emitted_artifacts: Vec<EmittedArtifactDigest>,
}

impl GuestImageValidationRecord {
    pub fn validate(&self) -> Result<(), ImageContractError> {
        self.profile.validate()?;
        if self.contract_version.trim().is_empty() {
            return Err(ImageContractError::EmptyContractVersion);
        }
        if self.emitted_artifacts.is_empty() {
            return Err(ImageContractError::MissingEmittedArtifacts);
        }
        for artifact in &self.emitted_artifacts {
            if artifact.label.trim().is_empty() {
                return Err(ImageContractError::EmptyArtifactLabel);
            }
            if artifact.path.as_os_str().is_empty() {
                return Err(ImageContractError::EmptyArtifactPath);
            }
            artifact.digest.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedOciManifest {
    pub image_ref: OciImageReference,
    pub image_index_digest: OciDigest,
    pub platform: OciPlatform,
    pub platform_manifest_digest: OciDigest,
}

impl ResolvedOciManifest {
    pub fn into_external_source(self) -> ExternalOciSource {
        ExternalOciSource {
            image_ref: self.image_ref.normalized(),
            image_index_digest: self.image_index_digest,
            platform: self.platform,
            platform_manifest_digest: self.platform_manifest_digest,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OciLayerDescriptor {
    pub media_type: String,
    pub digest: OciDigest,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OciPlatformManifest {
    pub media_type: String,
    pub config_digest: OciDigest,
    pub layers: Vec<OciLayerDescriptor>,
}

impl OciPlatformManifest {
    pub fn from_json(bytes: &[u8]) -> Result<Self, OciRootfsImportError> {
        let manifest: RegistryPlatformManifest =
            serde_json::from_slice(bytes).map_err(OciRootfsImportError::Json)?;
        let media_type = manifest.media_type.ok_or_else(|| {
            OciRootfsImportError::InvalidPlatformManifest(
                "platform manifest missing mediaType".to_string(),
            )
        })?;
        if !is_single_manifest_media_type(Some(&media_type)) {
            return Err(OciRootfsImportError::InvalidPlatformManifest(format!(
                "unsupported platform manifest mediaType {media_type}"
            )));
        }
        let layers = manifest
            .layers
            .into_iter()
            .map(|layer| OciLayerDescriptor {
                media_type: layer.media_type,
                digest: layer.digest,
                size: layer.size,
            })
            .collect::<Vec<_>>();
        if layers.is_empty() {
            return Err(OciRootfsImportError::NoRootfsLayers);
        }
        Ok(Self {
            media_type,
            config_digest: manifest.config.digest,
            layers,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OciLayerInput {
    pub descriptor: OciLayerDescriptor,
    pub path: PathBuf,
}

impl OciLayerInput {
    pub fn new(descriptor: OciLayerDescriptor, path: impl Into<PathBuf>) -> Self {
        Self {
            descriptor,
            path: path.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OciLayerImportRecord {
    pub media_type: String,
    pub digest: OciDigest,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedOciRootfs {
    pub root: PathBuf,
    pub applied_layers: Vec<OciLayerImportRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsOsRequirement {
    pub accepted_ids: Vec<String>,
    pub accepted_version_ids: Vec<String>,
}

impl RootfsOsRequirement {
    pub fn ubuntu_24_04() -> Self {
        Self {
            accepted_ids: vec!["ubuntu".to_string()],
            accepted_version_ids: vec!["24.04".to_string()],
        }
    }

    pub fn linux_any() -> Self {
        Self {
            accepted_ids: Vec::new(),
            accepted_version_ids: Vec::new(),
        }
    }

    fn validate(&self) -> Result<(), RootfsClassificationError> {
        if self.accepted_ids.iter().any(|id| id.trim().is_empty()) {
            return Err(RootfsClassificationError::EmptyOsReleaseId);
        }
        if self
            .accepted_version_ids
            .iter()
            .any(|version| version.trim().is_empty())
        {
            return Err(RootfsClassificationError::EmptyOsReleaseVersionId);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PackageManagerRequirement {
    None,
    AptDpkg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VfsGuestRequirements {
    pub requires_dev_directory: bool,
    pub requires_fuse_runtime_device: bool,
}

impl Default for VfsGuestRequirements {
    fn default() -> Self {
        Self {
            requires_dev_directory: true,
            requires_fuse_runtime_device: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VnetGuestRequirements {
    pub requires_etc_directory: bool,
}

impl Default for VnetGuestRequirements {
    fn default() -> Self {
        Self {
            requires_etc_directory: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsProfileSpec {
    pub profile: GuestImageProfile,
    pub os: RootfsOsRequirement,
    pub package_manager: PackageManagerRequirement,
    pub required_binaries: Vec<PathBuf>,
    pub vfs: VfsGuestRequirements,
    pub vnet: VnetGuestRequirements,
}

impl RootfsProfileSpec {
    pub fn for_profile(profile: GuestImageProfile) -> Self {
        let (os, package_manager, required_binaries) = match profile.init {
            InitProfile::UbuntuSystemd => (
                RootfsOsRequirement::ubuntu_24_04(),
                PackageManagerRequirement::AptDpkg,
                vec![PathBuf::from("/bin/sh")],
            ),
            _ => (
                RootfsOsRequirement::linux_any(),
                PackageManagerRequirement::None,
                vec![PathBuf::from("/bin/sh")],
            ),
        };
        Self {
            profile,
            os,
            package_manager,
            required_binaries,
            vfs: VfsGuestRequirements::default(),
            vnet: VnetGuestRequirements::default(),
        }
    }

    pub fn ubuntu_systemd(source: ExternalOciSource) -> Self {
        Self::for_profile(GuestImageProfile::ubuntu_systemd(source))
    }

    pub fn validate(&self) -> Result<(), RootfsClassificationError> {
        self.profile.validate()?;
        self.os.validate()?;
        for path in &self.required_binaries {
            validate_rootfs_requirement_path(path)?;
        }
        for path in &self.profile.required_mount_points {
            validate_rootfs_requirement_path(path)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootfsClassificationStatus {
    Ready,
    CompatibleWithAdaptation,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootfsRequirementStatus {
    Present,
    MissingButInstallable,
    NeedsCompatibilityLayer,
    NeedsRuntimeProvisioning,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootfsRequirementKind {
    RootDirectory,
    OsRelease,
    PackageManager,
    InitSystem,
    RequiredPackage,
    RequiredBinary,
    RequiredMountPoint,
    VfsDevDirectory,
    VfsFuseRuntimeDevice,
    VnetConfigDirectory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsRequirementFinding {
    pub kind: RootfsRequirementKind,
    pub status: RootfsRequirementStatus,
    pub path: Option<PathBuf>,
    pub package: Option<String>,
    pub evidence: String,
}

impl RootfsRequirementFinding {
    fn new(
        kind: RootfsRequirementKind,
        status: RootfsRequirementStatus,
        evidence: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            status,
            path: None,
            package: None,
            evidence: evidence.into(),
        }
    }

    fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    fn with_package(mut self, package: impl Into<String>) -> Self {
        self.package = Some(package.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsClassification {
    pub root: PathBuf,
    pub profile_name: String,
    pub status: RootfsClassificationStatus,
    pub findings: Vec<RootfsRequirementFinding>,
}

impl RootfsClassification {
    pub fn is_supported_foundation(&self) -> bool {
        self.status != RootfsClassificationStatus::Unsupported
    }

    pub fn findings_by_status(
        &self,
        status: RootfsRequirementStatus,
    ) -> impl Iterator<Item = &RootfsRequirementFinding> {
        self.findings
            .iter()
            .filter(move |finding| finding.status == status)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RootfsClassifier;

impl RootfsClassifier {
    pub fn new() -> Self {
        Self
    }

    pub fn classify(
        &self,
        root: impl AsRef<Path>,
        spec: &RootfsProfileSpec,
    ) -> Result<RootfsClassification, RootfsClassificationError> {
        spec.validate()?;
        let root = root.as_ref();
        let metadata =
            fs::symlink_metadata(root).map_err(|source| RootfsClassificationError::Io {
                path: Some(root.to_path_buf()),
                source,
            })?;
        if metadata.file_type().is_symlink() {
            return Err(RootfsClassificationError::RootMayNotBeSymlink {
                path: root.to_path_buf(),
            });
        }
        if !metadata.is_dir() {
            return Err(RootfsClassificationError::RootNotDirectory {
                path: root.to_path_buf(),
            });
        }

        let mut findings = vec![RootfsRequirementFinding::new(
            RootfsRequirementKind::RootDirectory,
            RootfsRequirementStatus::Present,
            "rootfs assembly root is a directory",
        )
        .with_path("/")];

        let os_release = read_rootfs_os_release(root)?;
        findings.push(classify_os_release(&spec.os, os_release.as_ref()));

        let package_manager = classify_package_manager(root, spec.package_manager)?;
        findings.push(package_manager.finding.clone());

        findings.extend(classify_init_system(
            root,
            spec.profile.init,
            package_manager.available,
        )?);
        findings.extend(classify_required_binaries(root, &spec.required_binaries)?);
        let installed_packages = read_installed_dpkg_packages(root)?;
        findings.extend(classify_required_packages(
            &spec.profile.required_packages,
            &installed_packages,
            package_manager.available,
        ));
        findings.extend(classify_required_mount_points(
            root,
            &spec.profile.required_mount_points,
        )?);
        findings.extend(classify_vfs_requirements(root, &spec.vfs)?);
        findings.extend(classify_vnet_requirements(root, &spec.vnet)?);

        let status = aggregate_classification_status(&findings);
        Ok(RootfsClassification {
            root: root.to_path_buf(),
            profile_name: spec.profile.name.clone(),
            status,
            findings,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsCompatibilityBackendEnv {
    pub motlie_backend: String,
    pub motlie_vfs_transport: String,
    pub motlie_vfs_host_cid: u32,
    pub motlie_vfs_port: u16,
    pub motlie_vfs_connect_timeout_ms: u64,
    pub motlie_vfs_connect_retry_ms: u64,
    pub motlie_net_backend: String,
    pub motlie_ssh_vsock_port: u16,
}

impl RootfsCompatibilityBackendEnv {
    pub fn for_backend(backend: impl Into<String>, net_backend: impl Into<String>) -> Self {
        Self {
            motlie_backend: backend.into(),
            motlie_vfs_transport: "vsock".to_string(),
            motlie_vfs_host_cid: MOTLIE_V15_VFS_HOST_CID,
            motlie_vfs_port: MOTLIE_V15_VFS_PORT,
            motlie_vfs_connect_timeout_ms: 60_000,
            motlie_vfs_connect_retry_ms: 250,
            motlie_net_backend: net_backend.into(),
            motlie_ssh_vsock_port: MOTLIE_V15_SSH_VSOCK_PORT,
        }
    }

    pub fn render(&self) -> Result<String, RootfsCompatibilityError> {
        self.validate()?;
        Ok(format!(
            concat!(
                "# Rendered by motlie-vmm rootfs compatibility assembler.\n",
                "# Common guest-visible schema; backend-specific values are the adaptation.\n",
                "MOTLIE_BACKEND={}\n",
                "MOTLIE_VFS_TRANSPORT={}\n",
                "MOTLIE_VFS_HOST_CID={}\n",
                "MOTLIE_VFS_PORT={}\n",
                "MOTLIE_VFS_CONNECT_TIMEOUT_MS={}\n",
                "MOTLIE_VFS_CONNECT_RETRY_MS={}\n",
                "MOTLIE_NET_BACKEND={}\n",
                "MOTLIE_SSH_VSOCK_PORT={}\n"
            ),
            self.motlie_backend,
            self.motlie_vfs_transport,
            self.motlie_vfs_host_cid,
            self.motlie_vfs_port,
            self.motlie_vfs_connect_timeout_ms,
            self.motlie_vfs_connect_retry_ms,
            self.motlie_net_backend,
            self.motlie_ssh_vsock_port
        ))
    }

    fn validate(&self) -> Result<(), RootfsCompatibilityError> {
        validate_env_token("MOTLIE_BACKEND", &self.motlie_backend)?;
        validate_env_token("MOTLIE_VFS_TRANSPORT", &self.motlie_vfs_transport)?;
        validate_env_token("MOTLIE_NET_BACKEND", &self.motlie_net_backend)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsMountSpec {
    pub tag: String,
    pub guest_path: PathBuf,
    pub read_only: bool,
}

impl RootfsMountSpec {
    pub fn new(tag: impl Into<String>, guest_path: impl Into<PathBuf>) -> Self {
        Self {
            tag: tag.into(),
            guest_path: guest_path.into(),
            read_only: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsPayloadFile {
    pub source: PathBuf,
    pub guest_path: PathBuf,
    pub mode: u32,
    pub link_paths: Vec<PathBuf>,
}

impl RootfsPayloadFile {
    pub fn new(source: impl Into<PathBuf>, guest_path: impl Into<PathBuf>, mode: u32) -> Self {
        Self {
            source: source.into(),
            guest_path: guest_path.into(),
            mode,
            link_paths: Vec::new(),
        }
    }

    pub fn with_link(mut self, link_path: impl Into<PathBuf>) -> Self {
        self.link_paths.push(link_path.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsUserSeed {
    pub user: String,
    pub uid: Option<u32>,
    pub gid: Option<u32>,
    pub home: PathBuf,
    pub ssh_principal: Option<String>,
    pub env: Vec<(String, String)>,
    pub passwordless_sudo: bool,
}

impl RootfsUserSeed {
    pub fn new(user: impl Into<String>, ssh_principal: impl Into<String>) -> Self {
        let user = user.into();
        Self {
            home: PathBuf::from(format!("/home/{user}")),
            user,
            uid: None,
            gid: None,
            ssh_principal: Some(ssh_principal.into()),
            env: Vec::new(),
            passwordless_sudo: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootfsPendingRequirementPolicy {
    Record,
    FailInstallable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsCompatibilityLayerSpec {
    pub profile_spec: RootfsProfileSpec,
    pub backend_env: RootfsCompatibilityBackendEnv,
    pub mounts: Vec<RootfsMountSpec>,
    pub guest_binaries: Vec<RootfsPayloadFile>,
    pub users: Vec<RootfsUserSeed>,
    pub ssh_user_ca_pubkey: Option<String>,
    pub enable_ch_egress_service: bool,
    pub pending_requirement_policy: RootfsPendingRequirementPolicy,
}

impl RootfsCompatibilityLayerSpec {
    pub fn new(
        profile_spec: RootfsProfileSpec,
        backend_env: RootfsCompatibilityBackendEnv,
    ) -> Self {
        Self {
            profile_spec,
            backend_env,
            mounts: Vec::new(),
            guest_binaries: Vec::new(),
            users: Vec::new(),
            ssh_user_ca_pubkey: None,
            enable_ch_egress_service: false,
            pending_requirement_policy: RootfsPendingRequirementPolicy::Record,
        }
    }

    fn validate(&self) -> Result<(), RootfsCompatibilityError> {
        self.profile_spec
            .validate()
            .map_err(RootfsCompatibilityError::Classification)?;
        self.backend_env.validate()?;
        if self.guest_binaries.is_empty() {
            return Err(RootfsCompatibilityError::MissingGuestBinaryPayload);
        }
        for mount in &self.mounts {
            validate_mount_tag(&mount.tag)?;
            validate_mount_guest_path(&mount.guest_path)?;
        }
        for payload in &self.guest_binaries {
            map_install_path_error(
                &payload.guest_path,
                validate_rootfs_requirement_path(&payload.guest_path),
            )?;
            for link_path in &payload.link_paths {
                map_install_path_error(link_path, validate_rootfs_requirement_path(link_path))?;
            }
        }
        if let Some(pubkey) = &self.ssh_user_ca_pubkey {
            validate_config_value("ssh_user_ca_pubkey", pubkey)?;
        }
        for user in &self.users {
            validate_user_seed(user)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootfsCompatibilityInstallKind {
    Directory,
    Config,
    GuestBinary,
    SupportScript,
    ProfileScript,
    ServiceUnit,
    ServiceEnablement,
    SshSeed,
    Sudoers,
    Symlink,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsCompatibilityInstallRecord {
    pub kind: RootfsCompatibilityInstallKind,
    pub path: PathBuf,
    pub mode: Option<u32>,
    pub source: Option<PathBuf>,
    pub target: Option<PathBuf>,
}

impl RootfsCompatibilityInstallRecord {
    fn new(
        kind: RootfsCompatibilityInstallKind,
        path: impl Into<PathBuf>,
        mode: Option<u32>,
    ) -> Self {
        Self {
            kind,
            path: path.into(),
            mode,
            source: None,
            target: None,
        }
    }

    fn with_source(mut self, source: impl Into<PathBuf>) -> Self {
        self.source = Some(source.into());
        self
    }

    fn with_target(mut self, target: impl Into<PathBuf>) -> Self {
        self.target = Some(target.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsCompatibilityPendingRequirement {
    pub kind: RootfsRequirementKind,
    pub status: RootfsRequirementStatus,
    pub path: Option<PathBuf>,
    pub package: Option<String>,
    pub evidence: String,
}

impl From<&RootfsRequirementFinding> for RootfsCompatibilityPendingRequirement {
    fn from(finding: &RootfsRequirementFinding) -> Self {
        Self {
            kind: finding.kind,
            status: finding.status,
            path: finding.path.clone(),
            package: finding.package.clone(),
            evidence: finding.evidence.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootfsCompatibilityAssemblyManifest {
    pub root: PathBuf,
    pub contract_version: String,
    pub profile: GuestImageProfile,
    pub classification: RootfsClassification,
    pub installed: Vec<RootfsCompatibilityInstallRecord>,
    pub pending_requirements: Vec<RootfsCompatibilityPendingRequirement>,
    pub backend_env_path: PathBuf,
    pub mounts_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RootfsCompatibilityAssembler;

impl RootfsCompatibilityAssembler {
    pub fn new() -> Self {
        Self
    }

    pub fn assemble(
        &self,
        root: impl AsRef<Path>,
        spec: &RootfsCompatibilityLayerSpec,
    ) -> Result<RootfsCompatibilityAssemblyManifest, RootfsCompatibilityError> {
        spec.validate()?;
        let root = root.as_ref();
        let classification = RootfsClassifier::new().classify(root, &spec.profile_spec)?;
        if !classification.is_supported_foundation() {
            let unsupported_findings = classification
                .findings
                .iter()
                .filter(|finding| finding.status == RootfsRequirementStatus::Unsupported)
                .count();
            return Err(RootfsCompatibilityError::UnsupportedRootfs {
                profile_name: classification.profile_name.clone(),
                unsupported_findings,
                classification,
            });
        }

        let pending_requirements = pending_requirements_from_classification(&classification);
        let installable_pending: Vec<_> = pending_requirements
            .iter()
            .filter(|finding| finding.status == RootfsRequirementStatus::MissingButInstallable)
            .cloned()
            .collect();
        if spec.pending_requirement_policy == RootfsPendingRequirementPolicy::FailInstallable
            && !installable_pending.is_empty()
        {
            return Err(RootfsCompatibilityError::InstallableRequirementsPending {
                requirements: installable_pending,
            });
        }

        let mut installed = Vec::new();
        install_required_directories(root, spec, &mut installed)?;
        install_guest_payloads(root, spec, &mut installed)?;
        install_builtin_support_files(root, spec.enable_ch_egress_service, &mut installed)?;
        install_backend_env(root, &spec.backend_env, &mut installed)?;
        install_mounts_yaml(root, &spec.mounts, &mut installed)?;
        install_ssh_and_user_seeds(root, spec, &mut installed)?;

        Ok(RootfsCompatibilityAssemblyManifest {
            root: root.to_path_buf(),
            contract_version: MOTLIE_V15_CONTRACT_VERSION.to_string(),
            profile: spec.profile_spec.profile.clone(),
            classification,
            installed,
            pending_requirements,
            backend_env_path: PathBuf::from(MOTLIE_V15_BACKEND_ENV_PATH),
            mounts_path: PathBuf::from(MOTLIE_V15_MOUNTS_PATH),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OciContentCache {
    root: PathBuf,
}

impl OciContentCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_path(&self, digest: &OciDigest) -> PathBuf {
        self.root
            .join("blobs")
            .join(digest.algorithm())
            .join(digest.encoded())
    }

    pub fn cached_blob(
        &self,
        digest: &OciDigest,
        expected_size: Option<u64>,
    ) -> Result<Option<CachedOciBlob>, OciRegistryError> {
        let path = self.blob_path(digest);
        if !path.exists() {
            return Ok(None);
        }
        verify_cached_blob(&path, digest, expected_size).map(Some)
    }

    pub fn store_blob(
        &self,
        digest: &OciDigest,
        bytes: &[u8],
    ) -> Result<CachedOciBlob, OciRegistryError> {
        if let Some(blob) = self.cached_blob(digest, Some(bytes.len() as u64))? {
            return Ok(blob);
        }
        let path = self.blob_path(digest);
        let parent = path.parent().ok_or_else(|| OciRegistryError::CachePath {
            path: path.clone(),
            reason: "cache blob path has no parent".to_string(),
        })?;
        fs::create_dir_all(parent).map_err(|source| OciRegistryError::CacheIo {
            path: Some(parent.to_path_buf()),
            source,
        })?;
        let tmp_path = cache_tmp_path(&path)?;
        let write_result = (|| {
            let mut file = File::create(&tmp_path).map_err(|source| OciRegistryError::CacheIo {
                path: Some(tmp_path.clone()),
                source,
            })?;
            file.write_all(bytes)
                .map_err(|source| OciRegistryError::CacheIo {
                    path: Some(tmp_path.clone()),
                    source,
                })?;
            file.sync_all()
                .map_err(|source| OciRegistryError::CacheIo {
                    path: Some(tmp_path.clone()),
                    source,
                })?;
            Ok::<(), OciRegistryError>(())
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(error);
        }
        finalize_cache_write(&tmp_path, &path, digest, Some(bytes.len() as u64))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedOciBlob {
    pub digest: OciDigest,
    pub path: PathBuf,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedOciPlatform {
    pub resolved: ResolvedOciManifest,
    pub manifest_blob: CachedOciBlob,
    pub manifest: OciPlatformManifest,
    pub layers: Vec<OciLayerInput>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OciRootfsImporter;

impl OciRootfsImporter {
    pub fn new() -> Self {
        Self
    }

    pub fn import_layers(
        &self,
        layers: &[OciLayerInput],
        assembly_root: impl AsRef<Path>,
    ) -> Result<ImportedOciRootfs, OciRootfsImportError> {
        if layers.is_empty() {
            return Err(OciRootfsImportError::NoRootfsLayers);
        }
        let assembly_root = assembly_root.as_ref();
        ensure_empty_assembly_root(assembly_root)?;

        let mut applied_layers = Vec::with_capacity(layers.len());
        for layer in layers {
            layer.descriptor.digest.validate()?;
            let compression = layer_compression(&layer.descriptor.media_type)?;
            verify_layer_blob(&layer.path, &layer.descriptor)?;
            apply_layer_blob(&layer.path, assembly_root, compression)?;
            applied_layers.push(OciLayerImportRecord {
                media_type: layer.descriptor.media_type.clone(),
                digest: layer.descriptor.digest.clone(),
                size: layer.descriptor.size,
            });
        }

        Ok(ImportedOciRootfs {
            root: assembly_root.to_path_buf(),
            applied_layers,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OciRegistryClient {
    client: reqwest::Client,
}

impl Default for OciRegistryClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OciRegistryClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    pub async fn resolve_ubuntu_systemd_source(
        &self,
        platform: OciPlatform,
    ) -> Result<ExternalOciSource, OciRegistryError> {
        let image_ref = OciImageReference::from_str(UBUNTU_SYSTEMD_SOURCE_REF)?;
        let resolved = self.resolve_manifest(&image_ref, platform).await?;
        Ok(resolved.into_external_source())
    }

    pub async fn resolve_manifest(
        &self,
        image_ref: &OciImageReference,
        platform: OciPlatform,
    ) -> Result<ResolvedOciManifest, OciRegistryError> {
        let manifest = self.fetch_manifest(image_ref).await?;
        let image_index_digest = digest_from_bytes(&manifest.body)?;
        if let Some(header_digest) = manifest.registry_digest {
            if header_digest != image_index_digest {
                return Err(OciRegistryError::DigestHeaderMismatch {
                    image_ref: image_ref.normalized(),
                    header_digest,
                    computed_digest: image_index_digest,
                });
            }
        }
        let platform_manifest_digest =
            select_platform_manifest_digest(&manifest.body, platform, image_ref)?;
        Ok(ResolvedOciManifest {
            image_ref: image_ref.clone(),
            image_index_digest,
            platform,
            platform_manifest_digest,
        })
    }

    pub async fn fetch_resolved_platform_to_cache(
        &self,
        resolved: &ResolvedOciManifest,
        cache: &OciContentCache,
    ) -> Result<CachedOciPlatform, OciRegistryError> {
        let manifest_blob = self
            .fetch_manifest_digest_to_cache(
                &resolved.image_ref,
                &resolved.platform_manifest_digest,
                cache,
            )
            .await?;
        let manifest_bytes =
            fs::read(&manifest_blob.path).map_err(|source| OciRegistryError::CacheIo {
                path: Some(manifest_blob.path.clone()),
                source,
            })?;
        let manifest = OciPlatformManifest::from_json(&manifest_bytes)?;
        let mut layers = Vec::with_capacity(manifest.layers.len());
        for descriptor in &manifest.layers {
            let cached = self
                .fetch_blob_to_cache(
                    &resolved.image_ref,
                    &descriptor.digest,
                    Some(descriptor.size),
                    cache,
                )
                .await?;
            layers.push(OciLayerInput::new(descriptor.clone(), cached.path));
        }
        Ok(CachedOciPlatform {
            resolved: resolved.clone(),
            manifest_blob,
            manifest,
            layers,
        })
    }

    pub async fn resolve_and_fetch_ubuntu_systemd_to_cache(
        &self,
        platform: OciPlatform,
        cache: &OciContentCache,
    ) -> Result<CachedOciPlatform, OciRegistryError> {
        let image_ref = OciImageReference::from_str(UBUNTU_SYSTEMD_SOURCE_REF)?;
        let resolved = self.resolve_manifest(&image_ref, platform).await?;
        self.fetch_resolved_platform_to_cache(&resolved, cache)
            .await
    }

    async fn fetch_manifest_digest_to_cache(
        &self,
        image_ref: &OciImageReference,
        digest: &OciDigest,
        cache: &OciContentCache,
    ) -> Result<CachedOciBlob, OciRegistryError> {
        if let Some(blob) = cache.cached_blob(digest, None)? {
            return Ok(blob);
        }
        let digest_ref = image_ref.with_digest(digest.clone());
        let manifest = self.fetch_manifest(&digest_ref).await?;
        let computed_digest = digest_from_bytes(&manifest.body)?;
        if let Some(header_digest) = manifest.registry_digest {
            if header_digest != computed_digest {
                return Err(OciRegistryError::DigestHeaderMismatch {
                    image_ref: digest_ref.normalized(),
                    header_digest,
                    computed_digest,
                });
            }
        }
        if &computed_digest != digest {
            return Err(OciRegistryError::ContentDigestMismatch {
                image_ref: digest_ref.normalized(),
                expected_digest: digest.clone(),
                computed_digest,
            });
        }
        cache.store_blob(digest, &manifest.body)
    }

    async fn fetch_blob_to_cache(
        &self,
        image_ref: &OciImageReference,
        digest: &OciDigest,
        expected_size: Option<u64>,
        cache: &OciContentCache,
    ) -> Result<CachedOciBlob, OciRegistryError> {
        if let Some(blob) = cache.cached_blob(digest, expected_size)? {
            return Ok(blob);
        }
        let mut response = self.fetch_blob_once(image_ref, digest, None).await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            let challenge = response
                .headers()
                .get(reqwest::header::WWW_AUTHENTICATE)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| OciRegistryError::MissingAuthChallenge {
                    image_ref: image_ref.normalized(),
                })?;
            let token = self.fetch_bearer_token(challenge, image_ref).await?;
            response = self
                .fetch_blob_once(image_ref, digest, Some(&token))
                .await?;
        }
        self.cache_blob_response(image_ref, digest, expected_size, cache, response)
            .await
    }

    async fn fetch_blob_once(
        &self,
        image_ref: &OciImageReference,
        digest: &OciDigest,
        bearer_token: Option<&str>,
    ) -> Result<reqwest::Response, OciRegistryError> {
        let url = blob_url(image_ref, digest);
        let mut request = self.client.get(&url);
        if let Some(token) = bearer_token {
            request = request.bearer_auth(token);
        }
        request
            .send()
            .await
            .map_err(|source| OciRegistryError::Request {
                image_ref: image_ref.normalized(),
                source,
            })
    }

    async fn cache_blob_response(
        &self,
        image_ref: &OciImageReference,
        digest: &OciDigest,
        expected_size: Option<u64>,
        cache: &OciContentCache,
        mut response: reqwest::Response,
    ) -> Result<CachedOciBlob, OciRegistryError> {
        let status = response.status();
        let digest_header = response
            .headers()
            .get("Docker-Content-Digest")
            .and_then(|value| value.to_str().ok())
            .map(OciDigest::new)
            .transpose()?;
        if !status.is_success() {
            let body = response
                .text()
                .await
                .map_err(|source| OciRegistryError::Request {
                    image_ref: image_ref.normalized(),
                    source,
                })?;
            return Err(OciRegistryError::RegistryStatus {
                image_ref: image_ref.normalized(),
                status: status.as_u16(),
                body,
            });
        }
        let path = cache.blob_path(digest);
        let parent = path.parent().ok_or_else(|| OciRegistryError::CachePath {
            path: path.clone(),
            reason: "cache blob path has no parent".to_string(),
        })?;
        fs::create_dir_all(parent).map_err(|source| OciRegistryError::CacheIo {
            path: Some(parent.to_path_buf()),
            source,
        })?;
        let tmp_path = cache_tmp_path(&path)?;
        let mut file = tokio::fs::File::create(&tmp_path).await.map_err(|source| {
            OciRegistryError::CacheIo {
                path: Some(tmp_path.clone()),
                source,
            }
        })?;
        let mut hasher = Sha256::new();
        let mut size = 0_u64;
        let write_result = async {
            while let Some(chunk) =
                response
                    .chunk()
                    .await
                    .map_err(|source| OciRegistryError::Request {
                        image_ref: image_ref.normalized(),
                        source,
                    })?
            {
                size += chunk.len() as u64;
                hasher.update(&chunk);
                file.write_all(&chunk)
                    .await
                    .map_err(|source| OciRegistryError::CacheIo {
                        path: Some(tmp_path.clone()),
                        source,
                    })?;
            }
            file.sync_all()
                .await
                .map_err(|source| OciRegistryError::CacheIo {
                    path: Some(tmp_path.clone()),
                    source,
                })?;
            Ok::<(), OciRegistryError>(())
        }
        .await;
        if let Err(error) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(error);
        }
        let computed_digest = OciDigest::new(format!("sha256:{}", hex::encode(hasher.finalize())))?;
        if let Some(header_digest) = digest_header {
            if header_digest != computed_digest {
                let _ = fs::remove_file(&tmp_path);
                return Err(OciRegistryError::DigestHeaderMismatch {
                    image_ref: image_ref.normalized(),
                    header_digest,
                    computed_digest,
                });
            }
        }
        if &computed_digest != digest {
            let _ = fs::remove_file(&tmp_path);
            return Err(OciRegistryError::ContentDigestMismatch {
                image_ref: image_ref.normalized(),
                expected_digest: digest.clone(),
                computed_digest,
            });
        }
        if let Some(expected) = expected_size {
            if size != expected {
                let _ = fs::remove_file(&tmp_path);
                return Err(OciRegistryError::CacheSizeMismatch {
                    path: tmp_path,
                    expected,
                    actual: size,
                });
            }
        }
        finalize_cache_write(&tmp_path, &path, digest, expected_size)
    }

    async fn fetch_manifest(
        &self,
        image_ref: &OciImageReference,
    ) -> Result<RegistryManifestResponse, OciRegistryError> {
        let mut response = self.fetch_manifest_once(image_ref, None).await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            let challenge = response
                .headers()
                .get(reqwest::header::WWW_AUTHENTICATE)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| OciRegistryError::MissingAuthChallenge {
                    image_ref: image_ref.normalized(),
                })?;
            let token = self.fetch_bearer_token(challenge, image_ref).await?;
            response = self.fetch_manifest_once(image_ref, Some(&token)).await?;
        }
        manifest_response_from_reqwest(image_ref, response).await
    }

    async fn fetch_manifest_once(
        &self,
        image_ref: &OciImageReference,
        bearer_token: Option<&str>,
    ) -> Result<reqwest::Response, OciRegistryError> {
        let url = manifest_url(image_ref);
        let mut request = self
            .client
            .get(&url)
            .header(reqwest::header::ACCEPT, OCI_MANIFEST_ACCEPT);
        if let Some(token) = bearer_token {
            request = request.bearer_auth(token);
        }
        request
            .send()
            .await
            .map_err(|source| OciRegistryError::Request {
                image_ref: image_ref.normalized(),
                source,
            })
    }

    async fn fetch_bearer_token(
        &self,
        challenge: &str,
        image_ref: &OciImageReference,
    ) -> Result<String, OciRegistryError> {
        let challenge = parse_bearer_challenge(challenge)?;
        let mut url = reqwest::Url::parse(&challenge.realm).map_err(|source| {
            OciRegistryError::InvalidAuthChallenge {
                challenge: challenge.raw.clone(),
                reason: source.to_string(),
            }
        })?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(service) = &challenge.service {
                query.append_pair("service", service);
            }
            let default_scope;
            let scope = if let Some(scope) = challenge.scope.as_deref() {
                scope
            } else {
                default_scope = image_ref.pull_scope();
                &default_scope
            };
            query.append_pair("scope", scope);
        }
        let response =
            self.client
                .get(url)
                .send()
                .await
                .map_err(|source| OciRegistryError::Request {
                    image_ref: image_ref.normalized(),
                    source,
                })?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|source| OciRegistryError::Request {
                image_ref: image_ref.normalized(),
                source,
            })?;
        if !status.is_success() {
            return Err(OciRegistryError::RegistryStatus {
                image_ref: image_ref.normalized(),
                status: status.as_u16(),
                body,
            });
        }
        let token: RegistryTokenResponse =
            serde_json::from_str(&body).map_err(|source| OciRegistryError::Json {
                image_ref: image_ref.normalized(),
                source,
            })?;
        token
            .token
            .or(token.access_token)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| OciRegistryError::MissingToken {
                image_ref: image_ref.normalized(),
            })
    }
}

#[derive(Debug, Error)]
pub enum OciRegistryError {
    #[error(transparent)]
    Contract(#[from] ImageContractError),
    #[error(transparent)]
    Import(#[from] OciRootfsImportError),
    #[error("registry request for {image_ref} failed: {source}")]
    Request {
        image_ref: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("registry returned HTTP {status} for {image_ref}: {body}")]
    RegistryStatus {
        image_ref: String,
        status: u16,
        body: String,
    },
    #[error("registry response for {image_ref} had invalid JSON: {source}")]
    Json {
        image_ref: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("registry response for {image_ref} is missing a bearer auth challenge")]
    MissingAuthChallenge { image_ref: String },
    #[error("unsupported registry auth challenge '{challenge}': {reason}")]
    InvalidAuthChallenge { challenge: String, reason: String },
    #[error("registry auth response for {image_ref} did not include a token")]
    MissingToken { image_ref: String },
    #[error(
        "registry digest header for {image_ref} was {header_digest}, but computed digest was {computed_digest}"
    )]
    DigestHeaderMismatch {
        image_ref: String,
        header_digest: OciDigest,
        computed_digest: OciDigest,
    },
    #[error("registry content for {image_ref} was {computed_digest}, expected {expected_digest}")]
    ContentDigestMismatch {
        image_ref: String,
        expected_digest: OciDigest,
        computed_digest: OciDigest,
    },
    #[error("manifest for {image_ref} does not contain platform {platform}")]
    PlatformNotFound {
        image_ref: String,
        platform: OciPlatform,
    },
    #[error(
        "manifest for {image_ref} is a single-platform image manifest; platform {platform} cannot be verified without config inspection"
    )]
    UnverifiedSingleManifest {
        image_ref: String,
        platform: OciPlatform,
    },
    #[error("invalid OCI cache path {path:?}: {reason}")]
    CachePath { path: PathBuf, reason: String },
    #[error("OCI cache I/O error at {path:?}: {source}")]
    CacheIo {
        path: Option<PathBuf>,
        #[source]
        source: io::Error,
    },
    #[error("cached OCI blob {path:?} digest was {actual}, expected {expected}")]
    CacheDigestMismatch {
        path: PathBuf,
        expected: OciDigest,
        actual: OciDigest,
    },
    #[error("cached OCI blob {path:?} size was {actual}, expected {expected}")]
    CacheSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
}

#[derive(Debug, Error)]
pub enum OciRootfsImportError {
    #[error(transparent)]
    Contract(#[from] ImageContractError),
    #[error("platform manifest JSON is invalid: {0}")]
    Json(#[source] serde_json::Error),
    #[error("invalid platform manifest: {0}")]
    InvalidPlatformManifest(String),
    #[error("OCI rootfs import requires at least one layer")]
    NoRootfsLayers,
    #[error("unsupported OCI layer media type {media_type}")]
    UnsupportedLayerMediaType { media_type: String },
    #[error("assembly root is not a directory: {path:?}")]
    AssemblyRootNotDirectory { path: PathBuf },
    #[error("assembly root must be empty before import: {path:?}")]
    AssemblyRootNotEmpty { path: PathBuf },
    #[error("unsafe OCI layer path {path:?}: {reason}")]
    UnsafeLayerPath { path: PathBuf, reason: String },
    #[error("layer blob {path:?} size was {actual}, expected {expected}")]
    LayerSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("layer blob {path:?} digest was {actual}, expected {expected}")]
    LayerDigestMismatch {
        path: PathBuf,
        expected: OciDigest,
        actual: OciDigest,
    },
    #[error(
        "unsupported layer digest algorithm {algorithm} for {path:?}; only sha256 is currently supported"
    )]
    UnsupportedLayerDigestAlgorithm { path: PathBuf, algorithm: String },
    #[error("I/O error at {path:?}: {source}")]
    Io {
        path: Option<PathBuf>,
        #[source]
        source: io::Error,
    },
}

#[derive(Debug, Error)]
pub enum RootfsClassificationError {
    #[error(transparent)]
    Contract(#[from] ImageContractError),
    #[error("rootfs path is not a directory: {path:?}")]
    RootNotDirectory { path: PathBuf },
    #[error("rootfs path may not be a symlink: {path:?}")]
    RootMayNotBeSymlink { path: PathBuf },
    #[error("rootfs OS release accepted IDs cannot contain an empty value")]
    EmptyOsReleaseId,
    #[error("rootfs OS release accepted VERSION_IDs cannot contain an empty value")]
    EmptyOsReleaseVersionId,
    #[error("invalid rootfs requirement path {path:?}: {reason}")]
    InvalidRequirementPath { path: PathBuf, reason: String },
    #[error("rootfs symlink {path:?} target {target:?} escapes the guest root")]
    SymlinkEscapesRoot { path: PathBuf, target: PathBuf },
    #[error("rootfs symlink resolution exceeded the limit while resolving {path:?}")]
    SymlinkResolutionLimit { path: PathBuf },
    #[error("I/O error at {path:?}: {source}")]
    Io {
        path: Option<PathBuf>,
        #[source]
        source: io::Error,
    },
}

#[derive(Debug, Error)]
pub enum RootfsCompatibilityError {
    #[error(transparent)]
    Classification(#[from] RootfsClassificationError),
    #[error("rootfs is unsupported for profile {profile_name}: {unsupported_findings} unsupported finding(s)")]
    UnsupportedRootfs {
        profile_name: String,
        unsupported_findings: usize,
        classification: RootfsClassification,
    },
    #[error("rootfs compatibility assembly requires at least one guest binary payload")]
    MissingGuestBinaryPayload,
    #[error("host payload {path:?} is not a regular file")]
    HostPayloadNotFile { path: PathBuf },
    #[error("installable rootfs requirements remain pending: {requirements:?}")]
    InstallableRequirementsPending {
        requirements: Vec<RootfsCompatibilityPendingRequirement>,
    },
    #[error("invalid rootfs install path {path:?}: {reason}")]
    InvalidInstallPath { path: PathBuf, reason: String },
    #[error("refusing to replace directory at guest path {path:?}")]
    ExistingDirectory { path: PathBuf },
    #[error("invalid compatibility config value {field}={value:?}: {reason}")]
    InvalidConfigValue {
        field: String,
        value: String,
        reason: String,
    },
    #[error("rootfs compatibility symlink creation is unsupported on this host platform")]
    UnsupportedHostSymlink,
    #[error("I/O error at {path:?}: {source}")]
    Io {
        path: Option<PathBuf>,
        #[source]
        source: io::Error,
    },
}

const OCI_IMAGE_MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
const DOCKER_IMAGE_MANIFEST_MEDIA_TYPE: &str =
    "application/vnd.docker.distribution.manifest.v2+json";
const OCI_LAYER_TAR_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar";
const OCI_LAYER_GZIP_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+gzip";
const DOCKER_LAYER_GZIP_MEDIA_TYPE: &str = "application/vnd.docker.image.rootfs.diff.tar.gzip";
const OCI_MANIFEST_ACCEPT: &str = concat!(
    "application/vnd.oci.image.index.v1+json, ",
    "application/vnd.docker.distribution.manifest.list.v2+json, ",
    "application/vnd.oci.image.manifest.v1+json, ",
    "application/vnd.docker.distribution.manifest.v2+json"
);

#[derive(Debug)]
struct RegistryManifestResponse {
    body: bytes::Bytes,
    registry_digest: Option<OciDigest>,
}

#[derive(Debug, Deserialize)]
struct RegistryTokenResponse {
    token: Option<String>,
    access_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RegistryManifestIndex {
    #[serde(rename = "mediaType")]
    media_type: Option<String>,
    manifests: Option<Vec<RegistryManifestDescriptor>>,
}

#[derive(Debug, Deserialize)]
struct RegistryManifestDescriptor {
    #[serde(rename = "mediaType")]
    media_type: Option<String>,
    digest: OciDigest,
    platform: Option<RegistryManifestPlatform>,
}

#[derive(Debug, Deserialize)]
struct RegistryManifestPlatform {
    os: String,
    architecture: String,
}

#[derive(Debug, Deserialize)]
struct RegistryPlatformManifest {
    #[serde(rename = "mediaType")]
    media_type: Option<String>,
    config: RegistryContentDescriptor,
    layers: Vec<RegistryContentDescriptor>,
}

#[derive(Debug, Deserialize)]
struct RegistryContentDescriptor {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: OciDigest,
    size: u64,
}

#[derive(Debug)]
struct BearerChallenge {
    raw: String,
    realm: String,
    service: Option<String>,
    scope: Option<String>,
}

#[derive(Debug, Clone)]
struct PackageManagerClassification {
    available: bool,
    finding: RootfsRequirementFinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootfsPathKind {
    Missing,
    File,
    Directory,
    Other,
}

const ROOTFS_SYMLINK_LIMIT: usize = 40;
const MOTLIE_VFS_GUEST_SERVICE: &str = r#"[Unit]
Description=motlie-vmm v1.5 guest filesystem mounter
After=local-fs.target cloud-final.service
Wants=cloud-final.service

[Service]
Type=simple
ExecStart=/usr/local/bin/motlie-vfs-guest --mounts /etc/motlie-vfs/mounts.yaml --backend-env /etc/motlie/v1.5/backend.env
Restart=on-failure
RestartSec=2
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=cloud-init.target
"#;
const MOTLIE_AGENT_STATE_SERVICE: &str = r#"[Unit]
Description=Link agent state into mounted guest home
After=motlie-vfs-guest.service
Requires=motlie-vfs-guest.service
ConditionPathIsDirectory=/agent-state

[Service]
Type=oneshot
ExecStart=/usr/local/bin/motlie-agent-state-setup
RemainAfterExit=yes

[Install]
WantedBy=cloud-init.target
"#;
const MOTLIE_VMM_VSOCK_SSH_SERVICE: &str = r#"[Unit]
Description=motlie-vmm vsock-to-SSH bridge (socat, guest to host)
After=ssh.service
Requires=ssh.service
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart=/usr/local/bin/motlie-vmm-vsock-ssh-loop
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
"#;
const MOTLIE_VMM_EGRESS_SERVICE: &str = r#"[Unit]
Description=motlie-vmm v1.5 CH egress NIC setup
After=systemd-networkd.service
Wants=systemd-networkd.service
ConditionPathExists=/etc/motlie-vmm/egress.mac

[Service]
Type=oneshot
ExecStart=/usr/local/bin/motlie-vmm-egress-setup
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"#;
const MOTLIE_AGENT_STATE_SETUP_SCRIPT: &str = r#"#!/bin/sh
set -eu

# MOTLIE_CONVERGENCE_AGENT_STATE_SETUP_V3
# This script is immutable base-image content. It must not chown VFS-backed
# /agent-state or /home paths because ownership is presented by the active VFS
# layer for the guest uid/gid.

setup_user() {
    user_name="$1"
    home_dir="/home/$user_name"
    config_dir="$home_dir/.config"
    codex_dst="$home_dir/.codex"
    claude_dst="$home_dir/.claude"
    claude_code_dst="$config_dir/claude-code"

    [ -d "$home_dir" ] || return 0

    for mount_path in "$codex_dst" "$claude_dst" "$claude_code_dst"; do
        umount "$mount_path" >/dev/null 2>&1 || true
    done

    install -d -m 0755 "$config_dir"
    install -d -m 0700 /agent-state/codex /agent-state/claude /agent-state/claude-code /agent-state/codex/sqlite

    rm -rf "$codex_dst" "$claude_dst" "$claude_code_dst"
    install -d -m 0700 "$codex_dst" "$claude_dst" "$claude_code_dst"

    mount --bind /agent-state/codex "$codex_dst"
    mount --bind /agent-state/claude "$claude_dst"
    mount --bind /agent-state/claude-code "$claude_code_dst"
}

for home_dir in /home/*; do
    [ -d "$home_dir" ] || continue
    user_name="$(basename "$home_dir")"
    case "$user_name" in
        admin|ubuntu) continue ;;
    esac
    if id -u "$user_name" >/dev/null 2>&1; then
        setup_user "$user_name"
    fi
done
"#;
const MOTLIE_VMM_VSOCK_SSH_LOOP_SCRIPT: &str = r#"#!/bin/sh
set -eu

BACKEND_ENV="${MOTLIE_BACKEND_ENV:-/etc/motlie/v1.5/backend.env}"
if [ -f "$BACKEND_ENV" ]; then
    . "$BACKEND_ENV"
fi
SSH_HOST_CID="${MOTLIE_SSH_HOST_CID:-${MOTLIE_VFS_HOST_CID:-2}}"
SSH_VSOCK_PORT="${MOTLIE_SSH_VSOCK_PORT:-2222}"

while true; do
    /usr/bin/socat "VSOCK-CONNECT:${SSH_HOST_CID}:${SSH_VSOCK_PORT}" TCP:127.0.0.1:22 || true
    sleep 1
done
"#;
const MOTLIE_VMM_EGRESS_SETUP_SCRIPT: &str = r#"#!/bin/sh
set -eu

# MOTLIE_VMM_V15_CH_EGRESS_SETUP_V1
# CH-specific adaptation for the common v1.5 image contract. The immutable
# rootfs installs this once; launch-time code only seeds /etc/motlie-vmm/egress.*
# values into the per-guest overlay.

EGRESS_MAC_FILE="/etc/motlie-vmm/egress.mac"
EGRESS_IPV4_FILE="/etc/motlie-vmm/egress.ipv4"
EGRESS_GATEWAY_FILE="/etc/motlie-vmm/egress.gateway"
EGRESS_DNS_FILE="/etc/motlie-vmm/egress.dns"
EGRESS_IFACE=""

if [ ! -f "$EGRESS_MAC_FILE" ]; then
    echo "motlie-vmm-egress-setup: missing $EGRESS_MAC_FILE" >&2
    exit 1
fi

EGRESS_MAC="$(cat "$EGRESS_MAC_FILE")"
[ -n "$EGRESS_MAC" ] || {
    echo "motlie-vmm-egress-setup: empty egress MAC" >&2
    exit 1
}

EGRESS_IPV4="$(cat "$EGRESS_IPV4_FILE" 2>/dev/null || true)"
EGRESS_GATEWAY="$(cat "$EGRESS_GATEWAY_FILE" 2>/dev/null || true)"
EGRESS_DNS="$(cat "$EGRESS_DNS_FILE" 2>/dev/null || true)"
[ -n "$EGRESS_IPV4" ] || EGRESS_IPV4="10.0.2.15"
[ -n "$EGRESS_GATEWAY" ] || EGRESS_GATEWAY="10.0.2.2"
[ -n "$EGRESS_DNS" ] || EGRESS_DNS="10.0.2.3"
EGRESS_NETWORK="${EGRESS_IPV4%.*}.0"

for _attempt in $(seq 1 30); do
    for candidate in /sys/class/net/*; do
        [ -e "$candidate/address" ] || continue
        candidate_mac="$(cat "$candidate/address" 2>/dev/null || true)"
        if [ "$candidate_mac" = "$EGRESS_MAC" ]; then
            EGRESS_IFACE="$(basename "$candidate")"
            break
        fi
    done
    [ -n "$EGRESS_IFACE" ] && break
    sleep 1
done

[ -n "$EGRESS_IFACE" ] || {
    echo "motlie-vmm-egress-setup: interface with MAC $EGRESS_MAC not found" >&2
    exit 1
}

mkdir -p /run
ln -sf "/sys/class/net/$EGRESS_IFACE" /run/motlie-vmm-egress.link
ip link set "$EGRESS_IFACE" up || exit 0
ip addr replace "$EGRESS_IPV4/24" dev "$EGRESS_IFACE"
ip route replace "$EGRESS_NETWORK/24" dev "$EGRESS_IFACE" scope link src "$EGRESS_IPV4"
ip route replace default via "$EGRESS_GATEWAY" dev "$EGRESS_IFACE" metric 100
cat > /etc/resolv.conf <<EOF
nameserver $EGRESS_DNS
options edns0
EOF
"#;
const MOTLIE_AGENT_STATE_PROFILE: &str = r#"agent_state_root=/agent-state
codex_root="$agent_state_root/codex"
codex_sqlite_root="$codex_root/sqlite"
claude_root="$agent_state_root/claude"
claude_code_root="$agent_state_root/claude-code"
if [ -d "$agent_state_root" ] && [ -n "${HOME:-}" ] && [ -d "$HOME" ] && [ "${USER:-}" != "root" ] && [ "${HOME#"/home/"}" != "$HOME" ]; then
    mkdir -p "$codex_root" "$codex_sqlite_root" "$claude_root" "$claude_code_root" "$HOME/.config" >/dev/null 2>&1 || true
    export CODEX_HOME="$codex_root"
    export CODEX_SQLITE_HOME="$codex_sqlite_root"
fi
"#;
const MOTLIE_DOTENV_PROFILE: &str = r#"if [ -f "$HOME/.env" ]; then
    set -a
    . "$HOME/.env"
    set +a
fi
"#;
const MOTLIE_APT_FORCE_IPV4: &str = r#"Acquire::ForceIPv4 "true";
"#;
const MOTLIE_SSHD_CA_CONFIG: &str = r#"TrustedUserCAKeys /etc/ssh/ca/user_ca.pub
AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u
"#;

fn validate_rootfs_requirement_path(path: &Path) -> Result<(), RootfsClassificationError> {
    if !path.is_absolute() {
        return Err(RootfsClassificationError::InvalidRequirementPath {
            path: path.to_path_buf(),
            reason: "path must be absolute".to_string(),
        });
    }
    for component in path.components() {
        match component {
            Component::RootDir | Component::CurDir | Component::Normal(_) => {}
            Component::ParentDir => {
                return Err(RootfsClassificationError::InvalidRequirementPath {
                    path: path.to_path_buf(),
                    reason: "parent traversal is not allowed".to_string(),
                });
            }
            Component::Prefix(_) => {
                return Err(RootfsClassificationError::InvalidRequirementPath {
                    path: path.to_path_buf(),
                    reason: "platform path prefixes are not allowed".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn guest_path_components(path: &Path) -> Result<Vec<PathBuf>, RootfsClassificationError> {
    validate_rootfs_requirement_path(path)?;
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(value) => components.push(PathBuf::from(value)),
            Component::ParentDir | Component::Prefix(_) => unreachable!("validated above"),
        }
    }
    Ok(components)
}

fn host_path_from_guest_components(root: &Path, components: &[PathBuf]) -> PathBuf {
    let mut path = root.to_path_buf();
    for component in components {
        path.push(component);
    }
    path
}

fn push_symlink_target_components(
    components: &mut Vec<PathBuf>,
    target: &Path,
    link_path: &Path,
) -> Result<(), RootfsClassificationError> {
    for component in target.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(value) => components.push(PathBuf::from(value)),
            Component::ParentDir => {
                if components.pop().is_none() {
                    return Err(RootfsClassificationError::SymlinkEscapesRoot {
                        path: link_path.to_path_buf(),
                        target: target.to_path_buf(),
                    });
                }
            }
            Component::Prefix(_) => {
                return Err(RootfsClassificationError::InvalidRequirementPath {
                    path: target.to_path_buf(),
                    reason: "platform path prefixes are not allowed in symlink targets".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn resolve_symlink_target_components(
    parent_components: &[PathBuf],
    target: &Path,
    link_path: &Path,
) -> Result<Vec<PathBuf>, RootfsClassificationError> {
    let mut components = if target.is_absolute() {
        Vec::new()
    } else {
        parent_components.to_vec()
    };
    push_symlink_target_components(&mut components, target, link_path)?;
    Ok(components)
}

fn guest_absolute_path_from_components(components: &[PathBuf]) -> PathBuf {
    let mut path = PathBuf::from("/");
    for component in components {
        path.push(component);
    }
    path
}

fn resolve_rootfs_path(
    root: &Path,
    absolute_path: &Path,
) -> Result<Option<PathBuf>, RootfsClassificationError> {
    let mut resolved = Vec::<PathBuf>::new();
    let mut pending = guest_path_components(absolute_path)?;
    let mut symlink_count = 0usize;

    while !pending.is_empty() {
        let component = pending.remove(0);
        let mut candidate = resolved.clone();
        candidate.push(component);
        let candidate_host_path = host_path_from_guest_components(root, &candidate);
        let metadata = match fs::symlink_metadata(&candidate_host_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(RootfsClassificationError::Io {
                    path: Some(candidate_host_path),
                    source,
                });
            }
        };

        if metadata.file_type().is_symlink() {
            symlink_count += 1;
            if symlink_count > ROOTFS_SYMLINK_LIMIT {
                return Err(RootfsClassificationError::SymlinkResolutionLimit {
                    path: absolute_path.to_path_buf(),
                });
            }
            let target = fs::read_link(&candidate_host_path).map_err(|source| {
                RootfsClassificationError::Io {
                    path: Some(candidate_host_path),
                    source,
                }
            })?;
            let parent_components = &candidate[..candidate.len().saturating_sub(1)];
            let mut target_components = resolve_symlink_target_components(
                parent_components,
                &target,
                &guest_absolute_path_from_components(&candidate),
            )?;
            target_components.extend(pending);
            pending = target_components;
            resolved.clear();
        } else {
            resolved = candidate;
        }
    }
    Ok(Some(host_path_from_guest_components(root, &resolved)))
}

fn classify_path_kind(
    root: &Path,
    absolute_path: &Path,
) -> Result<RootfsPathKind, RootfsClassificationError> {
    let Some(path) = resolve_rootfs_path(root, absolute_path)? else {
        return Ok(RootfsPathKind::Missing);
    };
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_dir() => Ok(RootfsPathKind::Directory),
        Ok(metadata) if metadata.is_file() => Ok(RootfsPathKind::File),
        Ok(_) => Ok(RootfsPathKind::Other),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(RootfsPathKind::Missing),
        Err(source) => Err(RootfsClassificationError::Io {
            path: Some(path),
            source,
        }),
    }
}

fn find_existing_rootfs_file(
    root: &Path,
    paths: &[&str],
) -> Result<Option<PathBuf>, RootfsClassificationError> {
    for path in paths {
        let absolute = Path::new(path);
        if classify_path_kind(root, absolute)? == RootfsPathKind::File {
            return Ok(Some(PathBuf::from(path)));
        }
    }
    Ok(None)
}

fn rootfs_resolved_path_is_file(path: &Path) -> Result<bool, RootfsClassificationError> {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.is_file())
        .map_err(|source| RootfsClassificationError::Io {
            path: Some(path.to_path_buf()),
            source,
        })
}

fn rootfs_file_resolves_to_any(
    root: &Path,
    path: &Path,
    expected_paths: &[&str],
) -> Result<Option<PathBuf>, RootfsClassificationError> {
    let Some(resolved_path) = resolve_rootfs_path(root, path)? else {
        return Ok(None);
    };
    if !rootfs_resolved_path_is_file(&resolved_path)? {
        return Ok(None);
    }
    for expected_path in expected_paths {
        let expected_absolute = Path::new(expected_path);
        if classify_path_kind(root, expected_absolute)? == RootfsPathKind::File
            && resolve_rootfs_path(root, expected_absolute)?.as_ref() == Some(&resolved_path)
        {
            return Ok(Some(PathBuf::from(expected_path)));
        }
    }
    Ok(None)
}

fn read_rootfs_file_optional(
    root: &Path,
    absolute_path: &Path,
) -> Result<Option<String>, RootfsClassificationError> {
    let Some(path) = resolve_rootfs_path(root, absolute_path)? else {
        return Ok(None);
    };
    match fs::read_to_string(&path) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(RootfsClassificationError::Io {
            path: Some(path),
            source,
        }),
    }
}

fn read_rootfs_os_release(
    root: &Path,
) -> Result<Option<HashMap<String, String>>, RootfsClassificationError> {
    for path in ["/etc/os-release", "/usr/lib/os-release"] {
        if let Some(contents) = read_rootfs_file_optional(root, Path::new(path))? {
            return Ok(Some(parse_os_release(&contents)));
        }
    }
    Ok(None)
}

fn parse_os_release(contents: &str) -> HashMap<String, String> {
    let mut values = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let value = raw_value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        values.insert(key.trim().to_string(), value);
    }
    values
}

fn pending_requirements_from_classification(
    classification: &RootfsClassification,
) -> Vec<RootfsCompatibilityPendingRequirement> {
    classification
        .findings
        .iter()
        .filter(|finding| {
            matches!(
                finding.status,
                RootfsRequirementStatus::MissingButInstallable
                    | RootfsRequirementStatus::NeedsRuntimeProvisioning
            )
        })
        .map(RootfsCompatibilityPendingRequirement::from)
        .collect()
}

fn install_required_directories(
    root: &Path,
    spec: &RootfsCompatibilityLayerSpec,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    let mut dirs = BTreeSet::from([
        PathBuf::from("/opt/motlie/v1.5/guest/bin"),
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/etc/motlie/v1.5"),
        PathBuf::from("/etc/motlie-vfs"),
        PathBuf::from("/etc/profile.d"),
        PathBuf::from("/etc/systemd/system"),
        PathBuf::from("/etc/systemd/system/cloud-init.target.wants"),
        PathBuf::from("/etc/systemd/system/multi-user.target.wants"),
    ]);
    if spec.profile_spec.vfs.requires_dev_directory {
        dirs.insert(PathBuf::from("/dev"));
    }
    if spec.profile_spec.vnet.requires_etc_directory {
        dirs.insert(PathBuf::from("/etc"));
    }
    if spec.enable_ch_egress_service {
        dirs.insert(PathBuf::from("/etc/motlie-vmm"));
    }
    for path in &spec.profile_spec.profile.required_mount_points {
        dirs.insert(path.clone());
    }
    for mount in &spec.mounts {
        dirs.insert(mount.guest_path.clone());
    }
    if spec.ssh_user_ca_pubkey.is_some() {
        dirs.insert(PathBuf::from("/etc/ssh/ca"));
        dirs.insert(PathBuf::from("/etc/ssh/sshd_config.d"));
    }
    if spec.users.iter().any(|user| user.ssh_principal.is_some()) {
        dirs.insert(PathBuf::from("/etc/ssh/auth_principals"));
    }
    if spec.users.iter().any(|user| user.passwordless_sudo) {
        dirs.insert(PathBuf::from("/etc/sudoers.d"));
    }
    for user in &spec.users {
        dirs.insert(user.home.clone());
    }

    for dir in dirs {
        rootfs_ensure_dir(root, &dir, 0o755)?;
        installed.push(RootfsCompatibilityInstallRecord::new(
            RootfsCompatibilityInstallKind::Directory,
            dir,
            Some(0o755),
        ));
    }
    Ok(())
}

fn install_guest_payloads(
    root: &Path,
    spec: &RootfsCompatibilityLayerSpec,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    for payload in &spec.guest_binaries {
        rootfs_copy_file(root, &payload.source, &payload.guest_path, payload.mode)?;
        installed.push(
            RootfsCompatibilityInstallRecord::new(
                RootfsCompatibilityInstallKind::GuestBinary,
                payload.guest_path.clone(),
                Some(payload.mode),
            )
            .with_source(payload.source.clone()),
        );

        let mut link_paths = payload.link_paths.clone();
        if payload.guest_path.as_path() == Path::new(MOTLIE_V15_GUEST_BIN_OPT)
            && !link_paths
                .iter()
                .any(|path| path == Path::new(MOTLIE_V15_GUEST_BIN_COMPAT))
        {
            link_paths.push(PathBuf::from(MOTLIE_V15_GUEST_BIN_COMPAT));
        }
        for link_path in link_paths {
            rootfs_create_symlink(root, &link_path, &payload.guest_path)?;
            installed.push(
                RootfsCompatibilityInstallRecord::new(
                    RootfsCompatibilityInstallKind::Symlink,
                    link_path,
                    None,
                )
                .with_target(payload.guest_path.clone()),
            );
        }
    }
    Ok(())
}

fn install_builtin_support_files(
    root: &Path,
    enable_ch_egress_service: bool,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    install_builtin_file(
        root,
        "/opt/motlie/v1.5/guest/bin/motlie-agent-state-setup",
        MOTLIE_AGENT_STATE_SETUP_SCRIPT,
        0o755,
        RootfsCompatibilityInstallKind::SupportScript,
        installed,
    )?;
    install_compat_symlink(
        root,
        "/usr/local/bin/motlie-agent-state-setup",
        "/opt/motlie/v1.5/guest/bin/motlie-agent-state-setup",
        installed,
    )?;
    install_builtin_file(
        root,
        "/opt/motlie/v1.5/guest/bin/motlie-vmm-vsock-ssh-loop",
        MOTLIE_VMM_VSOCK_SSH_LOOP_SCRIPT,
        0o755,
        RootfsCompatibilityInstallKind::SupportScript,
        installed,
    )?;
    install_compat_symlink(
        root,
        "/usr/local/bin/motlie-vmm-vsock-ssh-loop",
        "/opt/motlie/v1.5/guest/bin/motlie-vmm-vsock-ssh-loop",
        installed,
    )?;
    install_builtin_file(
        root,
        "/etc/systemd/system/motlie-vfs-guest.service",
        MOTLIE_VFS_GUEST_SERVICE,
        0o644,
        RootfsCompatibilityInstallKind::ServiceUnit,
        installed,
    )?;
    install_builtin_file(
        root,
        "/etc/systemd/system/motlie-agent-state.service",
        MOTLIE_AGENT_STATE_SERVICE,
        0o644,
        RootfsCompatibilityInstallKind::ServiceUnit,
        installed,
    )?;
    install_builtin_file(
        root,
        "/etc/systemd/system/motlie-vmm-vsock-ssh.service",
        MOTLIE_VMM_VSOCK_SSH_SERVICE,
        0o644,
        RootfsCompatibilityInstallKind::ServiceUnit,
        installed,
    )?;
    install_service_enablement(
        root,
        "/etc/systemd/system/cloud-init.target.wants/motlie-vfs-guest.service",
        "../motlie-vfs-guest.service",
        installed,
    )?;
    install_service_enablement(
        root,
        "/etc/systemd/system/cloud-init.target.wants/motlie-agent-state.service",
        "../motlie-agent-state.service",
        installed,
    )?;
    install_service_enablement(
        root,
        "/etc/systemd/system/multi-user.target.wants/motlie-vmm-vsock-ssh.service",
        "../motlie-vmm-vsock-ssh.service",
        installed,
    )?;
    install_builtin_file(
        root,
        "/etc/profile.d/agent-state.sh",
        MOTLIE_AGENT_STATE_PROFILE,
        0o644,
        RootfsCompatibilityInstallKind::ProfileScript,
        installed,
    )?;
    install_builtin_file(
        root,
        "/etc/profile.d/dotenv.sh",
        MOTLIE_DOTENV_PROFILE,
        0o644,
        RootfsCompatibilityInstallKind::ProfileScript,
        installed,
    )?;
    install_builtin_file(
        root,
        "/etc/apt/apt.conf.d/99motlie-force-ipv4",
        MOTLIE_APT_FORCE_IPV4,
        0o644,
        RootfsCompatibilityInstallKind::Config,
        installed,
    )?;

    if enable_ch_egress_service {
        install_builtin_file(
            root,
            "/opt/motlie/v1.5/guest/bin/motlie-vmm-egress-setup",
            MOTLIE_VMM_EGRESS_SETUP_SCRIPT,
            0o755,
            RootfsCompatibilityInstallKind::SupportScript,
            installed,
        )?;
        install_compat_symlink(
            root,
            "/usr/local/bin/motlie-vmm-egress-setup",
            "/opt/motlie/v1.5/guest/bin/motlie-vmm-egress-setup",
            installed,
        )?;
        install_builtin_file(
            root,
            "/etc/systemd/system/motlie-vmm-egress.service",
            MOTLIE_VMM_EGRESS_SERVICE,
            0o644,
            RootfsCompatibilityInstallKind::ServiceUnit,
            installed,
        )?;
        install_service_enablement(
            root,
            "/etc/systemd/system/multi-user.target.wants/motlie-vmm-egress.service",
            "../motlie-vmm-egress.service",
            installed,
        )?;
    }
    Ok(())
}

fn install_backend_env(
    root: &Path,
    backend_env: &RootfsCompatibilityBackendEnv,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    rootfs_write_file(
        root,
        Path::new(MOTLIE_V15_BACKEND_ENV_PATH),
        backend_env.render()?.as_bytes(),
        0o644,
    )?;
    installed.push(RootfsCompatibilityInstallRecord::new(
        RootfsCompatibilityInstallKind::Config,
        MOTLIE_V15_BACKEND_ENV_PATH,
        Some(0o644),
    ));
    Ok(())
}

fn install_mounts_yaml(
    root: &Path,
    mounts: &[RootfsMountSpec],
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    rootfs_write_file(
        root,
        Path::new(MOTLIE_V15_MOUNTS_PATH),
        render_rootfs_mounts_yaml(mounts)?.as_bytes(),
        0o644,
    )?;
    installed.push(RootfsCompatibilityInstallRecord::new(
        RootfsCompatibilityInstallKind::Config,
        MOTLIE_V15_MOUNTS_PATH,
        Some(0o644),
    ));
    Ok(())
}

fn install_ssh_and_user_seeds(
    root: &Path,
    spec: &RootfsCompatibilityLayerSpec,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    if let Some(pubkey) = &spec.ssh_user_ca_pubkey {
        rootfs_write_file(
            root,
            Path::new("/etc/ssh/ca/user_ca.pub"),
            format!("{pubkey}\n").as_bytes(),
            0o644,
        )?;
        installed.push(RootfsCompatibilityInstallRecord::new(
            RootfsCompatibilityInstallKind::SshSeed,
            "/etc/ssh/ca/user_ca.pub",
            Some(0o644),
        ));
        rootfs_write_file(
            root,
            Path::new(MOTLIE_V15_SSHD_CA_CONFIG_PATH),
            MOTLIE_SSHD_CA_CONFIG.as_bytes(),
            0o644,
        )?;
        installed.push(RootfsCompatibilityInstallRecord::new(
            RootfsCompatibilityInstallKind::SshSeed,
            MOTLIE_V15_SSHD_CA_CONFIG_PATH,
            Some(0o644),
        ));
    }

    let sudoers = render_sudoers(&spec.users)?;
    if !sudoers.is_empty() {
        rootfs_write_file(
            root,
            Path::new("/etc/sudoers.d/90-motlie-vmm"),
            sudoers.as_bytes(),
            0o440,
        )?;
        installed.push(RootfsCompatibilityInstallRecord::new(
            RootfsCompatibilityInstallKind::Sudoers,
            "/etc/sudoers.d/90-motlie-vmm",
            Some(0o440),
        ));
    }

    for user in &spec.users {
        if let Some(principal) = &user.ssh_principal {
            let path = PathBuf::from(format!("/etc/ssh/auth_principals/{}", user.user));
            rootfs_write_file(root, &path, format!("{principal}\n").as_bytes(), 0o644)?;
            installed.push(RootfsCompatibilityInstallRecord::new(
                RootfsCompatibilityInstallKind::SshSeed,
                path,
                Some(0o644),
            ));
        }
        if !user.env.is_empty() {
            let path = user.home.join(".env");
            rootfs_write_file(root, &path, render_user_env(user)?.as_bytes(), 0o600)?;
            installed.push(RootfsCompatibilityInstallRecord::new(
                RootfsCompatibilityInstallKind::Config,
                path,
                Some(0o600),
            ));
        }
    }
    Ok(())
}

fn install_builtin_file(
    root: &Path,
    guest_path: &str,
    contents: &str,
    mode: u32,
    kind: RootfsCompatibilityInstallKind,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    rootfs_write_file(root, Path::new(guest_path), contents.as_bytes(), mode)?;
    installed.push(RootfsCompatibilityInstallRecord::new(
        kind,
        guest_path,
        Some(mode),
    ));
    Ok(())
}

fn install_compat_symlink(
    root: &Path,
    link_path: &str,
    target_path: &str,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    rootfs_create_symlink(root, Path::new(link_path), Path::new(target_path))?;
    installed.push(
        RootfsCompatibilityInstallRecord::new(
            RootfsCompatibilityInstallKind::Symlink,
            link_path,
            None,
        )
        .with_target(target_path),
    );
    Ok(())
}

fn install_service_enablement(
    root: &Path,
    link_path: &str,
    target_path: &str,
    installed: &mut Vec<RootfsCompatibilityInstallRecord>,
) -> Result<(), RootfsCompatibilityError> {
    rootfs_create_symlink(root, Path::new(link_path), Path::new(target_path))?;
    installed.push(
        RootfsCompatibilityInstallRecord::new(
            RootfsCompatibilityInstallKind::ServiceEnablement,
            link_path,
            None,
        )
        .with_target(target_path),
    );
    Ok(())
}

fn render_rootfs_mounts_yaml(
    mounts: &[RootfsMountSpec],
) -> Result<String, RootfsCompatibilityError> {
    let mut out = String::from("mounts:\n");
    for mount in mounts {
        validate_mount_tag(&mount.tag)?;
        validate_mount_guest_path(&mount.guest_path)?;
        writeln!(&mut out, "  - tag: {}", mount.tag).expect("writing to String cannot fail");
        writeln!(&mut out, "    guest_path: {}", mount.guest_path.display())
            .expect("writing to String cannot fail");
        writeln!(&mut out, "    read_only: {}", mount.read_only)
            .expect("writing to String cannot fail");
    }
    Ok(out)
}

fn render_sudoers(users: &[RootfsUserSeed]) -> Result<String, RootfsCompatibilityError> {
    let mut out = String::new();
    for user in users.iter().filter(|user| user.passwordless_sudo) {
        validate_user_name(&user.user)?;
        writeln!(&mut out, "{} ALL=(ALL) NOPASSWD:ALL", user.user)
            .expect("writing to String cannot fail");
    }
    Ok(out)
}

fn render_user_env(user: &RootfsUserSeed) -> Result<String, RootfsCompatibilityError> {
    let mut out = String::from("# Rendered by motlie-vmm rootfs compatibility assembler.\n");
    for (key, value) in &user.env {
        validate_env_key(key)?;
        validate_config_value(key, value)?;
        writeln!(&mut out, "{key}={}", shell_quote(value)).expect("writing to String cannot fail");
    }
    Ok(out)
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn validate_user_seed(user: &RootfsUserSeed) -> Result<(), RootfsCompatibilityError> {
    validate_user_name(&user.user)?;
    map_install_path_error(&user.home, validate_rootfs_requirement_path(&user.home))?;
    if let Some(principal) = &user.ssh_principal {
        validate_config_value("ssh_principal", principal)?;
    }
    for (key, value) in &user.env {
        validate_env_key(key)?;
        validate_config_value(key, value)?;
    }
    Ok(())
}

fn validate_user_name(value: &str) -> Result<(), RootfsCompatibilityError> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(RootfsCompatibilityError::InvalidConfigValue {
            field: "user".to_string(),
            value: value.to_string(),
            reason: "user names must be non-empty ASCII alphanumeric, '_' or '-'".to_string(),
        });
    }
    Ok(())
}

fn validate_env_key(value: &str) -> Result<(), RootfsCompatibilityError> {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return Err(RootfsCompatibilityError::InvalidConfigValue {
            field: "env key".to_string(),
            value: value.to_string(),
            reason: "environment key cannot be empty".to_string(),
        });
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(RootfsCompatibilityError::InvalidConfigValue {
            field: "env key".to_string(),
            value: value.to_string(),
            reason: "environment keys must be shell-safe identifiers".to_string(),
        });
    }
    Ok(())
}

fn validate_env_token(field: &str, value: &str) -> Result<(), RootfsCompatibilityError> {
    validate_config_value(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(RootfsCompatibilityError::InvalidConfigValue {
            field: field.to_string(),
            value: value.to_string(),
            reason: "value must be an ASCII token containing only letters, digits, '_', '-' or '.'"
                .to_string(),
        });
    }
    Ok(())
}

fn validate_mount_tag(value: &str) -> Result<(), RootfsCompatibilityError> {
    validate_config_value("mount tag", value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(RootfsCompatibilityError::InvalidConfigValue {
            field: "mount tag".to_string(),
            value: value.to_string(),
            reason:
                "mount tags must be ASCII tokens containing only letters, digits, '_', '-' or '.'"
                    .to_string(),
        });
    }
    Ok(())
}

fn validate_mount_guest_path(path: &Path) -> Result<(), RootfsCompatibilityError> {
    map_install_path_error(path, validate_rootfs_requirement_path(path))?;
    let components = install_path_components(path)?;
    if components.is_empty() {
        return Err(RootfsCompatibilityError::InvalidInstallPath {
            path: path.to_path_buf(),
            reason: "mount guest path cannot be the guest root".to_string(),
        });
    }
    for component in components {
        let Some(value) = component.to_str() else {
            return Err(RootfsCompatibilityError::InvalidInstallPath {
                path: path.to_path_buf(),
                reason: "mount guest path components must be UTF-8".to_string(),
            });
        };
        validate_mount_path_component(path, value)?;
    }
    Ok(())
}

fn validate_mount_path_component(
    path: &Path,
    component: &str,
) -> Result<(), RootfsCompatibilityError> {
    if component.is_empty()
        || !component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(RootfsCompatibilityError::InvalidInstallPath {
            path: path.to_path_buf(),
            reason: "mount guest path components must use only letters, digits, '_', '-' or '.'"
                .to_string(),
        });
    }
    Ok(())
}

fn validate_config_value(field: &str, value: &str) -> Result<(), RootfsCompatibilityError> {
    if value.trim().is_empty() {
        return Err(RootfsCompatibilityError::InvalidConfigValue {
            field: field.to_string(),
            value: value.to_string(),
            reason: "value cannot be empty".to_string(),
        });
    }
    if value.bytes().any(|byte| matches!(byte, b'\n' | b'\r' | 0)) {
        return Err(RootfsCompatibilityError::InvalidConfigValue {
            field: field.to_string(),
            value: value.to_string(),
            reason: "newlines and NUL bytes are not allowed".to_string(),
        });
    }
    Ok(())
}

fn map_install_path_error<T>(
    path: &Path,
    result: Result<T, RootfsClassificationError>,
) -> Result<T, RootfsCompatibilityError> {
    result.map_err(|error| match error {
        RootfsClassificationError::InvalidRequirementPath { reason, .. } => {
            RootfsCompatibilityError::InvalidInstallPath {
                path: path.to_path_buf(),
                reason,
            }
        }
        other => RootfsCompatibilityError::Classification(other),
    })
}

fn install_path_components(path: &Path) -> Result<Vec<PathBuf>, RootfsCompatibilityError> {
    map_install_path_error(path, guest_path_components(path))
}

fn rootfs_ensure_dir(
    root: &Path,
    guest_path: &Path,
    mode: u32,
) -> Result<(), RootfsCompatibilityError> {
    let components = install_path_components(guest_path)?;
    rootfs_ensure_dir_components(root, &components, mode)?;
    Ok(())
}

fn rootfs_ensure_dir_components(
    root: &Path,
    components: &[PathBuf],
    mode: u32,
) -> Result<PathBuf, RootfsCompatibilityError> {
    let mut current = root.to_path_buf();
    for component in components {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(RootfsCompatibilityError::InvalidInstallPath {
                    path: current.clone(),
                    reason: "parent directories may not be symlinks".to_string(),
                });
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => {
                return Err(RootfsCompatibilityError::InvalidInstallPath {
                    path: current.clone(),
                    reason: "path component exists but is not a directory".to_string(),
                });
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&current).map_err(|source| RootfsCompatibilityError::Io {
                    path: Some(current.clone()),
                    source,
                })?;
                rootfs_set_mode(&current, mode)?;
            }
            Err(source) => {
                return Err(RootfsCompatibilityError::Io {
                    path: Some(current.clone()),
                    source,
                });
            }
        }
    }
    Ok(current)
}

fn rootfs_host_path_for_create(
    root: &Path,
    guest_path: &Path,
) -> Result<PathBuf, RootfsCompatibilityError> {
    let components = install_path_components(guest_path)?;
    if components.is_empty() {
        return Err(RootfsCompatibilityError::InvalidInstallPath {
            path: guest_path.to_path_buf(),
            reason: "cannot replace the guest root".to_string(),
        });
    }
    rootfs_ensure_dir_components(root, &components[..components.len() - 1], 0o755)?;
    Ok(host_path_from_guest_components(root, &components))
}

fn rootfs_write_file(
    root: &Path,
    guest_path: &Path,
    contents: &[u8],
    mode: u32,
) -> Result<(), RootfsCompatibilityError> {
    let target = rootfs_host_path_for_create(root, guest_path)?;
    remove_existing_file_or_symlink(&target, guest_path)?;
    let mut file = File::create(&target).map_err(|source| RootfsCompatibilityError::Io {
        path: Some(target.clone()),
        source,
    })?;
    file.write_all(contents)
        .map_err(|source| RootfsCompatibilityError::Io {
            path: Some(target.clone()),
            source,
        })?;
    rootfs_set_mode(&target, mode)
}

fn rootfs_copy_file(
    root: &Path,
    source: &Path,
    guest_path: &Path,
    mode: u32,
) -> Result<(), RootfsCompatibilityError> {
    let metadata = fs::metadata(source).map_err(|source_error| RootfsCompatibilityError::Io {
        path: Some(source.to_path_buf()),
        source: source_error,
    })?;
    if !metadata.is_file() {
        return Err(RootfsCompatibilityError::HostPayloadNotFile {
            path: source.to_path_buf(),
        });
    }
    let target = rootfs_host_path_for_create(root, guest_path)?;
    remove_existing_file_or_symlink(&target, guest_path)?;
    fs::copy(source, &target).map_err(|source_error| RootfsCompatibilityError::Io {
        path: Some(target.clone()),
        source: source_error,
    })?;
    rootfs_set_mode(&target, mode)
}

fn rootfs_create_symlink(
    root: &Path,
    link_path: &Path,
    target_path: &Path,
) -> Result<(), RootfsCompatibilityError> {
    let target = rootfs_host_path_for_create(root, link_path)?;
    remove_existing_file_or_symlink(&target, link_path)?;
    create_host_symlink(target_path, &target)
}

fn remove_existing_file_or_symlink(
    host_path: &Path,
    guest_path: &Path,
) -> Result<(), RootfsCompatibilityError> {
    match fs::symlink_metadata(host_path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            Err(RootfsCompatibilityError::ExistingDirectory {
                path: guest_path.to_path_buf(),
            })
        }
        Ok(_) => fs::remove_file(host_path).map_err(|source| RootfsCompatibilityError::Io {
            path: Some(host_path.to_path_buf()),
            source,
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(RootfsCompatibilityError::Io {
            path: Some(host_path.to_path_buf()),
            source,
        }),
    }
}

#[cfg(unix)]
fn rootfs_set_mode(path: &Path, mode: u32) -> Result<(), RootfsCompatibilityError> {
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|source| {
        RootfsCompatibilityError::Io {
            path: Some(path.to_path_buf()),
            source,
        }
    })
}

#[cfg(not(unix))]
fn rootfs_set_mode(_path: &Path, _mode: u32) -> Result<(), RootfsCompatibilityError> {
    Ok(())
}

#[cfg(unix)]
fn create_host_symlink(target: &Path, link: &Path) -> Result<(), RootfsCompatibilityError> {
    std::os::unix::fs::symlink(target, link).map_err(|source| RootfsCompatibilityError::Io {
        path: Some(link.to_path_buf()),
        source,
    })
}

#[cfg(not(unix))]
fn create_host_symlink(_target: &Path, _link: &Path) -> Result<(), RootfsCompatibilityError> {
    Err(RootfsCompatibilityError::UnsupportedHostSymlink)
}

fn classify_os_release(
    requirement: &RootfsOsRequirement,
    os_release: Option<&HashMap<String, String>>,
) -> RootfsRequirementFinding {
    let Some(os_release) = os_release else {
        return RootfsRequirementFinding::new(
            RootfsRequirementKind::OsRelease,
            RootfsRequirementStatus::Unsupported,
            "missing /etc/os-release and /usr/lib/os-release",
        );
    };
    let id = os_release.get("ID").cloned().unwrap_or_default();
    let version_id = os_release.get("VERSION_ID").cloned().unwrap_or_default();
    let id_matches = requirement.accepted_ids.is_empty()
        || requirement
            .accepted_ids
            .iter()
            .any(|accepted| accepted == &id);
    let version_matches = requirement.accepted_version_ids.is_empty()
        || requirement
            .accepted_version_ids
            .iter()
            .any(|accepted| accepted == &version_id);
    if id_matches && version_matches {
        RootfsRequirementFinding::new(
            RootfsRequirementKind::OsRelease,
            RootfsRequirementStatus::Present,
            format!("ID={id}, VERSION_ID={version_id}"),
        )
    } else {
        RootfsRequirementFinding::new(
            RootfsRequirementKind::OsRelease,
            RootfsRequirementStatus::Unsupported,
            format!(
                "ID={id}, VERSION_ID={version_id}; expected IDs {:?}, versions {:?}",
                requirement.accepted_ids, requirement.accepted_version_ids
            ),
        )
    }
}

fn classify_package_manager(
    root: &Path,
    requirement: PackageManagerRequirement,
) -> Result<PackageManagerClassification, RootfsClassificationError> {
    match requirement {
        PackageManagerRequirement::None => Ok(PackageManagerClassification {
            available: false,
            finding: RootfsRequirementFinding::new(
                RootfsRequirementKind::PackageManager,
                RootfsRequirementStatus::Present,
                "no package manager required by profile",
            ),
        }),
        PackageManagerRequirement::AptDpkg => {
            let apt_get = find_existing_rootfs_file(root, &["/usr/bin/apt-get", "/bin/apt-get"])?;
            let dpkg = find_existing_rootfs_file(root, &["/usr/bin/dpkg", "/bin/dpkg"])?;
            match (apt_get, dpkg) {
                (Some(apt_get), Some(dpkg)) => Ok(PackageManagerClassification {
                    available: true,
                    finding: RootfsRequirementFinding::new(
                        RootfsRequirementKind::PackageManager,
                        RootfsRequirementStatus::Present,
                        format!("apt-get at {apt_get:?}, dpkg at {dpkg:?}"),
                    ),
                }),
                (apt_get, dpkg) => Ok(PackageManagerClassification {
                    available: false,
                    finding: RootfsRequirementFinding::new(
                        RootfsRequirementKind::PackageManager,
                        RootfsRequirementStatus::Unsupported,
                        format!("apt-get={apt_get:?}, dpkg={dpkg:?}"),
                    ),
                }),
            }
        }
    }
}

fn classify_init_system(
    root: &Path,
    init: InitProfile,
    package_manager_available: bool,
) -> Result<Vec<RootfsRequirementFinding>, RootfsClassificationError> {
    let finding = match init {
        InitProfile::UbuntuSystemd => {
            let systemd = find_existing_rootfs_file(
                root,
                &["/usr/lib/systemd/systemd", "/lib/systemd/systemd"],
            )?;
            if let Some(path) = systemd {
                RootfsRequirementFinding::new(
                    RootfsRequirementKind::InitSystem,
                    RootfsRequirementStatus::Present,
                    format!("systemd indicator at {path:?}"),
                )
            } else if let Some(resolved) = rootfs_file_resolves_to_any(
                root,
                Path::new("/sbin/init"),
                &["/usr/lib/systemd/systemd", "/lib/systemd/systemd"],
            )? {
                RootfsRequirementFinding::new(
                    RootfsRequirementKind::InitSystem,
                    RootfsRequirementStatus::Present,
                    format!("/sbin/init resolves to systemd at {resolved:?}"),
                )
            } else if package_manager_available {
                RootfsRequirementFinding::new(
                    RootfsRequirementKind::InitSystem,
                    RootfsRequirementStatus::MissingButInstallable,
                    "systemd not present; apt/dpkg can install profile init packages",
                )
            } else {
                RootfsRequirementFinding::new(
                    RootfsRequirementKind::InitSystem,
                    RootfsRequirementStatus::Unsupported,
                    "systemd not present and package manager is unavailable",
                )
            }
        }
        InitProfile::Unsupported => RootfsRequirementFinding::new(
            RootfsRequirementKind::InitSystem,
            RootfsRequirementStatus::Unsupported,
            "profile declares unsupported init",
        ),
        other => RootfsRequirementFinding::new(
            RootfsRequirementKind::InitSystem,
            RootfsRequirementStatus::Unsupported,
            format!("classifier support for init profile {other:?} is not implemented"),
        ),
    };
    Ok(vec![finding])
}

fn classify_required_binaries(
    root: &Path,
    required_binaries: &[PathBuf],
) -> Result<Vec<RootfsRequirementFinding>, RootfsClassificationError> {
    required_binaries
        .iter()
        .map(|path| {
            let kind = classify_path_kind(root, path)?;
            let (status, evidence) = match kind {
                RootfsPathKind::File => (
                    RootfsRequirementStatus::Present,
                    format!("required binary {path:?} exists"),
                ),
                RootfsPathKind::Missing => (
                    RootfsRequirementStatus::Unsupported,
                    format!("required binary {path:?} is missing"),
                ),
                RootfsPathKind::Directory | RootfsPathKind::Other => (
                    RootfsRequirementStatus::Unsupported,
                    format!("required binary {path:?} is not a regular file"),
                ),
            };
            Ok(RootfsRequirementFinding::new(
                RootfsRequirementKind::RequiredBinary,
                status,
                evidence,
            )
            .with_path(path.clone()))
        })
        .collect()
}

fn read_installed_dpkg_packages(
    root: &Path,
) -> Result<BTreeSet<String>, RootfsClassificationError> {
    let Some(status) = read_rootfs_file_optional(root, Path::new("/var/lib/dpkg/status"))? else {
        return Ok(BTreeSet::new());
    };
    Ok(parse_dpkg_status(&status))
}

fn parse_dpkg_status(contents: &str) -> BTreeSet<String> {
    let mut packages = BTreeSet::new();
    let mut package_name: Option<String> = None;
    let mut installed = false;
    for line in contents.lines().chain(std::iter::once("")) {
        let line = line.trim();
        if line.is_empty() {
            if installed {
                if let Some(name) = package_name.take() {
                    packages.insert(name);
                }
            }
            package_name = None;
            installed = false;
            continue;
        }
        if let Some(value) = line.strip_prefix("Package:") {
            package_name = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("Status:") {
            installed = value.trim() == "install ok installed";
        }
    }
    packages
}

fn classify_required_packages(
    required_packages: &[String],
    installed_packages: &BTreeSet<String>,
    package_manager_available: bool,
) -> Vec<RootfsRequirementFinding> {
    required_packages
        .iter()
        .map(|package| {
            if installed_packages.contains(package) {
                RootfsRequirementFinding::new(
                    RootfsRequirementKind::RequiredPackage,
                    RootfsRequirementStatus::Present,
                    format!("package {package} is installed"),
                )
                .with_package(package.clone())
            } else if package_manager_available {
                RootfsRequirementFinding::new(
                    RootfsRequirementKind::RequiredPackage,
                    RootfsRequirementStatus::MissingButInstallable,
                    format!("package {package} is not installed but apt/dpkg is available"),
                )
                .with_package(package.clone())
            } else {
                RootfsRequirementFinding::new(
                    RootfsRequirementKind::RequiredPackage,
                    RootfsRequirementStatus::Unsupported,
                    format!("package {package} is missing and no package manager is available"),
                )
                .with_package(package.clone())
            }
        })
        .collect()
}

fn classify_required_mount_points(
    root: &Path,
    mount_points: &[PathBuf],
) -> Result<Vec<RootfsRequirementFinding>, RootfsClassificationError> {
    mount_points
        .iter()
        .map(|path| {
            let kind = classify_path_kind(root, path)?;
            let (status, evidence) = match kind {
                RootfsPathKind::Directory => (
                    RootfsRequirementStatus::Present,
                    format!("mount point {path:?} exists as a directory"),
                ),
                RootfsPathKind::Missing => (
                    RootfsRequirementStatus::NeedsCompatibilityLayer,
                    format!("mount point {path:?} must be created by compatibility layer"),
                ),
                RootfsPathKind::File | RootfsPathKind::Other => (
                    RootfsRequirementStatus::Unsupported,
                    format!("mount point {path:?} exists but is not a directory"),
                ),
            };
            Ok(RootfsRequirementFinding::new(
                RootfsRequirementKind::RequiredMountPoint,
                status,
                evidence,
            )
            .with_path(path.clone()))
        })
        .collect()
}

fn classify_vfs_requirements(
    root: &Path,
    requirements: &VfsGuestRequirements,
) -> Result<Vec<RootfsRequirementFinding>, RootfsClassificationError> {
    let mut findings = Vec::new();
    if requirements.requires_dev_directory {
        findings.push(match classify_path_kind(root, Path::new("/dev"))? {
            RootfsPathKind::Directory => RootfsRequirementFinding::new(
                RootfsRequirementKind::VfsDevDirectory,
                RootfsRequirementStatus::Present,
                "/dev exists as a directory",
            )
            .with_path("/dev"),
            RootfsPathKind::Missing => RootfsRequirementFinding::new(
                RootfsRequirementKind::VfsDevDirectory,
                RootfsRequirementStatus::NeedsCompatibilityLayer,
                "/dev must be created by compatibility layer or backend emitter",
            )
            .with_path("/dev"),
            RootfsPathKind::File | RootfsPathKind::Other => RootfsRequirementFinding::new(
                RootfsRequirementKind::VfsDevDirectory,
                RootfsRequirementStatus::Unsupported,
                "/dev exists but is not a directory",
            )
            .with_path("/dev"),
        });
    }
    if requirements.requires_fuse_runtime_device {
        let fuse_status = match classify_path_kind(root, Path::new("/dev/fuse"))? {
            RootfsPathKind::File | RootfsPathKind::Other => (
                RootfsRequirementStatus::Present,
                "/dev/fuse exists in the rootfs".to_string(),
            ),
            RootfsPathKind::Missing => (
                RootfsRequirementStatus::NeedsRuntimeProvisioning,
                "/dev/fuse must be supplied by backend runtime provisioning".to_string(),
            ),
            RootfsPathKind::Directory => (
                RootfsRequirementStatus::Unsupported,
                "/dev/fuse exists but is a directory".to_string(),
            ),
        };
        findings.push(
            RootfsRequirementFinding::new(
                RootfsRequirementKind::VfsFuseRuntimeDevice,
                fuse_status.0,
                fuse_status.1,
            )
            .with_path("/dev/fuse"),
        );
    }
    Ok(findings)
}

fn classify_vnet_requirements(
    root: &Path,
    requirements: &VnetGuestRequirements,
) -> Result<Vec<RootfsRequirementFinding>, RootfsClassificationError> {
    let mut findings = Vec::new();
    if requirements.requires_etc_directory {
        findings.push(match classify_path_kind(root, Path::new("/etc"))? {
            RootfsPathKind::Directory => RootfsRequirementFinding::new(
                RootfsRequirementKind::VnetConfigDirectory,
                RootfsRequirementStatus::Present,
                "/etc exists for resolver and network configuration",
            )
            .with_path("/etc"),
            RootfsPathKind::Missing => RootfsRequirementFinding::new(
                RootfsRequirementKind::VnetConfigDirectory,
                RootfsRequirementStatus::Unsupported,
                "/etc is missing; VNET resolver/routing configuration has no safe target",
            )
            .with_path("/etc"),
            RootfsPathKind::File | RootfsPathKind::Other => RootfsRequirementFinding::new(
                RootfsRequirementKind::VnetConfigDirectory,
                RootfsRequirementStatus::Unsupported,
                "/etc exists but is not a directory",
            )
            .with_path("/etc"),
        });
    }
    Ok(findings)
}

fn aggregate_classification_status(
    findings: &[RootfsRequirementFinding],
) -> RootfsClassificationStatus {
    if findings
        .iter()
        .any(|finding| finding.status == RootfsRequirementStatus::Unsupported)
    {
        RootfsClassificationStatus::Unsupported
    } else if findings
        .iter()
        .any(|finding| finding.status != RootfsRequirementStatus::Present)
    {
        RootfsClassificationStatus::CompatibleWithAdaptation
    } else {
        RootfsClassificationStatus::Ready
    }
}

async fn manifest_response_from_reqwest(
    image_ref: &OciImageReference,
    response: reqwest::Response,
) -> Result<RegistryManifestResponse, OciRegistryError> {
    let status = response.status();
    let digest_header = response
        .headers()
        .get("Docker-Content-Digest")
        .and_then(|value| value.to_str().ok())
        .map(OciDigest::new)
        .transpose()?;
    let body = response
        .bytes()
        .await
        .map_err(|source| OciRegistryError::Request {
            image_ref: image_ref.normalized(),
            source,
        })?;
    if !status.is_success() {
        return Err(OciRegistryError::RegistryStatus {
            image_ref: image_ref.normalized(),
            status: status.as_u16(),
            body: String::from_utf8_lossy(&body).into_owned(),
        });
    }
    Ok(RegistryManifestResponse {
        body,
        registry_digest: digest_header,
    })
}

fn manifest_url(image_ref: &OciImageReference) -> String {
    format!(
        "https://{}/v2/{}/manifests/{}",
        image_ref.registry_api_host(),
        image_ref.repository,
        image_ref.reference.registry_reference()
    )
}

fn blob_url(image_ref: &OciImageReference, digest: &OciDigest) -> String {
    format!(
        "https://{}/v2/{}/blobs/{}",
        image_ref.registry_api_host(),
        image_ref.repository,
        digest
    )
}

fn cache_tmp_path(path: &Path) -> Result<PathBuf, OciRegistryError> {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| OciRegistryError::CachePath {
            path: path.to_path_buf(),
            reason: "cache blob path has no file name".to_string(),
        })?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    Ok(path.with_file_name(format!(".{file_name}.tmp-{}-{nanos}", std::process::id())))
}

fn finalize_cache_write(
    tmp_path: &Path,
    path: &Path,
    digest: &OciDigest,
    expected_size: Option<u64>,
) -> Result<CachedOciBlob, OciRegistryError> {
    if path.exists() {
        let _ = fs::remove_file(tmp_path);
        return verify_cached_blob(path, digest, expected_size);
    }
    fs::rename(tmp_path, path).map_err(|source| OciRegistryError::CacheIo {
        path: Some(path.to_path_buf()),
        source,
    })?;
    verify_cached_blob(path, digest, expected_size)
}

fn verify_cached_blob(
    path: &Path,
    digest: &OciDigest,
    expected_size: Option<u64>,
) -> Result<CachedOciBlob, OciRegistryError> {
    let metadata = fs::metadata(path).map_err(|source| OciRegistryError::CacheIo {
        path: Some(path.to_path_buf()),
        source,
    })?;
    if !metadata.is_file() {
        return Err(OciRegistryError::CachePath {
            path: path.to_path_buf(),
            reason: "cached blob path is not a regular file".to_string(),
        });
    }
    if let Some(expected) = expected_size {
        if metadata.len() != expected {
            return Err(OciRegistryError::CacheSizeMismatch {
                path: path.to_path_buf(),
                expected,
                actual: metadata.len(),
            });
        }
    }
    let actual = digest_from_file_for_cache(path)?;
    if &actual != digest {
        return Err(OciRegistryError::CacheDigestMismatch {
            path: path.to_path_buf(),
            expected: digest.clone(),
            actual,
        });
    }
    Ok(CachedOciBlob {
        digest: digest.clone(),
        path: path.to_path_buf(),
        size: metadata.len(),
    })
}

fn digest_from_file_for_cache(path: &Path) -> Result<OciDigest, OciRegistryError> {
    let mut file = File::open(path).map_err(|source| OciRegistryError::CacheIo {
        path: Some(path.to_path_buf()),
        source,
    })?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher).map_err(|source| OciRegistryError::CacheIo {
        path: Some(path.to_path_buf()),
        source,
    })?;
    OciDigest::new(format!("sha256:{}", hex::encode(hasher.finalize())))
        .map_err(OciRegistryError::Contract)
}

fn parse_image_reference(value: &str) -> Result<OciImageReference, ImageContractError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ImageContractError::EmptyImageRef);
    }
    if value.contains("://") {
        return Err(ImageContractError::InvalidImageRef {
            value: value.to_string(),
            reason: "image reference must not include a URL scheme".to_string(),
        });
    }

    let (name, digest) = split_optional_digest(value)?;
    let (name, tag) = split_optional_tag(name)?;
    let (registry, repository) = split_registry_repository(name, value)?;
    let reference = if let Some(digest) = digest {
        OciImageReferenceKind::Digest(OciDigest::new(digest)?)
    } else {
        OciImageReferenceKind::Tag(tag.unwrap_or_else(|| "latest".to_string()))
    };
    validate_image_reference_parts(value, &registry, &repository, &reference)?;
    Ok(OciImageReference {
        registry,
        repository,
        reference,
    })
}

fn split_optional_digest(value: &str) -> Result<(&str, Option<&str>), ImageContractError> {
    let mut parts = value.split('@');
    let Some(name) = parts.next() else {
        return Err(ImageContractError::EmptyImageRef);
    };
    let digest = parts.next();
    if parts.next().is_some() {
        return Err(ImageContractError::InvalidImageRef {
            value: value.to_string(),
            reason: "image reference contains more than one digest separator".to_string(),
        });
    }
    Ok((name, digest))
}

fn split_optional_tag(value: &str) -> Result<(&str, Option<String>), ImageContractError> {
    let last_slash = value.rfind('/');
    let last_colon = value.rfind(':');
    if let Some(colon) = last_colon {
        if last_slash.map(|slash| colon > slash).unwrap_or(true) {
            let name = &value[..colon];
            let tag = &value[colon + 1..];
            if tag.is_empty() {
                return Err(ImageContractError::InvalidImageRef {
                    value: value.to_string(),
                    reason: "tag cannot be empty".to_string(),
                });
            }
            return Ok((name, Some(tag.to_string())));
        }
    }
    Ok((value, None))
}

fn split_registry_repository(
    name: &str,
    original: &str,
) -> Result<(String, String), ImageContractError> {
    if name.is_empty() {
        return Err(ImageContractError::InvalidImageRef {
            value: original.to_string(),
            reason: "repository name cannot be empty".to_string(),
        });
    }
    let mut components = name.split('/');
    let first = components.next().unwrap_or_default();
    let first_is_registry =
        first.contains('.') || first.contains(':') || first.eq_ignore_ascii_case("localhost");
    let (registry, repository) = if first_is_registry {
        let rest = components.collect::<Vec<_>>();
        if rest.is_empty() {
            return Err(ImageContractError::InvalidImageRef {
                value: original.to_string(),
                reason: "repository name cannot be empty".to_string(),
            });
        }
        (first.to_string(), rest.join("/"))
    } else {
        ("docker.io".to_string(), name.to_string())
    };
    let repository = if registry == "docker.io" && !repository.contains('/') {
        format!("library/{repository}")
    } else {
        repository
    };
    Ok((registry, repository))
}

fn validate_image_reference_parts(
    original: &str,
    registry: &str,
    repository: &str,
    reference: &OciImageReferenceKind,
) -> Result<(), ImageContractError> {
    if registry.trim().is_empty() || repository.trim().is_empty() {
        return Err(ImageContractError::InvalidImageRef {
            value: original.to_string(),
            reason: "registry and repository are required".to_string(),
        });
    }
    if repository
        .split('/')
        .any(|part| part.is_empty() || !part.bytes().all(is_repository_byte))
    {
        return Err(ImageContractError::InvalidImageRef {
            value: original.to_string(),
            reason: "repository must use lowercase OCI path components".to_string(),
        });
    }
    if let OciImageReferenceKind::Tag(tag) = reference {
        if tag.is_empty()
            || tag.len() > 128
            || !tag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
        {
            return Err(ImageContractError::InvalidImageRef {
                value: original.to_string(),
                reason: "tag contains unsupported characters".to_string(),
            });
        }
    }
    Ok(())
}

fn is_repository_byte(byte: u8) -> bool {
    byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
}

fn digest_from_bytes(bytes: &[u8]) -> Result<OciDigest, ImageContractError> {
    let digest = Sha256::digest(bytes);
    OciDigest::new(format!("sha256:{}", hex::encode(digest)))
}

fn digest_from_reader(reader: &mut impl Read) -> Result<OciDigest, OciRootfsImportError> {
    let mut hasher = Sha256::new();
    io::copy(reader, &mut hasher)
        .map_err(|source| OciRootfsImportError::Io { path: None, source })?;
    OciDigest::new(format!("sha256:{}", hex::encode(hasher.finalize())))
        .map_err(OciRootfsImportError::Contract)
}

fn verify_layer_blob(
    path: &Path,
    descriptor: &OciLayerDescriptor,
) -> Result<(), OciRootfsImportError> {
    if descriptor.digest.algorithm() != "sha256" {
        return Err(OciRootfsImportError::UnsupportedLayerDigestAlgorithm {
            path: path.to_path_buf(),
            algorithm: descriptor.digest.algorithm().to_string(),
        });
    }
    let metadata = fs::metadata(path).map_err(|source| OciRootfsImportError::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    if metadata.len() != descriptor.size {
        return Err(OciRootfsImportError::LayerSizeMismatch {
            path: path.to_path_buf(),
            expected: descriptor.size,
            actual: metadata.len(),
        });
    }
    let mut file = File::open(path).map_err(|source| OciRootfsImportError::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    let actual = digest_from_reader(&mut file)?;
    if actual != descriptor.digest {
        return Err(OciRootfsImportError::LayerDigestMismatch {
            path: path.to_path_buf(),
            expected: descriptor.digest.clone(),
            actual,
        });
    }
    Ok(())
}

fn ensure_empty_assembly_root(root: &Path) -> Result<(), OciRootfsImportError> {
    if root.exists() {
        if !root.is_dir() {
            return Err(OciRootfsImportError::AssemblyRootNotDirectory {
                path: root.to_path_buf(),
            });
        }
        if fs::read_dir(root)
            .map_err(|source| OciRootfsImportError::Io {
                path: Some(root.to_path_buf()),
                source,
            })?
            .next()
            .is_some()
        {
            return Err(OciRootfsImportError::AssemblyRootNotEmpty {
                path: root.to_path_buf(),
            });
        }
    } else {
        fs::create_dir_all(root).map_err(|source| OciRootfsImportError::Io {
            path: Some(root.to_path_buf()),
            source,
        })?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OciLayerCompression {
    Tar,
    GzipTar,
}

fn layer_compression(media_type: &str) -> Result<OciLayerCompression, OciRootfsImportError> {
    match media_type {
        OCI_LAYER_TAR_MEDIA_TYPE => Ok(OciLayerCompression::Tar),
        OCI_LAYER_GZIP_MEDIA_TYPE | DOCKER_LAYER_GZIP_MEDIA_TYPE => {
            Ok(OciLayerCompression::GzipTar)
        }
        _ => Err(OciRootfsImportError::UnsupportedLayerMediaType {
            media_type: media_type.to_string(),
        }),
    }
}

fn apply_layer_blob(
    path: &Path,
    root: &Path,
    compression: OciLayerCompression,
) -> Result<(), OciRootfsImportError> {
    let file = File::open(path).map_err(|source| OciRootfsImportError::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    match compression {
        OciLayerCompression::Tar => apply_tar_layer(file, root, path),
        OciLayerCompression::GzipTar => apply_tar_layer(GzDecoder::new(file), root, path),
    }
}

fn apply_tar_layer(
    reader: impl Read,
    root: &Path,
    layer_path: &Path,
) -> Result<(), OciRootfsImportError> {
    let mut archive = tar::Archive::new(reader);
    for entry in archive
        .entries()
        .map_err(|source| OciRootfsImportError::Io {
            path: Some(layer_path.to_path_buf()),
            source,
        })?
    {
        let mut entry = entry.map_err(|source| OciRootfsImportError::Io {
            path: Some(layer_path.to_path_buf()),
            source,
        })?;
        let raw_path = entry
            .path()
            .map_err(|source| OciRootfsImportError::Io {
                path: Some(layer_path.to_path_buf()),
                source,
            })?
            .into_owned();
        let relative_path = sanitize_layer_path(&raw_path)?;
        if apply_whiteout(root, &relative_path)? {
            continue;
        }
        if !entry
            .unpack_in(root)
            .map_err(|source| OciRootfsImportError::Io {
                path: Some(layer_path.to_path_buf()),
                source,
            })?
        {
            return Err(OciRootfsImportError::UnsafeLayerPath {
                path: raw_path,
                reason: "tar entry would unpack outside the assembly root".to_string(),
            });
        }
    }
    Ok(())
}

fn sanitize_layer_path(path: &Path) -> Result<PathBuf, OciRootfsImportError> {
    let mut sanitized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => sanitized.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(OciRootfsImportError::UnsafeLayerPath {
                    path: path.to_path_buf(),
                    reason: "absolute paths, prefixes, and parent traversal are not allowed"
                        .to_string(),
                });
            }
        }
    }
    if sanitized.as_os_str().is_empty() {
        return Err(OciRootfsImportError::UnsafeLayerPath {
            path: path.to_path_buf(),
            reason: "empty layer path".to_string(),
        });
    }
    Ok(sanitized)
}

fn apply_whiteout(root: &Path, relative_path: &Path) -> Result<bool, OciRootfsImportError> {
    let Some(file_name) = relative_path.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    if file_name == ".wh..wh..opq" {
        let parent = relative_path.parent().unwrap_or_else(|| Path::new(""));
        remove_directory_contents(root, parent)?;
        return Ok(true);
    }
    let Some(target_name) = file_name.strip_prefix(".wh.") else {
        return Ok(false);
    };
    if target_name.is_empty() {
        return Err(OciRootfsImportError::UnsafeLayerPath {
            path: relative_path.to_path_buf(),
            reason: "whiteout target cannot be empty".to_string(),
        });
    }
    let mut target = relative_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    target.push(target_name);
    remove_path_if_exists(root, &target)?;
    Ok(true)
}

fn remove_directory_contents(root: &Path, relative_dir: &Path) -> Result<(), OciRootfsImportError> {
    ensure_relative_parent_has_no_symlink(root, relative_dir)?;
    let target = root.join(relative_dir);
    match fs::symlink_metadata(&target) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(OciRootfsImportError::UnsafeLayerPath {
                path: relative_dir.to_path_buf(),
                reason: "opaque whiteout target is a symlink".to_string(),
            });
        }
        Ok(metadata) if metadata.is_dir() => {
            for entry in fs::read_dir(&target).map_err(|source| OciRootfsImportError::Io {
                path: Some(target.clone()),
                source,
            })? {
                let entry = entry.map_err(|source| OciRootfsImportError::Io {
                    path: Some(target.clone()),
                    source,
                })?;
                remove_path(&entry.path())?;
            }
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(OciRootfsImportError::Io {
                path: Some(target),
                source,
            });
        }
    }
    Ok(())
}

fn remove_path_if_exists(root: &Path, relative_path: &Path) -> Result<(), OciRootfsImportError> {
    ensure_relative_parent_has_no_symlink(root, relative_path)?;
    let target = root.join(relative_path);
    match fs::symlink_metadata(&target) {
        Ok(_) => remove_path(&target),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(OciRootfsImportError::Io {
            path: Some(target),
            source,
        }),
    }
}

fn ensure_relative_parent_has_no_symlink(
    root: &Path,
    relative_path: &Path,
) -> Result<(), OciRootfsImportError> {
    let Some(parent) = relative_path.parent() else {
        return Ok(());
    };
    let mut current = root.to_path_buf();
    for component in parent.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(OciRootfsImportError::UnsafeLayerPath {
                    path: relative_path.to_path_buf(),
                    reason: "whiteout parent resolves through a symlink".to_string(),
                });
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(OciRootfsImportError::Io {
                    path: Some(current),
                    source,
                });
            }
        }
    }
    Ok(())
}

fn remove_path(path: &Path) -> Result<(), OciRootfsImportError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| OciRootfsImportError::Io {
        path: Some(path.to_path_buf()),
        source,
    })?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path).map_err(|source| OciRootfsImportError::Io {
            path: Some(path.to_path_buf()),
            source,
        })
    } else {
        fs::remove_file(path).map_err(|source| OciRootfsImportError::Io {
            path: Some(path.to_path_buf()),
            source,
        })
    }
}

fn select_platform_manifest_digest(
    bytes: &[u8],
    platform: OciPlatform,
    image_ref: &OciImageReference,
) -> Result<OciDigest, OciRegistryError> {
    let manifest: RegistryManifestIndex =
        serde_json::from_slice(bytes).map_err(|source| OciRegistryError::Json {
            image_ref: image_ref.normalized(),
            source,
        })?;
    let media_type = manifest.media_type.as_deref();
    let manifests = manifest.manifests.unwrap_or_default();
    if is_single_manifest_media_type(media_type) {
        return Err(OciRegistryError::UnverifiedSingleManifest {
            image_ref: image_ref.normalized(),
            platform,
        });
    }
    if manifests.is_empty() {
        return Err(OciRegistryError::PlatformNotFound {
            image_ref: image_ref.normalized(),
            platform,
        });
    }
    manifests
        .into_iter()
        .find(|manifest| {
            manifest
                .platform
                .as_ref()
                .map(|candidate| {
                    candidate.os == platform.os.to_string()
                        && candidate.architecture == platform.architecture.oci_architecture()
                })
                .unwrap_or(false)
                && manifest
                    .media_type
                    .as_deref()
                    .map(|media_type| is_single_manifest_media_type(Some(media_type)))
                    .unwrap_or(true)
        })
        .map(|manifest| manifest.digest)
        .ok_or_else(|| OciRegistryError::PlatformNotFound {
            image_ref: image_ref.normalized(),
            platform,
        })
}

fn is_single_manifest_media_type(media_type: Option<&str>) -> bool {
    matches!(
        media_type,
        Some(OCI_IMAGE_MANIFEST_MEDIA_TYPE) | Some(DOCKER_IMAGE_MANIFEST_MEDIA_TYPE)
    )
}

fn parse_bearer_challenge(value: &str) -> Result<BearerChallenge, OciRegistryError> {
    let trimmed = value.trim();
    let Some(params) = trimmed.strip_prefix("Bearer ") else {
        return Err(OciRegistryError::InvalidAuthChallenge {
            challenge: value.to_string(),
            reason: "only Bearer auth challenges are supported".to_string(),
        });
    };
    let mut realm = None;
    let mut service = None;
    let mut scope = None;
    for part in split_quoted_csv(params) {
        let Some((key, raw_value)) = part.split_once('=') else {
            continue;
        };
        let parsed_value = raw_value.trim().trim_matches('"').to_string();
        match key.trim() {
            "realm" => realm = Some(parsed_value),
            "service" => service = Some(parsed_value),
            "scope" => scope = Some(parsed_value),
            _ => {}
        }
    }
    let realm = realm.ok_or_else(|| OciRegistryError::InvalidAuthChallenge {
        challenge: value.to_string(),
        reason: "Bearer challenge is missing realm".to_string(),
    })?;
    Ok(BearerChallenge {
        raw: value.to_string(),
        realm,
        service,
        scope,
    })
}

fn split_quoted_csv(value: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    for (idx, byte) in value.bytes().enumerate() {
        match byte {
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => {
                parts.push(value[start..idx].trim());
                start = idx + 1;
            }
            _ => {}
        }
    }
    parts.push(value[start..].trim());
    parts
}

fn validate_hex_digest(
    value: &str,
    encoded: &str,
    expected_len: usize,
    algorithm: &str,
) -> Result<(), ImageContractError> {
    if encoded.len() != expected_len || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ImageContractError::InvalidDigest {
            value: value.to_string(),
            reason: format!("{algorithm} digest must be {expected_len} hexadecimal characters"),
        });
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ImageContractError {
    #[error("OCI image reference cannot be empty")]
    EmptyImageRef,
    #[error("invalid OCI image reference '{value}': {reason}")]
    InvalidImageRef { value: String, reason: String },
    #[error("OCI digest cannot be empty")]
    EmptyDigest,
    #[error("invalid OCI digest '{value}': {reason}")]
    InvalidDigest { value: String, reason: String },
    #[error("guest image profile name cannot be empty")]
    EmptyProfileName,
    #[error("guest image profile '{profile}' expected init {expected:?}, got {actual:?}")]
    ProfileInitMismatch {
        profile: String,
        expected: InitProfile,
        actual: InitProfile,
    },
    #[error(
        "guest image profile '{profile}' expected source image '{expected_image_ref}', got '{actual_image_ref}'"
    )]
    ProfileSourceMismatch {
        profile: String,
        expected_image_ref: String,
        actual_image_ref: String,
    },
    #[error("guest image contract version cannot be empty")]
    EmptyContractVersion,
    #[error("guest image profile contains an empty required package")]
    EmptyRequiredPackage,
    #[error("guest image profile contains an empty required mount point")]
    EmptyRequiredMountPoint,
    #[error("guest image profile required mount point must be absolute: {0:?}")]
    RelativeRequiredMountPoint(PathBuf),
    #[error("guest image validation record must contain emitted artifact digests")]
    MissingEmittedArtifacts,
    #[error("emitted artifact label cannot be empty")]
    EmptyArtifactLabel,
    #[error("emitted artifact path cannot be empty")]
    EmptyArtifactPath,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;

    fn digest(byte: char) -> OciDigest {
        OciDigest::new(format!("sha256:{}", byte.to_string().repeat(64))).unwrap()
    }

    fn tar_bytes(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (path, contents) in files {
            let mut header = tar::Header::new_gnu();
            header.set_mode(0o644);
            header.set_size(contents.len() as u64);
            header.set_cksum();
            builder.append_data(&mut header, *path, *contents).unwrap();
        }
        builder.finish().unwrap();
        builder.into_inner().unwrap()
    }

    fn gzip_bytes(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap()
    }

    fn layer_descriptor(media_type: &str, bytes: &[u8]) -> OciLayerDescriptor {
        OciLayerDescriptor {
            media_type: media_type.to_string(),
            digest: digest_from_bytes(bytes).unwrap(),
            size: bytes.len() as u64,
        }
    }

    fn write_blob(root: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = root.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }

    fn write_rootfs_file(root: &Path, absolute_path: &str, contents: &str) {
        let path = root.join(absolute_path.trim_start_matches('/'));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    fn create_rootfs_dir(root: &Path, absolute_path: &str) {
        fs::create_dir_all(root.join(absolute_path.trim_start_matches('/'))).unwrap();
    }

    fn write_dpkg_status(root: &Path, packages: &[&str]) {
        let mut status = String::new();
        for package in packages {
            status.push_str(&format!(
                "Package: {package}\nStatus: install ok installed\n\n"
            ));
        }
        write_rootfs_file(root, "/var/lib/dpkg/status", &status);
    }

    fn ubuntu_classifier_source() -> ExternalOciSource {
        ExternalOciSource::ubuntu_systemd(OciPlatform::linux_amd64(), digest('a'), digest('b'))
    }

    fn write_ubuntu_systemd_foundation(root: &Path, packages: &[&str]) {
        write_rootfs_file(root, "/etc/os-release", "ID=ubuntu\nVERSION_ID=\"24.04\"\n");
        write_rootfs_file(root, "/usr/bin/apt-get", "#!/bin/sh\n");
        write_rootfs_file(root, "/usr/bin/dpkg", "#!/bin/sh\n");
        write_rootfs_file(root, "/bin/sh", "#!/bin/sh\n");
        write_rootfs_file(root, "/usr/lib/systemd/systemd", "");
        write_dpkg_status(root, packages);
        create_rootfs_dir(root, "/dev");
    }

    fn fake_guest_binary(root: &Path) -> PathBuf {
        let path = root.join("motlie-vfs-guest");
        fs::write(&path, b"#!/bin/sh\n").unwrap();
        path
    }

    fn installed_ubuntu_packages() -> Vec<&'static str> {
        vec![
            "bash",
            "bubblewrap",
            "ca-certificates",
            "cloud-init",
            "coreutils",
            "curl",
            "dbus",
            "dnsutils",
            "fuse3",
            "git",
            "iproute2",
            "iputils-ping",
            "libfuse3-3",
            "locales",
            "npm",
            "openssh-server",
            "python3",
            "socat",
            "strace",
            "sudo",
            "systemd",
            "systemd-sysv",
            "tmux",
            "vim",
            "wget",
        ]
    }

    fn rootfs_assembly_spec_for_profile(
        profile_spec: RootfsProfileSpec,
        binary: &Path,
    ) -> RootfsCompatibilityLayerSpec {
        let mut spec = RootfsCompatibilityLayerSpec::new(
            profile_spec,
            RootfsCompatibilityBackendEnv::for_backend("ch", "ch-vhost-user"),
        );
        spec.mounts = vec![
            RootfsMountSpec::new("alice-home", "/home/alice"),
            RootfsMountSpec::new("alice-workspace", "/workspace"),
            RootfsMountSpec::new("alice-agent-state", "/agent-state"),
        ];
        spec.guest_binaries.push(RootfsPayloadFile::new(
            binary,
            MOTLIE_V15_GUEST_BIN_OPT,
            0o755,
        ));
        spec.ssh_user_ca_pubkey = Some("ssh-ed25519 AAAATEST motlie-test-ca".to_string());
        let mut user = RootfsUserSeed::new("alice", "alice");
        user.env
            .push(("MOTLIE_GUEST".to_string(), "alice".to_string()));
        spec.users.push(user);
        spec.enable_ch_egress_service = true;
        spec
    }

    fn rootfs_assembly_spec(binary: &Path) -> RootfsCompatibilityLayerSpec {
        rootfs_assembly_spec_for_profile(
            RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source()),
            binary,
        )
    }

    fn rootfs_path(root: &Path, guest_path: &str) -> PathBuf {
        root.join(guest_path.trim_start_matches('/'))
    }

    fn finding_by_kind(
        classification: &RootfsClassification,
        kind: RootfsRequirementKind,
    ) -> &RootfsRequirementFinding {
        classification
            .findings
            .iter()
            .find(|finding| finding.kind == kind)
            .unwrap_or_else(|| panic!("missing finding for {kind:?}"))
    }

    fn package_finding<'a>(
        classification: &'a RootfsClassification,
        package: &str,
    ) -> &'a RootfsRequirementFinding {
        classification
            .findings
            .iter()
            .find(|finding| {
                finding.kind == RootfsRequirementKind::RequiredPackage
                    && finding.package.as_deref() == Some(package)
            })
            .unwrap_or_else(|| panic!("missing package finding for {package}"))
    }

    fn path_finding<'a>(
        classification: &'a RootfsClassification,
        kind: RootfsRequirementKind,
        path: &str,
    ) -> &'a RootfsRequirementFinding {
        classification
            .findings
            .iter()
            .find(|finding| {
                finding.kind == kind && finding.path.as_deref() == Some(Path::new(path))
            })
            .unwrap_or_else(|| panic!("missing {kind:?} finding for {path}"))
    }

    #[test]
    fn platform_defaults_follow_backend_contract() {
        assert_eq!(
            OciPlatform::default_for_v1_5_validation_backend(BackendKind::ChShell),
            OciPlatform::linux_amd64()
        );
        assert_eq!(
            OciPlatform::default_for_v1_5_validation_backend(BackendKind::Vz),
            OciPlatform::linux_arm64()
        );
    }

    #[test]
    fn digest_requires_algorithm_encoded_shape() {
        let digest = OciDigest::new(format!("sha256:{}", "a".repeat(64))).unwrap();
        assert_eq!(digest.algorithm(), "sha256");
        assert_eq!(digest.encoded(), "a".repeat(64));
        assert!(matches!(
            OciDigest::new("sha256:not/valid"),
            Err(ImageContractError::InvalidDigest { .. })
        ));
        assert!(matches!(
            OciDigest::new("sha256:0123456789abcdef"),
            Err(ImageContractError::InvalidDigest { .. })
        ));
        assert!(matches!(
            OciDigest::new("missing-colon"),
            Err(ImageContractError::InvalidDigest { .. })
        ));
    }

    #[test]
    fn image_reference_can_be_rewritten_to_digest_reference() {
        let image_ref = OciImageReference::from_str("ubuntu:24.04").unwrap();
        let digest_ref = image_ref.with_digest(digest('a'));

        assert_eq!(
            digest_ref.normalized(),
            format!("docker.io/library/ubuntu@{}", digest('a'))
        );
        assert!(matches!(
            digest_ref.reference,
            OciImageReferenceKind::Digest(ref value) if value == &digest('a')
        ));
    }

    #[test]
    fn image_reference_parses_docker_official_image() {
        let image_ref = OciImageReference::from_str("ubuntu:24.04").unwrap();

        assert_eq!(image_ref.registry, "docker.io");
        assert_eq!(image_ref.registry_api_host(), "registry-1.docker.io");
        assert_eq!(image_ref.repository, "library/ubuntu");
        assert_eq!(
            image_ref.reference,
            OciImageReferenceKind::Tag("24.04".to_string())
        );
        assert_eq!(image_ref.normalized(), UBUNTU_SYSTEMD_SOURCE_REF);
    }

    #[test]
    fn image_reference_parses_registry_port_and_digest() {
        let image_ref =
            OciImageReference::from_str(&format!("localhost:5000/team/repo@{}", digest('a')))
                .unwrap();

        assert_eq!(image_ref.registry, "localhost:5000");
        assert_eq!(image_ref.registry_api_host(), "localhost:5000");
        assert_eq!(image_ref.repository, "team/repo");
        assert!(matches!(
            image_ref.reference,
            OciImageReferenceKind::Digest(_)
        ));
    }

    #[test]
    fn image_reference_rejects_non_oci_repository_shape() {
        assert!(matches!(
            OciImageReference::from_str("docker.io/Library/Ubuntu:24.04"),
            Err(ImageContractError::InvalidImageRef { .. })
        ));
        assert!(matches!(
            OciImageReference::from_str("https://docker.io/library/ubuntu:24.04"),
            Err(ImageContractError::InvalidImageRef { .. })
        ));
    }

    #[test]
    fn digest_from_bytes_computes_full_sha256() {
        let digest = digest_from_bytes(b"hello").unwrap();

        assert_eq!(
            digest.as_ref(),
            "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn manifest_index_selects_requested_platform_digest() {
        let image_ref = OciImageReference::from_str(UBUNTU_SYSTEMD_SOURCE_REF).unwrap();
        let index = format!(
            r#"{{
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [
                    {{
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": "{}",
                        "platform": {{ "os": "linux", "architecture": "amd64" }}
                    }},
                    {{
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": "{}",
                        "platform": {{ "os": "linux", "architecture": "arm64" }}
                    }}
                ]
            }}"#,
            digest('a'),
            digest('b')
        );

        let selected = select_platform_manifest_digest(
            index.as_bytes(),
            OciPlatform::linux_arm64(),
            &image_ref,
        )
        .unwrap();

        assert_eq!(selected, digest('b'));
    }

    #[test]
    fn manifest_index_reports_missing_platform() {
        let image_ref = OciImageReference::from_str(UBUNTU_SYSTEMD_SOURCE_REF).unwrap();
        let index = format!(
            r#"{{
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [
                    {{
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": "{}",
                        "platform": {{ "os": "linux", "architecture": "amd64" }}
                    }}
                ]
            }}"#,
            digest('a')
        );

        assert!(matches!(
            select_platform_manifest_digest(
                index.as_bytes(),
                OciPlatform::linux_arm64(),
                &image_ref
            ),
            Err(OciRegistryError::PlatformNotFound { .. })
        ));
    }

    #[test]
    fn single_manifest_is_rejected_until_config_platform_is_verified() {
        let image_ref = OciImageReference::from_str(UBUNTU_SYSTEMD_SOURCE_REF).unwrap();
        let manifest = br#"{
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": { "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
            "layers": []
        }"#;

        assert!(matches!(
            select_platform_manifest_digest(manifest, OciPlatform::linux_amd64(), &image_ref),
            Err(OciRegistryError::UnverifiedSingleManifest { .. })
        ));
    }

    #[test]
    fn unknown_manifest_shape_without_descriptors_is_not_accepted() {
        let image_ref = OciImageReference::from_str(UBUNTU_SYSTEMD_SOURCE_REF).unwrap();

        assert!(matches!(
            select_platform_manifest_digest(b"{}", OciPlatform::linux_amd64(), &image_ref),
            Err(OciRegistryError::PlatformNotFound { .. })
        ));
    }

    #[test]
    fn bearer_challenge_parser_handles_quoted_commas() {
        let challenge = parse_bearer_challenge(
            r#"Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:library/ubuntu:pull""#,
        )
        .unwrap();

        assert_eq!(challenge.realm, "https://auth.docker.io/token");
        assert_eq!(challenge.service.as_deref(), Some("registry.docker.io"));
        assert_eq!(
            challenge.scope.as_deref(),
            Some("repository:library/ubuntu:pull")
        );
    }

    #[test]
    fn content_cache_uses_digest_addressed_paths_and_reuses_blobs() {
        let tempdir = tempfile::tempdir().unwrap();
        let cache = OciContentCache::new(tempdir.path().join("cache"));
        let bytes = b"cached layer bytes";
        let digest = digest_from_bytes(bytes).unwrap();

        let stored = cache.store_blob(&digest, bytes).unwrap();
        let cached = cache
            .cached_blob(&digest, Some(bytes.len() as u64))
            .unwrap()
            .unwrap();

        assert_eq!(stored, cached);
        assert_eq!(
            stored.path,
            cache
                .root()
                .join("blobs")
                .join("sha256")
                .join(digest.encoded())
        );
        assert_eq!(fs::read(stored.path).unwrap(), bytes);
    }

    #[test]
    fn content_cache_rejects_corrupt_cached_blobs() {
        let tempdir = tempfile::tempdir().unwrap();
        let cache = OciContentCache::new(tempdir.path().join("cache"));
        let digest = digest_from_bytes(b"expected").unwrap();
        let path = cache.blob_path(&digest);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"actual").unwrap();

        assert!(matches!(
            cache.cached_blob(&digest, Some("actual".len() as u64)),
            Err(OciRegistryError::CacheDigestMismatch { .. })
        ));
    }

    #[test]
    fn platform_manifest_records_rootfs_layers() {
        let manifest = format!(
            r#"{{
                "schemaVersion": 2,
                "mediaType": "{OCI_IMAGE_MANIFEST_MEDIA_TYPE}",
                "config": {{
                    "mediaType": "application/vnd.oci.image.config.v1+json",
                    "digest": "{}",
                    "size": 12
                }},
                "layers": [
                    {{
                        "mediaType": "{OCI_LAYER_GZIP_MEDIA_TYPE}",
                        "digest": "{}",
                        "size": 34
                    }}
                ]
            }}"#,
            digest('c'),
            digest('d')
        );

        let parsed = OciPlatformManifest::from_json(manifest.as_bytes()).unwrap();

        assert_eq!(parsed.media_type, OCI_IMAGE_MANIFEST_MEDIA_TYPE);
        assert_eq!(parsed.config_digest, digest('c'));
        assert_eq!(parsed.layers.len(), 1);
        assert_eq!(parsed.layers[0].digest, digest('d'));
        assert_eq!(parsed.layers[0].media_type, OCI_LAYER_GZIP_MEDIA_TYPE);
        assert_eq!(parsed.layers[0].size, 34);
    }

    #[test]
    fn rootfs_importer_applies_gzip_layers_and_whiteouts() {
        let tempdir = tempfile::tempdir().unwrap();
        let layer_root = tempdir.path().join("layers");
        fs::create_dir(&layer_root).unwrap();
        let rootfs = tempdir.path().join("rootfs");
        let first = gzip_bytes(&tar_bytes(&[
            ("etc/os-release", b"ID=ubuntu\n" as &[u8]),
            ("var/cache/apt/pkg", b"stale"),
        ]));
        let second = gzip_bytes(&tar_bytes(&[
            ("var/cache/apt/.wh.pkg", b""),
            ("workspace/.keep", b""),
        ]));
        let first_path = write_blob(&layer_root, "layer1.tar.gz", &first);
        let second_path = write_blob(&layer_root, "layer2.tar.gz", &second);
        let layers = vec![
            OciLayerInput::new(
                layer_descriptor(OCI_LAYER_GZIP_MEDIA_TYPE, &first),
                first_path,
            ),
            OciLayerInput::new(
                layer_descriptor(OCI_LAYER_GZIP_MEDIA_TYPE, &second),
                second_path,
            ),
        ];

        let imported = OciRootfsImporter::new()
            .import_layers(&layers, &rootfs)
            .unwrap();

        assert_eq!(imported.root, rootfs);
        assert_eq!(imported.applied_layers.len(), 2);
        assert_eq!(
            fs::read_to_string(imported.root.join("etc/os-release")).unwrap(),
            "ID=ubuntu\n"
        );
        assert!(imported.root.join("workspace/.keep").exists());
        assert!(!imported.root.join("var/cache/apt/pkg").exists());
    }

    #[test]
    fn rootfs_importer_applies_opaque_whiteout() {
        let tempdir = tempfile::tempdir().unwrap();
        let layer_root = tempdir.path().join("layers");
        fs::create_dir(&layer_root).unwrap();
        let rootfs = tempdir.path().join("rootfs");
        let first = tar_bytes(&[("var/lib/app/a", b"a" as &[u8]), ("var/lib/app/b", b"b")]);
        let second = tar_bytes(&[("var/lib/app/.wh..wh..opq", b""), ("var/lib/app/c", b"c")]);
        let first_path = write_blob(&layer_root, "layer1.tar", &first);
        let second_path = write_blob(&layer_root, "layer2.tar", &second);
        let layers = vec![
            OciLayerInput::new(
                layer_descriptor(OCI_LAYER_TAR_MEDIA_TYPE, &first),
                first_path,
            ),
            OciLayerInput::new(
                layer_descriptor(OCI_LAYER_TAR_MEDIA_TYPE, &second),
                second_path,
            ),
        ];

        let imported = OciRootfsImporter::new()
            .import_layers(&layers, &rootfs)
            .unwrap();

        assert!(!imported.root.join("var/lib/app/a").exists());
        assert!(!imported.root.join("var/lib/app/b").exists());
        assert_eq!(
            fs::read_to_string(imported.root.join("var/lib/app/c")).unwrap(),
            "c"
        );
    }

    #[test]
    fn rootfs_importer_rejects_empty_whiteout_targets() {
        for whiteout_path in [".wh.", "dir/.wh."] {
            let tempdir = tempfile::tempdir().unwrap();
            let layer_root = tempdir.path().join("layers");
            fs::create_dir(&layer_root).unwrap();
            let rootfs = tempdir.path().join("rootfs");
            let layer = tar_bytes(&[(whiteout_path, b"" as &[u8])]);
            let layer_path = write_blob(&layer_root, "layer.tar", &layer);

            let error = OciRootfsImporter::new()
                .import_layers(
                    &[OciLayerInput::new(
                        layer_descriptor(OCI_LAYER_TAR_MEDIA_TYPE, &layer),
                        layer_path,
                    )],
                    &rootfs,
                )
                .unwrap_err();

            assert!(
                matches!(error, OciRootfsImportError::UnsafeLayerPath { .. }),
                "expected UnsafeLayerPath for {whiteout_path}, got {error:?}"
            );
            assert!(
                rootfs.exists(),
                "empty whiteout target must not remove rootfs for {whiteout_path}"
            );
        }
    }

    #[test]
    fn rootfs_classifier_marks_ubuntu_foundation_compatible_with_adaptation() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        write_rootfs_file(root, "/etc/os-release", "ID=ubuntu\nVERSION_ID=\"24.04\"\n");
        write_rootfs_file(root, "/usr/bin/apt-get", "");
        write_rootfs_file(root, "/usr/bin/dpkg", "");
        write_rootfs_file(root, "/bin/sh", "");
        write_dpkg_status(root, &["base-files", "ca-certificates"]);
        create_rootfs_dir(root, "/dev");

        let spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        let classification = RootfsClassifier::new().classify(root, &spec).unwrap();

        assert_eq!(
            classification.status,
            RootfsClassificationStatus::CompatibleWithAdaptation
        );
        assert!(classification.is_supported_foundation());
        assert_eq!(
            finding_by_kind(&classification, RootfsRequirementKind::OsRelease).status,
            RootfsRequirementStatus::Present
        );
        assert_eq!(
            package_finding(&classification, "git").status,
            RootfsRequirementStatus::MissingButInstallable
        );
        assert_eq!(
            path_finding(
                &classification,
                RootfsRequirementKind::RequiredMountPoint,
                "/workspace"
            )
            .status,
            RootfsRequirementStatus::NeedsCompatibilityLayer
        );
        assert_eq!(
            path_finding(
                &classification,
                RootfsRequirementKind::VfsFuseRuntimeDevice,
                "/dev/fuse"
            )
            .status,
            RootfsRequirementStatus::NeedsRuntimeProvisioning
        );
    }

    #[cfg(unix)]
    #[test]
    fn rootfs_classifier_rejects_symlink_escape_outside_root() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir_all(root.join("etc")).unwrap();
        fs::write(tempdir.path().join("outside-os-release"), "ID=ubuntu\n").unwrap();
        symlink("../../outside-os-release", root.join("etc/os-release")).unwrap();

        let spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        let error = RootfsClassifier::new().classify(&root, &spec).unwrap_err();

        assert!(
            matches!(error, RootfsClassificationError::SymlinkEscapesRoot { .. }),
            "expected symlink escape rejection, got {error:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rootfs_classifier_resolves_valid_in_root_ubuntu_symlinks() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        create_rootfs_dir(root, "/etc");
        create_rootfs_dir(root, "/usr/bin");
        create_rootfs_dir(root, "/usr/lib/systemd");
        create_rootfs_dir(root, "/dev");
        write_rootfs_file(
            root,
            "/usr/lib/os-release",
            "ID=ubuntu\nVERSION_ID=\"24.04\"\n",
        );
        write_rootfs_file(root, "/usr/bin/apt-get", "");
        write_rootfs_file(root, "/usr/bin/dpkg", "");
        write_rootfs_file(root, "/usr/bin/sh", "");
        write_rootfs_file(root, "/usr/lib/systemd/systemd", "");
        symlink("/usr/lib/os-release", root.join("etc/os-release")).unwrap();
        symlink("usr/bin", root.join("bin")).unwrap();
        symlink("usr/lib", root.join("lib")).unwrap();

        let spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        let classification = RootfsClassifier::new().classify(root, &spec).unwrap();

        assert_ne!(
            classification.status,
            RootfsClassificationStatus::Unsupported
        );
        assert_eq!(
            finding_by_kind(&classification, RootfsRequirementKind::OsRelease).status,
            RootfsRequirementStatus::Present
        );
        assert_eq!(
            path_finding(
                &classification,
                RootfsRequirementKind::RequiredBinary,
                "/bin/sh"
            )
            .status,
            RootfsRequirementStatus::Present
        );
        assert_eq!(
            finding_by_kind(&classification, RootfsRequirementKind::InitSystem).status,
            RootfsRequirementStatus::Present
        );
    }

    #[test]
    fn rootfs_classifier_does_not_accept_arbitrary_sbin_init_as_systemd() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        write_rootfs_file(root, "/etc/os-release", "ID=ubuntu\nVERSION_ID=\"24.04\"\n");
        write_rootfs_file(root, "/usr/bin/apt-get", "");
        write_rootfs_file(root, "/usr/bin/dpkg", "");
        write_rootfs_file(root, "/bin/sh", "");
        write_rootfs_file(root, "/sbin/init", "#!/bin/sh\n");
        create_rootfs_dir(root, "/dev");

        let spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        let classification = RootfsClassifier::new().classify(root, &spec).unwrap();

        assert_eq!(
            finding_by_kind(&classification, RootfsRequirementKind::InitSystem).status,
            RootfsRequirementStatus::MissingButInstallable
        );
    }

    #[test]
    fn rootfs_classifier_rejects_package_manager_directories() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        write_rootfs_file(root, "/etc/os-release", "ID=ubuntu\nVERSION_ID=\"24.04\"\n");
        create_rootfs_dir(root, "/usr/bin/apt-get");
        create_rootfs_dir(root, "/usr/bin/dpkg");
        write_rootfs_file(root, "/bin/sh", "");
        write_rootfs_file(root, "/usr/lib/systemd/systemd", "");
        create_rootfs_dir(root, "/dev");

        let spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        let classification = RootfsClassifier::new().classify(root, &spec).unwrap();

        assert_eq!(
            finding_by_kind(&classification, RootfsRequirementKind::PackageManager).status,
            RootfsRequirementStatus::Unsupported
        );
        assert_eq!(
            classification.status,
            RootfsClassificationStatus::Unsupported
        );
    }

    #[test]
    fn rootfs_classifier_does_not_accept_systemd_directory() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        write_rootfs_file(root, "/etc/os-release", "ID=ubuntu\nVERSION_ID=\"24.04\"\n");
        write_rootfs_file(root, "/usr/bin/apt-get", "");
        write_rootfs_file(root, "/usr/bin/dpkg", "");
        write_rootfs_file(root, "/bin/sh", "");
        create_rootfs_dir(root, "/usr/lib/systemd/systemd");
        create_rootfs_dir(root, "/dev");

        let spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        let classification = RootfsClassifier::new().classify(root, &spec).unwrap();

        assert_eq!(
            finding_by_kind(&classification, RootfsRequirementKind::InitSystem).status,
            RootfsRequirementStatus::MissingButInstallable
        );
    }

    #[test]
    fn rootfs_classifier_rejects_wrong_os_for_ubuntu_profile() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        write_rootfs_file(root, "/etc/os-release", "ID=alpine\nVERSION_ID=\"3.20\"\n");
        write_rootfs_file(root, "/usr/bin/apt-get", "");
        write_rootfs_file(root, "/usr/bin/dpkg", "");
        write_rootfs_file(root, "/bin/sh", "");
        create_rootfs_dir(root, "/dev");

        let spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        let classification = RootfsClassifier::new().classify(root, &spec).unwrap();

        assert_eq!(
            classification.status,
            RootfsClassificationStatus::Unsupported
        );
        assert!(!classification.is_supported_foundation());
        assert_eq!(
            finding_by_kind(&classification, RootfsRequirementKind::OsRelease).status,
            RootfsRequirementStatus::Unsupported
        );
    }

    #[test]
    fn rootfs_classifier_honors_custom_profile_requirements() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        write_rootfs_file(root, "/etc/os-release", "ID=ubuntu\nVERSION_ID=\"24.04\"\n");
        write_rootfs_file(root, "/usr/bin/apt-get", "");
        write_rootfs_file(root, "/usr/bin/dpkg", "");
        write_rootfs_file(root, "/usr/lib/systemd/systemd", "");
        write_dpkg_status(root, &["base-files"]);
        create_rootfs_dir(root, "/dev");
        create_rootfs_dir(root, "/project");

        let mut profile = GuestImageProfile::ubuntu_systemd(ubuntu_classifier_source());
        profile.required_packages = vec!["base-files".to_string()];
        profile.required_mount_points = vec![PathBuf::from("/project")];
        let mut spec = RootfsProfileSpec::for_profile(profile);
        spec.required_binaries.clear();
        spec.vfs.requires_fuse_runtime_device = false;
        let classification = RootfsClassifier::new().classify(root, &spec).unwrap();

        assert_eq!(classification.status, RootfsClassificationStatus::Ready);
        assert_eq!(
            package_finding(&classification, "base-files").status,
            RootfsRequirementStatus::Present
        );
        assert!(
            classification
                .findings
                .iter()
                .all(|finding| finding.package.as_deref() != Some("git")),
            "custom package set must not inherit v1.5 example package choices"
        );
    }

    #[test]
    fn rootfs_profile_spec_rejects_relative_binary_requirements() {
        let mut spec = RootfsProfileSpec::ubuntu_systemd(ubuntu_classifier_source());
        spec.required_binaries = vec![PathBuf::from("bin/sh")];

        assert!(matches!(
            spec.validate(),
            Err(RootfsClassificationError::InvalidRequirementPath { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rootfs_assembler_installs_v15_compatibility_contract() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir(&root).unwrap();
        write_ubuntu_systemd_foundation(&root, &installed_ubuntu_packages());
        let binary = fake_guest_binary(tempdir.path());
        let spec = rootfs_assembly_spec(&binary);

        let manifest = RootfsCompatibilityAssembler::new()
            .assemble(&root, &spec)
            .unwrap();

        assert_eq!(manifest.contract_version, MOTLIE_V15_CONTRACT_VERSION);
        assert_eq!(manifest.profile.name, UBUNTU_SYSTEMD_PROFILE);
        assert!(
            manifest.pending_requirements.iter().any(|finding| {
                finding.kind == RootfsRequirementKind::VfsFuseRuntimeDevice
                    && finding.status == RootfsRequirementStatus::NeedsRuntimeProvisioning
            }),
            "runtime /dev/fuse provisioning should remain explicit in the manifest"
        );
        assert_eq!(
            fs::read(rootfs_path(&root, MOTLIE_V15_GUEST_BIN_OPT)).unwrap(),
            b"#!/bin/sh\n"
        );
        assert_eq!(
            fs::read_link(rootfs_path(&root, MOTLIE_V15_GUEST_BIN_COMPAT)).unwrap(),
            PathBuf::from(MOTLIE_V15_GUEST_BIN_OPT)
        );
        let backend_env =
            fs::read_to_string(rootfs_path(&root, MOTLIE_V15_BACKEND_ENV_PATH)).unwrap();
        assert!(backend_env.contains("MOTLIE_BACKEND=ch\n"));
        assert!(backend_env.contains("MOTLIE_NET_BACKEND=ch-vhost-user\n"));
        let mounts_yaml = fs::read_to_string(rootfs_path(&root, MOTLIE_V15_MOUNTS_PATH)).unwrap();
        assert!(mounts_yaml.contains("tag: alice-home"));
        assert!(mounts_yaml.contains("guest_path: /home/alice"));
        assert!(rootfs_path(&root, "/etc/systemd/system/motlie-vfs-guest.service").exists());
        assert_eq!(
            fs::read_link(rootfs_path(
                &root,
                "/etc/systemd/system/cloud-init.target.wants/motlie-vfs-guest.service"
            ))
            .unwrap(),
            PathBuf::from("../motlie-vfs-guest.service")
        );
        assert!(
            fs::symlink_metadata(rootfs_path(
                &root,
                "/usr/local/bin/motlie-agent-state-setup"
            ))
            .unwrap()
            .file_type()
            .is_symlink(),
            "compatibility scripts should be symlinks, not duplicate files"
        );
        assert_eq!(
            fs::read_link(rootfs_path(
                &root,
                "/usr/local/bin/motlie-agent-state-setup"
            ))
            .unwrap(),
            PathBuf::from("/opt/motlie/v1.5/guest/bin/motlie-agent-state-setup")
        );
        assert!(
            fs::read_to_string(rootfs_path(&root, "/etc/profile.d/agent-state.sh"))
                .unwrap()
                .contains("CODEX_HOME")
        );
        let sshd_ca_config =
            fs::read_to_string(rootfs_path(&root, MOTLIE_V15_SSHD_CA_CONFIG_PATH)).unwrap();
        assert!(sshd_ca_config.contains("TrustedUserCAKeys /etc/ssh/ca/user_ca.pub"));
        assert!(sshd_ca_config.contains("AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u"));
        assert_eq!(
            fs::read_to_string(rootfs_path(&root, "/etc/ssh/auth_principals/alice")).unwrap(),
            "alice\n"
        );
        assert!(
            fs::read_to_string(rootfs_path(&root, "/etc/sudoers.d/90-motlie-vmm"))
                .unwrap()
                .contains("alice ALL=(ALL) NOPASSWD:ALL")
        );
        assert!(
            manifest.installed.iter().any(|record| {
                record.kind == RootfsCompatibilityInstallKind::ServiceEnablement
                    && record.path
                        == PathBuf::from(
                            "/etc/systemd/system/multi-user.target.wants/motlie-vmm-egress.service",
                        )
            }),
            "CH egress service enablement should be recorded when requested"
        );
    }

    #[test]
    fn rootfs_assembler_uses_backend_env_for_ssh_vsock_port() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir(&root).unwrap();
        write_ubuntu_systemd_foundation(&root, &installed_ubuntu_packages());
        let binary = fake_guest_binary(tempdir.path());
        let mut spec = rootfs_assembly_spec(&binary);
        spec.backend_env.motlie_ssh_vsock_port = 2444;

        RootfsCompatibilityAssembler::new()
            .assemble(&root, &spec)
            .unwrap();

        let backend_env =
            fs::read_to_string(rootfs_path(&root, MOTLIE_V15_BACKEND_ENV_PATH)).unwrap();
        assert!(backend_env.contains("MOTLIE_SSH_VSOCK_PORT=2444\n"));
        let loop_script = fs::read_to_string(rootfs_path(
            &root,
            "/opt/motlie/v1.5/guest/bin/motlie-vmm-vsock-ssh-loop",
        ))
        .unwrap();
        assert!(loop_script.contains("SSH_VSOCK_PORT=\"${MOTLIE_SSH_VSOCK_PORT:-2222}\""));
        assert!(loop_script.contains("VSOCK-CONNECT:${SSH_HOST_CID}:${SSH_VSOCK_PORT}"));
        assert!(
            !loop_script.contains("VSOCK-CONNECT:2:2222"),
            "script must not hardcode the default SSH vsock port"
        );
    }

    #[test]
    fn rootfs_assembler_rejects_unsafe_mount_yaml_values() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir(&root).unwrap();
        write_ubuntu_systemd_foundation(&root, &installed_ubuntu_packages());
        let binary = fake_guest_binary(tempdir.path());

        let mut bad_tag = rootfs_assembly_spec(&binary);
        bad_tag.mounts[0].tag = "alice:home".to_string();
        let error = RootfsCompatibilityAssembler::new()
            .assemble(&root, &bad_tag)
            .unwrap_err();
        assert!(
            matches!(error, RootfsCompatibilityError::InvalidConfigValue { .. }),
            "expected strict mount tag validation, got {error:?}"
        );

        let mut bad_path = rootfs_assembly_spec(&binary);
        bad_path.mounts[0].guest_path = PathBuf::from("/home/alice workspace");
        let error = RootfsCompatibilityAssembler::new()
            .assemble(&root, &bad_path)
            .unwrap_err();
        assert!(
            matches!(error, RootfsCompatibilityError::InvalidInstallPath { .. }),
            "expected strict mount guest path validation, got {error:?}"
        );
    }

    #[test]
    fn rootfs_assembler_records_installable_requirements_without_mutating_packages() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir(&root).unwrap();
        write_ubuntu_systemd_foundation(
            &root,
            &[
                "ca-certificates",
                "curl",
                "iproute2",
                "openssh-server",
                "sudo",
            ],
        );
        let binary = fake_guest_binary(tempdir.path());
        let spec = rootfs_assembly_spec(&binary);

        let manifest = RootfsCompatibilityAssembler::new()
            .assemble(&root, &spec)
            .unwrap();

        assert!(
            manifest.pending_requirements.iter().any(|finding| {
                finding.kind == RootfsRequirementKind::RequiredPackage
                    && finding.status == RootfsRequirementStatus::MissingButInstallable
                    && finding.package.as_deref() == Some("git")
            }),
            "missing installable packages must be manifest evidence, not silent success"
        );
        let status = fs::read_to_string(rootfs_path(&root, "/var/lib/dpkg/status")).unwrap();
        assert!(
            !status.contains("Package: git\n"),
            "the assembler must not claim package installation happened"
        );
    }

    #[test]
    fn rootfs_assembler_can_fail_when_installable_requirements_are_pending() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir(&root).unwrap();
        write_ubuntu_systemd_foundation(&root, &["ca-certificates"]);
        let binary = fake_guest_binary(tempdir.path());
        let mut spec = rootfs_assembly_spec(&binary);
        spec.pending_requirement_policy = RootfsPendingRequirementPolicy::FailInstallable;

        let error = RootfsCompatibilityAssembler::new()
            .assemble(&root, &spec)
            .unwrap_err();

        assert!(
            matches!(
                error,
                RootfsCompatibilityError::InstallableRequirementsPending { .. }
            ),
            "expected pending installable requirements to fail, got {error:?}"
        );
    }

    #[test]
    fn rootfs_assembler_rejects_unsupported_foundations() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir(&root).unwrap();
        write_rootfs_file(&root, "/etc/os-release", "ID=alpine\nVERSION_ID=\"3.20\"\n");
        write_rootfs_file(&root, "/usr/bin/apt-get", "#!/bin/sh\n");
        write_rootfs_file(&root, "/usr/bin/dpkg", "#!/bin/sh\n");
        write_rootfs_file(&root, "/bin/sh", "#!/bin/sh\n");
        create_rootfs_dir(&root, "/dev");
        let binary = fake_guest_binary(tempdir.path());
        let spec = rootfs_assembly_spec(&binary);

        let error = RootfsCompatibilityAssembler::new()
            .assemble(&root, &spec)
            .unwrap_err();

        assert!(
            matches!(error, RootfsCompatibilityError::UnsupportedRootfs { .. }),
            "expected unsupported rootfs error, got {error:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rootfs_assembler_rejects_symlink_parents_for_writes() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().join("rootfs");
        fs::create_dir(&root).unwrap();
        write_ubuntu_systemd_foundation(&root, &installed_ubuntu_packages());
        fs::create_dir(tempdir.path().join("outside")).unwrap();
        symlink("../outside", root.join("etc/motlie")).unwrap();
        let binary = fake_guest_binary(tempdir.path());
        let spec = rootfs_assembly_spec(&binary);

        let error = RootfsCompatibilityAssembler::new()
            .assemble(&root, &spec)
            .unwrap_err();

        assert!(
            matches!(error, RootfsCompatibilityError::InvalidInstallPath { .. }),
            "expected symlink parent rejection, got {error:?}"
        );
    }

    #[test]
    fn rootfs_importer_rejects_digest_mismatch() {
        let tempdir = tempfile::tempdir().unwrap();
        let layer_root = tempdir.path().join("layers");
        fs::create_dir(&layer_root).unwrap();
        let rootfs = tempdir.path().join("rootfs");
        let layer = tar_bytes(&[("etc/os-release", b"ID=ubuntu\n" as &[u8])]);
        let layer_path = write_blob(&layer_root, "layer.tar", &layer);
        let descriptor = OciLayerDescriptor {
            media_type: OCI_LAYER_TAR_MEDIA_TYPE.to_string(),
            digest: digest('e'),
            size: layer.len() as u64,
        };

        assert!(matches!(
            OciRootfsImporter::new()
                .import_layers(&[OciLayerInput::new(descriptor, layer_path)], &rootfs),
            Err(OciRootfsImportError::LayerDigestMismatch { .. })
        ));
    }

    #[test]
    fn rootfs_importer_rejects_unsupported_layer_digest_algorithm() {
        let tempdir = tempfile::tempdir().unwrap();
        let layer_root = tempdir.path().join("layers");
        fs::create_dir(&layer_root).unwrap();
        let rootfs = tempdir.path().join("rootfs");
        let layer = tar_bytes(&[("etc/os-release", b"ID=ubuntu\n" as &[u8])]);
        let layer_path = write_blob(&layer_root, "layer.tar", &layer);
        let descriptor = OciLayerDescriptor {
            media_type: OCI_LAYER_TAR_MEDIA_TYPE.to_string(),
            digest: OciDigest::new(format!("sha512:{}", "a".repeat(128))).unwrap(),
            size: layer.len() as u64,
        };

        assert!(matches!(
            OciRootfsImporter::new()
                .import_layers(&[OciLayerInput::new(descriptor, layer_path)], &rootfs),
            Err(OciRootfsImportError::UnsupportedLayerDigestAlgorithm { .. })
        ));
    }

    #[test]
    fn rootfs_importer_rejects_unsafe_layer_paths() {
        assert!(matches!(
            sanitize_layer_path(Path::new("../escape")),
            Err(OciRootfsImportError::UnsafeLayerPath { .. })
        ));
        assert!(matches!(
            sanitize_layer_path(Path::new("/absolute")),
            Err(OciRootfsImportError::UnsafeLayerPath { .. })
        ));
    }

    #[test]
    fn rootfs_importer_requires_empty_assembly_root() {
        let tempdir = tempfile::tempdir().unwrap();
        let layer_root = tempdir.path().join("layers");
        fs::create_dir(&layer_root).unwrap();
        let rootfs = tempdir.path().join("rootfs");
        fs::create_dir(&rootfs).unwrap();
        fs::write(rootfs.join("stale"), b"stale").unwrap();
        let layer = tar_bytes(&[("etc/os-release", b"ID=ubuntu\n" as &[u8])]);
        let layer_path = write_blob(&layer_root, "layer.tar", &layer);

        assert!(matches!(
            OciRootfsImporter::new().import_layers(
                &[OciLayerInput::new(
                    layer_descriptor(OCI_LAYER_TAR_MEDIA_TYPE, &layer),
                    layer_path,
                )],
                &rootfs
            ),
            Err(OciRootfsImportError::AssemblyRootNotEmpty { .. })
        ));
    }

    #[tokio::test]
    #[ignore = "requires external registry network access"]
    async fn resolves_ubuntu_systemd_source_from_registry() {
        let resolver = OciRegistryClient::new();
        for platform in [OciPlatform::linux_amd64(), OciPlatform::linux_arm64()] {
            let source = resolver
                .resolve_ubuntu_systemd_source(platform)
                .await
                .unwrap();

            assert_eq!(source.image_ref, UBUNTU_SYSTEMD_SOURCE_REF);
            assert_eq!(source.platform, platform);
            assert!(source.image_index_digest.as_ref().starts_with("sha256:"));
            assert!(source
                .platform_manifest_digest
                .as_ref()
                .starts_with("sha256:"));
            source.validate().unwrap();
        }
    }

    #[tokio::test]
    #[ignore = "downloads Ubuntu OCI layer blobs and unpacks the rootfs"]
    async fn fetches_ubuntu_platform_layers_to_cache_and_imports_rootfs() {
        let tempdir = tempfile::tempdir().unwrap();
        let cache = OciContentCache::new(tempdir.path().join("cache"));
        let resolver = OciRegistryClient::new();
        let cached = resolver
            .resolve_and_fetch_ubuntu_systemd_to_cache(OciPlatform::linux_amd64(), &cache)
            .await
            .unwrap();

        assert_eq!(
            cached.resolved.image_ref.normalized(),
            UBUNTU_SYSTEMD_SOURCE_REF
        );
        assert!(!cached.layers.is_empty());
        assert!(cached.manifest_blob.path.exists());
        for layer in &cached.layers {
            assert!(layer.path.exists());
            verify_layer_blob(&layer.path, &layer.descriptor).unwrap();
        }

        let imported = OciRootfsImporter::new()
            .import_layers(&cached.layers, tempdir.path().join("rootfs"))
            .unwrap();
        let os_release = fs::read_to_string(imported.root.join("etc/os-release")).unwrap();
        assert!(os_release.contains("ID=ubuntu"));
        let profile = GuestImageProfile::ubuntu_systemd(cached.resolved.into_external_source());
        let spec = RootfsProfileSpec::for_profile(profile);
        let classification = RootfsClassifier::new()
            .classify(&imported.root, &spec)
            .unwrap();
        assert!(classification.is_supported_foundation());
        assert_ne!(
            classification.status,
            RootfsClassificationStatus::Unsupported
        );

        let binary = fake_guest_binary(tempdir.path());
        let assembly_spec = rootfs_assembly_spec_for_profile(spec, &binary);
        let manifest = RootfsCompatibilityAssembler::new()
            .assemble(&imported.root, &assembly_spec)
            .unwrap();
        assert_eq!(manifest.contract_version, MOTLIE_V15_CONTRACT_VERSION);
        assert!(imported
            .root
            .join(MOTLIE_V15_BACKEND_ENV_PATH.trim_start_matches('/'))
            .exists());
        assert!(
            manifest.pending_requirements.iter().any(|finding| {
                matches!(
                    finding.status,
                    RootfsRequirementStatus::MissingButInstallable
                        | RootfsRequirementStatus::NeedsRuntimeProvisioning
                )
            }),
            "live Ubuntu OCI import should preserve pending install/runtime evidence"
        );
    }

    #[test]
    fn ubuntu_profile_records_source_identity() {
        let source =
            ExternalOciSource::ubuntu_systemd(OciPlatform::linux_amd64(), digest('a'), digest('b'));
        let profile = GuestImageProfile::ubuntu_systemd(source);

        assert_eq!(profile.name, UBUNTU_SYSTEMD_PROFILE);
        assert_eq!(profile.source.image_ref, UBUNTU_SYSTEMD_SOURCE_REF);
        assert_eq!(profile.source.platform.to_string(), "linux/amd64");
        assert!(profile
            .required_packages
            .iter()
            .any(|pkg| pkg == "openssh-server"));
        assert!(profile
            .required_mount_points
            .iter()
            .any(|path| path == &PathBuf::from("/workspace")));
        profile.validate().unwrap();
    }

    #[test]
    fn profile_required_mount_points_must_be_absolute() {
        let source =
            ExternalOciSource::ubuntu_systemd(OciPlatform::linux_amd64(), digest('a'), digest('b'));
        let mut profile = GuestImageProfile::ubuntu_systemd(source);
        profile.required_mount_points = vec![PathBuf::from("workspace")];

        assert_eq!(
            profile.validate().unwrap_err(),
            ImageContractError::RelativeRequiredMountPoint(PathBuf::from("workspace"))
        );
    }

    #[test]
    fn ubuntu_profile_rejects_mismatched_source() {
        let mut source =
            ExternalOciSource::ubuntu_systemd(OciPlatform::linux_amd64(), digest('a'), digest('b'));
        source.image_ref = "docker.io/library/alpine:3".to_string();
        let profile = GuestImageProfile::ubuntu_systemd(source);

        assert_eq!(
            profile.validate().unwrap_err(),
            ImageContractError::ProfileSourceMismatch {
                profile: UBUNTU_SYSTEMD_PROFILE.to_string(),
                expected_image_ref: UBUNTU_SYSTEMD_SOURCE_REF.to_string(),
                actual_image_ref: "docker.io/library/alpine:3".to_string(),
            }
        );
    }

    #[test]
    fn validation_record_revalidates_embedded_profile_digests() {
        let mut profile = GuestImageProfile::ubuntu_systemd(ExternalOciSource::ubuntu_systemd(
            OciPlatform::linux_arm64(),
            digest('a'),
            digest('b'),
        ));
        profile.source.image_index_digest = OciDigest("sha256:0123456789abcdef".to_string());
        let record = GuestImageValidationRecord {
            profile,
            contract_version: "v1.5".to_string(),
            backend_kind: BackendKind::Vz,
            emitted_artifacts: vec![EmittedArtifactDigest {
                label: "rootfs".to_string(),
                path: PathBuf::from("artifacts/base/rootfs.squashfs"),
                digest: digest('c'),
            }],
        };

        assert!(matches!(
            record.validate(),
            Err(ImageContractError::InvalidDigest { .. })
        ));
    }

    #[test]
    fn validation_record_requires_backend_artifact_digests() {
        let record = GuestImageValidationRecord {
            profile: GuestImageProfile::ubuntu_systemd(ExternalOciSource::ubuntu_systemd(
                OciPlatform::linux_arm64(),
                digest('a'),
                digest('b'),
            )),
            contract_version: "v1.5".to_string(),
            backend_kind: BackendKind::Vz,
            emitted_artifacts: vec![],
        };

        assert_eq!(
            record.validate().unwrap_err(),
            ImageContractError::MissingEmittedArtifacts
        );
    }
}
