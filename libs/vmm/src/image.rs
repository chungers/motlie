use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend::BackendKind;

/// First supported external OCI import profile.
pub const UBUNTU_SYSTEMD_PROFILE: &str = "ubuntu-systemd";
pub const UBUNTU_SYSTEMD_SOURCE_REF: &str = "docker.io/library/ubuntu:24.04";

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
        Ok(Self(trimmed.to_string()))
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
                "ca-certificates".to_string(),
                "curl".to_string(),
                "git".to_string(),
                "iproute2".to_string(),
                "openssh-server".to_string(),
                "sudo".to_string(),
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
    pub profile_name: String,
    pub contract_version: String,
    pub source: ExternalOciSource,
    pub backend_kind: BackendKind,
    pub emitted_artifacts: Vec<EmittedArtifactDigest>,
}

impl GuestImageValidationRecord {
    pub fn validate(&self) -> Result<(), ImageContractError> {
        if self.profile_name.trim().is_empty() {
            return Err(ImageContractError::EmptyProfileName);
        }
        if self.contract_version.trim().is_empty() {
            return Err(ImageContractError::EmptyContractVersion);
        }
        self.source.validate()?;
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
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ImageContractError {
    #[error("OCI image reference cannot be empty")]
    EmptyImageRef,
    #[error("OCI digest cannot be empty")]
    EmptyDigest,
    #[error("invalid OCI digest '{value}': {reason}")]
    InvalidDigest { value: String, reason: String },
    #[error("guest image profile name cannot be empty")]
    EmptyProfileName,
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

    fn digest(byte: char) -> OciDigest {
        OciDigest::new(format!("sha256:{}", byte.to_string().repeat(64))).unwrap()
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
        assert!(OciDigest::new("sha256:0123456789abcdef").is_ok());
        assert!(matches!(
            OciDigest::new("sha256:not/valid"),
            Err(ImageContractError::InvalidDigest { .. })
        ));
        assert!(matches!(
            OciDigest::new("missing-colon"),
            Err(ImageContractError::InvalidDigest { .. })
        ));
    }

    #[test]
    fn ubuntu_profile_records_source_identity() {
        let source =
            ExternalOciSource::ubuntu_systemd(OciPlatform::linux_amd64(), digest('a'), digest('b'));
        let profile = GuestImageProfile::ubuntu_systemd(source);

        assert_eq!(profile.name, UBUNTU_SYSTEMD_PROFILE);
        assert_eq!(profile.source.image_ref, UBUNTU_SYSTEMD_SOURCE_REF);
        assert_eq!(profile.source.platform.to_string(), "linux/amd64");
        assert!(
            profile
                .required_packages
                .iter()
                .any(|pkg| pkg == "openssh-server")
        );
        assert!(
            profile
                .required_mount_points
                .iter()
                .any(|path| path == &PathBuf::from("/workspace"))
        );
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
    fn validation_record_requires_backend_artifact_digests() {
        let record = GuestImageValidationRecord {
            profile_name: UBUNTU_SYSTEMD_PROFILE.to_string(),
            contract_version: "v1.5".to_string(),
            source: ExternalOciSource::ubuntu_systemd(
                OciPlatform::linux_arm64(),
                digest('a'),
                digest('b'),
            ),
            backend_kind: BackendKind::Vz,
            emitted_artifacts: vec![],
        };

        assert_eq!(
            record.validate().unwrap_err(),
            ImageContractError::MissingEmittedArtifacts
        );
    }
}
