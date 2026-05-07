use std::fs::{self, File};
use std::io::{self, Read, Write};
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
        Ok(_) | Err(_) => {}
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
            assert!(
                source
                    .platform_manifest_digest
                    .as_ref()
                    .starts_with("sha256:")
            );
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
