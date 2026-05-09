use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use motlie_vmm::image::MOTLIE_V15_CONTRACT_VERSION;
use serde::{Deserialize, Serialize};

const DEFAULT_MANIFEST_NAME: &str = "mbuild-manifest.json";

fn main() {
    if let Err(error) = run() {
        eprintln!("mbuild: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let command = parse_args(env::args().skip(1).collect())?;
    match command {
        Command::Build {
            config,
            target,
            out,
        } => build(&config, target, &out),
        Command::Validate { config, artifact } => validate(&config, &artifact),
        Command::Help => {
            print_usage();
            Ok(())
        }
    }
}

fn build(config_path: &Path, target: ImageTarget, out: &Path) -> Result<(), Box<dyn Error>> {
    let config = load_config(config_path)?;
    config.validate_for_target(target)?;
    fs::create_dir_all(out)?;

    let manifest = ImageBuildManifest::from_config(config_path, target, out, &config);
    let manifest_path = out.join(DEFAULT_MANIFEST_NAME);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", manifest_path.display());
    Ok(())
}

fn validate(config_path: &Path, artifact: &Path) -> Result<(), Box<dyn Error>> {
    let config = load_config(config_path)?;
    config.validate()?;

    let manifest_path = artifact.join(DEFAULT_MANIFEST_NAME);
    let manifest: ImageBuildManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.contract_version != config.version {
        return Err(format!(
            "manifest contract version {} does not match config version {}",
            manifest.contract_version, config.version
        )
        .into());
    }
    if !config.emitters.contains(&manifest.target) {
        return Err(format!(
            "manifest target {} is not declared by config emitters",
            manifest.target.as_str()
        )
        .into());
    }
    if manifest.stages.is_empty() {
        return Err("manifest does not contain stage records".into());
    }
    println!("validated {}", manifest_path.display());
    Ok(())
}

fn load_config(path: &Path) -> Result<ImageBuildConfig, Box<dyn Error>> {
    let config: ImageBuildConfig = serde_yaml::from_slice(&fs::read(path)?)?;
    config.validate()?;
    Ok(config)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Build {
        config: PathBuf,
        target: ImageTarget,
        out: PathBuf,
    },
    Validate {
        config: PathBuf,
        artifact: PathBuf,
    },
    Help,
}

fn parse_args(args: Vec<String>) -> Result<Command, Box<dyn Error>> {
    let Some(command) = args.first().map(String::as_str) else {
        return Ok(Command::Help);
    };
    match command {
        "build" => parse_build_args(&args[1..]),
        "validate" => parse_validate_args(&args[1..]),
        "-h" | "--help" | "help" => Ok(Command::Help),
        other => Err(format!("unknown command {other:?}").into()),
    }
}

fn parse_build_args(args: &[String]) -> Result<Command, Box<dyn Error>> {
    let mut config = None;
    let mut target = None;
    let mut out = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--config" => config = iter.next().map(PathBuf::from),
            "--target" => {
                target = Some(
                    iter.next()
                        .ok_or("--target requires an argument")?
                        .parse::<ImageTarget>()?,
                );
            }
            "--out" => out = iter.next().map(PathBuf::from),
            "-h" | "--help" => return Ok(Command::Help),
            other => return Err(format!("unexpected build argument {other:?}").into()),
        }
    }
    Ok(Command::Build {
        config: config.ok_or("build requires --config PATH")?,
        target: target.ok_or("build requires --target ch|vz")?,
        out: out.ok_or("build requires --out PATH")?,
    })
}

fn parse_validate_args(args: &[String]) -> Result<Command, Box<dyn Error>> {
    let mut config = None;
    let mut artifact = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--config" => config = iter.next().map(PathBuf::from),
            "--artifact" => artifact = iter.next().map(PathBuf::from),
            "-h" | "--help" => return Ok(Command::Help),
            other => return Err(format!("unexpected validate argument {other:?}").into()),
        }
    }
    Ok(Command::Validate {
        config: config.ok_or("validate requires --config PATH")?,
        artifact: artifact.ok_or("validate requires --artifact PATH")?,
    })
}

fn print_usage() {
    println!(
        "\
Usage:
  mbuild build --config PATH --target ch|vz --out DIR
  mbuild validate --config PATH --artifact DIR

The current implementation consumes the v1.5 Dockerfile-like config and emits
a machine-readable stage manifest. Package installation and CH/VZ artifact
emission are explicit follow-on stages, not launch-time side effects."
    );
}

#[derive(Debug, Clone, Deserialize)]
struct ImageBuildConfig {
    version: String,
    source: SourceStage,
    package_stage: PackageStage,
    immutable_payloads: Vec<PayloadSpec>,
    sshd_policy: SshdPolicy,
    services: Vec<ServiceSpec>,
    immutable_files: Vec<String>,
    seed_files: Vec<String>,
    emitters: Vec<ImageTarget>,
    validation: Vec<String>,
}

impl ImageBuildConfig {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        require_eq("version", &self.version, MOTLIE_V15_CONTRACT_VERSION)?;
        self.source.validate()?;
        self.package_stage.validate()?;
        if self.immutable_payloads.is_empty() {
            return Err("immutable_payloads must not be empty".into());
        }
        for payload in &self.immutable_payloads {
            payload.validate()?;
        }
        self.sshd_policy.validate()?;
        if self.services.is_empty() {
            return Err("services must not be empty".into());
        }
        for service in &self.services {
            service.validate()?;
        }
        if self.immutable_files.is_empty() {
            return Err("immutable_files must not be empty".into());
        }
        if self.seed_files.is_empty() {
            return Err("seed_files must not be empty".into());
        }
        if self.emitters.is_empty() {
            return Err("emitters must not be empty".into());
        }
        if self.validation.is_empty() {
            return Err("validation must not be empty".into());
        }
        Ok(())
    }

    fn validate_for_target(&self, target: ImageTarget) -> Result<(), Box<dyn Error>> {
        self.validate()?;
        if !self.emitters.contains(&target) {
            return Err(format!("target {} is not declared in emitters", target.as_str()).into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SourceStage {
    image: String,
    profile: String,
    platform: String,
    digest_policy: String,
}

impl SourceStage {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        require_non_empty("source.image", &self.image)?;
        require_eq("source.profile", &self.profile, "ubuntu-systemd")?;
        require_non_empty("source.platform", &self.platform)?;
        require_eq("source.digest_policy", &self.digest_policy, "pinned")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PackageStage {
    manager: String,
    update: bool,
    install: Vec<String>,
    clean: bool,
}

impl PackageStage {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        require_eq("package_stage.manager", &self.manager, "apt")?;
        if self.install.is_empty() {
            return Err("package_stage.install must not be empty".into());
        }
        for package in &self.install {
            require_token("package_stage.install", package)?;
        }
        let _ = self.update;
        let _ = self.clean;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PayloadSpec {
    label: String,
    source: PathBuf,
    guest_path: String,
    mode: String,
    links: Vec<String>,
}

impl PayloadSpec {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        require_token("immutable_payloads.label", &self.label)?;
        require_non_empty(
            "immutable_payloads.source",
            &self.source.display().to_string(),
        )?;
        require_abs_path("immutable_payloads.guest_path", &self.guest_path)?;
        if self.mode.is_empty() {
            return Err("immutable_payloads.mode must not be empty".into());
        }
        for link in &self.links {
            require_abs_path("immutable_payloads.links", link)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SshdPolicy {
    trusted_user_ca_keys: String,
    authorized_principals_file: String,
    force_command: Option<String>,
}

impl SshdPolicy {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        require_abs_path(
            "sshd_policy.trusted_user_ca_keys",
            &self.trusted_user_ca_keys,
        )?;
        require_abs_path(
            "sshd_policy.authorized_principals_file",
            &self.authorized_principals_file,
        )?;
        if let Some(force_command) = &self.force_command {
            require_non_empty("sshd_policy.force_command", force_command)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ServiceSpec {
    name: String,
    enable: String,
}

impl ServiceSpec {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        require_non_empty("services.name", &self.name)?;
        if !self.name.ends_with(".service") {
            return Err(format!("service name {:?} must end with .service", self.name).into());
        }
        require_non_empty("services.enable", &self.enable)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ImageTarget {
    Ch,
    Vz,
}

impl ImageTarget {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ch => "ch",
            Self::Vz => "vz",
        }
    }
}

impl std::fmt::Display for ImageTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for ImageTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ch" => Ok(Self::Ch),
            "vz" => Ok(Self::Vz),
            other => Err(format!("unknown target {other:?}; expected ch or vz")),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageBuildManifest {
    contract_version: String,
    config_path: PathBuf,
    target: ImageTarget,
    output_dir: PathBuf,
    source: ManifestSource,
    stages: Vec<StageRecord>,
    immutable_files: Vec<String>,
    seed_files: Vec<String>,
    validation: Vec<String>,
}

impl ImageBuildManifest {
    fn from_config(
        config_path: &Path,
        target: ImageTarget,
        out: &Path,
        config: &ImageBuildConfig,
    ) -> Self {
        Self {
            contract_version: config.version.clone(),
            config_path: config_path.to_path_buf(),
            target,
            output_dir: out.to_path_buf(),
            source: ManifestSource {
                image: config.source.image.clone(),
                profile: config.source.profile.clone(),
                platform: config.source.platform.clone(),
                digest_policy: config.source.digest_policy.clone(),
            },
            stages: vec![
                StageRecord::declared("source", "base image and platform digest policy"),
                StageRecord::declared("import", "OCI rootfs import from resolved source"),
                StageRecord::declared("classify", "rootfs compatibility classification"),
                StageRecord::declared("package", "explicit package-manager installation stage"),
                StageRecord::declared("immutable-layer", "VMM guest payload and service layer"),
                StageRecord::declared("policy", "sshd and image hardening policy"),
                StageRecord::declared("seed", "per-guest seed or overlay emission"),
                StageRecord::declared("backend-emitter", "CH/VZ backend artifact emission"),
                StageRecord::declared("validation", "post-boot conformance validation"),
            ],
            immutable_files: config.immutable_files.clone(),
            seed_files: config.seed_files.clone(),
            validation: config.validation.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestSource {
    image: String,
    profile: String,
    platform: String,
    digest_policy: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StageRecord {
    name: String,
    status: String,
    summary: String,
}

impl StageRecord {
    fn declared(name: &str, summary: &str) -> Self {
        Self {
            name: name.to_string(),
            status: "declared".to_string(),
            summary: summary.to_string(),
        }
    }
}

fn require_eq(field: &str, actual: &str, expected: &str) -> Result<(), Box<dyn Error>> {
    if actual != expected {
        return Err(format!("{field} must be {expected:?}, got {actual:?}").into());
    }
    Ok(())
}

fn require_non_empty(field: &str, value: &str) -> Result<(), Box<dyn Error>> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty").into());
    }
    Ok(())
}

fn require_abs_path(field: &str, value: &str) -> Result<(), Box<dyn Error>> {
    require_non_empty(field, value)?;
    if !value.starts_with('/') {
        return Err(format!("{field} must be an absolute guest path, got {value:?}").into());
    }
    Ok(())
}

fn require_token(field: &str, value: &str) -> Result<(), Box<dyn Error>> {
    require_non_empty(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(format!("{field} contains unsupported token {value:?}").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_build_command() {
        let command = parse_args(vec![
            "build".to_string(),
            "--config".to_string(),
            "motlie-image.yaml".to_string(),
            "--target".to_string(),
            "ch".to_string(),
            "--out".to_string(),
            "artifacts/ch".to_string(),
        ])
        .unwrap();

        assert_eq!(
            command,
            Command::Build {
                config: PathBuf::from("motlie-image.yaml"),
                target: ImageTarget::Ch,
                out: PathBuf::from("artifacts/ch")
            }
        );
    }

    #[test]
    fn parses_validate_command() {
        let command = parse_args(vec![
            "validate".to_string(),
            "--config".to_string(),
            "motlie-image.yaml".to_string(),
            "--artifact".to_string(),
            "artifacts/ch".to_string(),
        ])
        .unwrap();

        assert_eq!(
            command,
            Command::Validate {
                config: PathBuf::from("motlie-image.yaml"),
                artifact: PathBuf::from("artifacts/ch")
            }
        );
    }
}
