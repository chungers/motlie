use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use motlie_vmm::image::{
    RootfsCloudInitSeed, RootfsCompatibilityBackendEnv, RootfsMountSpec,
    RootfsSeedOverlayAssembler, RootfsSeedOverlayManifest, RootfsSeedOverlaySpec, RootfsUserSeed,
    MOTLIE_V15_CONTRACT_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_MANIFEST_NAME: &str = "mbuild-manifest.json";
const DEFAULT_SEED_MANIFEST_NAME: &str = "mbuild-seed-manifest.json";
const DEFAULT_VALIDATION_MANIFEST_NAME: &str = "mbuild-validation-manifest.json";
const ADAPTER_LOG_NAME: &str = "mbuild-adapter.log";
const VALIDATION_LOG_NAME: &str = "mbuild-validation.log";

fn main() {
    if let Err(error) = run() {
        eprintln!("mbuild: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Build {
            config,
            target,
            out,
            repo_root,
            plan_only,
            adapter_arg,
        } => build(BuildOptions {
            config_path: config,
            target,
            out,
            repo_root,
            plan_only,
            adapter_args: adapter_arg,
        }),
        Commands::Seed {
            config,
            target,
            guest,
            uid,
            gid,
            hostname,
            out,
            ssh_ca_pubkey,
        } => seed(SeedOptions {
            config_path: config,
            target,
            guest,
            uid,
            gid,
            hostname,
            out,
            ssh_ca_pubkey,
        }),
        Commands::Validate {
            config,
            artifact,
            require_executed,
            repo_root,
            scenario,
            harness_bin,
        } => validate(ValidationOptions {
            config_path: config,
            artifact,
            require_executed,
            repo_root,
            scenario,
            harness_bin,
        }),
    }
}

#[derive(Debug)]
struct BuildOptions {
    config_path: PathBuf,
    target: ImageTarget,
    out: PathBuf,
    repo_root: Option<PathBuf>,
    plan_only: bool,
    adapter_args: Vec<String>,
}

fn build(options: BuildOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    config.validate_for_target(options.target)?;
    fs::create_dir_all(&options.out)?;

    let repo_root = options.repo_root.clone().unwrap_or_else(default_repo_root);
    let adapter = if options.plan_only {
        None
    } else {
        Some(run_backend_adapter(&repo_root, &config, &options)?)
    };
    let artifacts = if options.plan_only {
        Vec::new()
    } else {
        collect_artifacts(&options.out, DEFAULT_MANIFEST_NAME)?
    };

    let manifest = ImageBuildManifest::from_config(
        &options.config_path,
        options.target,
        &options.out,
        &config,
        adapter,
        artifacts,
    );
    let manifest_path = options.out.join(DEFAULT_MANIFEST_NAME);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", manifest_path.display());
    Ok(())
}

#[derive(Debug)]
struct SeedOptions {
    config_path: PathBuf,
    target: ImageTarget,
    guest: String,
    uid: Option<u32>,
    gid: Option<u32>,
    hostname: Option<String>,
    out: PathBuf,
    ssh_ca_pubkey: Option<PathBuf>,
}

fn seed(options: SeedOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    config.validate_for_target(options.target)?;
    fs::create_dir_all(&options.out)?;

    let hostname = options
        .hostname
        .unwrap_or_else(|| format!("motlie-{}", options.guest));
    let mut spec = RootfsSeedOverlaySpec::new(
        RootfsCloudInitSeed::new(&options.guest, hostname),
        backend_env_for_target(options.target),
    );
    spec.mounts = vec![
        RootfsMountSpec::new(
            format!("{}-home", options.guest),
            format!("/home/{}", options.guest),
        ),
        RootfsMountSpec::new(
            format!("{}-workspace", options.guest),
            PathBuf::from("/workspace"),
        ),
        RootfsMountSpec::new(
            format!("{}-agent-state", options.guest),
            PathBuf::from("/agent-state"),
        ),
    ];
    let mut user = RootfsUserSeed::new(&options.guest, &options.guest);
    user.uid = options.uid;
    user.gid = options.gid;
    spec.users.push(user);
    if let Some(path) = &options.ssh_ca_pubkey {
        spec.ssh_user_ca_pubkey = Some(fs::read_to_string(path)?);
    }

    let seed_manifest = RootfsSeedOverlayAssembler::new().assemble(&options.out, &spec)?;
    let manifest = ImageSeedManifest::from_seed_overlay(
        &options.config_path,
        options.target,
        &options.out,
        &config,
        &seed_manifest,
    );
    let manifest_path = options.out.join(DEFAULT_SEED_MANIFEST_NAME);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", manifest_path.display());
    Ok(())
}

#[derive(Debug)]
struct ValidationOptions {
    config_path: PathBuf,
    artifact: PathBuf,
    require_executed: bool,
    repo_root: Option<PathBuf>,
    scenario: Option<PathBuf>,
    harness_bin: Option<PathBuf>,
}

fn validate(options: ValidationOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    config.validate()?;

    let manifest_path = options.artifact.join(DEFAULT_MANIFEST_NAME);
    let manifest: ImageBuildManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.contract_version != config.version {
        bail!(
            "manifest contract version {} does not match config version {}",
            manifest.contract_version,
            config.version
        );
    }
    if !config.emitters.contains(&manifest.target) {
        bail!(
            "manifest target {} is not declared by config emitters",
            manifest.target.as_str()
        );
    }
    if manifest.stages.is_empty() {
        bail!("manifest does not contain stage records");
    }
    if options.require_executed && manifest.adapter.is_none() {
        bail!("manifest was created in --plan-only mode; no backend adapter evidence");
    }
    if options.require_executed && manifest.artifacts.is_empty() {
        bail!("executed manifest does not contain artifact digests");
    }
    if let Some(scenario) = &options.scenario {
        let repo_root = options.repo_root.clone().unwrap_or_else(default_repo_root);
        let validation =
            run_harness_validation(&repo_root, &manifest, &options.artifact, scenario, &options)?;
        let ok = validation.exit_status == 0;
        let validation_path = options.artifact.join(DEFAULT_VALIDATION_MANIFEST_NAME);
        fs::write(&validation_path, serde_json::to_vec_pretty(&validation)?)?;
        if !ok {
            bail!(
                "harness validation failed with status {}; see {}",
                validation.exit_status,
                validation.log_path.display()
            );
        }
        println!("validated {}", validation_path.display());
        return Ok(());
    }
    println!("validated {}", manifest_path.display());
    Ok(())
}

fn load_config(path: &Path) -> Result<ImageBuildConfig> {
    let config: ImageBuildConfig = serde_yaml::from_slice(&fs::read(path)?)?;
    config.validate()?;
    Ok(config)
}

fn default_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("bins/mbuild must live two levels below the repository root")
        .to_path_buf()
}

fn run_backend_adapter(
    repo_root: &Path,
    config: &ImageBuildConfig,
    options: &BuildOptions,
) -> Result<AdapterRecord> {
    let script = repo_root.join("libs/vmm/examples/v1.5/build-image.sh");
    if !script.is_file() {
        bail!("v1.5 backend adapter not found: {}", script.display());
    }
    let log_path = options.out.join(ADAPTER_LOG_NAME);
    let log = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&log_path)?;
    let stderr = log.try_clone()?;
    let package_include = config.package_stage.install.join(",");
    let started_at_unix_seconds = unix_now();
    let mut command = Command::new("bash");
    command
        .arg(&script)
        .arg("--backend")
        .arg(options.target.as_str())
        .args(&options.adapter_args)
        .current_dir(repo_root)
        .env("MOTLIE_V15_ARTIFACTS_DIR", &options.out)
        .env("MOTLIE_V15_BUILD_CONFIG", &options.config_path)
        .env("MOTLIE_V15_PACKAGE_INCLUDE", &package_include)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));
    let status = command.status()?;
    let completed_at_unix_seconds = unix_now();
    let exit_status = status.code().unwrap_or(-1);
    if !status.success() {
        bail!(
            "backend adapter failed with status {exit_status}; see {}",
            log_path.display()
        );
    }

    let mut command_line = vec![
        "bash".to_string(),
        script.display().to_string(),
        "--backend".to_string(),
        options.target.as_str().to_string(),
    ];
    command_line.extend(options.adapter_args.clone());

    Ok(AdapterRecord {
        kind: "v1.5-shell-adapter".to_string(),
        command: command_line,
        log_path,
        exit_status,
        started_at_unix_seconds,
        completed_at_unix_seconds,
        package_include: config.package_stage.install.clone(),
    })
}

fn run_harness_validation(
    repo_root: &Path,
    manifest: &ImageBuildManifest,
    artifact: &Path,
    scenario: &Path,
    options: &ValidationOptions,
) -> Result<HarnessValidationRecord> {
    let log_path = artifact.join(VALIDATION_LOG_NAME);
    let log = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&log_path)?;
    let stderr = log.try_clone()?;
    let started_at_unix_seconds = unix_now();

    let mut command_line = Vec::new();
    let mut command = if let Some(harness_bin) = &options.harness_bin {
        command_line.push(harness_bin.display().to_string());
        Command::new(harness_bin)
    } else {
        command_line.extend([
            "cargo".to_string(),
            "run".to_string(),
            "-p".to_string(),
            "motlie-vmm".to_string(),
            "--example".to_string(),
            "harness_v1_5".to_string(),
            "--".to_string(),
        ]);
        let mut command = Command::new("cargo");
        command.args(["run", "-p", "motlie-vmm", "--example", "harness_v1_5", "--"]);
        command
    };

    let backend = manifest.target.as_str().to_string();
    command_line.extend([
        "--backend".to_string(),
        backend.clone(),
        "scenario".to_string(),
        scenario.display().to_string(),
    ]);
    command
        .args(["--backend", &backend, "scenario"])
        .arg(scenario)
        .current_dir(repo_root)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));

    match manifest.target {
        ImageTarget::Ch => {
            command.env("MOTLIE_V15_CH_BASE_ARTIFACTS_DIR", artifact.join("base"));
        }
        ImageTarget::Vz => {
            command.env("MOTLIE_VZ_ARTIFACTS_DIR", artifact);
            let default_vm_dir = artifact.join("motlie-v1-5-base-iter.vm");
            if default_vm_dir.join("disk.img").exists() && default_vm_dir.join("nvram.bin").exists()
            {
                command.env("MOTLIE_VZ_BASE_VM_DIR", default_vm_dir);
            }
        }
    }

    let status = command.status()?;
    let completed_at_unix_seconds = unix_now();
    Ok(HarnessValidationRecord {
        contract_version: manifest.contract_version.clone(),
        target: manifest.target,
        artifact_dir: artifact.to_path_buf(),
        scenario: scenario.to_path_buf(),
        command: command_line,
        log_path,
        exit_status: status.code().unwrap_or(-1),
        started_at_unix_seconds,
        completed_at_unix_seconds,
    })
}

fn collect_artifacts(root: &Path, skip_file_name: &str) -> Result<Vec<ManifestArtifact>> {
    let mut artifacts = Vec::new();
    if root.exists() {
        collect_artifacts_recursive(root, root, skip_file_name, &mut artifacts)?;
    }
    artifacts.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(artifacts)
}

fn collect_artifacts_recursive(
    base: &Path,
    dir: &Path,
    skip_file_name: &str,
    artifacts: &mut Vec<ManifestArtifact>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_artifacts_recursive(base, &path, skip_file_name, artifacts)?;
        } else if file_type.is_file() {
            if path.file_name().and_then(|name| name.to_str()) == Some(skip_file_name) {
                continue;
            }
            let metadata = entry.metadata()?;
            let relative = path.strip_prefix(base).unwrap_or(&path).to_path_buf();
            artifacts.push(ManifestArtifact {
                label: artifact_label(&relative),
                path: relative,
                size_bytes: metadata.len(),
                sha256: sha256_file(&path)?,
            });
        }
    }
    Ok(())
}

fn artifact_label(path: &Path) -> String {
    path.to_string_lossy().replace('/', ":")
}

fn sha256_file(path: &Path) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn backend_env_for_target(target: ImageTarget) -> RootfsCompatibilityBackendEnv {
    match target {
        ImageTarget::Ch => RootfsCompatibilityBackendEnv::for_backend("ch", "ch-vhost-user"),
        ImageTarget::Vz => RootfsCompatibilityBackendEnv::for_backend("vz", "vz-userspace"),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Parser)]
#[command(
    name = "mbuild",
    about = "Motlie v1.5 image builder",
    version,
    after_help = "mbuild consumes the checked-in v1.5 image config. The build command can execute the current CH/VZ backend adapters or run in --plan-only mode; seed regenerates per-guest seed artifacts without rebuilding the immutable image."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
enum Commands {
    /// Build or plan backend artifacts from the image config.
    Build {
        /// Dockerfile-like v1.5 image build config.
        #[arg(long)]
        config: PathBuf,
        /// Backend artifact target to declare.
        #[arg(long, value_enum)]
        target: ImageTarget,
        /// Output directory for mbuild-manifest.json.
        #[arg(long)]
        out: PathBuf,
        /// Repository root that contains libs/vmm/examples/v1.5.
        #[arg(long)]
        repo_root: Option<PathBuf>,
        /// Validate and manifest the staged build without running backend scripts.
        #[arg(long)]
        plan_only: bool,
        /// Extra argument forwarded to the backend adapter script.
        #[arg(long = "adapter-arg")]
        adapter_arg: Vec<String>,
    },
    /// Emit per-guest seed artifacts without rebuilding the immutable image.
    Seed {
        /// Dockerfile-like v1.5 image build config.
        #[arg(long)]
        config: PathBuf,
        /// Backend seed target.
        #[arg(long, value_enum)]
        target: ImageTarget,
        /// Guest user/identity name.
        #[arg(long)]
        guest: String,
        /// Guest UID to encode into seed ownership.
        #[arg(long)]
        uid: Option<u32>,
        /// Guest GID to encode into seed ownership.
        #[arg(long)]
        gid: Option<u32>,
        /// Guest hostname. Defaults to motlie-<guest>.
        #[arg(long)]
        hostname: Option<String>,
        /// Output directory for seed files and mbuild-seed-manifest.json.
        #[arg(long)]
        out: PathBuf,
        /// SSH user CA public key to include in the seed overlay.
        #[arg(long)]
        ssh_ca_pubkey: Option<PathBuf>,
    },
    /// Validate an emitted mbuild manifest against the image config.
    Validate {
        /// Dockerfile-like v1.5 image build config.
        #[arg(long)]
        config: PathBuf,
        /// Artifact directory containing mbuild-manifest.json.
        #[arg(long)]
        artifact: PathBuf,
        /// Require evidence that backend adapters ran and produced artifacts.
        #[arg(long)]
        require_executed: bool,
        /// Repository root used when delegating live validation to harness_v1_5.
        #[arg(long)]
        repo_root: Option<PathBuf>,
        /// Optional v1.5 harness scenario to run after manifest validation.
        #[arg(long)]
        scenario: Option<PathBuf>,
        /// Optional prebuilt harness_v1_5 binary. Defaults to cargo run.
        #[arg(long)]
        harness_bin: Option<PathBuf>,
    },
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
    fn validate(&self) -> Result<()> {
        require_eq("version", &self.version, MOTLIE_V15_CONTRACT_VERSION)?;
        self.source.validate()?;
        self.package_stage.validate()?;
        if self.immutable_payloads.is_empty() {
            bail!("immutable_payloads must not be empty");
        }
        for payload in &self.immutable_payloads {
            payload.validate()?;
        }
        self.sshd_policy.validate()?;
        if self.services.is_empty() {
            bail!("services must not be empty");
        }
        for service in &self.services {
            service.validate()?;
        }
        if self.immutable_files.is_empty() {
            bail!("immutable_files must not be empty");
        }
        if self.seed_files.is_empty() {
            bail!("seed_files must not be empty");
        }
        if self.emitters.is_empty() {
            bail!("emitters must not be empty");
        }
        if self.validation.is_empty() {
            bail!("validation must not be empty");
        }
        Ok(())
    }

    fn validate_for_target(&self, target: ImageTarget) -> Result<()> {
        self.validate()?;
        if !self.emitters.contains(&target) {
            bail!("target {} is not declared in emitters", target.as_str());
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
    fn validate(&self) -> Result<()> {
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
    fn validate(&self) -> Result<()> {
        require_eq("package_stage.manager", &self.manager, "apt")?;
        if self.install.is_empty() {
            bail!("package_stage.install must not be empty");
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
    fn validate(&self) -> Result<()> {
        require_token("immutable_payloads.label", &self.label)?;
        require_non_empty(
            "immutable_payloads.source",
            &self.source.display().to_string(),
        )?;
        require_abs_path("immutable_payloads.guest_path", &self.guest_path)?;
        if self.mode.is_empty() {
            bail!("immutable_payloads.mode must not be empty");
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
    fn validate(&self) -> Result<()> {
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
    fn validate(&self) -> Result<()> {
        require_non_empty("services.name", &self.name)?;
        if !self.name.ends_with(".service") {
            bail!("service name {:?} must end with .service", self.name);
        }
        require_non_empty("services.enable", &self.enable)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ValueEnum)]
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

#[derive(Debug, Serialize, Deserialize)]
struct ImageBuildManifest {
    contract_version: String,
    config_path: PathBuf,
    target: ImageTarget,
    output_dir: PathBuf,
    source: ManifestSource,
    package_stage: ManifestPackageStage,
    stages: Vec<StageRecord>,
    immutable_files: Vec<String>,
    seed_files: Vec<String>,
    validation: Vec<String>,
    adapter: Option<AdapterRecord>,
    artifacts: Vec<ManifestArtifact>,
    pending_runtime_requirements: Vec<String>,
}

impl ImageBuildManifest {
    fn from_config(
        config_path: &Path,
        target: ImageTarget,
        out: &Path,
        config: &ImageBuildConfig,
        adapter: Option<AdapterRecord>,
        artifacts: Vec<ManifestArtifact>,
    ) -> Self {
        let stages = if let Some(adapter) = &adapter {
            executed_stages(adapter)
        } else {
            declared_stages()
        };
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
            package_stage: ManifestPackageStage {
                manager: config.package_stage.manager.clone(),
                update: config.package_stage.update,
                install: config.package_stage.install.clone(),
                clean: config.package_stage.clean,
            },
            stages,
            immutable_files: config.immutable_files.clone(),
            seed_files: config.seed_files.clone(),
            validation: config.validation.clone(),
            adapter,
            artifacts,
            pending_runtime_requirements: vec![
                "live v1.5 harness validation must be run after boot".to_string(),
                "per-guest seed artifacts are generated with `mbuild seed`".to_string(),
            ],
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
struct ManifestPackageStage {
    manager: String,
    update: bool,
    install: Vec<String>,
    clean: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdapterRecord {
    kind: String,
    command: Vec<String>,
    log_path: PathBuf,
    exit_status: i32,
    started_at_unix_seconds: u64,
    completed_at_unix_seconds: u64,
    package_include: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestArtifact {
    label: String,
    path: PathBuf,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct HarnessValidationRecord {
    contract_version: String,
    target: ImageTarget,
    artifact_dir: PathBuf,
    scenario: PathBuf,
    command: Vec<String>,
    log_path: PathBuf,
    exit_status: i32,
    started_at_unix_seconds: u64,
    completed_at_unix_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct StageRecord {
    name: String,
    status: String,
    summary: String,
    evidence: Vec<String>,
}

impl StageRecord {
    fn declared(name: &str, summary: &str) -> Self {
        Self {
            name: name.to_string(),
            status: "declared".to_string(),
            summary: summary.to_string(),
            evidence: Vec::new(),
        }
    }

    fn delegated(name: &str, summary: &str, adapter: &AdapterRecord) -> Self {
        Self {
            name: name.to_string(),
            status: "delegated".to_string(),
            summary: summary.to_string(),
            evidence: vec![format!("adapter_log={}", adapter.log_path.display())],
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageSeedManifest {
    contract_version: String,
    config_path: PathBuf,
    target: ImageTarget,
    output_dir: PathBuf,
    seed_files: Vec<String>,
    rootfs_seed_overlay: RootfsSeedOverlayManifest,
    artifacts: Vec<ManifestArtifact>,
}

impl ImageSeedManifest {
    fn from_seed_overlay(
        config_path: &Path,
        target: ImageTarget,
        out: &Path,
        config: &ImageBuildConfig,
        rootfs_seed_overlay: &RootfsSeedOverlayManifest,
    ) -> Self {
        Self {
            contract_version: config.version.clone(),
            config_path: config_path.to_path_buf(),
            target,
            output_dir: out.to_path_buf(),
            seed_files: config.seed_files.clone(),
            rootfs_seed_overlay: rootfs_seed_overlay.clone(),
            artifacts: collect_artifacts(out, DEFAULT_SEED_MANIFEST_NAME).unwrap_or_default(),
        }
    }
}

fn declared_stages() -> Vec<StageRecord> {
    vec![
        StageRecord::declared("source", "base image and platform digest policy"),
        StageRecord::declared("import", "OCI rootfs import from resolved source"),
        StageRecord::declared("classify", "rootfs compatibility classification"),
        StageRecord::declared("package", "explicit package-manager installation stage"),
        StageRecord::declared("immutable-layer", "VMM guest payload and service layer"),
        StageRecord::declared("policy", "sshd and image hardening policy"),
        StageRecord::declared("seed", "per-guest seed or overlay emission"),
        StageRecord::declared("backend-emitter", "CH/VZ backend artifact emission"),
        StageRecord::declared("validation", "post-boot conformance validation"),
    ]
}

fn executed_stages(adapter: &AdapterRecord) -> Vec<StageRecord> {
    vec![
        StageRecord::delegated(
            "source",
            "source contract consumed by backend adapter",
            adapter,
        ),
        StageRecord::delegated(
            "import",
            "backend adapter materialized rootfs input",
            adapter,
        ),
        StageRecord::delegated(
            "classify",
            "backend adapter checked guest contract inputs",
            adapter,
        ),
        StageRecord::delegated(
            "package",
            "package-manager stage executed by backend adapter",
            adapter,
        ),
        StageRecord::delegated(
            "immutable-layer",
            "VMM guest payload and services installed by backend adapter",
            adapter,
        ),
        StageRecord::delegated(
            "policy",
            "sshd, sudo, service, and image policy applied by backend adapter",
            adapter,
        ),
        StageRecord::declared(
            "seed",
            "per-guest seed is regenerated separately with `mbuild seed`",
        ),
        StageRecord::delegated(
            "backend-emitter",
            "backend boot artifacts emitted by adapter",
            adapter,
        ),
        StageRecord::declared(
            "validation",
            "post-boot harness validation is required after artifact build",
        ),
    ]
}

fn require_eq(field: &str, actual: &str, expected: &str) -> Result<()> {
    if actual != expected {
        bail!("{field} must be {expected:?}, got {actual:?}");
    }
    Ok(())
}

fn require_non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(())
}

fn require_abs_path(field: &str, value: &str) -> Result<()> {
    require_non_empty(field, value)?;
    if !value.starts_with('/') {
        bail!("{field} must be an absolute guest path, got {value:?}");
    }
    Ok(())
}

fn require_token(field: &str, value: &str) -> Result<()> {
    require_non_empty(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        bail!("{field} contains unsupported token {value:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_build_command() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "build",
            "--config",
            "motlie-image.yaml",
            "--target",
            "ch",
            "--out",
            "artifacts/ch",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Build {
                config: PathBuf::from("motlie-image.yaml"),
                target: ImageTarget::Ch,
                out: PathBuf::from("artifacts/ch"),
                repo_root: None,
                plan_only: false,
                adapter_arg: Vec::new()
            }
        );
    }

    #[test]
    fn parses_validate_command() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "validate",
            "--config",
            "motlie-image.yaml",
            "--artifact",
            "artifacts/ch",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Validate {
                config: PathBuf::from("motlie-image.yaml"),
                artifact: PathBuf::from("artifacts/ch"),
                require_executed: false,
                repo_root: None,
                scenario: None,
                harness_bin: None
            }
        );
    }

    #[test]
    fn parses_seed_command() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "seed",
            "--config",
            "motlie-image.yaml",
            "--target",
            "vz",
            "--guest",
            "alice",
            "--uid",
            "2001",
            "--gid",
            "2001",
            "--out",
            "artifacts/seed/alice",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Seed {
                config: PathBuf::from("motlie-image.yaml"),
                target: ImageTarget::Vz,
                guest: "alice".to_string(),
                uid: Some(2001),
                gid: Some(2001),
                hostname: None,
                out: PathBuf::from("artifacts/seed/alice"),
                ssh_ca_pubkey: None
            }
        );
    }

    #[test]
    fn help_mentions_current_scope() {
        let error = Cli::try_parse_from(["mbuild", "--help"]).unwrap_err();

        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
        assert!(error.to_string().contains("mbuild consumes"));
    }
}
