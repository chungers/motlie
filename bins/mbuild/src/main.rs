use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process;
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use motlie_vmm::image::{
    v1_5, ExternalOciSource, GuestImageProfile, OciContentCache, OciDigest, OciImageReference,
    OciPlatform, OciRegistryClient, OciRootfsImporter, RootfsCloudInitSeed,
    RootfsCompatibilityAssembler, RootfsCompatibilityBackendEnv, RootfsCompatibilityLayerSpec,
    RootfsMountSpec, RootfsPayloadFile, RootfsPendingRequirementPolicy, RootfsProfileSpec,
    RootfsSeedOverlayAssembler, RootfsSeedOverlayManifest, RootfsSeedOverlaySpec, RootfsUserSeed,
    UBUNTU_SYSTEMD_PROFILE,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{error, info, instrument};
use tracing_subscriber::EnvFilter;

const DEFAULT_MANIFEST_NAME: &str = "mbuild-manifest.json";
const DEFAULT_SEED_MANIFEST_NAME: &str = "mbuild-seed-manifest.json";
const DEFAULT_VALIDATION_MANIFEST_NAME: &str = "mbuild-validation-manifest.json";
const ADAPTER_LOG_NAME: &str = "mbuild-adapter.log";
const CH_EMITTER_LOG_NAME: &str = "mbuild-ch-emitter.log";
const CH_PACKAGE_STAGE_LOG_NAME: &str = "mbuild-package-stage.log";
const CH_PACKAGE_STAGE_SCRIPT_NAME: &str = "mbuild-apt-stage.sh";
const CH_BUILD_RESULT_NAME: &str = "ch-build-result.json";
const CH_GUEST_CONTRACT_NAME: &str = "guest-contract.json";
const CH_ROOTFS_NAME: &str = "rootfs.squashfs";
const VALIDATION_LOG_NAME: &str = "mbuild-validation.log";
const OCI_CACHE_DIR_ENV: &str = "MOTLIE_MBUILD_OCI_CACHE_DIR";
const CH_KERNEL_RELEASE_DEFAULT: &str = "ch-release-v6.16.9-20251112";

fn main() {
    init_tracing();
    if let Err(error) = run() {
        error!(error = %error, "mbuild failed");
        std::process::exit(1);
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
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
            rootfs_tarball,
        } => build(BuildOptions {
            config_path: config,
            target,
            out,
            repo_root,
            plan_only,
            adapter_args: adapter_arg,
            rootfs_tarball,
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
    target: BackendTargetId,
    out: PathBuf,
    repo_root: Option<PathBuf>,
    plan_only: bool,
    adapter_args: Vec<String>,
    rootfs_tarball: Option<PathBuf>,
}

#[instrument(skip(options), fields(target = %options.target, config = %options.config_path.display()))]
fn build(options: BuildOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    let emitter = config.validate_for_target(&options.target)?;
    let rootfs_tarball = if let Some(path) = &options.rootfs_tarball {
        if emitter.adapter.env.rootfs_tarball.is_none() {
            bail!(
                "target {} does not declare an adapter rootfs_tarball env",
                options.target
            );
        }
        Some(rootfs_tarball_record(path)?)
    } else {
        None
    };
    fs::create_dir_all(&options.out)?;

    let adapter = if options.plan_only {
        None
    } else {
        let repo_root = resolve_repo_root(options.repo_root.as_deref())?;
        Some(run_build_execution(
            &repo_root,
            &config,
            emitter,
            &options,
            rootfs_tarball.as_ref(),
        )?)
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
    info!(path = %manifest_path.display(), "wrote mbuild manifest");
    Ok(())
}

#[derive(Debug)]
struct SeedOptions {
    config_path: PathBuf,
    target: BackendTargetId,
    guest: String,
    uid: Option<u32>,
    gid: Option<u32>,
    hostname: Option<String>,
    out: PathBuf,
    ssh_ca_pubkey: Option<PathBuf>,
}

#[instrument(skip(options), fields(target = %options.target, guest = %options.guest, config = %options.config_path.display()))]
fn seed(options: SeedOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    let emitter = config.validate_for_target(&options.target)?;
    fs::create_dir_all(&options.out)?;

    let hostname = options
        .hostname
        .unwrap_or_else(|| format!("motlie-{}", options.guest));
    let seed_stage = config.seed.render(&options.guest)?;
    let mut spec = RootfsSeedOverlaySpec::new(
        RootfsCloudInitSeed::new(&options.guest, hostname),
        emitter.backend_env(),
    );
    spec.mounts = seed_stage.mounts;
    let mut user = RootfsUserSeed::new(&options.guest, seed_stage.ssh_principal);
    user.home = seed_stage.user_home;
    user.uid = options.uid;
    user.gid = options.gid;
    spec.users.push(user);
    if let Some(path) = &options.ssh_ca_pubkey {
        spec.ssh_user_ca_pubkey = Some(fs::read_to_string(path)?);
    }

    let seed_manifest = RootfsSeedOverlayAssembler::new().assemble(&options.out, &spec)?;
    let manifest = ImageSeedManifest::from_seed_overlay(
        &options.config_path,
        options.target.clone(),
        &options.out,
        &config,
        &seed_manifest,
    )?;
    let manifest_path = options.out.join(DEFAULT_SEED_MANIFEST_NAME);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    info!(path = %manifest_path.display(), "wrote mbuild seed manifest");
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

#[instrument(skip(options), fields(config = %options.config_path.display(), artifact = %options.artifact.display()))]
fn validate(options: ValidationOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    config.validate()?;

    let manifest_path = options.artifact.join(DEFAULT_MANIFEST_NAME);
    let manifest: ImageBuildManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let emitter = manifest.validate_against_config(&config)?;
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
        let repo_root = resolve_repo_root(options.repo_root.as_deref())?;
        let validation = run_harness_validation(
            &repo_root,
            &manifest,
            emitter,
            &options.artifact,
            scenario,
            &options,
        )?;
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
        info!(path = %validation_path.display(), "validated mbuild harness scenario");
        return Ok(());
    }
    info!(path = %manifest_path.display(), "validated mbuild manifest");
    Ok(())
}

#[instrument(fields(path = %path.display()))]
fn load_config(path: &Path) -> Result<ImageBuildConfig> {
    let config: ImageBuildConfig = serde_yaml::from_slice(&fs::read(path)?)?;
    config.validate()?;
    Ok(config)
}

fn resolve_repo_root(provided: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = provided {
        return validate_repo_root(path);
    }
    if let Some(path) = env::var_os("MOTLIE_REPO_ROOT") {
        return validate_repo_root(Path::new(&path));
    }

    let output = Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            let cargo_toml = String::from_utf8(output.stdout)
                .context("cargo locate-project returned non-UTF-8 output")?;
            let cargo_toml = PathBuf::from(cargo_toml.trim());
            if let Some(repo_root) = cargo_toml.parent() {
                return validate_repo_root(repo_root);
            }
        }
    }

    bail!("repo root is required; pass --repo-root or set MOTLIE_REPO_ROOT")
}

fn validate_repo_root(path: &Path) -> Result<PathBuf> {
    if !path.is_dir() {
        bail!("repo root is not a directory: {}", path.display());
    }
    if !path.join("Cargo.toml").is_file() {
        bail!("repo root does not contain Cargo.toml: {}", path.display());
    }
    Ok(path.to_path_buf())
}

fn run_build_execution(
    repo_root: &Path,
    config: &ImageBuildConfig,
    emitter: &BackendEmitterSpec,
    options: &BuildOptions,
    rootfs_tarball: Option<&RootfsTarballRecord>,
) -> Result<AdapterRecord> {
    if config.source.kind == SourceKind::ExternalOci && options.target.as_str() == "ch" {
        return run_ch_external_oci_build(repo_root, config, emitter, options);
    }
    run_backend_adapter(repo_root, config, emitter, options, rootfs_tarball)
}

#[instrument(
    skip(repo_root, config, _emitter, options),
    fields(target = %options.target, out = %options.out.display())
)]
fn run_ch_external_oci_build(
    repo_root: &Path,
    config: &ImageBuildConfig,
    _emitter: &BackendEmitterSpec,
    options: &BuildOptions,
) -> Result<AdapterRecord> {
    let started_at_unix_seconds = unix_now()?;
    let log_path = options.out.join(CH_EMITTER_LOG_NAME);
    fs::write(&log_path, b"")?;
    append_log(&log_path, "=== mbuild CH external-OCI builder ===\n")?;

    let platform = parse_source_platform(&config.source.platform)?;
    let source = resolve_external_source(config, platform)?;
    append_log(
        &log_path,
        &format!(
            "source={} platform={} index={} manifest={}\n",
            source.image_ref,
            source.platform,
            source.image_index_digest,
            source.platform_manifest_digest
        ),
    )?;

    let host = ChHostTarget::detect()?;
    if platform != host.platform {
        bail!(
            "CH external-OCI build currently requires host-native platform {}; config resolved {}",
            host.platform,
            platform
        );
    }

    let work_root = create_work_root("mbuild-ch-rootfs")?;
    let rootfs_dir = work_root.join("rootfs");
    let cache_dir = oci_cache_dir(repo_root)?;
    let base_dir = options.out.join("base");
    fs::create_dir_all(&base_dir)?;

    let imported = import_external_oci_rootfs(&source, &cache_dir, &rootfs_dir)?;
    fs::write(
        options.out.join("mbuild-oci-import.json"),
        serde_json::to_vec_pretty(&imported)?,
    )?;

    let guest_binaries = build_guest_binaries(repo_root, &host, &log_path)?;
    run_apt_package_stage(&rootfs_dir, config, &options.out)?;

    let assembly_manifest =
        assemble_immutable_rootfs(&rootfs_dir, config, &source, &guest_binaries)?;
    fs::write(
        options.out.join("mbuild-rootfs-assembly.json"),
        serde_json::to_vec_pretty(&assembly_manifest)?,
    )?;

    install_ch_boot_adaptations(repo_root, &rootfs_dir, &source, &host, &guest_binaries.vfs)?;
    write_ch_guest_contract(
        &rootfs_dir.join("opt/motlie/v1.5/guest/guest-contract.json"),
        &source,
        &host,
        Path::new(host.kernel_image),
        &guest_binaries.vfs,
    )?;
    let rootfs_path = base_dir.join(CH_ROOTFS_NAME);
    emit_squashfs(
        &rootfs_dir,
        &rootfs_path,
        &options.out.join(CH_EMITTER_LOG_NAME),
    )?;
    let kernel_path = emit_ch_kernel(&base_dir, &host, &log_path)?;
    let contract_path = base_dir.join(CH_GUEST_CONTRACT_NAME);
    write_ch_guest_contract(
        &contract_path,
        &source,
        &host,
        &kernel_path,
        &guest_binaries.vfs,
    )?;
    write_ch_build_result(
        &options.out,
        &contract_path,
        &kernel_path,
        &rootfs_path,
        &guest_binaries.vfs,
    )?;

    let completed_at_unix_seconds = unix_now()?;
    append_log(&log_path, "=== CH external-OCI build complete ===\n")?;
    let _ = fs::remove_dir_all(&work_root);

    Ok(AdapterRecord {
        kind: "external-oci-ch-emitter".to_string(),
        command: vec![
            "mbuild".to_string(),
            "build".to_string(),
            "--target".to_string(),
            options.target.to_string(),
        ],
        log_path,
        exit_status: 0,
        started_at_unix_seconds,
        completed_at_unix_seconds,
        package_include: config.package_stage.install.clone(),
        materialized_source: None,
        external_oci_source: Some(source),
        rootfs_tarball: None,
    })
}

#[instrument(
    skip(repo_root, config, emitter, options),
    fields(target = %options.target, out = %options.out.display())
)]
fn run_backend_adapter(
    repo_root: &Path,
    config: &ImageBuildConfig,
    emitter: &BackendEmitterSpec,
    options: &BuildOptions,
    rootfs_tarball: Option<&RootfsTarballRecord>,
) -> Result<AdapterRecord> {
    let log_path = options.out.join(ADAPTER_LOG_NAME);
    let log = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&log_path)?;
    let stderr = log.try_clone()?;
    let package_include = config.package_stage.install.join(",");
    let started_at_unix_seconds = unix_now()?;
    let mut command = Command::new(&emitter.adapter.program);
    command
        .args(&emitter.adapter.args)
        .args(&options.adapter_args)
        .current_dir(repo_root)
        .env(&emitter.adapter.env.artifact_dir, &options.out)
        .env(&emitter.adapter.env.build_config, &options.config_path)
        .env(
            &emitter.adapter.env.package_manager,
            config.package_stage.manager.as_str(),
        )
        .env(
            &emitter.adapter.env.package_update,
            bool_env(config.package_stage.update),
        )
        .env(&emitter.adapter.env.package_include, &package_include)
        .env(
            &emitter.adapter.env.package_clean,
            bool_env(config.package_stage.clean),
        )
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));
    if let (Some(rootfs_tarball), Some(env_name)) =
        (rootfs_tarball, emitter.adapter.env.rootfs_tarball.as_ref())
    {
        command.env(env_name, &rootfs_tarball.canonical_path);
    }
    let mut command_line = vec![emitter.adapter.program.clone()];
    command_line.extend(emitter.adapter.args.clone());
    command_line.extend(options.adapter_args.clone());
    info!(
        target = %options.target,
        log = %log_path.display(),
        command = ?command_line,
        "running backend adapter"
    );
    let status = command.status()?;
    let completed_at_unix_seconds = unix_now()?;
    let exit_status = status.code().unwrap_or(-1);
    if !status.success() {
        bail!(
            "backend adapter failed with status {exit_status}; see {}",
            log_path.display()
        );
    }

    Ok(AdapterRecord {
        kind: "v1.5-shell-adapter".to_string(),
        command: command_line,
        log_path,
        exit_status,
        started_at_unix_seconds,
        completed_at_unix_seconds,
        package_include: config.package_stage.install.clone(),
        materialized_source: emitter.materialized_source.clone(),
        external_oci_source: None,
        rootfs_tarball: rootfs_tarball.cloned(),
    })
}

#[derive(Debug, Clone)]
struct ChHostTarget {
    platform: OciPlatform,
    rust_target: &'static str,
    deb_arch: &'static str,
    kernel_image: &'static str,
    kernel_asset: &'static str,
}

impl ChHostTarget {
    fn detect() -> Result<Self> {
        match env::consts::ARCH {
            "x86_64" => Ok(Self {
                platform: OciPlatform::linux_amd64(),
                rust_target: "x86_64-unknown-linux-gnu",
                deb_arch: "amd64",
                kernel_image: "vmlinux.bin",
                kernel_asset: "vmlinux",
            }),
            "aarch64" => Ok(Self {
                platform: OciPlatform::linux_arm64(),
                rust_target: "aarch64-unknown-linux-gnu",
                deb_arch: "arm64",
                kernel_image: "Image",
                kernel_asset: "Image-arm64",
            }),
            other => bail!("unsupported CH host architecture for v1.5 build: {other}"),
        }
    }
}

fn parse_source_platform(value: &str) -> Result<OciPlatform> {
    match value {
        "linux/amd64" | "amd64" => Ok(OciPlatform::linux_amd64()),
        "linux/arm64" | "arm64" => Ok(OciPlatform::linux_arm64()),
        "host-native" | "host-native-linux" => Ok(ChHostTarget::detect()?.platform),
        other => bail!(
            "unsupported source.platform {other:?}; expected linux/amd64, linux/arm64, or host-native-linux"
        ),
    }
}

fn resolve_external_source(
    config: &ImageBuildConfig,
    platform: OciPlatform,
) -> Result<ExternalOciSource> {
    if config.source.profile.as_str() != UBUNTU_SYSTEMD_PROFILE {
        bail!(
            "external-oci source.profile {:?} is not implemented; expected {}",
            config.source.profile.as_str(),
            UBUNTU_SYSTEMD_PROFILE
        );
    }
    let image_ref = OciImageReference::from_str(&config.source.image)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("create OCI registry runtime")?;
    let resolved = runtime.block_on(async {
        OciRegistryClient::new()
            .resolve_manifest(&image_ref, platform)
            .await
    })?;

    if let Some(expected) = &config.source.image_index_digest {
        if expected != &resolved.image_index_digest {
            bail!(
                "resolved source.image_index_digest {} does not match pinned {}",
                resolved.image_index_digest,
                expected
            );
        }
    }
    if let Some(expected) = &config.source.platform_manifest_digest {
        if expected != &resolved.platform_manifest_digest {
            bail!(
                "resolved source.platform_manifest_digest {} does not match pinned {}",
                resolved.platform_manifest_digest,
                expected
            );
        }
    }

    Ok(resolved.into_external_source())
}

fn import_external_oci_rootfs(
    source: &ExternalOciSource,
    cache_dir: &Path,
    rootfs_dir: &Path,
) -> Result<motlie_vmm::image::ImportedOciRootfs> {
    let image_ref = OciImageReference::from_str(&source.image_ref)?;
    let resolved = motlie_vmm::image::ResolvedOciManifest {
        image_ref,
        image_index_digest: source.image_index_digest.clone(),
        platform: source.platform,
        platform_manifest_digest: source.platform_manifest_digest.clone(),
    };
    let cache = OciContentCache::new(cache_dir);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("create OCI fetch runtime")?;
    let cached = runtime.block_on(async {
        OciRegistryClient::new()
            .fetch_resolved_platform_to_cache(&resolved, &cache)
            .await
    })?;
    Ok(OciRootfsImporter::new().import_layers(&cached.layers, rootfs_dir)?)
}

fn oci_cache_dir(repo_root: &Path) -> Result<PathBuf> {
    if let Some(path) = env::var_os(OCI_CACHE_DIR_ENV) {
        return Ok(PathBuf::from(path));
    }
    Ok(repo_root.join("target/mbuild-cache/oci"))
}

fn create_work_root(prefix: &str) -> Result<PathBuf> {
    let path = env::temp_dir().join(format!("{prefix}-{}-{}", process::id(), unix_now()?));
    fs::create_dir_all(&path)
        .with_context(|| format!("create work directory {}", path.display()))?;
    Ok(path)
}

fn subordinate_id_range(path: &str) -> Result<(u32, u32)> {
    let user =
        env::var("USER").context("USER is required for rootless package-stage subid lookup")?;
    let contents = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    for line in contents.lines() {
        let mut fields = line.split(':');
        let Some(name) = fields.next() else {
            continue;
        };
        if name != user {
            continue;
        }
        let start = fields
            .next()
            .context("subid entry missing start")?
            .parse::<u32>()
            .with_context(|| format!("parse {path} start for {user}"))?;
        let count = fields
            .next()
            .context("subid entry missing count")?
            .parse::<u32>()
            .with_context(|| format!("parse {path} count for {user}"))?;
        if count < 1024 {
            bail!("{path} entry for {user} is too small for package-stage uid/gid mapping");
        }
        return Ok((start, count));
    }
    bail!("{path} has no entry for {user}; rootless package stage requires subordinate ids")
}

#[derive(Debug, Clone)]
struct GuestBinaries {
    vfs: PathBuf,
    ssh_bridge: PathBuf,
}

fn build_guest_binaries(
    repo_root: &Path,
    host: &ChHostTarget,
    log_path: &Path,
) -> Result<GuestBinaries> {
    let target_dir = repo_root
        .join("target")
        .join(host.rust_target)
        .join("release");
    let vfs = target_dir.join("motlie-vfs-guest-v1_5");
    let ssh_bridge = target_dir.join("motlie-vsock-ssh-bridge-v1_5");
    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--manifest-path")
        .arg(repo_root.join("libs/vmm/Cargo.toml"))
        .arg("--release")
        .arg("--target")
        .arg(host.rust_target)
        .arg("--no-default-features")
        .arg("--features")
        .arg("guest-vfs")
        .arg("--bin")
        .arg("motlie-vfs-guest-v1_5")
        .arg("--bin")
        .arg("motlie-vsock-ssh-bridge-v1_5")
        .current_dir(repo_root);
    run_logged_command(&mut command, log_path, "build v1.5 guest binaries")?;
    if !vfs.is_file() {
        bail!("guest binary was not produced at {}", vfs.display());
    }
    if !ssh_bridge.is_file() {
        bail!(
            "SSH bridge binary was not produced at {}",
            ssh_bridge.display()
        );
    }
    let marker = Command::new(&vfs)
        .arg("--contract")
        .output()
        .with_context(|| format!("run guest binary contract check {}", vfs.display()))?;
    let marker_stdout =
        String::from_utf8(marker.stdout).context("guest binary contract marker is not UTF-8")?;
    if marker_stdout.trim() != v1_5::MOTLIE_V15_GUEST_MOUNTER_MARKER {
        bail!(
            "guest binary contract marker mismatch: expected {}, got {:?}",
            v1_5::MOTLIE_V15_GUEST_MOUNTER_MARKER,
            marker_stdout.trim()
        );
    }
    Ok(GuestBinaries { vfs, ssh_bridge })
}

fn run_apt_package_stage(rootfs_dir: &Path, config: &ImageBuildConfig, out: &Path) -> Result<()> {
    if config.package_stage.manager.as_str() != "apt" {
        bail!(
            "CH external-OCI package stage only implements apt, got {}",
            config.package_stage.manager.as_str()
        );
    }
    let script_path = out.join(CH_PACKAGE_STAGE_SCRIPT_NAME);
    let log_path = out.join(CH_PACKAGE_STAGE_LOG_NAME);
    fs::write(&script_path, apt_stage_script())?;
    set_mode(&script_path, 0o755)?;
    let (subuid_start, subuid_count) = subordinate_id_range("/etc/subuid")?;
    let (subgid_start, subgid_count) = subordinate_id_range("/etc/subgid")?;
    let uid_count = subuid_count.min(65_535);
    let gid_count = subgid_count.min(65_535);

    let mut command = Command::new("unshare");
    command
        .arg("--user")
        .arg("--map-root-user")
        .arg(format!("--map-users=1:{subuid_start}:{uid_count}"))
        .arg(format!("--map-groups=1:{subgid_start}:{gid_count}"))
        .arg("--setgroups")
        .arg("allow")
        .arg("--mount")
        .arg("--pid")
        .arg("--fork")
        .arg("--mount-proc")
        .arg("bash")
        .arg(&script_path)
        .arg(rootfs_dir)
        .arg(bool_env(config.package_stage.update))
        .arg(bool_env(config.package_stage.clean));
    for package in &config.package_stage.install {
        command.arg(package);
    }
    command.arg("--");
    for package in &config.package_stage.npm_global {
        command.arg(&package.package);
    }
    command.arg("--");
    for package in &config.package_stage.npm_global {
        for binary in &package.binaries {
            command.arg(binary);
        }
    }
    run_logged_command(&mut command, &log_path, "apt/npm package stage")
}

fn apt_stage_script() -> &'static str {
    r#"#!/usr/bin/env bash
set -euo pipefail
rootfs="$(readlink -f "$1")"
update="$2"
clean="$3"
shift 3

apt_packages=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    apt_packages+=("$1")
    shift
done
[[ $# -gt 0 ]] && shift
npm_packages=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    npm_packages+=("$1")
    shift
done
[[ $# -gt 0 ]] && shift
npm_binaries=("$@")

mkdir -p "$rootfs/proc" "$rootfs/sys" "$rootfs/dev" "$rootfs/etc" "$rootfs/tmp"
mount --bind /proc "$rootfs/proc"
mount --rbind /sys "$rootfs/sys"
mount --rbind /dev "$rootfs/dev"
cleanup() {
    umount -l "$rootfs/dev" >/dev/null 2>&1 || true
    umount -l "$rootfs/sys" >/dev/null 2>&1 || true
    umount -l "$rootfs/proc" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ -f /etc/resolv.conf ]]; then
    cp /etc/resolv.conf "$rootfs/etc/resolv.conf"
fi

export DEBIAN_FRONTEND=noninteractive
if [[ "$update" == "1" ]]; then
    chroot "$rootfs" env DEBIAN_FRONTEND=noninteractive apt-get -o APT::Sandbox::User=root update
fi
if [[ ${#apt_packages[@]} -gt 0 ]]; then
    chroot "$rootfs" env DEBIAN_FRONTEND=noninteractive apt-get -o APT::Sandbox::User=root install -y --no-install-recommends "${apt_packages[@]}"
fi

mkdir -p "$rootfs/opt/motlie/npm"
if [[ ${#npm_packages[@]} -gt 0 ]]; then
    chroot "$rootfs" env npm_config_update_notifier=false npm install -g --prefix /opt/motlie/npm "${npm_packages[@]}"
    mkdir -p "$rootfs/usr/local/bin"
    for binary in "${npm_binaries[@]}"; do
        chroot "$rootfs" ln -sf "/opt/motlie/npm/bin/$binary" "/usr/local/bin/$binary"
    done
fi

printf 'en_US.UTF-8 UTF-8\n' > "$rootfs/etc/locale.gen"
chroot "$rootfs" locale-gen en_US.UTF-8
chroot "$rootfs" update-locale LANG=en_US.UTF-8
chroot "$rootfs" ssh-keygen -A
sed -i 's/#PermitRootLogin.*/PermitRootLogin prohibit-password/' "$rootfs/etc/ssh/sshd_config" || true
systemctl --root="$rootfs" enable ssh >/dev/null
systemctl --root="$rootfs" enable systemd-networkd >/dev/null
systemctl --root="$rootfs" disable systemd-networkd-wait-online.service >/dev/null 2>&1 || true
systemctl --root="$rootfs" disable apt-daily.service apt-daily.timer apt-daily-upgrade.service apt-daily-upgrade.timer unattended-upgrades.service >/dev/null 2>&1 || true
systemctl --root="$rootfs" mask apt-daily.service apt-daily-upgrade.service unattended-upgrades.service >/dev/null 2>&1 || true
grep -qxF user_allow_other "$rootfs/etc/fuse.conf" 2>/dev/null || printf '%s\n' user_allow_other >> "$rootfs/etc/fuse.conf"
if [[ "$clean" == "1" ]]; then
    chroot "$rootfs" apt-get clean
    rm -rf "$rootfs/var/lib/apt/lists"/* "$rootfs/var/cache/apt/archives"/* "$rootfs/root/.npm"
    mkdir -p "$rootfs/var/cache/apt/archives/partial"
fi
truncate -s 0 "$rootfs/etc/machine-id"
"#
}

fn assemble_immutable_rootfs(
    rootfs_dir: &Path,
    config: &ImageBuildConfig,
    source: &ExternalOciSource,
    guest_binaries: &GuestBinaries,
) -> Result<motlie_vmm::image::RootfsCompatibilityAssemblyManifest> {
    let mut profile = GuestImageProfile::ubuntu_systemd(source.clone());
    profile.required_packages = config.package_stage.install.clone();
    let profile_spec = RootfsProfileSpec::for_profile(profile);
    let mut spec = RootfsCompatibilityLayerSpec::new(profile_spec);
    spec.pending_requirement_policy = RootfsPendingRequirementPolicy::FailInstallable;
    spec.enable_ch_egress_service = true;
    for payload in &config.immutable_payloads {
        let mode = parse_octal_mode(&payload.mode)?;
        let source = if payload.guest_path == v1_5::MOTLIE_V15_GUEST_BIN_OPT {
            guest_binaries.vfs.clone()
        } else if payload.guest_path == "/opt/motlie/v1.5/guest/bin/motlie-vsock-ssh-bridge" {
            guest_binaries.ssh_bridge.clone()
        } else {
            payload.source.clone()
        };
        let mut file = RootfsPayloadFile::new(source, PathBuf::from(&payload.guest_path), mode);
        for link in &payload.links {
            file = file.with_link(PathBuf::from(link));
        }
        spec.guest_binaries.push(file);
    }
    Ok(RootfsCompatibilityAssembler::new().assemble(rootfs_dir, &spec)?)
}

fn parse_octal_mode(value: &str) -> Result<u32> {
    u32::from_str_radix(value.trim_start_matches('0'), 8)
        .with_context(|| format!("invalid octal mode {value:?}"))
}

fn install_ch_boot_adaptations(
    repo_root: &Path,
    rootfs_dir: &Path,
    source: &ExternalOciSource,
    host: &ChHostTarget,
    guest_binary: &Path,
) -> Result<()> {
    let example_dir = repo_root.join("libs/vmm/examples/v1.5");
    copy_into_rootfs(
        &example_dir.join("overlay-init"),
        rootfs_dir,
        "/sbin/overlay-init",
        0o755,
    )?;
    copy_into_rootfs(
        &example_dir.join("99_motlie_ch.cfg"),
        rootfs_dir,
        "/etc/cloud/cloud.cfg.d/99_motlie_ch.cfg",
        0o644,
    )?;
    write_rootfs_file(
        rootfs_dir,
        "/etc/fstab",
        b"proc        /proc    proc    defaults        0      0\nsysfs       /sys     sysfs   defaults        0      0\ndevtmpfs    /dev     devtmpfs defaults       0      0\n",
        0o644,
    )?;
    write_rootfs_file(
        rootfs_dir,
        "/etc/motd",
        ch_motd(source, host).as_bytes(),
        0o644,
    )?;
    let _ = guest_binary;
    Ok(())
}

fn ch_motd(source: &ExternalOciSource, host: &ChHostTarget) -> String {
    format!(
        "motlie v1.5 CH guest\nsource={}\nplatform={}\narch={}\n",
        source.image_ref, source.platform, host.deb_arch
    )
}

fn emit_squashfs(rootfs_dir: &Path, rootfs_path: &Path, log_path: &Path) -> Result<()> {
    if let Some(parent) = rootfs_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let script = "set -euo pipefail\nrm -f \"$2\"\ntar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C \"$1\" -cf - . | tar2sqfs -f \"$2\"\n";
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(script)
        .arg("mbuild-squashfs")
        .arg(rootfs_dir)
        .arg(rootfs_path);
    run_logged_command(&mut command, log_path, "emit CH squashfs")
}

fn emit_ch_kernel(base_dir: &Path, host: &ChHostTarget, log_path: &Path) -> Result<PathBuf> {
    let kernel_path = base_dir.join(host.kernel_image);
    if kernel_path.is_file() {
        return Ok(kernel_path);
    }
    let release =
        env::var("CH_KERNEL_RELEASE").unwrap_or_else(|_| CH_KERNEL_RELEASE_DEFAULT.to_string());
    let url = env::var("CH_KERNEL_URL").unwrap_or_else(|_| {
        format!(
            "https://github.com/cloud-hypervisor/linux/releases/download/{release}/{}",
            host.kernel_asset
        )
    });
    let mut command = Command::new("wget");
    command
        .arg("-q")
        .arg("--show-progress")
        .arg("-O")
        .arg(&kernel_path)
        .arg(url);
    run_logged_command(&mut command, log_path, "download CH kernel")?;
    Ok(kernel_path)
}

fn write_ch_guest_contract(
    path: &Path,
    source: &ExternalOciSource,
    host: &ChHostTarget,
    kernel_path: &Path,
    guest_binary: &Path,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let value = serde_json::json!({
        "contract_version": v1_5::MOTLIE_V15_CONTRACT_VERSION,
        "packaging_backend": "ch-squashfs",
        "guest_arch": host.deb_arch,
        "kernel_image": kernel_path.file_name().and_then(|name| name.to_str()).unwrap_or(host.kernel_image),
        "source": {
            "kind": "external-oci",
            "image": source.image_ref.clone(),
            "platform": source.platform.to_string(),
            "image_index_digest": source.image_index_digest.to_string(),
            "platform_manifest_digest": source.platform_manifest_digest.to_string(),
        },
        "guest_contract": {
            "motlie_vfs_guest_path": v1_5::MOTLIE_V15_GUEST_BIN_OPT,
            "motlie_vfs_guest_compat_path": v1_5::MOTLIE_V15_GUEST_BIN_COMPAT,
            "motlie_vfs_guest_marker": v1_5::MOTLIE_V15_GUEST_MOUNTER_MARKER,
            "motlie_vfs_guest_build_features": v1_5::MOTLIE_V15_GUEST_BUILD_FEATURES,
            "guest_binary": guest_binary.display().to_string(),
            "backend_env": v1_5::MOTLIE_V15_BACKEND_ENV_PATH,
            "mounts": v1_5::MOTLIE_V15_MOUNTS_PATH,
            "agent_state": "/agent-state"
        },
        "launch_contract": {
            "builds_allowed": false,
            "forbidden_first_contact_tools": ["apt", "apt-get", "cargo", "npm", "rustup"]
        }
    });
    fs::write(path, serde_json::to_vec_pretty(&value)?)?;
    Ok(())
}

fn write_ch_build_result(
    out: &Path,
    contract_path: &Path,
    kernel_path: &Path,
    rootfs_path: &Path,
    guest_binary: &Path,
) -> Result<()> {
    let value = serde_json::json!({
        "ok": true,
        "backend": "ch",
        "contract": contract_path.display().to_string(),
        "kernel": kernel_path.display().to_string(),
        "rootfs": rootfs_path.display().to_string(),
        "guest_binary": guest_binary.display().to_string(),
    });
    fs::write(
        out.join(CH_BUILD_RESULT_NAME),
        serde_json::to_vec_pretty(&value)?,
    )?;
    Ok(())
}

fn copy_into_rootfs(source: &Path, rootfs_dir: &Path, guest_path: &str, mode: u32) -> Result<()> {
    let bytes = fs::read(source)
        .with_context(|| format!("read CH adaptation source {}", source.display()))?;
    write_rootfs_file(rootfs_dir, guest_path, &bytes, mode)
}

fn write_rootfs_file(rootfs_dir: &Path, guest_path: &str, bytes: &[u8], mode: u32) -> Result<()> {
    let relative = guest_path
        .strip_prefix('/')
        .with_context(|| format!("guest path must be absolute: {guest_path}"))?;
    let path = rootfs_dir.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, bytes)?;
    set_mode(&path, mode)
}

fn set_mode(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
    }
    Ok(())
}

fn append_log(path: &Path, text: &str) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    use std::io::Write as _;
    file.write_all(text.as_bytes())?;
    Ok(())
}

fn run_logged_command(command: &mut Command, log_path: &Path, label: &str) -> Result<()> {
    append_log(
        log_path,
        &format!("\n--- {label} ---\ncommand={command:?}\n"),
    )?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let stderr = log.try_clone()?;
    let status = command
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr))
        .status()
        .with_context(|| format!("run {label}"))?;
    if !status.success() {
        bail!(
            "{label} failed with status {}; see {}",
            status.code().unwrap_or(-1),
            log_path.display()
        );
    }
    Ok(())
}

#[instrument(
    skip(repo_root, manifest, emitter, artifact, options),
    fields(target = %manifest.target, scenario = %scenario.display())
)]
fn run_harness_validation(
    repo_root: &Path,
    manifest: &ImageBuildManifest,
    emitter: &BackendEmitterSpec,
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
    let started_at_unix_seconds = unix_now()?;

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

    emitter.validation.apply_to_command(&mut command, artifact);
    info!(
        target = %manifest.target,
        scenario = %scenario.display(),
        log = %log_path.display(),
        command = ?command_line,
        "running harness validation"
    );

    let status = command.status()?;
    let completed_at_unix_seconds = unix_now()?;
    Ok(HarnessValidationRecord {
        contract_version: manifest.contract_version.clone(),
        target: manifest.target.clone(),
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
            let relative = path
                .strip_prefix(base)
                .with_context(|| {
                    format!(
                        "artifact {} is not under artifact root {}",
                        path.display(),
                        base.display()
                    )
                })?
                .to_path_buf();
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

fn rootfs_tarball_record(path: &Path) -> Result<RootfsTarballRecord> {
    let canonical_path = fs::canonicalize(path)
        .with_context(|| format!("rootfs tarball cannot be canonicalized: {}", path.display()))?;
    let metadata = fs::metadata(&canonical_path).with_context(|| {
        format!(
            "rootfs tarball metadata is not readable: {}",
            canonical_path.display()
        )
    })?;
    if !metadata.is_file() {
        bail!(
            "rootfs tarball is not a file: {}",
            canonical_path.display()
        );
    }
    let sha256 = sha256_file(&canonical_path).with_context(|| {
        format!(
            "failed to compute rootfs tarball sha256: {}",
            canonical_path.display()
        )
    })?;
    Ok(RootfsTarballRecord {
        canonical_path,
        size_bytes: metadata.len(),
        sha256,
    })
}

fn unix_now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_secs())
}

fn bool_env(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
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
        #[arg(long)]
        target: BackendTargetId,
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
        /// Assembled rootfs tarball to hand to a backend emitter.
        #[arg(long)]
        rootfs_tarball: Option<PathBuf>,
    },
    /// Emit per-guest seed artifacts without rebuilding the immutable image.
    Seed {
        /// Dockerfile-like v1.5 image build config.
        #[arg(long)]
        config: PathBuf,
        /// Backend seed target.
        #[arg(long)]
        target: BackendTargetId,
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
#[serde(deny_unknown_fields)]
struct ImageBuildConfig {
    version: String,
    source: SourceStage,
    package_stage: PackageStage,
    immutable_payloads: Vec<PayloadSpec>,
    sshd_policy: SshdPolicy,
    services: Vec<ServiceSpec>,
    immutable_files: Vec<String>,
    seed_files: Vec<String>,
    seed: SeedStage,
    emitters: Vec<BackendEmitterSpec>,
    validation: Vec<String>,
}

impl ImageBuildConfig {
    fn validate(&self) -> Result<()> {
        require_eq("version", &self.version, v1_5::MOTLIE_V15_CONTRACT_VERSION)?;
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
        self.seed.validate()?;
        if self.emitters.is_empty() {
            bail!("emitters must not be empty");
        }
        let mut ids = BTreeSet::new();
        for emitter in &self.emitters {
            emitter.validate()?;
            if !ids.insert(emitter.id.clone()) {
                bail!("duplicate emitter id {}", emitter.id);
            }
        }
        if self.validation.is_empty() {
            bail!("validation must not be empty");
        }
        Ok(())
    }

    fn validate_for_target(&self, target: &BackendTargetId) -> Result<&BackendEmitterSpec> {
        self.validate()?;
        self.emitter(target)
            .with_context(|| format!("target {} is not declared in emitters", target.as_str()))
    }

    fn emitter(&self, target: &BackendTargetId) -> Option<&BackendEmitterSpec> {
        self.emitters.iter().find(|emitter| &emitter.id == target)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceStage {
    kind: SourceKind,
    image: String,
    profile: ProfileId,
    platform: String,
    digest_policy: DigestPolicy,
    #[serde(default)]
    image_index_digest: Option<OciDigest>,
    #[serde(default)]
    platform_manifest_digest: Option<OciDigest>,
}

impl SourceStage {
    fn validate(&self) -> Result<()> {
        require_non_empty("source.image", &self.image)?;
        self.profile.validate("source.profile")?;
        require_non_empty("source.platform", &self.platform)?;
        match self.kind {
            SourceKind::ExternalOci => {
                if self.digest_policy == DigestPolicy::AdapterVerified {
                    bail!("source.digest_policy adapter-verified is only valid for transitional-adapter sources");
                }
                if self.digest_policy == DigestPolicy::Pinned
                    && (self.image_index_digest.is_none()
                        || self.platform_manifest_digest.is_none())
                {
                    bail!("source.digest_policy pinned requires source.image_index_digest and source.platform_manifest_digest");
                }
            }
            SourceKind::TransitionalAdapter => {
                if self.digest_policy != DigestPolicy::AdapterVerified {
                    bail!("source.digest_policy for transitional-adapter sources must be adapter-verified");
                }
                if self.image_index_digest.is_some() || self.platform_manifest_digest.is_some() {
                    bail!("transitional-adapter sources must not declare OCI digest pins");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum SourceKind {
    ExternalOci,
    TransitionalAdapter,
}

impl std::fmt::Display for SourceKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExternalOci => formatter.write_str("external-oci"),
            Self::TransitionalAdapter => formatter.write_str("transitional-adapter"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageStage {
    manager: PackageManagerId,
    update: bool,
    install: Vec<String>,
    #[serde(default)]
    npm_global: Vec<NpmGlobalPackage>,
    clean: bool,
}

impl PackageStage {
    fn validate(&self) -> Result<()> {
        self.manager.validate("package_stage.manager")?;
        let strategy = package_manager_strategy(self.manager.as_str()).with_context(|| {
            format!(
                "package_stage.manager {:?} is not registered; registered managers: {}",
                self.manager.as_str(),
                PACKAGE_MANAGER_STRATEGIES
                    .iter()
                    .map(|strategy| strategy.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        if strategy.support != PackageManagerSupport::Implemented {
            bail!(
                "package_stage.manager {:?} is reserved but not implemented by current mbuild adapters",
                self.manager.as_str()
            );
        }
        if self.install.is_empty() {
            bail!("package_stage.install must not be empty");
        }
        for package in &self.install {
            (strategy.validate_package)("package_stage.install", package)?;
        }
        for package in &self.npm_global {
            package.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NpmGlobalPackage {
    package: String,
    binaries: Vec<String>,
}

impl NpmGlobalPackage {
    fn validate(&self) -> Result<()> {
        require_non_empty("package_stage.npm_global.package", &self.package)?;
        if self
            .package
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || matches!(byte, b',' | b';'))
        {
            bail!(
                "package_stage.npm_global.package contains unsupported npm package spec {:?}",
                self.package
            );
        }
        if self.binaries.is_empty() {
            bail!("package_stage.npm_global.binaries must not be empty");
        }
        for binary in &self.binaries {
            require_token("package_stage.npm_global.binaries", binary)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
struct ServiceSpec {
    name: String,
    enable: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedStage {
    user_home: String,
    ssh_principal: String,
    mounts: Vec<SeedMountTemplate>,
}

impl SeedStage {
    fn validate(&self) -> Result<()> {
        require_seed_template("seed.user_home", &self.user_home)?;
        require_seed_template("seed.ssh_principal", &self.ssh_principal)?;
        if self.mounts.is_empty() {
            bail!("seed.mounts must not be empty");
        }
        for mount in &self.mounts {
            mount.validate()?;
        }
        Ok(())
    }

    fn render(&self, guest: &str) -> Result<RenderedSeedStage> {
        require_token("seed guest", guest)?;
        let user_home = render_seed_template(&self.user_home, guest);
        require_abs_path("seed.user_home", &user_home)?;
        let ssh_principal = render_seed_template(&self.ssh_principal, guest);
        require_token("seed.ssh_principal", &ssh_principal)?;
        let mut mounts = Vec::with_capacity(self.mounts.len());
        for mount in &self.mounts {
            mounts.push(mount.render(guest)?);
        }
        Ok(RenderedSeedStage {
            user_home: PathBuf::from(user_home),
            ssh_principal,
            mounts,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedMountTemplate {
    tag: String,
    guest_path: String,
    #[serde(default)]
    read_only: bool,
}

impl SeedMountTemplate {
    fn validate(&self) -> Result<()> {
        require_seed_template("seed.mounts.tag", &self.tag)?;
        require_seed_template("seed.mounts.guest_path", &self.guest_path)?;
        Ok(())
    }

    fn render(&self, guest: &str) -> Result<RootfsMountSpec> {
        let tag = render_seed_template(&self.tag, guest);
        require_token("seed.mounts.tag", &tag)?;
        let guest_path = render_seed_template(&self.guest_path, guest);
        require_abs_path("seed.mounts.guest_path", &guest_path)?;
        let mut spec = RootfsMountSpec::new(tag, PathBuf::from(guest_path));
        spec.read_only = self.read_only;
        Ok(spec)
    }
}

#[derive(Debug, Clone)]
struct RenderedSeedStage {
    user_home: PathBuf,
    ssh_principal: String,
    mounts: Vec<RootfsMountSpec>,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
#[serde(transparent)]
struct BackendTargetId(String);

impl BackendTargetId {
    fn parse(value: String) -> std::result::Result<Self, String> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err("backend target id must not be empty".to_string());
        }
        if !trimmed
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        {
            return Err(format!(
                "backend target id contains unsupported token {value:?}"
            ));
        }
        Ok(Self(trimmed.to_string()))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for BackendTargetId {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::parse(value.to_string())
    }
}

impl std::fmt::Display for BackendTargetId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
struct ProfileId(String);

impl ProfileId {
    fn validate(&self, field: &str) -> Result<()> {
        require_token(field, &self.0)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum DigestPolicy {
    Pinned,
    Floating,
    AdapterVerified,
}

impl std::fmt::Display for DigestPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pinned => formatter.write_str("pinned"),
            Self::Floating => formatter.write_str("floating"),
            Self::AdapterVerified => formatter.write_str("adapter-verified"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
struct PackageManagerId(String);

impl PackageManagerId {
    fn validate(&self, field: &str) -> Result<()> {
        require_token(field, &self.0)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
struct PackageManagerStrategy {
    id: &'static str,
    support: PackageManagerSupport,
    validate_package: fn(&str, &str) -> Result<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageManagerSupport {
    Implemented,
    Reserved,
}

const PACKAGE_MANAGER_STRATEGIES: &[PackageManagerStrategy] = &[
    PackageManagerStrategy {
        id: "apt",
        support: PackageManagerSupport::Implemented,
        validate_package: validate_apt_package_spec,
    },
    PackageManagerStrategy {
        id: "apk",
        support: PackageManagerSupport::Reserved,
        validate_package: validate_reserved_package_spec,
    },
    PackageManagerStrategy {
        id: "dnf",
        support: PackageManagerSupport::Reserved,
        validate_package: validate_reserved_package_spec,
    },
    PackageManagerStrategy {
        id: "zypper",
        support: PackageManagerSupport::Reserved,
        validate_package: validate_reserved_package_spec,
    },
    PackageManagerStrategy {
        id: "pacman",
        support: PackageManagerSupport::Reserved,
        validate_package: validate_reserved_package_spec,
    },
];

fn package_manager_strategy(id: &str) -> Option<&'static PackageManagerStrategy> {
    PACKAGE_MANAGER_STRATEGIES
        .iter()
        .find(|strategy| strategy.id == id)
}

fn validate_reserved_package_spec(field: &str, value: &str) -> Result<()> {
    require_non_empty(field, value)
}

fn validate_apt_package_spec(field: &str, value: &str) -> Result<()> {
    require_non_empty(field, value)?;
    if value
        .bytes()
        .any(|byte| byte.is_ascii_whitespace() || matches!(byte, b',' | b'/'))
    {
        bail!("{field} contains unsupported apt package spec {value:?}");
    }

    let mut equals = value.split('=');
    let name_arch = equals.next().unwrap_or_default();
    let version = equals.next();
    if equals.next().is_some() {
        bail!("{field} contains unsupported apt package spec {value:?}: multiple '=' separators");
    }
    if let Some(version) = version {
        validate_apt_version(field, value, version)?;
    }

    let mut name_arch_parts = name_arch.split(':');
    let name = name_arch_parts.next().unwrap_or_default();
    let arch = name_arch_parts.next();
    if name_arch_parts.next().is_some() {
        bail!("{field} contains unsupported apt package spec {value:?}: multiple ':' separators before version");
    }
    validate_apt_package_name(field, value, name)?;
    if let Some(arch) = arch {
        validate_apt_arch(field, value, arch)?;
    }
    Ok(())
}

fn validate_apt_package_name(field: &str, full: &str, name: &str) -> Result<()> {
    require_non_empty(field, name)?;
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        bail!("{field} contains unsupported apt package spec {full:?}: empty package name");
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        bail!("{field} contains unsupported apt package spec {full:?}: package name must start with lowercase ASCII or digit");
    }
    if !bytes.all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')
    }) {
        bail!("{field} contains unsupported apt package spec {full:?}: unsupported package name character");
    }
    Ok(())
}

fn validate_apt_arch(field: &str, full: &str, arch: &str) -> Result<()> {
    require_non_empty(field, arch)?;
    let mut bytes = arch.bytes();
    let Some(first) = bytes.next() else {
        bail!(
            "{field} contains unsupported apt package spec {full:?}: empty architecture qualifier"
        );
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        bail!("{field} contains unsupported apt package spec {full:?}: architecture must start with lowercase ASCII or digit");
    }
    if !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-') {
        bail!("{field} contains unsupported apt package spec {full:?}: unsupported architecture character");
    }
    Ok(())
}

fn validate_apt_version(field: &str, full: &str, version: &str) -> Result<()> {
    require_non_empty(field, version)?;
    if !version.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'+' | b'-' | b':' | b'~')
    }) {
        bail!(
            "{field} contains unsupported apt package spec {full:?}: unsupported version character"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BackendEmitterSpec {
    id: BackendTargetId,
    materialized_source: Option<AdapterMaterializedSource>,
    adapter: BackendAdapterSpec,
    seed: BackendSeedSpec,
    validation: BackendValidationSpec,
}

impl BackendEmitterSpec {
    fn validate(&self) -> Result<()> {
        require_token("emitters.id", self.id.as_str())?;
        if let Some(source) = &self.materialized_source {
            source.validate()?;
        }
        self.adapter.validate()?;
        self.seed.validate()?;
        self.validation.validate()?;
        Ok(())
    }

    fn backend_env(&self) -> RootfsCompatibilityBackendEnv {
        RootfsCompatibilityBackendEnv::for_backend(
            &self.seed.motlie_backend,
            &self.seed.motlie_net_backend,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AdapterMaterializedSource {
    image: String,
    profile: String,
    platform: String,
    materializer: String,
}

impl AdapterMaterializedSource {
    fn validate(&self) -> Result<()> {
        require_non_empty("emitters.materialized_source.image", &self.image)?;
        require_token("emitters.materialized_source.profile", &self.profile)?;
        require_token("emitters.materialized_source.platform", &self.platform)?;
        require_token(
            "emitters.materialized_source.materializer",
            &self.materializer,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BackendAdapterSpec {
    program: String,
    #[serde(default)]
    args: Vec<String>,
    env: BackendAdapterEnvSpec,
}

impl BackendAdapterSpec {
    fn validate(&self) -> Result<()> {
        require_non_empty("emitters.adapter.program", &self.program)?;
        for arg in &self.args {
            require_non_empty("emitters.adapter.args", arg)?;
        }
        self.env.validate()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BackendAdapterEnvSpec {
    artifact_dir: String,
    build_config: String,
    package_manager: String,
    package_update: String,
    package_include: String,
    package_clean: String,
    #[serde(default)]
    rootfs_tarball: Option<String>,
}

impl BackendAdapterEnvSpec {
    fn validate(&self) -> Result<()> {
        require_env_name("emitters.adapter.env.artifact_dir", &self.artifact_dir)?;
        require_env_name("emitters.adapter.env.build_config", &self.build_config)?;
        require_env_name(
            "emitters.adapter.env.package_manager",
            &self.package_manager,
        )?;
        require_env_name("emitters.adapter.env.package_update", &self.package_update)?;
        require_env_name(
            "emitters.adapter.env.package_include",
            &self.package_include,
        )?;
        require_env_name("emitters.adapter.env.package_clean", &self.package_clean)?;
        if let Some(env_name) = &self.rootfs_tarball {
            require_env_name("emitters.adapter.env.rootfs_tarball", env_name)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BackendSeedSpec {
    motlie_backend: String,
    motlie_net_backend: String,
}

impl BackendSeedSpec {
    fn validate(&self) -> Result<()> {
        require_token("emitters.seed.motlie_backend", &self.motlie_backend)?;
        require_token("emitters.seed.motlie_net_backend", &self.motlie_net_backend)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BackendValidationSpec {
    artifact_dir_env: Option<String>,
    artifact_dir_suffix: Option<PathBuf>,
    base_vm_dir_env: Option<String>,
    base_vm_dir_suffix: Option<PathBuf>,
    #[serde(default)]
    base_vm_dir_required_files: Vec<PathBuf>,
}

impl BackendValidationSpec {
    fn validate(&self) -> Result<()> {
        if let Some(env_name) = &self.artifact_dir_env {
            require_env_name("emitters.validation.artifact_dir_env", env_name)?;
        }
        if let Some(env_name) = &self.base_vm_dir_env {
            require_env_name("emitters.validation.base_vm_dir_env", env_name)?;
        }
        Ok(())
    }

    fn apply_to_command(&self, command: &mut Command, artifact: &Path) {
        if let Some(env_name) = &self.artifact_dir_env {
            let suffix = self.artifact_dir_suffix.as_deref().unwrap_or(Path::new(""));
            command.env(env_name, artifact.join(suffix));
        }
        if let Some(env_name) = &self.base_vm_dir_env {
            let suffix = self.base_vm_dir_suffix.as_deref().unwrap_or(Path::new(""));
            let path = artifact.join(suffix);
            let ready = if self.base_vm_dir_required_files.is_empty() {
                path.exists()
            } else {
                self.base_vm_dir_required_files
                    .iter()
                    .all(|required| path.join(required).exists())
            };
            if ready {
                command.env(env_name, path);
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageBuildManifest {
    contract_version: String,
    config_path: PathBuf,
    target: BackendTargetId,
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
        target: BackendTargetId,
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
            source: ManifestSource::from_config(config),
            package_stage: ManifestPackageStage::from_config(config),
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

    fn validate_against_config<'a>(
        &self,
        config: &'a ImageBuildConfig,
    ) -> Result<&'a BackendEmitterSpec> {
        require_manifest_match("contract_version", &self.contract_version, &config.version)?;
        let emitter = config.validate_for_target(&self.target)?;
        require_manifest_match("source", &self.source, &ManifestSource::from_config(config))?;
        require_manifest_match(
            "package_stage",
            &self.package_stage,
            &ManifestPackageStage::from_config(config),
        )?;
        require_manifest_match(
            "immutable_files",
            &self.immutable_files,
            &config.immutable_files,
        )?;
        require_manifest_match("seed_files", &self.seed_files, &config.seed_files)?;
        require_manifest_match("validation", &self.validation, &config.validation)?;
        if let Some(adapter) = &self.adapter {
            require_manifest_match(
                "adapter.materialized_source",
                &adapter.materialized_source,
                &emitter.materialized_source,
            )?;
            require_manifest_match(
                "adapter.package_include",
                &adapter.package_include,
                &config.package_stage.install,
            )?;
        }
        Ok(emitter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestSource {
    kind: String,
    image: String,
    profile: String,
    platform: String,
    digest_policy: String,
    image_index_digest: Option<OciDigest>,
    platform_manifest_digest: Option<OciDigest>,
}

impl ManifestSource {
    fn from_config(config: &ImageBuildConfig) -> Self {
        Self {
            kind: config.source.kind.to_string(),
            image: config.source.image.clone(),
            profile: config.source.profile.as_str().to_string(),
            platform: config.source.platform.clone(),
            digest_policy: config.source.digest_policy.to_string(),
            image_index_digest: config.source.image_index_digest.clone(),
            platform_manifest_digest: config.source.platform_manifest_digest.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestPackageStage {
    manager: String,
    update: bool,
    install: Vec<String>,
    npm_global: Vec<NpmGlobalPackage>,
    clean: bool,
}

impl ManifestPackageStage {
    fn from_config(config: &ImageBuildConfig) -> Self {
        Self {
            manager: config.package_stage.manager.as_str().to_string(),
            update: config.package_stage.update,
            install: config.package_stage.install.clone(),
            npm_global: config.package_stage.npm_global.clone(),
            clean: config.package_stage.clean,
        }
    }
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
    materialized_source: Option<AdapterMaterializedSource>,
    #[serde(default)]
    external_oci_source: Option<ExternalOciSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rootfs_tarball: Option<RootfsTarballRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RootfsTarballRecord {
    canonical_path: PathBuf,
    size_bytes: u64,
    sha256: String,
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
    target: BackendTargetId,
    artifact_dir: PathBuf,
    scenario: PathBuf,
    command: Vec<String>,
    log_path: PathBuf,
    exit_status: i32,
    started_at_unix_seconds: u64,
    completed_at_unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn executed(name: &str, summary: &str, adapter: &AdapterRecord) -> Self {
        Self {
            name: name.to_string(),
            status: "executed".to_string(),
            summary: summary.to_string(),
            evidence: stage_evidence(adapter),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageSeedManifest {
    contract_version: String,
    config_path: PathBuf,
    target: BackendTargetId,
    output_dir: PathBuf,
    seed_files: Vec<String>,
    rootfs_seed_overlay: RootfsSeedOverlayManifest,
    artifacts: Vec<ManifestArtifact>,
}

impl ImageSeedManifest {
    fn from_seed_overlay(
        config_path: &Path,
        target: BackendTargetId,
        out: &Path,
        config: &ImageBuildConfig,
        rootfs_seed_overlay: &RootfsSeedOverlayManifest,
    ) -> Result<Self> {
        Ok(Self {
            contract_version: config.version.clone(),
            config_path: config_path.to_path_buf(),
            target,
            output_dir: out.to_path_buf(),
            seed_files: config.seed_files.clone(),
            rootfs_seed_overlay: rootfs_seed_overlay.clone(),
            artifacts: collect_artifacts(out, DEFAULT_SEED_MANIFEST_NAME)?,
        })
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
    if adapter.kind == "external-oci-ch-emitter" {
        return vec![
            StageRecord::executed(
                "source",
                "external OCI source resolved and digest-checked",
                adapter,
            ),
            StageRecord::executed(
                "import",
                "selected OCI platform layers fetched and imported",
                adapter,
            ),
            StageRecord::executed(
                "classify",
                "imported rootfs classified against the selected profile",
                adapter,
            ),
            StageRecord::executed(
                "package",
                "explicit apt/npm package stage executed before backend emit",
                adapter,
            ),
            StageRecord::executed(
                "immutable-layer",
                "VMM guest payload and reusable service layer installed",
                adapter,
            ),
            StageRecord::executed(
                "policy",
                "sshd, service, and image policy applied before boot",
                adapter,
            ),
            StageRecord::declared(
                "seed",
                "per-guest seed is regenerated separately with `mbuild seed`",
            ),
            StageRecord::executed(
                "backend-emitter",
                "CH boot artifacts emitted from assembled rootfs",
                adapter,
            ),
            StageRecord::declared(
                "validation",
                "post-boot harness validation is required after artifact build",
            ),
        ];
    }
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

fn stage_evidence(adapter: &AdapterRecord) -> Vec<String> {
    let mut evidence = vec![format!("log={}", adapter.log_path.display())];
    if let Some(source) = &adapter.external_oci_source {
        evidence.push(format!("image={}", source.image_ref));
        evidence.push(format!("platform={}", source.platform));
        evidence.push(format!("image_index_digest={}", source.image_index_digest));
        evidence.push(format!(
            "platform_manifest_digest={}",
            source.platform_manifest_digest
        ));
    }
    if let Some(rootfs_tarball) = &adapter.rootfs_tarball {
        evidence.push(format!(
            "rootfs_tarball={}",
            rootfs_tarball.canonical_path.display()
        ));
        evidence.push(format!(
            "rootfs_tarball_size_bytes={}",
            rootfs_tarball.size_bytes
        ));
        evidence.push(format!("rootfs_tarball_sha256={}", rootfs_tarball.sha256));
    }
    evidence
}

fn require_eq(field: &str, actual: &str, expected: &str) -> Result<()> {
    if actual != expected {
        bail!("{field} must be {expected:?}, got {actual:?}");
    }
    Ok(())
}

fn require_manifest_match<T>(field: &str, actual: &T, expected: &T) -> Result<()>
where
    T: std::fmt::Debug + PartialEq,
{
    if actual != expected {
        bail!("manifest {field} does not match config: manifest={actual:?}, config={expected:?}");
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

fn require_env_name(field: &str, value: &str) -> Result<()> {
    require_non_empty(field, value)?;
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        bail!("{field} must not be empty");
    };
    if !(first.is_ascii_uppercase() || first == b'_') {
        bail!("{field} must start with an uppercase ASCII letter or underscore");
    }
    if !bytes.all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_') {
        bail!("{field} must contain only uppercase ASCII letters, digits, or underscore");
    }
    Ok(())
}

fn require_seed_template(field: &str, value: &str) -> Result<()> {
    require_non_empty(field, value)?;
    let mut rest = value;
    loop {
        let open = rest.find('{');
        let close = rest.find('}');
        match (open, close) {
            (Some(open), Some(close)) if close < open => {
                bail!("{field} contains an unopened template placeholder");
            }
            (Some(open), _) => {
                let after_start = &rest[open + 1..];
                let Some(end) = after_start.find('}') else {
                    bail!("{field} contains an unclosed template placeholder");
                };
                let name = &after_start[..end];
                if name != "guest" {
                    bail!("{field} contains unsupported template placeholder {{{name}}}");
                }
                rest = &after_start[end + 1..];
            }
            (None, Some(_)) => bail!("{field} contains an unopened template placeholder"),
            (None, None) => break,
        }
    }
    Ok(())
}

fn render_seed_template(template: &str, guest: &str) -> String {
    template.replace("{guest}", guest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_config_fixture() -> ImageBuildConfig {
        ImageBuildConfig {
            version: v1_5::MOTLIE_V15_CONTRACT_VERSION.to_string(),
            source: SourceStage {
                kind: SourceKind::TransitionalAdapter,
                image: "examples/v1.5/current-shell-adapters".to_string(),
                profile: ProfileId("v1-5-shell-adapter".to_string()),
                platform: "host-native".to_string(),
                digest_policy: DigestPolicy::AdapterVerified,
                image_index_digest: None,
                platform_manifest_digest: None,
            },
            package_stage: PackageStage {
                manager: PackageManagerId("apt".to_string()),
                update: true,
                install: vec!["bash".to_string(), "g++".to_string()],
                npm_global: Vec::new(),
                clean: true,
            },
            immutable_payloads: vec![PayloadSpec {
                label: "motlie-vfs-guest".to_string(),
                source: PathBuf::from("target/release/motlie-vfs-guest-v1_5"),
                guest_path: v1_5::MOTLIE_V15_GUEST_BIN_OPT.to_string(),
                mode: "0755".to_string(),
                links: vec![v1_5::MOTLIE_V15_GUEST_BIN_COMPAT.to_string()],
            }],
            sshd_policy: SshdPolicy {
                trusted_user_ca_keys: "/etc/ssh/ca/user_ca.pub".to_string(),
                authorized_principals_file: "/etc/ssh/auth_principals/%u".to_string(),
                force_command: None,
            },
            services: vec![ServiceSpec {
                name: "motlie-vfs-guest.service".to_string(),
                enable: "cloud-init.target".to_string(),
            }],
            immutable_files: vec![v1_5::MOTLIE_V15_GUEST_BIN_OPT.to_string()],
            seed_files: vec![v1_5::MOTLIE_V15_BACKEND_ENV_PATH.to_string()],
            seed: SeedStage {
                user_home: "/home/{guest}".to_string(),
                ssh_principal: "{guest}".to_string(),
                mounts: vec![SeedMountTemplate {
                    tag: "{guest}-workspace".to_string(),
                    guest_path: "/workspace".to_string(),
                    read_only: false,
                }],
            },
            emitters: vec![BackendEmitterSpec {
                id: BackendTargetId("ch".to_string()),
                materialized_source: Some(AdapterMaterializedSource {
                    image: "debian:bookworm".to_string(),
                    profile: "debian-systemd".to_string(),
                    platform: "host-native-linux".to_string(),
                    materializer: "mmdebstrap".to_string(),
                }),
                adapter: BackendAdapterSpec {
                    program: "bash".to_string(),
                    args: vec!["libs/vmm/examples/v1.5/build-image.sh".to_string()],
                    env: BackendAdapterEnvSpec {
                        artifact_dir: "MOTLIE_V15_ARTIFACTS_DIR".to_string(),
                        build_config: "MOTLIE_V15_BUILD_CONFIG".to_string(),
                        package_manager: "MOTLIE_V15_PACKAGE_MANAGER".to_string(),
                        package_update: "MOTLIE_V15_PACKAGE_UPDATE".to_string(),
                        package_include: "MOTLIE_V15_PACKAGE_INCLUDE".to_string(),
                        package_clean: "MOTLIE_V15_PACKAGE_CLEAN".to_string(),
                        rootfs_tarball: None,
                    },
                },
                seed: BackendSeedSpec {
                    motlie_backend: "ch".to_string(),
                    motlie_net_backend: "ch-vhost-user".to_string(),
                },
                validation: BackendValidationSpec {
                    artifact_dir_env: Some("MOTLIE_V15_CH_BASE_ARTIFACTS_DIR".to_string()),
                    artifact_dir_suffix: Some(PathBuf::from("base")),
                    base_vm_dir_env: None,
                    base_vm_dir_suffix: None,
                    base_vm_dir_required_files: Vec::new(),
                },
            }],
            validation: vec!["vfs_memfs".to_string()],
        }
    }

    fn image_config_yaml(extra_top_level: &str, extra_source: &str) -> String {
        format!(
            r#"version: v1.5
source:
  kind: transitional-adapter
  image: examples/v1.5/current-shell-adapters
  profile: v1-5-shell-adapter
  platform: host-native
  digest_policy: adapter-verified
{extra_source}package_stage:
  manager: apt
  update: true
  install:
    - bash
  clean: true
immutable_payloads:
  - label: motlie-vfs-guest
    source: target/release/motlie-vfs-guest-v1_5
    guest_path: /opt/motlie/v1.5/guest/bin/motlie-vfs-guest
    mode: "0755"
    links:
      - /usr/local/bin/motlie-vfs-guest
sshd_policy:
  trusted_user_ca_keys: /etc/ssh/ca/user_ca.pub
  authorized_principals_file: /etc/ssh/auth_principals/%u
  force_command: null
services:
  - name: motlie-vfs-guest.service
    enable: cloud-init.target
immutable_files:
  - /opt/motlie/v1.5/guest/bin/motlie-vfs-guest
seed_files:
  - /etc/motlie/v1.5/backend.env
seed:
  user_home: /home/{{guest}}
  ssh_principal: "{{guest}}"
  mounts:
    - tag: "{{guest}}-workspace"
      guest_path: /workspace
emitters:
  - id: ch
    materialized_source:
      image: debian:bookworm
      profile: debian-systemd
      platform: host-native-linux
      materializer: mmdebstrap
    adapter:
      program: bash
      args: []
      env:
        artifact_dir: MOTLIE_V15_ARTIFACTS_DIR
        build_config: MOTLIE_V15_BUILD_CONFIG
        package_manager: MOTLIE_V15_PACKAGE_MANAGER
        package_update: MOTLIE_V15_PACKAGE_UPDATE
        package_include: MOTLIE_V15_PACKAGE_INCLUDE
        package_clean: MOTLIE_V15_PACKAGE_CLEAN
    seed:
      motlie_backend: ch
      motlie_net_backend: ch-vhost-user
    validation:
      artifact_dir_env: MOTLIE_V15_CH_BASE_ARTIFACTS_DIR
validation:
  - vfs_memfs
{extra_top_level}"#
        )
    }

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
                target: "ch".parse().unwrap(),
                out: PathBuf::from("artifacts/ch"),
                repo_root: None,
                plan_only: false,
                adapter_arg: Vec::new(),
                rootfs_tarball: None
            }
        );
    }

    #[test]
    fn parses_build_rootfs_tarball() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "build",
            "--config",
            "motlie-image.yaml",
            "--target",
            "vz",
            "--out",
            "artifacts/vz",
            "--rootfs-tarball",
            "artifacts/rootfs.tar",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Build {
                config: PathBuf::from("motlie-image.yaml"),
                target: "vz".parse().unwrap(),
                out: PathBuf::from("artifacts/vz"),
                repo_root: None,
                plan_only: false,
                adapter_arg: Vec::new(),
                rootfs_tarball: Some(PathBuf::from("artifacts/rootfs.tar"))
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
                target: "vz".parse().unwrap(),
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

    #[test]
    fn seed_stage_renders_configured_topology() {
        let stage = SeedStage {
            user_home: "/home/{guest}".to_string(),
            ssh_principal: "{guest}".to_string(),
            mounts: vec![
                SeedMountTemplate {
                    tag: "{guest}-home".to_string(),
                    guest_path: "/home/{guest}".to_string(),
                    read_only: false,
                },
                SeedMountTemplate {
                    tag: "{guest}-readonly".to_string(),
                    guest_path: "/readonly".to_string(),
                    read_only: true,
                },
            ],
        };

        let rendered = stage.render("alice").unwrap();

        assert_eq!(rendered.user_home, PathBuf::from("/home/alice"));
        assert_eq!(rendered.ssh_principal, "alice");
        assert_eq!(rendered.mounts[0].tag, "alice-home");
        assert_eq!(rendered.mounts[0].guest_path, PathBuf::from("/home/alice"));
        assert!(!rendered.mounts[0].read_only);
        assert_eq!(rendered.mounts[1].tag, "alice-readonly");
        assert_eq!(rendered.mounts[1].guest_path, PathBuf::from("/readonly"));
        assert!(rendered.mounts[1].read_only);
    }

    #[test]
    fn reserved_package_managers_are_not_executable() {
        let stage = PackageStage {
            manager: PackageManagerId("apk".to_string()),
            update: true,
            install: vec!["openssh".to_string()],
            npm_global: Vec::new(),
            clean: true,
        };

        let error = stage.validate().unwrap_err();

        assert!(error
            .to_string()
            .contains("reserved but not implemented by current mbuild adapters"));
    }

    #[test]
    fn apt_package_specs_allow_common_apt_syntax() {
        let stage = PackageStage {
            manager: PackageManagerId("apt".to_string()),
            update: true,
            install: vec![
                "g++".to_string(),
                "libstdc++6".to_string(),
                "foo:amd64".to_string(),
                "foo=version".to_string(),
                "foo:amd64=1:2.0+really-1~deb12u1".to_string(),
            ],
            npm_global: Vec::new(),
            clean: true,
        };

        stage.validate().unwrap();
    }

    #[test]
    fn config_rejects_unknown_top_level_fields() {
        let error =
            serde_yaml::from_str::<ImageBuildConfig>(&image_config_yaml("unexpected: true\n", ""))
                .unwrap_err();

        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn config_rejects_unknown_nested_fields() {
        let error = serde_yaml::from_str::<ImageBuildConfig>(&image_config_yaml(
            "",
            "  unexpected_source: true\n",
        ))
        .unwrap_err();

        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn manifest_validation_rejects_stale_package_config() {
        let config = image_config_fixture();
        let mut manifest = ImageBuildManifest::from_config(
            Path::new("motlie-image.yaml"),
            BackendTargetId("ch".to_string()),
            Path::new("artifacts/ch"),
            &config,
            None,
            Vec::new(),
        );
        manifest.package_stage.install = vec!["nano".to_string()];

        let error = manifest.validate_against_config(&config).unwrap_err();

        assert!(error
            .to_string()
            .contains("manifest package_stage does not match config"));
    }

    #[test]
    fn manifest_validation_rejects_stale_adapter_materialized_source() {
        let config = image_config_fixture();
        let manifest = ImageBuildManifest::from_config(
            Path::new("motlie-image.yaml"),
            BackendTargetId("ch".to_string()),
            Path::new("artifacts/ch"),
            &config,
            Some(AdapterRecord {
                kind: "v1.5-shell-adapter".to_string(),
                command: vec!["bash".to_string()],
                log_path: PathBuf::from("mbuild-adapter.log"),
                exit_status: 0,
                started_at_unix_seconds: 1,
                completed_at_unix_seconds: 2,
                package_include: config.package_stage.install.clone(),
                materialized_source: Some(AdapterMaterializedSource {
                    image: "ubuntu:24.04".to_string(),
                    profile: "ubuntu-systemd".to_string(),
                    platform: "linux/amd64".to_string(),
                    materializer: "oci-importer".to_string(),
                }),
                external_oci_source: None,
                rootfs_tarball: None,
            }),
            Vec::new(),
        );

        let error = manifest.validate_against_config(&config).unwrap_err();

        assert!(error
            .to_string()
            .contains("manifest adapter.materialized_source does not match config"));
    }
}
