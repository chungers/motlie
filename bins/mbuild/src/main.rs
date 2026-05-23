use std::collections::{BTreeMap, BTreeSet};
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
    v1_5, ExternalOciSource, GuestArchitecture, GuestImageProfile, OciBlobPushStatus,
    OciContentCache, OciDigest, OciImageReference, OciImageReferenceKind, OciPlatform,
    OciRegistryAuth, OciRegistryClient, OciRegistryError, OciRootfsImporter, RootfsCloudInitSeed,
    RootfsCompatibilityAssembler, RootfsCompatibilityBackendEnv, RootfsCompatibilityLayerSpec,
    RootfsMountSpec, RootfsPayloadFile, RootfsPendingRequirementPolicy, RootfsProfileSpec,
    RootfsSeedOverlayAssembler, RootfsSeedOverlayManifest, RootfsSeedOverlaySpec, RootfsUserSeed,
    ALPINE_OPENRC_PROFILE, ALPINE_OPENRC_SOURCE_REF, UBUNTU_SYSTEMD_PROFILE,
    UBUNTU_SYSTEMD_SOURCE_REF,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{error, info, instrument};
use tracing_subscriber::EnvFilter;

const DEFAULT_MANIFEST_NAME: &str = "mbuild-manifest.json";
const DEFAULT_SEED_MANIFEST_NAME: &str = "mbuild-seed-manifest.json";
const DEFAULT_VALIDATION_MANIFEST_NAME: &str = "mbuild-validation-manifest.json";
const DEFAULT_OCI_EXPORT_MANIFEST_NAME: &str = "mbuild-oci-export.json";
const DEFAULT_OCI_INDEX_MANIFEST_NAME: &str = "mbuild-oci-index.json";
const DEFAULT_RELEASE_EVIDENCE_NAME: &str = "mbuild-release-evidence.json";
const DEFAULT_OCI_PUSH_MANIFEST_NAME: &str = "mbuild-oci-push.json";
const ADAPTER_LOG_NAME: &str = "mbuild-adapter.log";
const CH_EMITTER_LOG_NAME: &str = "mbuild-ch-emitter.log";
const CH_PACKAGE_STAGE_LOG_NAME: &str = "mbuild-package-stage.log";
const CH_PACKAGE_STAGE_SCRIPT_NAME: &str = "mbuild-package-stage.sh";
const CH_BUILD_RESULT_NAME: &str = "ch-build-result.json";
const CH_GUEST_CONTRACT_NAME: &str = "guest-contract.json";
const CH_ROOTFS_NAME: &str = "rootfs.squashfs";
const COMMON_ROOTFS_TARBALL_NAME: &str = "assembled-rootfs.tar";
const COMMON_ROOTFS_RECORD_NAME: &str = "mbuild-common-rootfs.json";
const VALIDATION_LOG_NAME: &str = "mbuild-validation.log";
const OCI_CACHE_DIR_ENV: &str = "MOTLIE_MBUILD_OCI_CACHE_DIR";
const CROSS_ARCH_CHROOT_ENV: &str = "MOTLIE_MBUILD_ALLOW_CROSS_ARCH_CHROOT";
const CH_KERNEL_RELEASE_DEFAULT: &str = "ch-release-v6.16.9-20251112";
const OCI_LAYOUT_VERSION: &str = "1.0.0";
const OCI_IMAGE_INDEX_MEDIA_TYPE: &str = "application/vnd.oci.image.index.v1+json";
const OCI_IMAGE_MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
const OCI_IMAGE_CONFIG_MEDIA_TYPE: &str = "application/vnd.oci.image.config.v1+json";
const OCI_IMAGE_LAYER_TAR_MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar";
const OCI_MANIFEST_ARTIFACT_KIND: &str = "motlie.vm-image-artifact";

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
            oci_layout,
        } => build(BuildOptions {
            config_path: config,
            target,
            out,
            repo_root,
            plan_only,
            adapter_args: adapter_arg,
            rootfs_tarball,
            oci_layout,
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
        Commands::Oci { command } => match command {
            OciCommands::Export {
                config,
                artifact,
                out,
                tag,
            } => oci_export(OciExportOptions {
                config_path: config,
                artifact,
                out,
                tag,
            }),
            OciCommands::Validate {
                config,
                artifact,
                layout,
            } => oci_validate(OciValidateOptions {
                config_path: config,
                artifact,
                layout,
            }),
            OciCommands::Resolve { image, platform } => oci_resolve(&image, &platform),
            OciCommands::Index { out, image, layout } => oci_index(OciIndexOptions {
                out,
                image,
                layouts: layout,
            }),
            OciCommands::Push {
                layout,
                image,
                dry_run,
                allow_overwrite,
                username,
                password_env,
                token_env,
                out,
            } => oci_push(OciPushOptions {
                layout,
                image,
                dry_run,
                allow_overwrite,
                username,
                password_env,
                token_env,
                out,
            }),
            OciCommands::Evidence {
                config,
                artifact,
                layout,
                publish_ref,
                out,
            } => oci_release_evidence(OciReleaseEvidenceOptions {
                config_path: config,
                artifact,
                layout,
                publish_ref,
                out,
            }),
        },
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
    oci_layout: Option<PathBuf>,
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

    let manifest = load_build_manifest(&options.artifact)?;
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
    let manifest_path = options.artifact.join(DEFAULT_MANIFEST_NAME);
    info!(path = %manifest_path.display(), "validated mbuild manifest");
    Ok(())
}

#[derive(Debug)]
struct OciExportOptions {
    config_path: PathBuf,
    artifact: PathBuf,
    out: PathBuf,
    tag: Option<String>,
}

#[derive(Debug)]
struct OciValidateOptions {
    config_path: PathBuf,
    artifact: PathBuf,
    layout: PathBuf,
}

#[derive(Debug)]
struct OciIndexOptions {
    out: PathBuf,
    image: String,
    layouts: Vec<PathBuf>,
}

#[derive(Debug)]
struct OciPushOptions {
    layout: PathBuf,
    image: String,
    dry_run: bool,
    allow_overwrite: bool,
    username: Option<String>,
    password_env: Option<String>,
    token_env: Option<String>,
    out: Option<PathBuf>,
}

#[derive(Debug)]
struct OciReleaseEvidenceOptions {
    config_path: PathBuf,
    artifact: PathBuf,
    layout: PathBuf,
    publish_ref: Option<String>,
    out: Option<PathBuf>,
}

#[instrument(skip(options), fields(config = %options.config_path.display(), artifact = %options.artifact.display(), out = %options.out.display()))]
fn oci_export(options: OciExportOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    let manifest = load_build_manifest(&options.artifact)?;
    manifest.validate_against_config(&config)?;
    let adapter = manifest
        .adapter
        .as_ref()
        .context("OCI export requires an executed mbuild artifact, not a plan-only manifest")?;
    let source = adapter
        .external_oci_source
        .as_ref()
        .context("OCI export requires an external-oci source record in the build manifest")?;
    let rootfs = common_rootfs_tarball_from_artifact(&options.artifact, adapter)?;
    let export = write_oci_layout(
        &manifest,
        source,
        &rootfs,
        &options.artifact,
        &options.out,
        options.tag.as_deref(),
    )?;
    let export_manifest_path = options.out.join(DEFAULT_OCI_EXPORT_MANIFEST_NAME);
    fs::write(&export_manifest_path, serde_json::to_vec_pretty(&export)?)?;
    validate_oci_layout(&options.out, &export, Some(&config), Some(&manifest))?;
    info!(path = %export_manifest_path.display(), "wrote mbuild OCI export manifest");
    Ok(())
}

#[instrument(skip(options), fields(config = %options.config_path.display(), artifact = %options.artifact.display(), layout = %options.layout.display()))]
fn oci_validate(options: OciValidateOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    let build_manifest = load_build_manifest(&options.artifact)?;
    build_manifest.validate_against_config(&config)?;
    let export_manifest = load_oci_export_manifest(&options.layout)?;
    let validation = validate_oci_layout(
        &options.layout,
        &export_manifest,
        Some(&config),
        Some(&build_manifest),
    )?;
    info!(
        layout = %options.layout.display(),
        platform = %validation.platform,
        rootfs = %validation.rootfs_layer.digest,
        "validated mbuild OCI layout"
    );
    Ok(())
}

#[instrument(fields(image = %image, platform = %platform))]
fn oci_resolve(image: &str, platform: &str) -> Result<()> {
    let platform = parse_source_platform(platform)?;
    let image_ref = OciImageReference::from_str(image)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("create OCI registry runtime")?;
    let resolved = runtime.block_on(async {
        OciRegistryClient::new()
            .resolve_manifest(&image_ref, platform)
            .await
    })?;
    let value = serde_json::json!({
        "image": resolved.image_ref.to_string(),
        "platform": resolved.platform.to_string(),
        "image_index_digest": resolved.image_index_digest.to_string(),
        "platform_manifest_digest": resolved.platform_manifest_digest.to_string(),
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

#[instrument(skip(options), fields(out = %options.out.display(), image = %options.image))]
fn oci_index(options: OciIndexOptions) -> Result<()> {
    if options.layouts.is_empty() {
        bail!("oci index requires at least one --layout");
    }
    prepare_oci_output_dir(&options.out)?;
    fs::create_dir_all(options.out.join("blobs/sha256"))?;
    fs::write(
        options.out.join("oci-layout"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "imageLayoutVersion": OCI_LAYOUT_VERSION
        }))?,
    )?;

    let mut manifests = Vec::with_capacity(options.layouts.len());
    let mut platform_sources = BTreeMap::new();
    for layout in &options.layouts {
        let export = load_oci_export_manifest(layout)?;
        let validation = validate_oci_layout(layout, &export, None, None)?;
        let manifest_path = descriptor_path(layout, &validation.image_manifest)?;
        copy_oci_blob_to_layout(layout, &options.out, &validation.image_manifest)?;
        copy_oci_blob_to_layout(layout, &options.out, &validation.image_config)?;
        copy_oci_blob_to_layout(layout, &options.out, &validation.rootfs_layer)?;
        let platform = parse_source_platform(&validation.platform)?;
        if platform_sources
            .insert(platform.to_string(), validation.source.clone())
            .is_some()
        {
            bail!("duplicate OCI payload platform in index input: {platform}");
        }
        manifests.push(serde_json::json!({
            "mediaType": validation.image_manifest.media_type,
            "digest": validation.image_manifest.digest,
            "size": validation.image_manifest.size_bytes,
            "platform": {
                "architecture": platform.architecture.to_string(),
                "os": platform.os.to_string(),
            },
            "annotations": {
                "org.opencontainers.image.ref.name": options.image,
                "io.motlie.contract.version": validation.contract_version,
                "io.motlie.source.image_index_digest": validation.source.image_index_digest.as_ref().map(ToString::to_string).unwrap_or_default(),
                "io.motlie.source.platform_manifest_digest": validation.source.platform_manifest_digest.as_ref().map(ToString::to_string).unwrap_or_default(),
                "io.motlie.input.layout": layout.display().to_string(),
                "io.motlie.input.manifest": manifest_path.display().to_string(),
            }
        }));
    }
    manifests.sort_by(|left, right| {
        left["platform"]["architecture"]
            .as_str()
            .cmp(&right["platform"]["architecture"].as_str())
    });
    let index_value = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": OCI_IMAGE_INDEX_MEDIA_TYPE,
        "manifests": manifests,
        "annotations": {
            "org.opencontainers.image.ref.name": options.image,
            "io.motlie.contract.version": v1_5::MOTLIE_V15_CONTRACT_VERSION,
        }
    });
    let index_bytes = serde_json::to_vec_pretty(&index_value)?;
    fs::write(options.out.join("index.json"), &index_bytes)?;
    let index_manifest = OciMultiArchIndexManifest {
        image: options.image,
        output_dir: options.out.clone(),
        oci_layout_version: OCI_LAYOUT_VERSION.to_string(),
        image_index: OciBlobDescriptor {
            media_type: OCI_IMAGE_INDEX_MEDIA_TYPE.to_string(),
            digest: format!("sha256:{}", sha256_bytes(&index_bytes)),
            size_bytes: index_bytes.len() as u64,
            path: Some(PathBuf::from("index.json")),
        },
        platforms: platform_sources,
        created_at_unix_seconds: unix_now()?,
    };
    let manifest_path = options.out.join(DEFAULT_OCI_INDEX_MANIFEST_NAME);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&index_manifest)?)?;
    info!(path = %manifest_path.display(), "wrote mbuild OCI multi-arch index manifest");
    Ok(())
}

#[instrument(skip(options), fields(layout = %options.layout.display(), image = %options.image, dry_run = options.dry_run))]
fn oci_push(options: OciPushOptions) -> Result<()> {
    let image_ref = OciImageReference::from_str(&options.image)?;
    let publish_reference = match &image_ref.reference {
        OciImageReferenceKind::Tag(tag) => tag.clone(),
        OciImageReferenceKind::Digest(_) => {
            bail!("mbuild oci push requires a tag destination, not a digest reference")
        }
    };
    let plan = build_oci_push_plan(&options.layout, image_ref.clone())?;
    let out = options
        .out
        .clone()
        .unwrap_or_else(|| options.layout.join(DEFAULT_OCI_PUSH_MANIFEST_NAME));
    info!(
        layout = %options.layout.display(),
        image = %image_ref,
        blobs = plan.blobs.len(),
        manifests = plan.manifests.len(),
        dry_run = options.dry_run,
        "prepared native OCI registry push plan"
    );

    if options.dry_run {
        let evidence = OciPushEvidence::dry_run(&plan, &out)?;
        write_oci_push_evidence(&out, &evidence)?;
        return Ok(());
    }

    let auth = resolve_registry_auth(
        &image_ref,
        options.username.as_deref(),
        options.password_env.as_deref(),
        options.token_env.as_deref(),
    )?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("create OCI registry push runtime")?;
    let evidence = runtime.block_on(async {
        let client = OciRegistryClient::new();
        if !options.allow_overwrite {
            match client
                .fetch_manifest_digest_with_auth(&image_ref, &auth)
                .await
            {
                Ok(existing) => bail!(
                    "destination tag {} already exists with digest {}; rerun with --allow-overwrite to replace it",
                    image_ref,
                    existing
                ),
                Err(OciRegistryError::RegistryStatus { status: 404, .. }) => {}
                Err(error) => return Err(error.into()),
            }
        }

        let mut blob_records = Vec::with_capacity(plan.blobs.len());
        for descriptor in &plan.blobs {
            let digest = OciDigest::new(descriptor.digest.clone())?;
            let path = descriptor_path(&plan.layout, descriptor)?;
            let status = client
                .push_blob_from_path(
                    &image_ref,
                    &digest,
                    descriptor.size_bytes,
                    &path,
                    &auth,
                )
                .await?;
            blob_records.push(OciPushedBlobRecord {
                descriptor: descriptor.clone(),
                status,
            });
        }

        let mut manifest_records = Vec::with_capacity(plan.manifests.len() + 1);
        for descriptor in &plan.manifests {
            let path = descriptor_path(&plan.layout, descriptor)?;
            let bytes = tokio::fs::read(&path).await.with_context(|| {
                format!("read OCI manifest blob for push {}", path.display())
            })?;
            let result = client
                .push_manifest_bytes(
                    &image_ref,
                    &descriptor.digest,
                    &descriptor.media_type,
                    bytes,
                    &auth,
                )
                .await?;
            if result.computed_digest.as_ref() != descriptor.digest {
                bail!(
                    "pushed manifest digest mismatch for {}: computed {}, descriptor {}",
                    path.display(),
                    result.computed_digest,
                    descriptor.digest
                );
            }
            manifest_records.push(OciPushedManifestRecord {
                descriptor: descriptor.clone(),
                reference: descriptor.digest.clone(),
                registry_digest: result.registry_digest.map(|digest| digest.to_string()),
            });
        }

        let index_path = descriptor_path(&plan.layout, &plan.image_index)?;
        let index_bytes = tokio::fs::read(&index_path).await.with_context(|| {
            format!("read OCI index blob for push {}", index_path.display())
        })?;
        let index_result = client
            .push_manifest_bytes(
                &image_ref,
                &publish_reference,
                &plan.image_index.media_type,
                index_bytes,
                &auth,
            )
            .await?;
        if index_result.computed_digest.as_ref() != plan.image_index.digest {
            bail!(
                "pushed index digest mismatch for {}: computed {}, descriptor {}",
                index_path.display(),
                index_result.computed_digest,
                plan.image_index.digest
            );
        }
        manifest_records.push(OciPushedManifestRecord {
            descriptor: plan.image_index.clone(),
            reference: publish_reference.clone(),
            registry_digest: index_result.registry_digest.map(|digest| digest.to_string()),
        });

        let remote_digest = client
            .fetch_manifest_digest_with_auth(&image_ref, &auth)
            .await?;
        if remote_digest.as_ref() != plan.image_index.digest {
            bail!(
                "remote tag {} resolved to digest {}, expected {}",
                image_ref,
                remote_digest,
                plan.image_index.digest
            );
        }

        OciPushEvidence::from_push(&plan, blob_records, manifest_records, &remote_digest, &out)
    })?;
    write_oci_push_evidence(&out, &evidence)?;
    Ok(())
}

#[instrument(skip(options), fields(config = %options.config_path.display(), artifact = %options.artifact.display(), layout = %options.layout.display()))]
fn oci_release_evidence(options: OciReleaseEvidenceOptions) -> Result<()> {
    let config = load_config(&options.config_path)?;
    let build_manifest = load_build_manifest(&options.artifact)?;
    build_manifest.validate_against_config(&config)?;
    let export_manifest = load_oci_export_manifest(&options.layout)?;
    let validation = validate_oci_layout(
        &options.layout,
        &export_manifest,
        Some(&config),
        Some(&build_manifest),
    )?;
    let evidence = ReleaseArtifactEvidence::from_records(
        &build_manifest,
        &validation,
        options.publish_ref.as_deref(),
    )?;
    let out = options
        .out
        .unwrap_or_else(|| options.layout.join(DEFAULT_RELEASE_EVIDENCE_NAME));
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_vec_pretty(&evidence)?)?;
    info!(path = %out.display(), "wrote VM image artifact release evidence");
    Ok(())
}

#[instrument(fields(path = %path.display()))]
fn load_config(path: &Path) -> Result<ImageBuildConfig> {
    let config: ImageBuildConfig = serde_yaml::from_slice(&fs::read(path)?)?;
    config.validate()?;
    Ok(config)
}

fn load_build_manifest(artifact: &Path) -> Result<ImageBuildManifest> {
    let manifest_path = artifact.join(DEFAULT_MANIFEST_NAME);
    serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("read build manifest {}", manifest_path.display()))
}

fn load_oci_export_manifest(layout: &Path) -> Result<OciExportManifest> {
    let manifest_path = layout.join(DEFAULT_OCI_EXPORT_MANIFEST_NAME);
    serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("read OCI export manifest {}", manifest_path.display()))
}

fn load_oci_index_manifest(layout: &Path) -> Result<OciMultiArchIndexManifest> {
    let manifest_path = layout.join(DEFAULT_OCI_INDEX_MANIFEST_NAME);
    serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("read OCI index manifest {}", manifest_path.display()))
}

#[derive(Debug, Clone)]
struct ValidatedOciPayload {
    export: OciExportManifest,
    source: ExternalOciSource,
    rootfs_tarball: RootfsTarballRecord,
}

fn load_validated_oci_payload(
    oci_layout: &Path,
    config: &ImageBuildConfig,
) -> Result<ValidatedOciPayload> {
    let export = load_oci_export_manifest(oci_layout)?;
    let export = validate_oci_layout(oci_layout, &export, Some(config), None)?;
    let source = external_source_from_manifest(&export.source)?;
    let rootfs_blob = descriptor_path(oci_layout, &export.rootfs_layer)?;
    let rootfs_tarball = RootfsTarballRecord {
        canonical_path: fs::canonicalize(&rootfs_blob)?,
        size_bytes: export.rootfs_layer.size_bytes,
        sha256: digest_hex(&export.rootfs_layer.digest)?.to_string(),
    };

    Ok(ValidatedOciPayload {
        export,
        source,
        rootfs_tarball,
    })
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
    if let Some(oci_layout) = options.oci_layout.as_deref() {
        if config.source.kind != SourceKind::ExternalOci {
            bail!("--oci-layout requires source.kind external-oci");
        }
        if rootfs_tarball.is_some() {
            bail!("--oci-layout and --rootfs-tarball are mutually exclusive");
        }
        if options.target.as_str() == "ch" {
            return run_ch_oci_layout_build(repo_root, config, emitter, options, oci_layout);
        }
        return run_backend_adapter_oci_layout_build(
            repo_root, config, emitter, options, oci_layout,
        );
    }
    if config.source.kind == SourceKind::ExternalOci && options.target.as_str() == "ch" {
        return run_ch_external_oci_build(repo_root, config, emitter, options);
    }
    if config.source.kind == SourceKind::ExternalOci
        && config.source.profile.as_str() == ALPINE_OPENRC_PROFILE
        && emitter.adapter.env.rootfs_tarball.is_some()
        && rootfs_tarball.is_none()
    {
        bail!(
            "external-oci Alpine target {} requires --oci-layout or --rootfs-tarball; \
             the VZ adapter cannot synthesize an Alpine/OpenRC rootfs from the native \
             source VM without an assembled payload",
            options.target
        );
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

    let host = HostPlatformTarget::detect()?;
    let guest = ChGuestTarget::from_platform(platform)?;
    append_log(
        &log_path,
        &format!(
            "builder_host={} guest_platform={} guest_rust_target={} guest_deb_arch={}\n",
            host.platform, guest.platform, guest.rust_target, guest.deb_arch
        ),
    )?;
    preflight_guest_chroot_execution(&host, &guest, &log_path)?;

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

    let chroot_support = prepare_guest_chroot_execution(&rootfs_dir, &host, &guest, &log_path)?;
    let guest_binaries = build_guest_binaries(repo_root, &guest, &log_path)?;
    run_package_stage(&rootfs_dir, config, &guest, &options.out)?;

    let assembly_manifest =
        assemble_immutable_rootfs(&rootfs_dir, config, &source, &guest_binaries)?;
    fs::write(
        options.out.join("mbuild-rootfs-assembly.json"),
        serde_json::to_vec_pretty(&assembly_manifest)?,
    )?;
    verify_guest_binary_contract_marker(&rootfs_dir, &log_path)?;
    cleanup_guest_chroot_execution(&rootfs_dir, chroot_support.as_ref(), &log_path)?;
    let common_rootfs_tarball = emit_rootfs_tarball(
        &rootfs_dir,
        &options.out.join(COMMON_ROOTFS_TARBALL_NAME),
        &log_path,
    )?;
    write_common_rootfs_record(&options.out, &source, &common_rootfs_tarball)?;

    install_ch_boot_adaptations(repo_root, &rootfs_dir, &source, &guest, &guest_binaries.vfs)?;
    write_ch_guest_contract(
        &rootfs_dir.join("opt/motlie/v1.5/guest/guest-contract.json"),
        &source,
        &guest,
        Path::new(guest.kernel_image),
        &guest_binaries.vfs,
    )?;
    let rootfs_path = base_dir.join(CH_ROOTFS_NAME);
    emit_ch_rootfs_image(
        &rootfs_dir,
        &rootfs_path,
        &options.out.join(CH_EMITTER_LOG_NAME),
    )?;
    let kernel_path = emit_ch_kernel(&base_dir, &guest, &log_path)?;
    let contract_path = base_dir.join(CH_GUEST_CONTRACT_NAME);
    write_ch_guest_contract(
        &contract_path,
        &source,
        &guest,
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
        build_host: Some(BuildHostRecord::from(&host)),
        guest_target: Some(GuestTargetRecord::from(&guest)),
        rootfs_tarball: Some(common_rootfs_tarball),
    })
}

#[instrument(
    skip(repo_root, config, _emitter, options, oci_layout),
    fields(target = %options.target, out = %options.out.display(), layout = %oci_layout.display())
)]
fn run_ch_oci_layout_build(
    repo_root: &Path,
    config: &ImageBuildConfig,
    _emitter: &BackendEmitterSpec,
    options: &BuildOptions,
    oci_layout: &Path,
) -> Result<AdapterRecord> {
    let started_at_unix_seconds = unix_now()?;
    let log_path = options.out.join(CH_EMITTER_LOG_NAME);
    fs::write(&log_path, b"")?;
    append_log(&log_path, "=== mbuild CH OCI-layout builder ===\n")?;

    let payload = load_validated_oci_payload(oci_layout, config)?;
    let validation = &payload.export;
    let source = payload.source.clone();
    let platform = parse_source_platform(&validation.platform)?;
    let host = HostPlatformTarget::detect()?;
    let guest = ChGuestTarget::from_platform(platform)?;
    append_log(
        &log_path,
        &format!(
            "oci_layout={} tag={} platform={} image_index={} manifest={}\n",
            oci_layout.display(),
            validation.tag,
            source.platform,
            source.image_index_digest,
            source.platform_manifest_digest
        ),
    )?;
    append_log(
        &log_path,
        &format!(
            "builder_host={} guest_platform={} guest_rust_target={} guest_deb_arch={}\n",
            host.platform, guest.platform, guest.rust_target, guest.deb_arch
        ),
    )?;

    let work_root = create_work_root("mbuild-ch-oci-rootfs")?;
    let rootfs_dir = work_root.join("rootfs");
    fs::create_dir_all(&rootfs_dir)?;
    let rootfs_blob = &payload.rootfs_tarball.canonical_path;
    extract_rootfs_tarball(rootfs_blob, &rootfs_dir, &log_path)?;

    let base_dir = options.out.join("base");
    fs::create_dir_all(&base_dir)?;
    let rootfs_record = payload.rootfs_tarball.clone();
    write_common_rootfs_record(&options.out, &source, &rootfs_record)?;

    let guest_binary = rootfs_dir.join(
        v1_5::MOTLIE_V15_GUEST_BIN_OPT
            .strip_prefix('/')
            .unwrap_or(v1_5::MOTLIE_V15_GUEST_BIN_OPT),
    );
    if !guest_binary.is_file() {
        bail!(
            "OCI payload does not contain required guest binary {}",
            v1_5::MOTLIE_V15_GUEST_BIN_OPT
        );
    }
    install_ch_boot_adaptations(repo_root, &rootfs_dir, &source, &guest, &guest_binary)?;
    write_ch_guest_contract(
        &rootfs_dir.join("opt/motlie/v1.5/guest/guest-contract.json"),
        &source,
        &guest,
        Path::new(guest.kernel_image),
        &guest_binary,
    )?;
    let rootfs_path = base_dir.join(CH_ROOTFS_NAME);
    emit_ch_rootfs_image(
        &rootfs_dir,
        &rootfs_path,
        &options.out.join(CH_EMITTER_LOG_NAME),
    )?;
    let kernel_path = emit_ch_kernel(&base_dir, &guest, &log_path)?;
    let contract_path = base_dir.join(CH_GUEST_CONTRACT_NAME);
    write_ch_guest_contract(&contract_path, &source, &guest, &kernel_path, &guest_binary)?;
    write_ch_build_result(
        &options.out,
        &contract_path,
        &kernel_path,
        &rootfs_path,
        &guest_binary,
    )?;

    let completed_at_unix_seconds = unix_now()?;
    append_log(&log_path, "=== CH OCI-layout build complete ===\n")?;
    let _ = fs::remove_dir_all(&work_root);

    Ok(AdapterRecord {
        kind: "oci-payload-ch-emitter".to_string(),
        command: vec![
            "mbuild".to_string(),
            "build".to_string(),
            "--target".to_string(),
            options.target.to_string(),
            "--oci-layout".to_string(),
            oci_layout.display().to_string(),
        ],
        log_path,
        exit_status: 0,
        started_at_unix_seconds,
        completed_at_unix_seconds,
        package_include: config.package_stage.install.clone(),
        materialized_source: None,
        external_oci_source: Some(source),
        build_host: Some(BuildHostRecord::from(&host)),
        guest_target: Some(GuestTargetRecord::from(&guest)),
        rootfs_tarball: Some(rootfs_record),
    })
}

#[instrument(
    skip(repo_root, config, emitter, options, oci_layout),
    fields(target = %options.target, out = %options.out.display(), layout = %oci_layout.display())
)]
fn run_backend_adapter_oci_layout_build(
    repo_root: &Path,
    config: &ImageBuildConfig,
    emitter: &BackendEmitterSpec,
    options: &BuildOptions,
    oci_layout: &Path,
) -> Result<AdapterRecord> {
    if emitter.adapter.env.rootfs_tarball.is_none() {
        bail!(
            "target {} cannot consume --oci-layout because it does not declare an adapter rootfs_tarball env",
            options.target
        );
    }

    let payload = load_validated_oci_payload(oci_layout, config)?;
    let mut adapter = run_backend_adapter(
        repo_root,
        config,
        emitter,
        options,
        Some(&payload.rootfs_tarball),
    )?;
    adapter.kind = format!("oci-payload-{}-adapter", options.target);
    adapter.external_oci_source = Some(payload.source);
    Ok(adapter)
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
        build_host: None,
        guest_target: None,
        rootfs_tarball: rootfs_tarball.cloned(),
    })
}

#[derive(Debug, Clone)]
struct HostPlatformTarget {
    platform: OciPlatform,
    deb_arch: &'static str,
}

impl HostPlatformTarget {
    fn detect() -> Result<Self> {
        match env::consts::ARCH {
            "x86_64" => Ok(Self {
                platform: OciPlatform::linux_amd64(),
                deb_arch: "amd64",
            }),
            "aarch64" => Ok(Self {
                platform: OciPlatform::linux_arm64(),
                deb_arch: "arm64",
            }),
            other => bail!("unsupported CH build host architecture: {other}"),
        }
    }
}

#[derive(Debug, Clone)]
struct ChGuestTarget {
    platform: OciPlatform,
    rust_target: &'static str,
    deb_arch: &'static str,
    kernel_image: &'static str,
    kernel_asset: &'static str,
    qemu_binfmt_name: &'static str,
    qemu_static_binary: &'static str,
}

impl ChGuestTarget {
    fn from_platform(platform: OciPlatform) -> Result<Self> {
        match platform.architecture {
            GuestArchitecture::Amd64 => Ok(Self {
                platform: OciPlatform::linux_amd64(),
                rust_target: "x86_64-unknown-linux-musl",
                deb_arch: "amd64",
                kernel_image: "vmlinux.bin",
                kernel_asset: "bzImage-x86_64",
                qemu_binfmt_name: "qemu-x86_64",
                qemu_static_binary: "qemu-x86_64-static",
            }),
            GuestArchitecture::Arm64 => Ok(Self {
                platform: OciPlatform::linux_arm64(),
                rust_target: "aarch64-unknown-linux-musl",
                deb_arch: "arm64",
                kernel_image: "Image",
                kernel_asset: "Image-arm64",
                qemu_binfmt_name: "qemu-aarch64",
                qemu_static_binary: "qemu-aarch64-static",
            }),
        }
    }

    fn apk_arch(&self) -> &'static str {
        match self.platform.architecture {
            GuestArchitecture::Amd64 => "x86_64",
            GuestArchitecture::Arm64 => "aarch64",
        }
    }

    fn npm_arch(&self) -> &'static str {
        match self.platform.architecture {
            GuestArchitecture::Amd64 => "x64",
            GuestArchitecture::Arm64 => "arm64",
        }
    }
}

fn parse_source_platform(value: &str) -> Result<OciPlatform> {
    match value {
        "linux/amd64" | "amd64" => Ok(OciPlatform::linux_amd64()),
        "linux/arm64" | "arm64" => Ok(OciPlatform::linux_arm64()),
        "host-native" | "host-native-linux" => Ok(HostPlatformTarget::detect()?.platform),
        other => bail!(
            "unsupported source.platform {other:?}; expected linux/amd64, linux/arm64, or host-native-linux"
        ),
    }
}

fn resolve_external_source(
    config: &ImageBuildConfig,
    platform: OciPlatform,
) -> Result<ExternalOciSource> {
    match config.source.profile.as_str() {
        UBUNTU_SYSTEMD_PROFILE => require_eq(
            "source.image for ubuntu-systemd",
            &config.source.image,
            UBUNTU_SYSTEMD_SOURCE_REF,
        )?,
        ALPINE_OPENRC_PROFILE => require_eq(
            "source.image for alpine-openrc",
            &config.source.image,
            ALPINE_OPENRC_SOURCE_REF,
        )?,
        profile => bail!("unsupported source.profile {profile:?}"),
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
    guest: &ChGuestTarget,
    log_path: &Path,
) -> Result<GuestBinaries> {
    let target_dir = repo_root
        .join("target")
        .join(guest.rust_target)
        .join("release");
    let vfs = target_dir.join("motlie-vfs-guest-v1_5");
    let ssh_bridge = target_dir.join("motlie-vsock-ssh-bridge-v1_5");
    let rust_lld = rust_lld_path()?;
    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--manifest-path")
        .arg(repo_root.join("libs/vmm/Cargo.toml"))
        .arg("--release")
        .arg("--target")
        .arg(guest.rust_target)
        .arg("--no-default-features")
        .arg("--features")
        .arg("guest-vfs")
        .arg("--bin")
        .arg("motlie-vfs-guest-v1_5")
        .arg("--bin")
        .arg("motlie-vsock-ssh-bridge-v1_5")
        .current_dir(repo_root)
        .env(
            cargo_target_linker_env(guest.rust_target),
            rust_lld.as_os_str(),
        );
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
    Ok(GuestBinaries { vfs, ssh_bridge })
}

fn rust_lld_path() -> Result<PathBuf> {
    let output = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .context("locate Rust sysroot for rust-lld")?;
    if !output.status.success() {
        bail!(
            "rustc --print sysroot failed with status {}",
            output.status.code().unwrap_or(-1)
        );
    }
    let sysroot =
        String::from_utf8(output.stdout).context("rustc --print sysroot returned non-UTF-8")?;
    let host_triple = rust_host_triple()?;
    let path = PathBuf::from(sysroot.trim())
        .join("lib/rustlib")
        .join(host_triple)
        .join("bin/rust-lld");
    if path.is_file() {
        return Ok(path);
    }
    if let Some(path) = find_executable("rust-lld") {
        return Ok(path);
    }
    bail!("rust-lld was not found in the active Rust toolchain; install the rust-lld component")
}

fn rust_host_triple() -> Result<String> {
    let output = Command::new("rustc")
        .arg("-vV")
        .output()
        .context("locate Rust host triple")?;
    if !output.status.success() {
        bail!(
            "rustc -vV failed with status {}",
            output.status.code().unwrap_or(-1)
        );
    }
    let stdout = String::from_utf8(output.stdout).context("rustc -vV returned non-UTF-8")?;
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(ToOwned::to_owned)
        .context("rustc -vV did not report a host triple")
}

fn cargo_target_linker_env(rust_target: &str) -> String {
    format!(
        "CARGO_TARGET_{}_LINKER",
        rust_target.replace('-', "_").to_ascii_uppercase()
    )
}

#[derive(Debug, Clone)]
struct GuestChrootSupport {
    copied_qemu_guest_path: Option<PathBuf>,
}

fn preflight_guest_chroot_execution(
    host: &HostPlatformTarget,
    guest: &ChGuestTarget,
    log_path: &Path,
) -> Result<()> {
    if host.platform == guest.platform {
        return Ok(());
    }
    let qemu = cross_chroot_qemu_path(host, guest)?;
    append_log(
        log_path,
        &format!(
            "cross-arch chroot preflight passed: host={} guest={} qemu={} binfmt={}\n",
            host.platform,
            guest.platform,
            qemu.display(),
            guest.qemu_binfmt_name
        ),
    )?;
    Ok(())
}

fn prepare_guest_chroot_execution(
    rootfs_dir: &Path,
    host: &HostPlatformTarget,
    guest: &ChGuestTarget,
    log_path: &Path,
) -> Result<Option<GuestChrootSupport>> {
    if host.platform == guest.platform {
        append_log(
            log_path,
            &format!(
                "guest chroot execution is native: host={} guest={}\n",
                host.platform, guest.platform
            ),
        )?;
        return Ok(None);
    }
    let qemu = cross_chroot_qemu_path(host, guest)?;
    let guest_qemu_path = rootfs_dir.join("usr/bin").join(guest.qemu_static_binary);
    let copied_qemu_guest_path = if guest_qemu_path.exists() {
        None
    } else {
        if let Some(parent) = guest_qemu_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&qemu, &guest_qemu_path).with_context(|| {
            format!(
                "copy {} into guest rootfs at {}",
                qemu.display(),
                guest_qemu_path.display()
            )
        })?;
        set_mode(&guest_qemu_path, 0o755)?;
        Some(guest_qemu_path)
    };
    append_log(
        log_path,
        &format!(
            "guest chroot execution is cross-arch: host={} guest={} qemu={} binfmt={}\n",
            host.platform,
            guest.platform,
            qemu.display(),
            guest.qemu_binfmt_name
        ),
    )?;
    Ok(Some(GuestChrootSupport {
        copied_qemu_guest_path,
    }))
}

fn cross_chroot_qemu_path(host: &HostPlatformTarget, guest: &ChGuestTarget) -> Result<PathBuf> {
    if env::var_os(CROSS_ARCH_CHROOT_ENV).is_some_and(|value| value == "0") {
        bail!(
            "cross-architecture CH package staging is disabled by {CROSS_ARCH_CHROOT_ENV}=0; host={} guest={}",
            host.platform,
            guest.platform
        );
    }

    ensure_binfmt_enabled(guest)?;
    find_executable(guest.qemu_static_binary).with_context(|| {
        format!(
            "cross-architecture CH package staging requires {}; install qemu-user-static and enable binfmt for {}",
            guest.qemu_static_binary, guest.qemu_binfmt_name
        )
    })
}

fn cleanup_guest_chroot_execution(
    _rootfs_dir: &Path,
    support: Option<&GuestChrootSupport>,
    log_path: &Path,
) -> Result<()> {
    if let Some(path) = support.and_then(|support| support.copied_qemu_guest_path.as_ref()) {
        fs::remove_file(path)
            .with_context(|| format!("remove build-time qemu helper {}", path.display()))?;
        append_log(
            log_path,
            &format!("removed build-time qemu helper {}\n", path.display()),
        )?;
    }
    Ok(())
}

fn ensure_binfmt_enabled(guest: &ChGuestTarget) -> Result<()> {
    let entry = Path::new("/proc/sys/fs/binfmt_misc").join(guest.qemu_binfmt_name);
    let contents = fs::read_to_string(&entry).with_context(|| {
        format!(
            "cross-architecture CH package staging requires enabled binfmt entry {}; install qemu-user-static/binfmt-support or run the build on native {} hardware",
            entry.display(),
            guest.platform
        )
    })?;
    if !contents.lines().any(|line| line.trim() == "enabled") {
        bail!(
            "binfmt entry {} is not enabled; enable {} before cross-architecture CH package staging",
            entry.display(),
            guest.qemu_binfmt_name
        );
    }
    Ok(())
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let candidate = Path::new(name);
    if candidate.components().count() > 1 && candidate.is_file() {
        return Some(candidate.to_path_buf());
    }
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join(name))
            .find(|candidate| candidate.is_file())
    })
}

fn verify_guest_binary_contract_marker(rootfs_dir: &Path, log_path: &Path) -> Result<()> {
    let mut command = rootless_unshare_command()?;
    command
        .arg("chroot")
        .arg(rootfs_dir)
        .arg(v1_5::MOTLIE_V15_GUEST_BIN_OPT)
        .arg("--contract");
    append_log(
        log_path,
        &format!("\n--- verify installed guest binary contract marker ---\ncommand={command:?}\n"),
    )?;
    let output = command
        .output()
        .context("run guest binary contract check")?;
    append_log(
        log_path,
        &format!(
            "stdout={}\nstderr={}\n",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )?;
    if !output.status.success() {
        bail!(
            "guest binary contract check failed with status {}; see {}",
            output.status.code().unwrap_or(-1),
            log_path.display()
        );
    }
    let marker_stdout = String::from_utf8(output.stdout)
        .context("guest binary contract marker output is not UTF-8")?;
    if marker_stdout.trim() != v1_5::MOTLIE_V15_GUEST_MOUNTER_MARKER {
        bail!(
            "guest binary contract marker mismatch: expected {}, got {:?}",
            v1_5::MOTLIE_V15_GUEST_MOUNTER_MARKER,
            marker_stdout.trim()
        );
    }
    Ok(())
}

fn rootless_unshare_command() -> Result<Command> {
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
        .arg("--fork");
    Ok(command)
}

fn run_package_stage(
    rootfs_dir: &Path,
    config: &ImageBuildConfig,
    guest: &ChGuestTarget,
    out: &Path,
) -> Result<()> {
    let script_path = out.join(CH_PACKAGE_STAGE_SCRIPT_NAME);
    let log_path = out.join(CH_PACKAGE_STAGE_LOG_NAME);
    let stage_script = match config.package_stage.manager.as_str() {
        "apt" => apt_stage_script(),
        "apk" => apk_stage_script(),
        manager => bail!("CH external-OCI package stage does not implement {manager}"),
    };
    fs::write(&script_path, stage_script)?;
    set_mode(&script_path, 0o755)?;

    let mut command = rootless_unshare_command()?;
    command
        .arg("--mount-proc")
        .arg("bash")
        .arg(&script_path)
        .arg(rootfs_dir)
        .arg(bool_env(config.package_stage.update))
        .arg(bool_env(config.package_stage.clean));
    command.env("MOTLIE_MBUILD_GUEST_APK_ARCH", guest.apk_arch());
    command.env("MOTLIE_MBUILD_GUEST_NPM_ARCH", guest.npm_arch());
    command.env("MOTLIE_MBUILD_GUEST_PLATFORM", guest.platform.to_string());
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
    run_logged_command(
        &mut command,
        &log_path,
        &format!(
            "{}/npm package stage",
            config.package_stage.manager.as_str()
        ),
    )
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
    rm -f "$rootfs/tmp/mbuild-host-ca-certificates.crt" >/dev/null 2>&1 || true
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
    npm_env=(npm_config_update_notifier=false)
    if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
        cp /etc/ssl/certs/ca-certificates.crt "$rootfs/tmp/mbuild-host-ca-certificates.crt"
        npm_env+=(NODE_EXTRA_CA_CERTS=/tmp/mbuild-host-ca-certificates.crt npm_config_cafile=/tmp/mbuild-host-ca-certificates.crt)
    fi
    chroot "$rootfs" env "${npm_env[@]}" npm install -g --prefix /opt/motlie/npm "${npm_packages[@]}"
    rm -f "$rootfs/tmp/mbuild-host-ca-certificates.crt"
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

fn apk_stage_script() -> &'static str {
    r#"#!/usr/bin/env bash
set -euo pipefail
rootfs="$(readlink -f "$1")"
update="$2"
clean="$3"
shift 3

apk_packages=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    apk_packages+=("$1")
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

apk_arch="${MOTLIE_MBUILD_GUEST_APK_ARCH:-}"
if [[ -z "$apk_arch" && -f "$rootfs/etc/apk/arch" ]]; then
    apk_arch="$(cat "$rootfs/etc/apk/arch")"
fi
npm_arch="${MOTLIE_MBUILD_GUEST_NPM_ARCH:-}"
case "$apk_arch" in
    x86_64) [[ -n "$npm_arch" ]] || npm_arch="x64" ;;
    aarch64) [[ -n "$npm_arch" ]] || npm_arch="arm64" ;;
    *) echo "ERROR: unsupported or unknown Alpine guest apk arch: ${apk_arch:-<empty>}" >&2; exit 1 ;;
esac
case "$npm_arch" in
    x64|arm64) ;;
    *) echo "ERROR: unsupported or unknown guest npm arch: ${npm_arch:-<empty>}" >&2; exit 1 ;;
esac

find_apk_static() {
    if [[ -n "${MOTLIE_MBUILD_APK_STATIC:-}" ]]; then
        if [[ -x "$MOTLIE_MBUILD_APK_STATIC" ]]; then
            printf '%s\n' "$MOTLIE_MBUILD_APK_STATIC"
            return 0
        fi
        echo "ERROR: MOTLIE_MBUILD_APK_STATIC is not executable: $MOTLIE_MBUILD_APK_STATIC" >&2
        exit 1
    fi
    local candidate
    for candidate in apk.static /sbin/apk.static /usr/sbin/apk.static /usr/local/sbin/apk.static; do
        if [[ "$candidate" == */* ]]; then
            if [[ -x "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        elif command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    echo "ERROR: apk.static not found. Install apk-tools-static or set MOTLIE_MBUILD_APK_STATIC." >&2
    exit 1
}

restore_apk_recorded_modes() {
    local db="$rootfs/lib/apk/db/installed"
    [[ -f "$db" ]] || return 0
    local line field value dir="" file="" uid="" gid="" mode="" path=""
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -n "$line" && "$line" == *:* ]] || continue
        field="${line%%:*}"
        value="${line#*:}"
        case "$field" in
            F)
                dir="$value"
                file=""
                ;;
            M)
                IFS=: read -r uid gid mode <<< "$value"
                path="$rootfs/$dir"
                if [[ -n "$dir" && -n "$mode" && -e "$path" && ! -L "$path" ]]; then
                    chmod "$mode" "$path" || true
                fi
                ;;
            R)
                file="$value"
                ;;
            a)
                IFS=: read -r uid gid mode <<< "$value"
                path="$rootfs/$dir/$file"
                if [[ -n "$dir" && -n "$file" && -n "$mode" && -e "$path" && ! -L "$path" ]]; then
                    chmod "$mode" "$path" || true
                fi
                ;;
        esac
    done < "$db"
}

repair_claude_code_binary() {
    local claude_root="$rootfs/opt/motlie/npm/lib/node_modules/@anthropic-ai/claude-code"
    local optional_binary="$claude_root/node_modules/@anthropic-ai/claude-code-linux-${npm_arch}-musl/bin/claude"
    local wrapper="$claude_root/bin/claude.exe"
    if [[ -x "$optional_binary" && -e "$wrapper" ]]; then
        ln -sf "../node_modules/@anthropic-ai/claude-code-linux-${npm_arch}-musl/bin/claude" "$wrapper"
    fi
}

chroot_env=(env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
apk_static="$(find_apk_static)"
apk_root=("$apk_static" --root "$rootfs" --arch "$apk_arch" --keys-dir "$rootfs/etc/apk/keys" --repositories-file "$rootfs/etc/apk/repositories")

mkdir -p "$rootfs/proc" "$rootfs/sys" "$rootfs/dev" "$rootfs/etc" "$rootfs/tmp"
mount --bind /proc "$rootfs/proc"
mount --rbind /sys "$rootfs/sys"
mount --rbind /dev "$rootfs/dev"
cleanup() {
    rm -f "$rootfs/tmp/mbuild-host-ca-certificates.crt" >/dev/null 2>&1 || true
    umount -l "$rootfs/dev" >/dev/null 2>&1 || true
    umount -l "$rootfs/sys" >/dev/null 2>&1 || true
    umount -l "$rootfs/proc" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ -f /etc/resolv.conf ]]; then
    cp /etc/resolv.conf "$rootfs/etc/resolv.conf"
fi
if [[ -f "$rootfs/etc/apk/repositories" ]]; then
    sed -i 's#^https://dl-cdn.alpinelinux.org/#http://dl-cdn.alpinelinux.org/#' "$rootfs/etc/apk/repositories"
fi

bootstrap_packages=()
remaining_packages=()
for package in "${apk_packages[@]}"; do
    case "$package" in
        ca-certificates|ca-certificates-bundle) bootstrap_packages+=("$package") ;;
        *) remaining_packages+=("$package") ;;
    esac
done

if [[ ${#bootstrap_packages[@]} -gt 0 ]]; then
    "${apk_root[@]}" --no-check-certificate add --no-cache "${bootstrap_packages[@]}"
    chroot "$rootfs" "${chroot_env[@]}" update-ca-certificates >/dev/null 2>&1 || true
fi
if [[ "$update" == "1" ]]; then
    "${apk_root[@]}" update
fi
if [[ ${#remaining_packages[@]} -gt 0 ]]; then
    "${apk_root[@]}" add --no-cache "${remaining_packages[@]}"
fi
restore_apk_recorded_modes

mkdir -p "$rootfs/opt/motlie/npm"
if [[ ${#npm_packages[@]} -gt 0 ]]; then
    npm_env=(
        npm_config_update_notifier=false
        npm_config_platform=linux
        npm_config_arch="$npm_arch"
        npm_config_libc=musl
        npm_config_include=optional
        npm_config_optional=true
    )
    if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
        cp /etc/ssl/certs/ca-certificates.crt "$rootfs/tmp/mbuild-host-ca-certificates.crt"
        npm_env+=(NODE_EXTRA_CA_CERTS=/tmp/mbuild-host-ca-certificates.crt npm_config_cafile=/tmp/mbuild-host-ca-certificates.crt)
    fi
    chroot "$rootfs" "${chroot_env[@]}" "${npm_env[@]}" npm install -g --prefix /opt/motlie/npm "${npm_packages[@]}"
    rm -f "$rootfs/tmp/mbuild-host-ca-certificates.crt"
    repair_claude_code_binary
    mkdir -p "$rootfs/usr/local/bin"
    for binary in "${npm_binaries[@]}"; do
        chroot "$rootfs" "${chroot_env[@]}" ln -sf "/opt/motlie/npm/bin/$binary" "/usr/local/bin/$binary"
    done
fi

chroot "$rootfs" "${chroot_env[@]}" ssh-keygen -A
sed -i 's/#PermitRootLogin.*/PermitRootLogin prohibit-password/' "$rootfs/etc/ssh/sshd_config" || true
grep -qxF 'Include /etc/ssh/sshd_config.d/*.conf' "$rootfs/etc/ssh/sshd_config" 2>/dev/null || printf '\nInclude /etc/ssh/sshd_config.d/*.conf\n' >> "$rootfs/etc/ssh/sshd_config"
if [[ -x "$rootfs/sbin/rc-update" ]]; then
    mkdir -p "$rootfs/etc/runlevels/sysinit" "$rootfs/etc/runlevels/boot" "$rootfs/etc/runlevels/default"
    grep -qxF 'rc_logger="YES"' "$rootfs/etc/rc.conf" 2>/dev/null || printf '\nrc_logger="YES"\n' >> "$rootfs/etc/rc.conf"
    grep -qxF 'rc_verbose=yes' "$rootfs/etc/rc.conf" 2>/dev/null || printf 'rc_verbose=yes\n' >> "$rootfs/etc/rc.conf"
    for entry in \
        "devfs sysinit" \
        "procfs sysinit" \
        "sysfs sysinit" \
        "hostname boot" \
        "bootmisc boot" \
        "root boot" \
        "fsck boot" \
        "localmount boot" \
        "loopback boot" \
        "networking boot"; do
        service="${entry% *}"
        runlevel="${entry#* }"
        if [[ -f "$rootfs/etc/init.d/$service" ]]; then
            chroot "$rootfs" "${chroot_env[@]}" rc-update add "$service" "$runlevel"
        fi
    done
    for service in sshd dbus cloud-init-local cloud-init cloud-config cloud-final local; do
        if [[ -f "$rootfs/etc/init.d/$service" ]]; then
            chroot "$rootfs" "${chroot_env[@]}" rc-update add "$service" default
        fi
    done
fi
grep -qxF user_allow_other "$rootfs/etc/fuse.conf" 2>/dev/null || printf '%s\n' user_allow_other >> "$rootfs/etc/fuse.conf"
if [[ "$clean" == "1" ]]; then
    rm -rf "$rootfs/var/cache/apk"/* "$rootfs/root/.npm"
fi
"#
}

fn assemble_immutable_rootfs(
    rootfs_dir: &Path,
    config: &ImageBuildConfig,
    source: &ExternalOciSource,
    guest_binaries: &GuestBinaries,
) -> Result<motlie_vmm::image::RootfsCompatibilityAssemblyManifest> {
    let mut profile = guest_image_profile_from_config(config, source.clone())?;
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

fn guest_image_profile_from_config(
    config: &ImageBuildConfig,
    source: ExternalOciSource,
) -> Result<GuestImageProfile> {
    match config.source.profile.as_str() {
        UBUNTU_SYSTEMD_PROFILE => Ok(GuestImageProfile::ubuntu_systemd(source)),
        ALPINE_OPENRC_PROFILE => Ok(GuestImageProfile::alpine_openrc(source)),
        profile => bail!("unsupported source.profile {profile:?}"),
    }
}

fn parse_octal_mode(value: &str) -> Result<u32> {
    u32::from_str_radix(value.trim_start_matches('0'), 8)
        .with_context(|| format!("invalid octal mode {value:?}"))
}

fn install_ch_boot_adaptations(
    repo_root: &Path,
    rootfs_dir: &Path,
    source: &ExternalOciSource,
    guest: &ChGuestTarget,
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
        ch_motd(source, guest).as_bytes(),
        0o644,
    )?;
    let _ = guest_binary;
    Ok(())
}

fn ch_motd(source: &ExternalOciSource, guest: &ChGuestTarget) -> String {
    format!(
        "motlie v1.5 CH guest\nsource={}\nplatform={}\narch={}\n",
        source.image_ref, source.platform, guest.deb_arch
    )
}

fn emit_ch_rootfs_image(rootfs_dir: &Path, rootfs_path: &Path, log_path: &Path) -> Result<()> {
    if let Some(parent) = rootfs_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let script = r#"set -euo pipefail
rm -f "$2"
size_bytes="$(du -sb "$1" | awk '{print $1}')"
overhead=$((size_bytes / 4))
min_overhead=$((128 * 1024 * 1024))
if (( overhead < min_overhead )); then
    overhead=$min_overhead
fi
image_bytes=$((size_bytes + overhead))
image_mib=$(((image_bytes + 1048575) / 1048576))
truncate -s "${image_mib}M" "$2"
fakeroot sh -c 'chown -h -R 0:0 "$1"; mkfs.ext4 -F -d "$1" "$2" -q' mbuild-ext4 "$1" "$2"
"#;
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(script)
        .arg("mbuild-ch-ext4")
        .arg(rootfs_dir)
        .arg(rootfs_path);
    run_logged_command(&mut command, log_path, "emit CH ext4 rootfs image")
}

fn emit_rootfs_tarball(
    rootfs_dir: &Path,
    tarball_path: &Path,
    log_path: &Path,
) -> Result<RootfsTarballRecord> {
    if let Some(parent) = tarball_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let script = "set -euo pipefail\nrm -f \"$2\"\ntar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C \"$1\" -cf \"$2\" .\n";
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(script)
        .arg("mbuild-rootfs-tarball")
        .arg(rootfs_dir)
        .arg(tarball_path);
    run_logged_command(&mut command, log_path, "emit common rootfs tarball")?;
    rootfs_tarball_record(tarball_path)
}

fn emit_ch_kernel(base_dir: &Path, guest: &ChGuestTarget, log_path: &Path) -> Result<PathBuf> {
    let kernel_path = base_dir.join(guest.kernel_image);
    if kernel_path.is_file() {
        return Ok(kernel_path);
    }
    let release =
        env::var("CH_KERNEL_RELEASE").unwrap_or_else(|_| CH_KERNEL_RELEASE_DEFAULT.to_string());
    let url = env::var("CH_KERNEL_URL").unwrap_or_else(|_| {
        format!(
            "https://github.com/cloud-hypervisor/linux/releases/download/{release}/{}",
            guest.kernel_asset
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
    guest: &ChGuestTarget,
    kernel_path: &Path,
    guest_binary: &Path,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let value = serde_json::json!({
        "contract_version": v1_5::MOTLIE_V15_CONTRACT_VERSION,
        "packaging_backend": "ch-ext4",
        "guest_arch": guest.deb_arch,
        "guest_platform": guest.platform.to_string(),
        "guest_rust_target": guest.rust_target,
        "kernel_image": kernel_path.file_name().and_then(|name| name.to_str()).unwrap_or(guest.kernel_image),
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
            "forbidden_first_contact_tools": ["apk", "apt", "apt-get", "cargo", "npm", "rustup"]
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

fn write_common_rootfs_record(
    out: &Path,
    source: &ExternalOciSource,
    tarball: &RootfsTarballRecord,
) -> Result<()> {
    let value = serde_json::json!({
        "kind": "common-assembled-rootfs-tarball",
        "path": tarball.canonical_path.display().to_string(),
        "size_bytes": tarball.size_bytes,
        "sha256": tarball.sha256,
        "source": {
            "kind": "external-oci",
            "image": source.image_ref.clone(),
            "platform": source.platform.to_string(),
            "image_index_digest": source.image_index_digest.to_string(),
            "platform_manifest_digest": source.platform_manifest_digest.to_string(),
        },
        "consumers": {
            "ch": "CH emitter applies backend boot adaptations after this common rootfs snapshot",
            "vz": "VZ emitter consumes this tarball with --rootfs-tarball during image build"
        }
    });
    fs::write(
        out.join(COMMON_ROOTFS_RECORD_NAME),
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
        source: manifest.source.clone(),
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
            if is_unhashed_vm_disk_artifact(&relative) {
                continue;
            }
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

fn is_unhashed_vm_disk_artifact(relative: &Path) -> bool {
    // VZ disk images are boot substrates, not immutable rootfs payloads. Hashing
    // a 20+ GiB sparse disk during generic manifest collection makes successful
    // builds appear hung; the build-result/guest-contract records the source VM
    // and rootfs tarball digests that define reproducibility instead.
    relative.file_name().and_then(|name| name.to_str()) == Some("disk.img")
        && relative
            .parent()
            .and_then(|parent| parent.extension())
            .and_then(|extension| extension.to_str())
            == Some("vm")
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
        bail!("rootfs tarball is not a file: {}", canonical_path.display());
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

fn common_rootfs_tarball_from_artifact(
    artifact: &Path,
    adapter: &AdapterRecord,
) -> Result<RootfsTarballRecord> {
    let expected = adapter
        .rootfs_tarball
        .as_ref()
        .context("OCI export requires adapter.rootfs_tarball evidence")?;
    let local = rootfs_tarball_record(&artifact.join(COMMON_ROOTFS_TARBALL_NAME))?;
    if local.size_bytes != expected.size_bytes || local.sha256 != expected.sha256 {
        bail!(
            "assembled rootfs tarball does not match manifest evidence: local size={} sha256={}, manifest size={} sha256={}",
            local.size_bytes,
            local.sha256,
            expected.size_bytes,
            expected.sha256
        );
    }
    Ok(local)
}

fn write_oci_layout(
    manifest: &ImageBuildManifest,
    source: &ExternalOciSource,
    rootfs: &RootfsTarballRecord,
    artifact: &Path,
    out: &Path,
    tag: Option<&str>,
) -> Result<OciExportManifest> {
    prepare_oci_output_dir(out)?;
    let platform = parse_source_platform(&manifest.source.platform)?;
    let tag = tag
        .map(str::to_string)
        .unwrap_or_else(|| default_oci_ref_name(&platform));
    let blobs_dir = out.join("blobs/sha256");
    fs::create_dir_all(&blobs_dir)?;
    fs::write(
        out.join("oci-layout"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "imageLayoutVersion": OCI_LAYOUT_VERSION
        }))?,
    )?;

    let layer_blob = blobs_dir.join(&rootfs.sha256);
    fs::copy(&rootfs.canonical_path, &layer_blob).with_context(|| {
        format!(
            "copy assembled rootfs {} to OCI layer blob {}",
            rootfs.canonical_path.display(),
            layer_blob.display()
        )
    })?;
    let layer_descriptor = OciBlobDescriptor {
        media_type: OCI_IMAGE_LAYER_TAR_MEDIA_TYPE.to_string(),
        digest: format!("sha256:{}", rootfs.sha256),
        size_bytes: rootfs.size_bytes,
        path: Some(PathBuf::from("blobs/sha256").join(&rootfs.sha256)),
    };

    let labels = serde_json::json!({
        "org.opencontainers.image.title": "motlie-guest",
        "org.opencontainers.image.ref.name": tag,
        "io.motlie.contract.version": manifest.contract_version,
        "io.motlie.source.image": source.image_ref,
        "io.motlie.source.image_index_digest": source.image_index_digest.to_string(),
        "io.motlie.source.platform": source.platform.to_string(),
        "io.motlie.source.platform_manifest_digest": source.platform_manifest_digest.to_string(),
        "io.motlie.validation.profile": manifest.validation.join(","),
    });
    let config_value = serde_json::json!({
        "architecture": platform.architecture.to_string(),
        "os": platform.os.to_string(),
        "config": {
            "Labels": labels,
        },
        "rootfs": {
            "type": "layers",
            "diff_ids": [layer_descriptor.digest],
        },
        "history": [{
            "created_by": "mbuild oci export",
            "comment": "Motlie v1.5 assembled rootfs payload exported from mbuild artifact"
        }]
    });
    let config_descriptor =
        write_oci_json_blob(&blobs_dir, OCI_IMAGE_CONFIG_MEDIA_TYPE, &config_value)?;

    let manifest_value = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": OCI_IMAGE_MANIFEST_MEDIA_TYPE,
        "config": {
            "mediaType": config_descriptor.media_type,
            "digest": config_descriptor.digest,
            "size": config_descriptor.size_bytes,
        },
        "layers": [{
            "mediaType": layer_descriptor.media_type,
            "digest": layer_descriptor.digest,
            "size": layer_descriptor.size_bytes,
            "annotations": {
                "org.opencontainers.image.title": "motlie-v1.5-rootfs"
            }
        }],
        "annotations": {
            "org.opencontainers.image.ref.name": tag,
            "io.motlie.contract.version": manifest.contract_version,
            "io.motlie.source.image_index_digest": source.image_index_digest.to_string(),
            "io.motlie.source.platform_manifest_digest": source.platform_manifest_digest.to_string(),
        }
    });
    let image_manifest_descriptor =
        write_oci_json_blob(&blobs_dir, OCI_IMAGE_MANIFEST_MEDIA_TYPE, &manifest_value)?;

    let index_value = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": OCI_IMAGE_INDEX_MEDIA_TYPE,
        "manifests": [{
            "mediaType": image_manifest_descriptor.media_type,
            "digest": image_manifest_descriptor.digest,
            "size": image_manifest_descriptor.size_bytes,
            "platform": {
                "architecture": platform.architecture.to_string(),
                "os": platform.os.to_string(),
            },
            "annotations": {
                "org.opencontainers.image.ref.name": tag,
                "io.motlie.contract.version": manifest.contract_version,
                "io.motlie.source.image_index_digest": source.image_index_digest.to_string(),
                "io.motlie.source.platform_manifest_digest": source.platform_manifest_digest.to_string(),
            }
        }]
    });
    let index_bytes = serde_json::to_vec_pretty(&index_value)?;
    fs::write(out.join("index.json"), &index_bytes)?;
    let image_index = OciBlobDescriptor {
        media_type: OCI_IMAGE_INDEX_MEDIA_TYPE.to_string(),
        digest: format!("sha256:{}", sha256_bytes(&index_bytes)),
        size_bytes: index_bytes.len() as u64,
        path: Some(PathBuf::from("index.json")),
    };

    Ok(OciExportManifest {
        contract_version: manifest.contract_version.clone(),
        tag,
        source: manifest.source.clone(),
        input_artifact_dir: artifact.to_path_buf(),
        input_rootfs_tarball: rootfs.clone(),
        output_dir: out.to_path_buf(),
        oci_layout_version: OCI_LAYOUT_VERSION.to_string(),
        platform: platform.to_string(),
        image_index,
        image_manifest: image_manifest_descriptor,
        image_config: config_descriptor,
        rootfs_layer: layer_descriptor,
        created_at_unix_seconds: unix_now()?,
    })
}

fn prepare_oci_output_dir(out: &Path) -> Result<()> {
    if out.exists() {
        if !out.is_dir() {
            bail!(
                "OCI export output exists and is not a directory: {}",
                out.display()
            );
        }
        let mut entries = fs::read_dir(out)
            .with_context(|| format!("read OCI export output {}", out.display()))?;
        if entries.next().transpose()?.is_some() {
            bail!(
                "OCI export output directory must be empty to avoid stale blobs: {}",
                out.display()
            );
        }
    } else {
        fs::create_dir_all(out)
            .with_context(|| format!("create OCI export output {}", out.display()))?;
    }
    Ok(())
}

fn validate_oci_layout(
    layout: &Path,
    export: &OciExportManifest,
    config: Option<&ImageBuildConfig>,
    build_manifest: Option<&ImageBuildManifest>,
) -> Result<OciExportManifest> {
    if !layout.is_dir() {
        bail!("OCI layout is not a directory: {}", layout.display());
    }
    if export.oci_layout_version != OCI_LAYOUT_VERSION {
        bail!(
            "OCI export manifest layout version {} does not match expected {}",
            export.oci_layout_version,
            OCI_LAYOUT_VERSION
        );
    }
    let layout_json: serde_json::Value =
        serde_json::from_slice(&fs::read(layout.join("oci-layout"))?)?;
    let layout_version = layout_json
        .get("imageLayoutVersion")
        .and_then(|value| value.as_str())
        .context("oci-layout is missing imageLayoutVersion")?;
    if layout_version != OCI_LAYOUT_VERSION {
        bail!(
            "oci-layout imageLayoutVersion {layout_version:?} does not match expected {OCI_LAYOUT_VERSION:?}"
        );
    }
    if let Some(config) = config {
        require_manifest_match(
            "oci.contract_version",
            &export.contract_version,
            &config.version,
        )?;
        require_manifest_match(
            "oci.source",
            &export.source,
            &ManifestSource::from_config(config),
        )?;
        require_manifest_match("oci.platform", &export.platform, &config.source.platform)?;
    }
    if let Some(build_manifest) = build_manifest {
        require_manifest_match("oci.source", &export.source, &build_manifest.source)?;
        require_manifest_match(
            "oci.contract_version",
            &export.contract_version,
            &build_manifest.contract_version,
        )?;
        if export.input_artifact_dir != build_manifest.output_dir {
            require_artifact_consumed_oci_payload(build_manifest, export)?;
        }
        if let Some(adapter) = &build_manifest.adapter {
            if let Some(rootfs) = &adapter.rootfs_tarball {
                require_manifest_match(
                    "oci.input_rootfs_tarball.size_bytes",
                    &export.input_rootfs_tarball.size_bytes,
                    &rootfs.size_bytes,
                )?;
                require_manifest_match(
                    "oci.input_rootfs_tarball.sha256",
                    &export.input_rootfs_tarball.sha256,
                    &rootfs.sha256,
                )?;
            }
        }
    }

    verify_descriptor_blob(layout, &export.image_index)?;
    verify_descriptor_blob(layout, &export.image_manifest)?;
    verify_descriptor_blob(layout, &export.image_config)?;
    verify_descriptor_blob(layout, &export.rootfs_layer)?;

    let platform = parse_source_platform(&export.platform)?;
    let index_value = read_descriptor_json(layout, &export.image_index)?;
    let index_manifests = index_value
        .get("manifests")
        .and_then(|value| value.as_array())
        .context("index.json is missing manifests array")?;
    if index_manifests.len() != 1 {
        bail!(
            "mbuild OCI export expects exactly one platform manifest, found {}",
            index_manifests.len()
        );
    }
    let index_manifest = &index_manifests[0];
    require_json_str(
        index_manifest,
        "mediaType",
        &export.image_manifest.media_type,
        "index manifest mediaType",
    )?;
    require_json_str(
        index_manifest,
        "digest",
        &export.image_manifest.digest,
        "index manifest digest",
    )?;
    require_json_u64(
        index_manifest,
        "size",
        export.image_manifest.size_bytes,
        "index manifest size",
    )?;
    let index_platform = index_manifest
        .get("platform")
        .context("index manifest is missing platform")?;
    require_json_str(
        index_platform,
        "architecture",
        &platform.architecture.to_string(),
        "index platform architecture",
    )?;
    require_json_str(
        index_platform,
        "os",
        &platform.os.to_string(),
        "index platform os",
    )?;

    let manifest_value = read_descriptor_json(layout, &export.image_manifest)?;
    let config_value = manifest_value
        .get("config")
        .context("image manifest is missing config descriptor")?;
    require_json_str(
        config_value,
        "mediaType",
        &export.image_config.media_type,
        "config descriptor mediaType",
    )?;
    require_json_str(
        config_value,
        "digest",
        &export.image_config.digest,
        "config descriptor digest",
    )?;
    require_json_u64(
        config_value,
        "size",
        export.image_config.size_bytes,
        "config descriptor size",
    )?;
    let layers = manifest_value
        .get("layers")
        .and_then(|value| value.as_array())
        .context("image manifest is missing layers array")?;
    if layers.len() != 1 {
        bail!(
            "mbuild OCI export expects exactly one rootfs layer, found {}",
            layers.len()
        );
    }
    require_json_str(
        &layers[0],
        "mediaType",
        &export.rootfs_layer.media_type,
        "rootfs layer mediaType",
    )?;
    require_json_str(
        &layers[0],
        "digest",
        &export.rootfs_layer.digest,
        "rootfs layer digest",
    )?;
    require_json_u64(
        &layers[0],
        "size",
        export.rootfs_layer.size_bytes,
        "rootfs layer size",
    )?;

    let config_value = read_descriptor_json(layout, &export.image_config)?;
    require_json_str(
        &config_value,
        "architecture",
        &platform.architecture.to_string(),
        "config architecture",
    )?;
    require_json_str(&config_value, "os", &platform.os.to_string(), "config os")?;
    let diff_ids = config_value
        .get("rootfs")
        .and_then(|rootfs| rootfs.get("diff_ids"))
        .and_then(|value| value.as_array())
        .context("image config is missing rootfs.diff_ids")?;
    if !diff_ids
        .iter()
        .any(|value| value.as_str() == Some(export.rootfs_layer.digest.as_str()))
    {
        bail!(
            "image config rootfs.diff_ids does not include rootfs layer digest {}",
            export.rootfs_layer.digest
        );
    }
    Ok(export.clone())
}

fn build_oci_push_plan(layout: &Path, image_ref: OciImageReference) -> Result<OciPushPlan> {
    if layout.join(DEFAULT_OCI_INDEX_MANIFEST_NAME).exists() {
        let index = load_oci_index_manifest(layout)?;
        validate_oci_index_layout(layout, &index, image_ref)
    } else {
        let export = load_oci_export_manifest(layout)?;
        let validation = validate_oci_layout(layout, &export, None, None)?;
        Ok(OciPushPlan {
            layout: layout.to_path_buf(),
            image_ref,
            layout_kind: OciPushLayoutKind::SinglePlatformExport,
            image_index: validation.image_index,
            manifests: vec![validation.image_manifest],
            blobs: vec![validation.image_config, validation.rootfs_layer],
        })
    }
}

fn validate_oci_index_layout(
    layout: &Path,
    manifest: &OciMultiArchIndexManifest,
    image_ref: OciImageReference,
) -> Result<OciPushPlan> {
    validate_oci_layout_version(layout, &manifest.oci_layout_version)?;
    verify_descriptor_blob(layout, &manifest.image_index)?;
    let index_value = read_descriptor_json(layout, &manifest.image_index)?;
    require_json_str(
        &index_value,
        "mediaType",
        OCI_IMAGE_INDEX_MEDIA_TYPE,
        "multi-arch index mediaType",
    )?;
    let index_manifests = index_value
        .get("manifests")
        .and_then(|value| value.as_array())
        .context("multi-arch index.json is missing manifests array")?;
    if index_manifests.is_empty() {
        bail!("multi-arch OCI index contains no manifests");
    }

    let mut manifests = BTreeMap::new();
    let mut blobs = BTreeMap::new();
    for (idx, value) in index_manifests.iter().enumerate() {
        let descriptor = descriptor_from_json(value, &format!("index manifests[{idx}]"))?;
        verify_descriptor_blob(layout, &descriptor)?;
        collect_image_manifest_dependencies(layout, &descriptor, &mut blobs)?;
        insert_descriptor(&mut manifests, descriptor)?;
    }

    Ok(OciPushPlan {
        layout: layout.to_path_buf(),
        image_ref,
        layout_kind: OciPushLayoutKind::MultiArchIndex,
        image_index: manifest.image_index.clone(),
        manifests: manifests.into_values().collect(),
        blobs: blobs.into_values().collect(),
    })
}

fn validate_oci_layout_version(layout: &Path, expected: &str) -> Result<()> {
    if !layout.is_dir() {
        bail!("OCI layout is not a directory: {}", layout.display());
    }
    if expected != OCI_LAYOUT_VERSION {
        bail!(
            "OCI layout manifest version {} does not match expected {}",
            expected,
            OCI_LAYOUT_VERSION
        );
    }
    let layout_json: serde_json::Value =
        serde_json::from_slice(&fs::read(layout.join("oci-layout"))?)?;
    let layout_version = layout_json
        .get("imageLayoutVersion")
        .and_then(|value| value.as_str())
        .context("oci-layout is missing imageLayoutVersion")?;
    if layout_version != OCI_LAYOUT_VERSION {
        bail!(
            "oci-layout imageLayoutVersion {layout_version:?} does not match expected {OCI_LAYOUT_VERSION:?}"
        );
    }
    Ok(())
}

fn collect_image_manifest_dependencies(
    layout: &Path,
    manifest_descriptor: &OciBlobDescriptor,
    blobs: &mut BTreeMap<String, OciBlobDescriptor>,
) -> Result<()> {
    if manifest_descriptor.media_type != OCI_IMAGE_MANIFEST_MEDIA_TYPE {
        bail!(
            "OCI index references unsupported child manifest media type {} for {}",
            manifest_descriptor.media_type,
            manifest_descriptor.digest
        );
    }
    let manifest_value = read_descriptor_json(layout, manifest_descriptor)?;
    let config = manifest_value
        .get("config")
        .context("image manifest is missing config descriptor")?;
    let config_descriptor = descriptor_from_json(config, "image manifest config")?;
    verify_descriptor_blob(layout, &config_descriptor)?;
    insert_descriptor(blobs, config_descriptor)?;

    let layers = manifest_value
        .get("layers")
        .and_then(|value| value.as_array())
        .context("image manifest is missing layers array")?;
    if layers.is_empty() {
        bail!(
            "image manifest {} has no layers",
            manifest_descriptor.digest
        );
    }
    for (idx, value) in layers.iter().enumerate() {
        let layer = descriptor_from_json(value, &format!("image manifest layers[{idx}]"))?;
        verify_descriptor_blob(layout, &layer)?;
        insert_descriptor(blobs, layer)?;
    }
    Ok(())
}

fn descriptor_from_json(value: &serde_json::Value, context: &str) -> Result<OciBlobDescriptor> {
    let media_type = value
        .get("mediaType")
        .and_then(|value| value.as_str())
        .with_context(|| format!("{context} is missing mediaType"))?
        .to_string();
    let digest = value
        .get("digest")
        .and_then(|value| value.as_str())
        .with_context(|| format!("{context} is missing digest"))?
        .to_string();
    OciDigest::new(digest.clone()).with_context(|| format!("{context} has invalid digest"))?;
    let size_bytes = value
        .get("size")
        .and_then(|value| value.as_u64())
        .with_context(|| format!("{context} is missing size"))?;
    Ok(OciBlobDescriptor {
        media_type,
        digest,
        size_bytes,
        path: None,
    })
}

fn insert_descriptor(
    descriptors: &mut BTreeMap<String, OciBlobDescriptor>,
    descriptor: OciBlobDescriptor,
) -> Result<()> {
    if let Some(existing) = descriptors.get(&descriptor.digest) {
        if existing.media_type != descriptor.media_type
            || existing.size_bytes != descriptor.size_bytes
        {
            bail!(
                "OCI descriptor conflict for digest {}: existing media_type={} size={}, new media_type={} size={}",
                descriptor.digest,
                existing.media_type,
                existing.size_bytes,
                descriptor.media_type,
                descriptor.size_bytes
            );
        }
        return Ok(());
    }
    descriptors.insert(descriptor.digest.clone(), descriptor);
    Ok(())
}

fn resolve_registry_auth(
    image_ref: &OciImageReference,
    username: Option<&str>,
    password_env: Option<&str>,
    token_env: Option<&str>,
) -> Result<OciRegistryAuth> {
    resolve_registry_auth_with_env(image_ref, username, password_env, token_env, |name| {
        env::var(name).ok()
    })
}

fn resolve_registry_auth_with_env<F>(
    image_ref: &OciImageReference,
    username: Option<&str>,
    password_env: Option<&str>,
    token_env: Option<&str>,
    env_get: F,
) -> Result<OciRegistryAuth>
where
    F: Fn(&str) -> Option<String>,
{
    let is_ghcr = image_ref.registry == "ghcr.io";
    let username = username
        .map(str::to_string)
        .or_else(|| secret_env_value("MOTLIE_MBUILD_REGISTRY_USERNAME", &env_get))
        .or_else(|| {
            if is_ghcr {
                secret_env_value("GITHUB_ACTOR", &env_get)
            } else {
                None
            }
        });
    let explicit_password = password_env
        .map(|name| {
            read_required_secret_env_with(name, &env_get)
                .map(|value| (RegistrySecretKind::Password, name.to_string(), value))
        })
        .transpose()?;
    let explicit_token = token_env
        .map(|name| {
            read_required_secret_env_with(name, &env_get)
                .map(|value| (RegistrySecretKind::Token, name.to_string(), value))
        })
        .transpose()?;
    if explicit_password.is_some() && explicit_token.is_some() {
        bail!("use only one of --password-env or --token-env for OCI registry auth");
    }
    let secret = explicit_password
        .or(explicit_token)
        .or_else(|| default_registry_token_with_env(image_ref, &env_get));
    let Some((kind, _source, secret)) = secret else {
        if username.is_some() {
            bail!("registry username was provided but no password/token env var was available");
        }
        return Ok(OciRegistryAuth::Anonymous);
    };
    match kind {
        RegistrySecretKind::Password => {
            let username = username.with_context(|| {
                format!(
                    "registry password auth for {} requires --username or MOTLIE_MBUILD_REGISTRY_USERNAME{}",
                    image_ref,
                    if is_ghcr { "/GITHUB_ACTOR" } else { "" }
                )
            })?;
            Ok(OciRegistryAuth::basic(username, secret))
        }
        RegistrySecretKind::Token if is_ghcr => {
            let username = username.with_context(|| {
                "GHCR upload requires a username; pass --username or set MOTLIE_MBUILD_REGISTRY_USERNAME/GITHUB_ACTOR"
            })?;
            Ok(OciRegistryAuth::basic(username, secret))
        }
        RegistrySecretKind::Token => {
            if let Some(username) = username {
                Ok(OciRegistryAuth::basic(username, secret))
            } else {
                Ok(OciRegistryAuth::bearer(secret))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistrySecretKind {
    Password,
    Token,
}

fn read_required_secret_env_with<F>(name: &str, env_get: &F) -> Result<String>
where
    F: Fn(&str) -> Option<String>,
{
    let value =
        env_get(name).with_context(|| format!("registry auth env var {name} is not set"))?;
    if value.trim().is_empty() {
        bail!("registry auth env var {name} is empty");
    }
    Ok(value)
}

fn default_registry_token_with_env<F>(
    image_ref: &OciImageReference,
    env_get: &F,
) -> Option<(RegistrySecretKind, String, String)>
where
    F: Fn(&str) -> Option<String>,
{
    let default_names: &[&str] = if image_ref.registry == "ghcr.io" {
        &[
            "MOTLIE_MBUILD_REGISTRY_TOKEN",
            "GHCR_TOKEN",
            "CR_PAT",
            "GITHUB_TOKEN",
        ]
    } else {
        &["MOTLIE_MBUILD_REGISTRY_TOKEN"]
    };
    default_names.iter().find_map(|name| {
        secret_env_value(name, env_get)
            .map(|value| (RegistrySecretKind::Token, (*name).to_string(), value))
    })
}

fn secret_env_value<F>(name: &str, env_get: &F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    env_get(name).filter(|value| !value.trim().is_empty())
}

fn write_oci_push_evidence(path: &Path, evidence: &OciPushEvidence) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(evidence)?)?;
    info!(path = %path.display(), "wrote mbuild OCI push evidence");
    Ok(())
}

fn require_artifact_consumed_oci_payload(
    build_manifest: &ImageBuildManifest,
    export: &OciExportManifest,
) -> Result<()> {
    let adapter = build_manifest.adapter.as_ref().with_context(|| {
        format!(
            "OCI layout was exported from {}, but artifact {} has no adapter evidence",
            export.input_artifact_dir.display(),
            build_manifest.output_dir.display()
        )
    })?;
    let adapter_source = adapter.external_oci_source.as_ref().with_context(|| {
        format!(
            "OCI layout was exported from {}, but artifact {} does not record an external OCI source",
            export.input_artifact_dir.display(),
            build_manifest.output_dir.display()
        )
    })?;
    let export_source = external_source_from_manifest(&export.source)?;
    require_manifest_match(
        "adapter.external_oci_source",
        adapter_source,
        &export_source,
    )?;

    let rootfs = adapter.rootfs_tarball.as_ref().with_context(|| {
        format!(
            "OCI layout was exported from {}, but artifact {} does not record consumed rootfs evidence",
            export.input_artifact_dir.display(),
            build_manifest.output_dir.display()
        )
    })?;
    let layer_sha256 = digest_hex(&export.rootfs_layer.digest)?.to_string();
    require_manifest_match(
        "adapter.rootfs_tarball.size_bytes",
        &rootfs.size_bytes,
        &export.rootfs_layer.size_bytes,
    )?;
    require_manifest_match(
        "adapter.rootfs_tarball.sha256",
        &rootfs.sha256,
        &layer_sha256,
    )?;
    Ok(())
}

fn descriptor_path(layout: &Path, descriptor: &OciBlobDescriptor) -> Result<PathBuf> {
    if let Some(path) = &descriptor.path {
        if path.is_absolute() {
            bail!("OCI descriptor path must be relative: {}", path.display());
        }
        return Ok(layout.join(path));
    }
    let digest = digest_hex(&descriptor.digest)?;
    Ok(layout.join("blobs/sha256").join(digest))
}

fn verify_descriptor_blob(layout: &Path, descriptor: &OciBlobDescriptor) -> Result<()> {
    let path = descriptor_path(layout, descriptor)?;
    let metadata = fs::metadata(&path)
        .with_context(|| format!("read OCI descriptor blob {}", path.display()))?;
    if !metadata.is_file() {
        bail!("OCI descriptor path is not a file: {}", path.display());
    }
    if metadata.len() != descriptor.size_bytes {
        bail!(
            "OCI blob {} size mismatch: file={}, descriptor={}",
            path.display(),
            metadata.len(),
            descriptor.size_bytes
        );
    }
    let actual = format!("sha256:{}", sha256_file(&path)?);
    if actual != descriptor.digest {
        bail!(
            "OCI blob {} digest mismatch: file={}, descriptor={}",
            path.display(),
            actual,
            descriptor.digest
        );
    }
    Ok(())
}

fn read_descriptor_json(
    layout: &Path,
    descriptor: &OciBlobDescriptor,
) -> Result<serde_json::Value> {
    let path = descriptor_path(layout, descriptor)?;
    serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse OCI JSON blob {}", path.display()))
}

fn copy_oci_blob_to_layout(
    source_layout: &Path,
    dest_layout: &Path,
    descriptor: &OciBlobDescriptor,
) -> Result<()> {
    let source = descriptor_path(source_layout, descriptor)?;
    let dest = descriptor_path(dest_layout, descriptor)?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&source, &dest)
        .with_context(|| format!("copy OCI blob {} to {}", source.display(), dest.display()))?;
    verify_descriptor_blob(dest_layout, descriptor)
}

fn extract_rootfs_tarball(tarball: &Path, rootfs_dir: &Path, log_path: &Path) -> Result<()> {
    let script =
        "set -euo pipefail\nmkdir -p \"$2\"\ntar --preserve-permissions -C \"$2\" -xf \"$1\"\n";
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(script)
        .arg("mbuild-extract-rootfs")
        .arg(tarball)
        .arg(rootfs_dir);
    run_logged_command(&mut command, log_path, "extract OCI rootfs payload")
}

fn external_source_from_manifest(source: &ManifestSource) -> Result<ExternalOciSource> {
    if source.kind != SourceKind::ExternalOci.to_string() {
        bail!(
            "OCI payload source kind must be external-oci, got {}",
            source.kind
        );
    }
    let platform = parse_source_platform(&source.platform)?;
    let image_index_digest = source
        .image_index_digest
        .clone()
        .context("OCI payload source is missing image_index_digest")?;
    let platform_manifest_digest = source
        .platform_manifest_digest
        .clone()
        .context("OCI payload source is missing platform_manifest_digest")?;
    Ok(ExternalOciSource {
        image_ref: source.image.clone(),
        platform,
        image_index_digest,
        platform_manifest_digest,
    })
}

fn digest_hex(digest: &str) -> Result<&str> {
    digest
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .with_context(|| format!("unsupported OCI digest {digest:?}; expected sha256:<64 hex>"))
}

fn require_json_str(
    value: &serde_json::Value,
    key: &str,
    expected: &str,
    label: &str,
) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(|value| value.as_str())
        .with_context(|| format!("{label} is missing string field {key}"))?;
    if actual != expected {
        bail!("{label} mismatch: expected {expected:?}, got {actual:?}");
    }
    Ok(())
}

fn require_json_u64(
    value: &serde_json::Value,
    key: &str,
    expected: u64,
    label: &str,
) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(|value| value.as_u64())
        .with_context(|| format!("{label} is missing integer field {key}"))?;
    if actual != expected {
        bail!("{label} mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

fn write_oci_json_blob(
    blobs_dir: &Path,
    media_type: &str,
    value: &serde_json::Value,
) -> Result<OciBlobDescriptor> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let digest = sha256_bytes(&bytes);
    let path = blobs_dir.join(&digest);
    fs::write(&path, &bytes).with_context(|| format!("write OCI blob {}", path.display()))?;
    Ok(OciBlobDescriptor {
        media_type: media_type.to_string(),
        digest: format!("sha256:{digest}"),
        size_bytes: bytes.len() as u64,
        path: Some(PathBuf::from("blobs/sha256").join(digest)),
    })
}

fn default_oci_ref_name(platform: &OciPlatform) -> String {
    format!("motlie-guest:v1.5-{}", platform.architecture)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn unix_now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_secs())
}

fn git_source_commit() -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .context("read git source commit")?;
    if !output.status.success() {
        bail!(
            "git rev-parse HEAD failed with status {}",
            output.status.code().unwrap_or(-1)
        );
    }
    Ok(String::from_utf8(output.stdout)
        .context("git rev-parse HEAD returned non-UTF-8")?
        .trim()
        .to_string())
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
        /// Local OCI image layout containing a Motlie assembled rootfs payload to consume.
        #[arg(long)]
        oci_layout: Option<PathBuf>,
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
    /// OCI payload operations for issue #258 convergence.
    Oci {
        #[command(subcommand)]
        command: OciCommands,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
enum OciCommands {
    /// Export an executed mbuild rootfs handoff as an OCI image layout.
    Export {
        /// Dockerfile-like v1.5 image build config.
        #[arg(long)]
        config: PathBuf,
        /// Artifact directory containing mbuild-manifest.json and assembled-rootfs.tar.
        #[arg(long)]
        artifact: PathBuf,
        /// Output directory for the OCI image layout.
        #[arg(long)]
        out: PathBuf,
        /// Optional OCI ref-name annotation. Defaults to motlie-guest:v1.5-<arch>.
        #[arg(long)]
        tag: Option<String>,
    },
    /// Validate a local OCI image layout exported by mbuild.
    Validate {
        /// Dockerfile-like v1.5 image build config.
        #[arg(long)]
        config: PathBuf,
        /// Artifact directory containing mbuild-manifest.json.
        #[arg(long)]
        artifact: PathBuf,
        /// OCI image layout directory containing mbuild-oci-export.json.
        #[arg(long)]
        layout: PathBuf,
    },
    /// Resolve a registry image/platform to immutable OCI digests.
    Resolve {
        /// OCI image reference to resolve.
        #[arg(long)]
        image: String,
        /// OCI platform, for example linux/amd64 or linux/arm64.
        #[arg(long)]
        platform: String,
    },
    /// Combine validated per-arch mbuild OCI layouts into one local multi-arch OCI index.
    Index {
        /// Output directory for the multi-arch OCI image layout.
        #[arg(long)]
        out: PathBuf,
        /// Canonical image reference annotation for the multi-arch index.
        #[arg(long)]
        image: String,
        /// Per-architecture mbuild OCI layout input. Repeat once per platform.
        #[arg(long = "layout")]
        layout: Vec<PathBuf>,
    },
    /// Push a validated mbuild OCI layout directly to an OCI registry.
    Push {
        /// OCI image layout directory from `mbuild oci export` or `mbuild oci index`.
        #[arg(long)]
        layout: PathBuf,
        /// Destination registry image reference, for example ghcr.io/chungers/motlie-guest:v1.5.
        #[arg(long)]
        image: String,
        /// Validate and print the push plan without contacting the registry.
        #[arg(long)]
        dry_run: bool,
        /// Allow replacing an existing remote tag. Defaults to refusing overwrite.
        #[arg(long)]
        allow_overwrite: bool,
        /// Registry username for basic/token auth. Defaults to MOTLIE_MBUILD_REGISTRY_USERNAME; GHCR also accepts GITHUB_ACTOR.
        #[arg(long)]
        username: Option<String>,
        /// Environment variable containing a registry password/PAT for basic auth.
        #[arg(long)]
        password_env: Option<String>,
        /// Environment variable containing a registry token. Defaults to MOTLIE_MBUILD_REGISTRY_TOKEN; GHCR also uses GHCR_TOKEN, CR_PAT, or GITHUB_TOKEN.
        #[arg(long)]
        token_env: Option<String>,
        /// Output path for mbuild OCI push evidence. Defaults to <layout>/mbuild-oci-push.json.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Emit release-manifest-ready VM image artifact evidence.
    Evidence {
        /// Dockerfile-like v1.5 image build config.
        #[arg(long)]
        config: PathBuf,
        /// Artifact directory containing mbuild-manifest.json.
        #[arg(long)]
        artifact: PathBuf,
        /// OCI image layout directory containing mbuild-oci-export.json.
        #[arg(long)]
        layout: PathBuf,
        /// Optional published OCI reference or digest to include in evidence.
        #[arg(long)]
        publish_ref: Option<String>,
        /// Output path for evidence JSON. Defaults to <layout>/mbuild-release-evidence.json.
        #[arg(long)]
        out: Option<PathBuf>,
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
                    bail!(
                        "source.digest_policy adapter-verified is only valid for transitional-adapter sources"
                    );
                }
                if self.digest_policy == DigestPolicy::Pinned
                    && (self.image_index_digest.is_none()
                        || self.platform_manifest_digest.is_none())
                {
                    bail!(
                        "source.digest_policy pinned requires source.image_index_digest and source.platform_manifest_digest"
                    );
                }
            }
            SourceKind::TransitionalAdapter => {
                if self.digest_policy != DigestPolicy::AdapterVerified {
                    bail!(
                        "source.digest_policy for transitional-adapter sources must be adapter-verified"
                    );
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
        require_token("services.name", &self.name)?;
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
        support: PackageManagerSupport::Implemented,
        validate_package: validate_apk_package_spec,
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

fn validate_apk_package_spec(field: &str, value: &str) -> Result<()> {
    require_non_empty(field, value)?;
    if value
        .bytes()
        .any(|byte| byte.is_ascii_whitespace() || matches!(byte, b',' | b'/' | b';'))
    {
        bail!("{field} contains unsupported apk package spec {value:?}");
    }
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        bail!("{field} contains unsupported apk package spec {value:?}: empty package name");
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        bail!(
            "{field} contains unsupported apk package spec {value:?}: package name must start with lowercase ASCII or digit"
        );
    }
    if !bytes.all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'+' | b'-' | b'.' | b'_')
    }) {
        bail!(
            "{field} contains unsupported apk package spec {value:?}: unsupported package name character"
        );
    }
    Ok(())
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
        bail!(
            "{field} contains unsupported apt package spec {value:?}: multiple ':' separators before version"
        );
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
        bail!(
            "{field} contains unsupported apt package spec {full:?}: package name must start with lowercase ASCII or digit"
        );
    }
    if !bytes.all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')
    }) {
        bail!(
            "{field} contains unsupported apt package spec {full:?}: unsupported package name character"
        );
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
        bail!(
            "{field} contains unsupported apt package spec {full:?}: architecture must start with lowercase ASCII or digit"
        );
    }
    if !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-') {
        bail!(
            "{field} contains unsupported apt package spec {full:?}: unsupported architecture character"
        );
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
    build_host: Option<BuildHostRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    guest_target: Option<GuestTargetRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rootfs_tarball: Option<RootfsTarballRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildHostRecord {
    platform: String,
    deb_arch: String,
}

impl From<&HostPlatformTarget> for BuildHostRecord {
    fn from(target: &HostPlatformTarget) -> Self {
        Self {
            platform: target.platform.to_string(),
            deb_arch: target.deb_arch.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GuestTargetRecord {
    platform: String,
    rust_target: String,
    deb_arch: String,
    kernel_image: String,
    kernel_asset: String,
}

impl From<&ChGuestTarget> for GuestTargetRecord {
    fn from(target: &ChGuestTarget) -> Self {
        Self {
            platform: target.platform.to_string(),
            rust_target: target.rust_target.to_string(),
            deb_arch: target.deb_arch.to_string(),
            kernel_image: target.kernel_image.to_string(),
            kernel_asset: target.kernel_asset.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RootfsTarballRecord {
    canonical_path: PathBuf,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OciExportManifest {
    contract_version: String,
    tag: String,
    source: ManifestSource,
    input_artifact_dir: PathBuf,
    input_rootfs_tarball: RootfsTarballRecord,
    output_dir: PathBuf,
    oci_layout_version: String,
    platform: String,
    image_index: OciBlobDescriptor,
    image_manifest: OciBlobDescriptor,
    image_config: OciBlobDescriptor,
    rootfs_layer: OciBlobDescriptor,
    created_at_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OciBlobDescriptor {
    media_type: String,
    digest: String,
    size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OciMultiArchIndexManifest {
    image: String,
    output_dir: PathBuf,
    oci_layout_version: String,
    image_index: OciBlobDescriptor,
    platforms: BTreeMap<String, ManifestSource>,
    created_at_unix_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OciPushLayoutKind {
    SinglePlatformExport,
    MultiArchIndex,
}

#[derive(Debug, Clone)]
struct OciPushPlan {
    layout: PathBuf,
    image_ref: OciImageReference,
    layout_kind: OciPushLayoutKind,
    image_index: OciBlobDescriptor,
    manifests: Vec<OciBlobDescriptor>,
    blobs: Vec<OciBlobDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OciPushedBlobRecord {
    descriptor: OciBlobDescriptor,
    status: OciBlobPushStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OciPushedManifestRecord {
    descriptor: OciBlobDescriptor,
    reference: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    registry_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OciPushEvidence {
    image: String,
    layout_dir: PathBuf,
    layout_kind: OciPushLayoutKind,
    oci_layout_version: String,
    image_index: OciBlobDescriptor,
    blobs: Vec<OciPushedBlobRecord>,
    manifests: Vec<OciPushedManifestRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remote_digest: Option<String>,
    dry_run: bool,
    output_path: PathBuf,
    created_at_unix_seconds: u64,
}

impl OciPushEvidence {
    fn dry_run(plan: &OciPushPlan, output_path: &Path) -> Result<Self> {
        Ok(Self {
            image: plan.image_ref.to_string(),
            layout_dir: plan.layout.clone(),
            layout_kind: plan.layout_kind,
            oci_layout_version: OCI_LAYOUT_VERSION.to_string(),
            image_index: plan.image_index.clone(),
            blobs: plan
                .blobs
                .iter()
                .cloned()
                .map(|descriptor| OciPushedBlobRecord {
                    descriptor,
                    status: OciBlobPushStatus::DryRun,
                })
                .collect(),
            manifests: plan
                .manifests
                .iter()
                .cloned()
                .map(|descriptor| OciPushedManifestRecord {
                    reference: descriptor.digest.clone(),
                    descriptor,
                    registry_digest: None,
                })
                .chain(std::iter::once(OciPushedManifestRecord {
                    descriptor: plan.image_index.clone(),
                    reference: plan.image_ref.reference.registry_reference().to_string(),
                    registry_digest: None,
                }))
                .collect(),
            remote_digest: None,
            dry_run: true,
            output_path: output_path.to_path_buf(),
            created_at_unix_seconds: unix_now()?,
        })
    }

    fn from_push(
        plan: &OciPushPlan,
        blobs: Vec<OciPushedBlobRecord>,
        manifests: Vec<OciPushedManifestRecord>,
        remote_digest: &OciDigest,
        output_path: &Path,
    ) -> Result<Self> {
        Ok(Self {
            image: plan.image_ref.to_string(),
            layout_dir: plan.layout.clone(),
            layout_kind: plan.layout_kind,
            oci_layout_version: OCI_LAYOUT_VERSION.to_string(),
            image_index: plan.image_index.clone(),
            blobs,
            manifests,
            remote_digest: Some(remote_digest.to_string()),
            dry_run: false,
            output_path: output_path.to_path_buf(),
            created_at_unix_seconds: unix_now()?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReleaseArtifactEvidence {
    kind: String,
    contract_version: String,
    target: BackendTargetId,
    backend: String,
    package_engine: String,
    source_commit: String,
    config_path: PathBuf,
    artifact_dir: PathBuf,
    oci_layout_dir: PathBuf,
    publish_ref: Option<String>,
    source: ManifestSource,
    build_host: Option<BuildHostRecord>,
    guest_target: Option<GuestTargetRecord>,
    oci: ReleaseOciEvidence,
    artifacts: Vec<ManifestArtifact>,
    validation: Option<HarnessValidationRecord>,
    created_at_unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReleaseOciEvidence {
    tag: String,
    platform: String,
    image_index: OciBlobDescriptor,
    image_manifest: OciBlobDescriptor,
    image_config: OciBlobDescriptor,
    rootfs_layer: OciBlobDescriptor,
}

impl ReleaseArtifactEvidence {
    fn from_records(
        build_manifest: &ImageBuildManifest,
        oci_manifest: &OciExportManifest,
        publish_ref: Option<&str>,
    ) -> Result<Self> {
        let validation_path = build_manifest
            .output_dir
            .join(DEFAULT_VALIDATION_MANIFEST_NAME);
        let validation = if validation_path.is_file() {
            Some(
                serde_json::from_slice(&fs::read(&validation_path)?).with_context(|| {
                    format!("read validation manifest {}", validation_path.display())
                })?,
            )
        } else {
            None
        };
        let source_commit = git_source_commit().unwrap_or_else(|_| "unknown".to_string());
        Ok(Self {
            kind: OCI_MANIFEST_ARTIFACT_KIND.to_string(),
            contract_version: build_manifest.contract_version.clone(),
            target: build_manifest.target.clone(),
            backend: build_manifest.target.to_string(),
            package_engine: build_manifest
                .adapter
                .as_ref()
                .map(|adapter| adapter.kind.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            source_commit,
            config_path: build_manifest.config_path.clone(),
            artifact_dir: build_manifest.output_dir.clone(),
            oci_layout_dir: oci_manifest.output_dir.clone(),
            publish_ref: publish_ref.map(ToOwned::to_owned),
            source: oci_manifest.source.clone(),
            build_host: build_manifest
                .adapter
                .as_ref()
                .and_then(|adapter| adapter.build_host.clone()),
            guest_target: build_manifest
                .adapter
                .as_ref()
                .and_then(|adapter| adapter.guest_target.clone()),
            oci: ReleaseOciEvidence {
                tag: oci_manifest.tag.clone(),
                platform: oci_manifest.platform.clone(),
                image_index: oci_manifest.image_index.clone(),
                image_manifest: oci_manifest.image_manifest.clone(),
                image_config: oci_manifest.image_config.clone(),
                rootfs_layer: oci_manifest.rootfs_layer.clone(),
            },
            artifacts: build_manifest.artifacts.clone(),
            validation,
            created_at_unix_seconds: unix_now()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestArtifact {
    label: String,
    path: PathBuf,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HarnessValidationRecord {
    contract_version: String,
    target: BackendTargetId,
    artifact_dir: PathBuf,
    source: ManifestSource,
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
    if adapter.kind == "external-oci-ch-emitter" || adapter.kind == "oci-payload-ch-emitter" {
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
                "explicit package-manager/npm package stage executed before backend emit",
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
    if let Some(host) = &adapter.build_host {
        evidence.push(format!("build_host_platform={}", host.platform));
        evidence.push(format!("build_host_deb_arch={}", host.deb_arch));
    }
    if let Some(guest) = &adapter.guest_target {
        evidence.push(format!("guest_platform={}", guest.platform));
        evidence.push(format!("guest_rust_target={}", guest.rust_target));
        evidence.push(format!("guest_deb_arch={}", guest.deb_arch));
        evidence.push(format!("guest_kernel_image={}", guest.kernel_image));
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

    fn external_oci_config_yaml() -> String {
        format!(
            r#"version: v1.5
source:
  kind: external-oci
  image: docker.io/library/ubuntu:24.04
  profile: ubuntu-systemd
  platform: linux/arm64
  digest_policy: pinned
  image_index_digest: sha256:{}
  platform_manifest_digest: sha256:{}
package_stage:
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
"#,
            "a".repeat(64),
            "b".repeat(64)
        )
    }

    fn temp_test_dir(name: &str) -> PathBuf {
        let path = env::temp_dir().join(format!(
            "mbuild-{name}-{}-{}",
            process::id(),
            unix_now().unwrap()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn repo_root_fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("mbuild crate should live under repo/bins/mbuild")
            .to_path_buf()
    }

    struct OciExportFixture {
        root: PathBuf,
        layout: PathBuf,
    }

    fn fixture_env<const N: usize>(
        vars: [(&'static str, &'static str); N],
    ) -> impl Fn(&str) -> Option<String> {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    fn assert_basic_auth(auth: OciRegistryAuth, expected_username: &str, expected_password: &str) {
        match auth {
            OciRegistryAuth::Basic { username, password } => {
                assert_eq!(username, expected_username);
                assert_eq!(password, expected_password);
            }
            other => panic!("expected basic auth, got {other:?}"),
        }
    }

    fn assert_bearer_auth(auth: OciRegistryAuth, expected_token: &str) {
        match auth {
            OciRegistryAuth::Bearer { token } => assert_eq!(token, expected_token),
            other => panic!("expected bearer auth, got {other:?}"),
        }
    }

    fn write_oci_export_fixture(
        name: &str,
        platform: OciPlatform,
        rootfs_bytes: &[u8],
        tag: &str,
    ) -> OciExportFixture {
        let root = temp_test_dir(name);
        let config_path = root.join("motlie-image.yaml");
        let artifact = root.join("artifact");
        let layout = root.join("oci");
        fs::create_dir_all(&artifact).unwrap();
        fs::create_dir_all(&layout).unwrap();
        fs::write(
            &config_path,
            external_oci_config_yaml()
                .replace("platform: linux/arm64", &format!("platform: {platform}")),
        )
        .unwrap();
        fs::write(artifact.join(COMMON_ROOTFS_TARBALL_NAME), rootfs_bytes).unwrap();

        let config = load_config(&config_path).unwrap();
        let source = ExternalOciSource::ubuntu_systemd(
            platform,
            OciDigest::new(format!("sha256:{}", "a".repeat(64))).unwrap(),
            OciDigest::new(format!("sha256:{}", "b".repeat(64))).unwrap(),
        );
        let rootfs = rootfs_tarball_record(&artifact.join(COMMON_ROOTFS_TARBALL_NAME)).unwrap();
        let guest = ChGuestTarget::from_platform(platform).unwrap();
        let manifest = ImageBuildManifest::from_config(
            &config_path,
            BackendTargetId("ch".to_string()),
            &artifact,
            &config,
            Some(AdapterRecord {
                kind: "external-oci-ch-emitter".to_string(),
                command: vec!["mbuild".to_string(), "build".to_string()],
                log_path: artifact.join(CH_EMITTER_LOG_NAME),
                exit_status: 0,
                started_at_unix_seconds: 1,
                completed_at_unix_seconds: 2,
                package_include: config.package_stage.install.clone(),
                materialized_source: None,
                external_oci_source: Some(source),
                build_host: Some(BuildHostRecord {
                    platform: platform.to_string(),
                    deb_arch: guest.deb_arch.to_string(),
                }),
                guest_target: Some(GuestTargetRecord::from(&guest)),
                rootfs_tarball: Some(rootfs),
            }),
            Vec::new(),
        );
        fs::write(
            artifact.join(DEFAULT_MANIFEST_NAME),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        oci_export(OciExportOptions {
            config_path,
            artifact,
            out: layout.clone(),
            tag: Some(tag.to_string()),
        })
        .unwrap();

        OciExportFixture { root, layout }
    }

    fn release_config_path(name: &str) -> PathBuf {
        repo_root_fixture()
            .join("releases/vmm/v1.5/configs")
            .join(name)
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
                rootfs_tarball: None,
                oci_layout: None
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
                rootfs_tarball: Some(PathBuf::from("artifacts/rootfs.tar")),
                oci_layout: None
            }
        );
    }

    #[test]
    fn parses_build_oci_layout() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "build",
            "--config",
            "motlie-image.yaml",
            "--target",
            "vz",
            "--out",
            "artifacts/vz",
            "--oci-layout",
            "artifacts/oci-arm64",
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
                rootfs_tarball: None,
                oci_layout: Some(PathBuf::from("artifacts/oci-arm64"))
            }
        );
    }

    #[test]
    fn parses_oci_export_command() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "oci",
            "export",
            "--config",
            "motlie-image.yaml",
            "--artifact",
            "artifacts/ch",
            "--out",
            "artifacts/oci",
            "--tag",
            "motlie-guest:test",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Oci {
                command: OciCommands::Export {
                    config: PathBuf::from("motlie-image.yaml"),
                    artifact: PathBuf::from("artifacts/ch"),
                    out: PathBuf::from("artifacts/oci"),
                    tag: Some("motlie-guest:test".to_string())
                }
            }
        );
    }

    #[test]
    fn parses_oci_validate_command() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "oci",
            "validate",
            "--config",
            "motlie-image.yaml",
            "--artifact",
            "artifacts/ch",
            "--layout",
            "artifacts/oci",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Oci {
                command: OciCommands::Validate {
                    config: PathBuf::from("motlie-image.yaml"),
                    artifact: PathBuf::from("artifacts/ch"),
                    layout: PathBuf::from("artifacts/oci")
                }
            }
        );
    }

    #[test]
    fn parses_oci_index_command() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "oci",
            "index",
            "--out",
            "artifacts/index",
            "--image",
            "ghcr.io/chungers/motlie-guest:v1.5",
            "--layout",
            "artifacts/oci-amd64",
            "--layout",
            "artifacts/oci-arm64",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Oci {
                command: OciCommands::Index {
                    out: PathBuf::from("artifacts/index"),
                    image: "ghcr.io/chungers/motlie-guest:v1.5".to_string(),
                    layout: vec![
                        PathBuf::from("artifacts/oci-amd64"),
                        PathBuf::from("artifacts/oci-arm64")
                    ]
                }
            }
        );
    }

    #[test]
    fn parses_oci_push_command() {
        let cli = Cli::try_parse_from([
            "mbuild",
            "oci",
            "push",
            "--layout",
            "artifacts/index",
            "--image",
            "ghcr.io/chungers/motlie-guest:v1.5",
            "--dry-run",
            "--allow-overwrite",
            "--username",
            "octocat",
            "--token-env",
            "GHCR_TOKEN",
            "--out",
            "artifacts/index/mbuild-oci-push.json",
        ])
        .unwrap();

        assert_eq!(
            cli.command,
            Commands::Oci {
                command: OciCommands::Push {
                    layout: PathBuf::from("artifacts/index"),
                    image: "ghcr.io/chungers/motlie-guest:v1.5".to_string(),
                    dry_run: true,
                    allow_overwrite: true,
                    username: Some("octocat".to_string()),
                    password_env: None,
                    token_env: Some("GHCR_TOKEN".to_string()),
                    out: Some(PathBuf::from("artifacts/index/mbuild-oci-push.json")),
                }
            }
        );
    }

    #[test]
    fn registry_auth_github_defaults_apply_only_to_ghcr() {
        let ghcr = OciImageReference::from_str("ghcr.io/chungers/motlie-guest:v1").unwrap();
        let ghcr_auth = resolve_registry_auth_with_env(
            &ghcr,
            None,
            None,
            None,
            fixture_env([("GITHUB_ACTOR", "octocat"), ("GHCR_TOKEN", "ghcr-secret")]),
        )
        .unwrap();
        assert_basic_auth(ghcr_auth, "octocat", "ghcr-secret");

        let other =
            OciImageReference::from_str("registry.example.com/acme/motlie-guest:v1").unwrap();
        let other_auth = resolve_registry_auth_with_env(
            &other,
            None,
            None,
            None,
            fixture_env([("GITHUB_ACTOR", "octocat"), ("GHCR_TOKEN", "ghcr-secret")]),
        )
        .unwrap();
        assert!(other_auth.is_anonymous());
    }

    #[test]
    fn registry_auth_non_ghcr_uses_only_generic_default_token() {
        let image =
            OciImageReference::from_str("registry.example.com/acme/motlie-guest:v1").unwrap();
        let auth = resolve_registry_auth_with_env(
            &image,
            None,
            None,
            None,
            fixture_env([
                ("GITHUB_ACTOR", "octocat"),
                ("GHCR_TOKEN", "ghcr-secret"),
                ("MOTLIE_MBUILD_REGISTRY_TOKEN", "generic-secret"),
            ]),
        )
        .unwrap();

        assert_bearer_auth(auth, "generic-secret");
    }

    #[test]
    fn registry_auth_explicit_non_ghcr_token_does_not_use_github_actor() {
        let image =
            OciImageReference::from_str("registry.example.com/acme/motlie-guest:v1").unwrap();
        let auth = resolve_registry_auth_with_env(
            &image,
            None,
            None,
            Some("REGISTRY_TOKEN"),
            fixture_env([
                ("GITHUB_ACTOR", "octocat"),
                ("REGISTRY_TOKEN", "registry-secret"),
            ]),
        )
        .unwrap();

        assert_bearer_auth(auth, "registry-secret");
    }

    #[test]
    fn registry_auth_password_requires_registry_username() {
        let image =
            OciImageReference::from_str("registry.example.com/acme/motlie-guest:v1").unwrap();
        let error = resolve_registry_auth_with_env(
            &image,
            None,
            Some("REGISTRY_PASSWORD"),
            None,
            fixture_env([
                ("GITHUB_ACTOR", "octocat"),
                ("REGISTRY_PASSWORD", "registry-secret"),
            ]),
        )
        .unwrap_err();

        assert!(error.to_string().contains("requires --username"));
    }

    #[test]
    fn oci_push_dry_run_writes_single_platform_plan() {
        let fixture = write_oci_export_fixture(
            "oci-push-single",
            OciPlatform::linux_arm64(),
            b"fake-rootfs-tar-arm64",
            "motlie-guest:v1.5-arm64",
        );
        let out = fixture.root.join("push-evidence.json");

        oci_push(OciPushOptions {
            layout: fixture.layout.clone(),
            image: "ghcr.io/chungers/motlie-guest:v1.5-arm64".to_string(),
            dry_run: true,
            allow_overwrite: false,
            username: Some("dry-run-should-not-resolve-auth".to_string()),
            password_env: None,
            token_env: None,
            out: Some(out.clone()),
        })
        .unwrap();

        let evidence: OciPushEvidence = serde_json::from_slice(&fs::read(out).unwrap()).unwrap();
        assert!(evidence.dry_run);
        assert_eq!(
            evidence.layout_kind,
            OciPushLayoutKind::SinglePlatformExport
        );
        assert_eq!(evidence.blobs.len(), 2);
        assert!(evidence
            .blobs
            .iter()
            .all(|blob| blob.status == OciBlobPushStatus::DryRun));
        assert_eq!(evidence.manifests.len(), 2);
        assert_eq!(evidence.remote_digest, None);

        fs::remove_dir_all(fixture.root).unwrap();
    }

    #[test]
    fn oci_push_dry_run_writes_multi_arch_index_plan() {
        let arm64 = write_oci_export_fixture(
            "oci-push-index-arm64",
            OciPlatform::linux_arm64(),
            b"fake-rootfs-tar-arm64",
            "motlie-guest:v1.5-arm64",
        );
        let amd64 = write_oci_export_fixture(
            "oci-push-index-amd64",
            OciPlatform::linux_amd64(),
            b"fake-rootfs-tar-amd64",
            "motlie-guest:v1.5-amd64",
        );
        let index_root = temp_test_dir("oci-push-index");
        oci_index(OciIndexOptions {
            out: index_root.clone(),
            image: "ghcr.io/chungers/motlie-guest:v1.5".to_string(),
            layouts: vec![amd64.layout.clone(), arm64.layout.clone()],
        })
        .unwrap();
        let out = index_root.join("push-evidence.json");

        oci_push(OciPushOptions {
            layout: index_root.clone(),
            image: "ghcr.io/chungers/motlie-guest:v1.5".to_string(),
            dry_run: true,
            allow_overwrite: false,
            username: None,
            password_env: None,
            token_env: None,
            out: Some(out.clone()),
        })
        .unwrap();

        let evidence: OciPushEvidence = serde_json::from_slice(&fs::read(out).unwrap()).unwrap();
        assert!(evidence.dry_run);
        assert_eq!(evidence.layout_kind, OciPushLayoutKind::MultiArchIndex);
        assert_eq!(evidence.blobs.len(), 4);
        assert!(evidence
            .blobs
            .iter()
            .all(|blob| blob.status == OciBlobPushStatus::DryRun));
        assert_eq!(evidence.manifests.len(), 3);
        assert!(evidence
            .manifests
            .iter()
            .any(|manifest| manifest.reference == "v1.5"));

        fs::remove_dir_all(arm64.root).unwrap();
        fs::remove_dir_all(amd64.root).unwrap();
        fs::remove_dir_all(index_root).unwrap();
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
            manager: PackageManagerId("dnf".to_string()),
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
    fn apk_package_specs_allow_common_apk_names() {
        let stage = PackageStage {
            manager: PackageManagerId("apk".to_string()),
            update: true,
            install: vec![
                "openssh-server".to_string(),
                "py3-cloud-init".to_string(),
                "libstdc++".to_string(),
                "foo_bar".to_string(),
            ],
            npm_global: Vec::new(),
            clean: true,
        };

        stage.validate().unwrap();
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
    fn ch_guest_target_comes_from_requested_platform() {
        let amd64 = ChGuestTarget::from_platform(OciPlatform::linux_amd64()).unwrap();
        let arm64 = ChGuestTarget::from_platform(OciPlatform::linux_arm64()).unwrap();

        assert_eq!(amd64.platform, OciPlatform::linux_amd64());
        assert_eq!(amd64.rust_target, "x86_64-unknown-linux-musl");
        assert_eq!(amd64.deb_arch, "amd64");
        assert_eq!(amd64.kernel_image, "vmlinux.bin");
        assert_eq!(amd64.kernel_asset, "bzImage-x86_64");
        assert_eq!(amd64.apk_arch(), "x86_64");
        assert_eq!(amd64.npm_arch(), "x64");
        assert_eq!(arm64.platform, OciPlatform::linux_arm64());
        assert_eq!(arm64.rust_target, "aarch64-unknown-linux-musl");
        assert_eq!(arm64.deb_arch, "arm64");
        assert_eq!(arm64.kernel_image, "Image");
        assert_eq!(arm64.kernel_asset, "Image-arm64");
        assert_eq!(arm64.apk_arch(), "aarch64");
        assert_eq!(arm64.npm_arch(), "arm64");
    }

    #[test]
    fn cargo_target_linker_env_formats_target_triple() {
        assert_eq!(
            cargo_target_linker_env("x86_64-unknown-linux-musl"),
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER"
        );
    }

    #[test]
    fn alpine_release_configs_declare_vz_emitters() {
        for config_name in [
            "motlie-image.alpine-3.22.linux-arm64.yaml",
            "motlie-image.alpine-3.22.linux-amd64.yaml",
        ] {
            let config = load_config(&release_config_path(config_name)).unwrap();
            let target = BackendTargetId("vz".to_string());
            let emitter = config.validate_for_target(&target).unwrap();

            assert_eq!(emitter.seed.motlie_backend, "vz");
            assert_eq!(emitter.seed.motlie_net_backend, "vz-userspace");
            assert_eq!(
                emitter.adapter.env.rootfs_tarball.as_deref(),
                Some("MOTLIE_V15_ASSEMBLED_ROOTFS_TARBALL")
            );
        }
    }

    #[test]
    fn apk_stage_uses_apk_static_and_restores_recorded_modes() {
        let script = apk_stage_script();

        assert!(script.contains(r#"apk_static="$(find_apk_static)""#));
        assert!(script.contains(r#"--root "$rootfs" --arch "$apk_arch""#));
        assert!(script.contains("restore_apk_recorded_modes"));
        assert!(script.contains(r#"local db="$rootfs/lib/apk/db/installed""#));
        assert!(script.contains(r#"chmod "$mode" "$path""#));
        assert!(script.contains(
            "chroot_env=(env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)"
        ));
        assert!(script.contains(r#"chroot "$rootfs" "${chroot_env[@]}" rc-update add"#));
        assert!(script.contains(r#""root boot""#));
        assert!(script.contains(r#""fsck boot""#));
        assert!(script.contains(r#""networking boot""#));
    }

    #[test]
    fn apk_stage_uses_guest_npm_platform_and_repairs_claude() {
        let script = apk_stage_script();

        assert!(script.contains("npm_config_platform=linux"));
        assert!(script.contains(r#"npm_config_arch="$npm_arch""#));
        assert!(script.contains("npm_config_libc=musl"));
        assert!(script.contains("@anthropic-ai/claude-code-linux-${npm_arch}-musl/bin/claude"));
        assert!(script.contains(r#"local wrapper="$claude_root/bin/claude.exe""#));
        assert!(script.contains("repair_claude_code_binary"));
    }

    #[test]
    fn ch_launcher_uses_ext4_raw_rootfs_disks() {
        let launcher =
            fs::read_to_string(repo_root_fixture().join("libs/vmm/examples/v1.5/launch-ch.sh"))
                .unwrap();

        assert!(launcher.contains("rootfstype=ext4"));
        assert!(
            launcher.contains("path=$BASE_ARTIFACTS/rootfs.squashfs,readonly=on,image_type=raw")
        );
        assert!(launcher.contains("path=$RUNTIME_OVERLAY,image_type=raw"));
        assert!(!launcher.contains("rootfstype=squashfs"));
    }

    #[test]
    fn artifact_collection_skips_vz_disk_image_hashes() {
        let root = temp_test_dir("artifact-skip-vz-disk");
        let vm_dir = root.join("motlie-v1-5-base-iter.vm");
        fs::create_dir_all(&vm_dir).unwrap();
        fs::write(vm_dir.join("disk.img"), b"large sparse disk placeholder").unwrap();
        fs::write(vm_dir.join("nvram.bin"), b"nvram").unwrap();

        let artifacts = collect_artifacts(&root, DEFAULT_MANIFEST_NAME).unwrap();

        assert!(!artifacts
            .iter()
            .any(|artifact| artifact.path.ends_with("motlie-v1-5-base-iter.vm/disk.img")));
        assert!(artifacts.iter().any(|artifact| artifact
            .path
            .ends_with("motlie-v1-5-base-iter.vm/nvram.bin")));
    }

    #[test]
    fn alpine_vz_external_oci_requires_payload() {
        let config_path = release_config_path("motlie-image.alpine-3.22.linux-arm64.yaml");
        let config = load_config(&config_path).unwrap();
        let target = BackendTargetId("vz".to_string());
        let emitter = config.validate_for_target(&target).unwrap();
        let options = BuildOptions {
            config_path,
            target,
            out: temp_test_dir("alpine-vz-no-payload"),
            repo_root: None,
            plan_only: false,
            adapter_args: Vec::new(),
            rootfs_tarball: None,
            oci_layout: None,
        };

        let error = run_build_execution(Path::new("."), &config, emitter, &options, None)
            .expect_err("Alpine VZ must not silently fall back to the native source VM");

        assert!(error
            .to_string()
            .contains("requires --oci-layout or --rootfs-tarball"));
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
                build_host: None,
                guest_target: None,
                rootfs_tarball: None,
            }),
            Vec::new(),
        );

        let error = manifest.validate_against_config(&config).unwrap_err();

        assert!(error
            .to_string()
            .contains("manifest adapter.materialized_source does not match config"));
    }

    #[test]
    fn oci_export_writes_image_layout_from_common_rootfs() {
        let root = temp_test_dir("oci-export");
        let config_path = root.join("motlie-image.yaml");
        let artifact = root.join("artifact");
        let out = root.join("oci");
        fs::create_dir_all(&artifact).unwrap();
        fs::create_dir_all(&out).unwrap();
        fs::write(&config_path, external_oci_config_yaml()).unwrap();
        fs::write(
            artifact.join(COMMON_ROOTFS_TARBALL_NAME),
            b"fake-rootfs-tar",
        )
        .unwrap();

        let config = load_config(&config_path).unwrap();
        let source = ExternalOciSource::ubuntu_systemd(
            OciPlatform::linux_arm64(),
            OciDigest::new(format!("sha256:{}", "a".repeat(64))).unwrap(),
            OciDigest::new(format!("sha256:{}", "b".repeat(64))).unwrap(),
        );
        let rootfs = rootfs_tarball_record(&artifact.join(COMMON_ROOTFS_TARBALL_NAME)).unwrap();
        let manifest = ImageBuildManifest::from_config(
            &config_path,
            BackendTargetId("ch".to_string()),
            &artifact,
            &config,
            Some(AdapterRecord {
                kind: "external-oci-ch-emitter".to_string(),
                command: vec!["mbuild".to_string(), "build".to_string()],
                log_path: artifact.join(CH_EMITTER_LOG_NAME),
                exit_status: 0,
                started_at_unix_seconds: 1,
                completed_at_unix_seconds: 2,
                package_include: config.package_stage.install.clone(),
                materialized_source: None,
                external_oci_source: Some(source.clone()),
                build_host: Some(BuildHostRecord {
                    platform: OciPlatform::linux_arm64().to_string(),
                    deb_arch: "arm64".to_string(),
                }),
                guest_target: Some(GuestTargetRecord {
                    platform: OciPlatform::linux_arm64().to_string(),
                    rust_target: "aarch64-unknown-linux-musl".to_string(),
                    deb_arch: "arm64".to_string(),
                    kernel_image: "Image".to_string(),
                    kernel_asset: "Image-arm64".to_string(),
                }),
                rootfs_tarball: Some(rootfs.clone()),
            }),
            Vec::new(),
        );
        fs::write(
            artifact.join(DEFAULT_MANIFEST_NAME),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        oci_export(OciExportOptions {
            config_path: config_path.clone(),
            artifact: artifact.clone(),
            out: out.clone(),
            tag: Some("motlie-guest:test".to_string()),
        })
        .unwrap();

        assert!(out.join("oci-layout").is_file());
        assert!(out.join("index.json").is_file());
        assert!(out.join("blobs/sha256").join(&rootfs.sha256).is_file());
        let export: OciExportManifest =
            serde_json::from_slice(&fs::read(out.join(DEFAULT_OCI_EXPORT_MANIFEST_NAME)).unwrap())
                .unwrap();
        assert_eq!(export.tag, "motlie-guest:test");
        assert_eq!(export.platform, "linux/arm64");
        assert_eq!(
            export.rootfs_layer.digest,
            format!("sha256:{}", rootfs.sha256)
        );
        let payload = load_validated_oci_payload(&out, &config).unwrap();
        assert_eq!(payload.source.platform.to_string(), "linux/arm64");
        assert_eq!(payload.rootfs_tarball.sha256, rootfs.sha256);
        assert_eq!(payload.rootfs_tarball.size_bytes, rootfs.size_bytes);
        let consumer_artifact = root.join("consumer-artifact");
        let consumer_manifest = ImageBuildManifest::from_config(
            &config_path,
            BackendTargetId("ch".to_string()),
            &consumer_artifact,
            &config,
            Some(AdapterRecord {
                kind: "oci-payload-ch-emitter".to_string(),
                command: vec!["mbuild".to_string(), "build".to_string()],
                log_path: consumer_artifact.join(CH_EMITTER_LOG_NAME),
                exit_status: 0,
                started_at_unix_seconds: 3,
                completed_at_unix_seconds: 4,
                package_include: config.package_stage.install.clone(),
                materialized_source: None,
                external_oci_source: Some(source),
                build_host: Some(BuildHostRecord {
                    platform: OciPlatform::linux_arm64().to_string(),
                    deb_arch: "arm64".to_string(),
                }),
                guest_target: Some(GuestTargetRecord {
                    platform: OciPlatform::linux_arm64().to_string(),
                    rust_target: "aarch64-unknown-linux-musl".to_string(),
                    deb_arch: "arm64".to_string(),
                    kernel_image: "Image".to_string(),
                    kernel_asset: "Image-arm64".to_string(),
                }),
                rootfs_tarball: Some(rootfs),
            }),
            Vec::new(),
        );
        validate_oci_layout(&out, &export, Some(&config), Some(&consumer_manifest)).unwrap();

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rootfs_tarball_extract_preserves_setuid_mode() {
        let root = temp_test_dir("extract-setuid");
        let src = root.join("src");
        let sudo = src.join("usr/bin/sudo");
        fs::create_dir_all(sudo.parent().unwrap()).unwrap();
        fs::write(&sudo, b"sudo").unwrap();
        fs::set_permissions(&sudo, fs::Permissions::from_mode(0o4755)).unwrap();
        let tarball = root.join("rootfs.tar");
        let status = Command::new("tar")
            .arg("-C")
            .arg(&src)
            .arg("-cf")
            .arg(&tarball)
            .arg(".")
            .status()
            .unwrap();
        assert!(status.success());

        let out = root.join("out");
        extract_rootfs_tarball(&tarball, &out, &root.join("extract.log")).unwrap();

        let mode = fs::metadata(out.join("usr/bin/sudo"))
            .unwrap()
            .permissions()
            .mode()
            & 0o7777;
        assert_eq!(mode, 0o4755);

        fs::remove_dir_all(root).unwrap();
    }
}
