mod pty;
mod shell;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use motlie_vmm::backend::BackendError;
use motlie_vmm::ca::SshCa;
use motlie_vmm::guestfs::GuestFsError;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::observability::VmObservability;
use motlie_vmm::orchestrator::{
    LifecycleServices, OrchestratorError, PrepareRequest, ReadinessPolicy, ShutdownReport,
    VmHandle, boot, prepare,
};
use motlie_vmm::runtime::{
    ControlPlaneBacking, FilesystemBacking, HypervisorBacking, NetworkBacking, Runtime,
    RuntimeError,
};
use motlie_vmm::spec::{
    BootArtifacts, GuestMountSpec, GuestResources, GuestSpec, GuestSshAccess, GuestStorage,
    GuestUser, RuntimeNamespace, SoftwareProfile,
};
use motlie_vmm::ssh::{
    self, ExecOutput, PtyTranscriptEvent, SshProxyConfig, SshProxyError, new_guest_registry,
};
use pty::{PtyScenarioError, PtyScenarioResult};
use serde::Serialize;
use thiserror::Error;
use tokio::time::sleep;

type DynError = Box<dyn std::error::Error + Send + Sync>;

struct HarnessInstance {
    namespace: RuntimeNamespace,
    demo_root: PathBuf,
    socket_root: PathBuf,
    proxy_port: u16,
}

#[derive(Debug, Serialize)]
struct ScenarioCheck {
    name: String,
    detail: String,
}

#[derive(Debug, Serialize)]
struct ScenarioResult {
    status: ScenarioStatus,
    scenario: String,
    guest_id: String,
    pid: Option<u32>,
    proxy: String,
    shutdown: Option<ShutdownReport>,
    shutdown_forced: Option<String>,
    observability: Option<VmObservability>,
    checks: Vec<ScenarioCheck>,
    pty: Option<PtyScenarioResult>,
    pty_transcript: Option<Vec<PtyTranscriptEvent>>,
    error: Option<ScenarioFailure>,
    cleanup_error: Option<ScenarioFailure>,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ScenarioStatus {
    Passed,
    Failed,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FailureClass {
    Config,
    Artifact,
    Backend,
    Filesystem,
    Network,
    Ssh,
    Readiness,
    Pty,
    Assertion,
    Shutdown,
    Internal,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct ScenarioFailure {
    class: FailureClass,
    stage: &'static str,
    code: &'static str,
    message: String,
}

#[derive(Debug, Error)]
enum HarnessError {
    #[error("required artifact missing: {path}")]
    MissingArtifact { path: PathBuf },
    #[error("unsupported scenario: {0}")]
    UnsupportedScenario(String),
    #[error("prepare failed: {0}")]
    Prepare(#[source] OrchestratorError),
    #[error("boot failed: {0}")]
    Boot(#[source] OrchestratorError),
    #[error("readiness failed: {0}")]
    Ready(#[source] OrchestratorError),
    #[error("smoke exec '{check}' failed: {source}")]
    SmokeExec {
        check: &'static str,
        #[source]
        source: OrchestratorError,
    },
    #[error("smoke check '{check}' expected {expected}, got: {observed_excerpt}")]
    SmokeAssertion {
        check: &'static str,
        expected: String,
        observed_excerpt: String,
    },
    #[error("failed to persist harness artifact {path}: {reason}")]
    ArtifactPersist { path: PathBuf, reason: String },
    #[error(transparent)]
    Pty(#[from] PtyScenarioError),
    #[error("shutdown failed: {0}")]
    Shutdown(#[source] OrchestratorError),
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), DynError> {
    let mut args = std::env::args().skip(1);
    let mut scenario = "smoke".to_string();
    let mut root_override: Option<PathBuf> = None;
    let mut result_json_path: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--root requires a path".to_string())?;
                root_override = Some(PathBuf::from(value));
            }
            "--result-json" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--result-json requires a path".to_string())?;
                result_json_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other if other.starts_with("--root=") => {
                root_override = Some(PathBuf::from(other.trim_start_matches("--root=")));
            }
            other if other.starts_with("--result-json=") => {
                result_json_path = Some(PathBuf::from(other.trim_start_matches("--result-json=")));
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}").into());
            }
            other => {
                scenario = other.to_string();
            }
        }
    }

    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/v1.4");
    let artifacts_dir = base_dir.join("artifacts/base");
    ensure_file_exists(&artifacts_dir.join("rootfs.squashfs"))
        .map_err(|err| -> DynError { Box::new(err) })?;
    ensure_file_exists(&artifacts_dir.join("Image"))
        .map_err(|err| -> DynError { Box::new(err) })?;

    let root_dir = root_override.unwrap_or_else(RuntimeNamespace::root_from_env_or_temp);
    let instance = new_harness_instance(&root_dir)?;
    if scenario == "shell" {
        return shell::run_shell(&base_dir, &artifacts_dir, &instance).await;
    }

    let guest = demo_guest(
        "alice",
        &artifacts_dir,
        &instance.demo_root,
        &instance.namespace,
    );
    std::fs::create_dir_all(&instance.socket_root)?;

    seed_host_mounts(&guest)?;

    let ca = Arc::new(SshCa::new()?);
    let guest_registry = new_guest_registry();
    let proxy_config = SshProxyConfig {
        listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), instance.proxy_port),
    };
    let proxy = format!("ssh://localhost:{}", proxy_config.listen.port());
    tokio::spawn(ssh::run_proxy(
        proxy_config.clone(),
        Arc::clone(&guest_registry),
    ));
    print_instance_details(&instance, &proxy_config);
    let runtime = Arc::new(Runtime {
        hypervisor: HypervisorBacking::CloudHypervisorShell(
            motlie_vmm::backend::ch::shell::ChShellBackend::new(),
        ),
        filesystem: FilesystemBacking::MotlieVfs(
            motlie_vmm::backend::motlie::vfs::MotlieVfsBacking::new(),
        ),
        network: NetworkBacking::MotlieVnet(
            motlie_vmm::backend::motlie::vnet::MotlieVnetBacking::new(),
        ),
        control_plane: ControlPlaneBacking::MotlieSshProxy(
            motlie_vmm::backend::motlie::ssh_proxy::MotlieSshProxyBacking::new(
                Arc::clone(&ca),
                Arc::clone(&guest_registry),
            ),
        ),
    });

    let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig {
        socket_dir: instance.socket_root.clone(),
        ..GuestNetAllocatorConfig::default()
    });
    let prepared = prepare(
        PrepareRequest {
            guest,
            namespace: instance.namespace.clone(),
            network_modes: NetworkModes {
                admin: AdminNetMode::None,
                egress: EgressNetMode::VhostUser,
            },
            base_dir: base_dir.clone(),
            ssh_ca_pubkey: Some(ca.public_key_openssh()?),
        },
        &mut allocator,
    )
    .map_err(HarnessError::Prepare)?;
    let mut handle: Option<VmHandle> = None;
    let mut observability: Option<VmObservability> = None;
    let mut checks = Vec::new();
    let mut pty: Option<PtyScenarioResult> = None;
    let mut pty_transcript: Option<Vec<PtyTranscriptEvent>> = None;

    let run_error = {
        let run = async {
            let booted = boot(prepared, LifecycleServices { runtime })
                .await
                .map_err(HarnessError::Boot)?;
            handle = Some(booted);

            let active_handle = handle
                .as_ref()
                .expect("handle is set immediately after boot succeeds");
            active_handle
                .ready(&ReadinessPolicy::default())
                .await
                .map_err(HarnessError::Ready)?;
            observability = Some(active_handle.observability());

            match scenario.as_str() {
                "smoke" => {
                    checks = run_smoke(active_handle).await?;
                }
                "pty" => {
                    let transcript_path = observability
                        .as_ref()
                        .expect("observability exists after readiness")
                        .run_bundle
                        .capture_paths
                        .pty_transcript_json
                        .clone();
                    let pty_run =
                        pty::run_pty_smoke(active_handle, transcript_path.clone()).await?;
                    persist_json(&transcript_path, &pty_run.transcript).map_err(|source| {
                        HarnessError::ArtifactPersist {
                            path: transcript_path.clone(),
                            reason: source.to_string(),
                        }
                    })?;
                    pty = Some(pty_run.result);
                    pty_transcript = Some(pty_run.transcript);
                    checks.push(ScenarioCheck {
                        name: "pty".to_string(),
                        detail: "PTY banner, prompt, resize, transcript, and terminal-close checks passed"
                            .to_string(),
                    });
                }
                other => return Err(HarnessError::UnsupportedScenario(other.to_string())),
            }
            Ok::<(), HarnessError>(())
        };
        run.await.err()
    };

    if let Some(active_handle) = handle.as_ref() {
        observability = Some(active_handle.observability());
    }

    let mut error = run_error.as_ref().map(classify_failure);
    let mut cleanup_error = None;
    let mut shutdown = None;
    if let Some(active_handle) = handle.as_ref() {
        match active_handle.shutdown().await {
            Ok(report) => shutdown = Some(report),
            Err(err) => {
                let failure = classify_failure(&HarnessError::Shutdown(err));
                if error.is_none() {
                    error = Some(failure);
                } else {
                    cleanup_error = Some(failure);
                }
            }
        }
    }

    let result = ScenarioResult {
        status: if error.is_none() {
            ScenarioStatus::Passed
        } else {
            ScenarioStatus::Failed
        },
        scenario: scenario.clone(),
        guest_id: "alice".to_string(),
        pid: shutdown
            .as_ref()
            .and_then(|report| report.pid)
            .or_else(|| handle.as_ref().and_then(|h| h.pid)),
        proxy: proxy.clone(),
        shutdown: shutdown.clone(),
        shutdown_forced: shutdown
            .as_ref()
            .and_then(|report| report.forced.map(str::to_string)),
        observability,
        checks,
        pty,
        pty_transcript,
        error,
        cleanup_error,
    };
    if let Some(internal_path) = result
        .observability
        .as_ref()
        .map(|obs| obs.run_bundle.capture_paths.scenario_result_json.clone())
    {
        persist_json(&internal_path, &result)?;
    }
    if let Some(path) = result_json_path.as_ref() {
        persist_json(path, &result)?;
    }

    match result.status {
        ScenarioStatus::Passed => {
            println!(
                "v1.4 harness {} passed: guest={} pid={:?} forced={:?} proxy=127.0.0.1:{}",
                scenario,
                result.guest_id,
                result.pid,
                result.shutdown_forced,
                proxy_config.listen.port()
            );
            Ok(())
        }
        ScenarioStatus::Failed => {
            let failure = result
                .error
                .as_ref()
                .map(|err| format!("{} [{}:{}]", err.message, err.stage, err.code))
                .unwrap_or_else(|| "unknown harness failure".to_string());
            Err(failure.into())
        }
    }
}

async fn run_smoke(handle: &VmHandle) -> Result<Vec<ScenarioCheck>, HarnessError> {
    let mut checks = Vec::new();
    let hello = handle
        .exec("/bin/echo hello", Duration::from_secs(10))
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "hello",
            source,
        })?;
    ensure_contains("hello", &hello.stdout, "hello")?;
    checks.push(ScenarioCheck {
        name: "hello".to_string(),
        detail: "programmatic exec returned hello".to_string(),
    });

    let vfs = handle
        .exec(
            "/bin/sh -lc 'pwd && test -d /home/alice && test -d /workspace && test -d /agent-state && grep -q \"Alice workspace mounted from the host.\" /workspace/README.md && echo VFS_OK'",
            Duration::from_secs(10),
        )
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "vfs",
            source,
        })?;
    ensure_contains("vfs", &vfs.stdout, "VFS_OK")?;
    checks.push(ScenarioCheck {
        name: "vfs".to_string(),
        detail: "home, workspace, and agent-state mounts are visible".to_string(),
    });

    let route = exec_until_success(
        handle,
        r#"/bin/sh -lc 'ip route | grep -q "^default via 10.0.2.2 " && echo ROUTE_OK'"#,
        "ROUTE_OK",
        Duration::from_secs(10),
    )
    .await
    .map_err(|source| HarnessError::SmokeExec {
        check: "route",
        source,
    })?;
    ensure_contains("route", &route.stdout, "ROUTE_OK")?;
    checks.push(ScenarioCheck {
        name: "route".to_string(),
        detail: "default route points at Motlie vnet".to_string(),
    });

    let outbound = exec_until_success(
        handle,
        r#"/bin/sh -lc 'code=$(curl -s -o /dev/null -w "%{http_code}" https://example.com); test "$code" = 200 && echo HTTPS_OK'"#,
        "HTTPS_OK",
        Duration::from_secs(20),
    )
    .await
    .map_err(|source| HarnessError::SmokeExec {
        check: "https",
        source,
    })?;
    ensure_contains("https", &outbound.stdout, "HTTPS_OK")?;
    checks.push(ScenarioCheck {
        name: "https".to_string(),
        detail: "outbound HTTPS fetch succeeded".to_string(),
    });
    Ok(checks)
}

fn ensure_file_exists(path: &Path) -> Result<(), HarnessError> {
    if path.exists() {
        Ok(())
    } else {
        Err(HarnessError::MissingArtifact {
            path: path.to_path_buf(),
        })
    }
}

fn ensure_contains(check: &'static str, stdout: &str, needle: &str) -> Result<(), HarnessError> {
    if stdout.contains(needle) {
        Ok(())
    } else {
        Err(HarnessError::SmokeAssertion {
            check,
            expected: format!("output containing '{needle}'"),
            observed_excerpt: excerpt(stdout),
        })
    }
}

fn persist_json<T: Serialize>(path: &Path, value: &T) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn classify_failure(error: &HarnessError) -> ScenarioFailure {
    match error {
        HarnessError::MissingArtifact { .. } => ScenarioFailure {
            class: FailureClass::Artifact,
            stage: "setup",
            code: "artifact_missing",
            message: error.to_string(),
        },
        HarnessError::UnsupportedScenario(_) => ScenarioFailure {
            class: FailureClass::Config,
            stage: "setup",
            code: "unsupported_scenario",
            message: error.to_string(),
        },
        HarnessError::Prepare(source) => classify_orchestrator("prepare", source),
        HarnessError::Boot(source) => classify_orchestrator("boot", source),
        HarnessError::Ready(source) => classify_orchestrator("ready", source),
        HarnessError::SmokeExec { source, .. } => classify_orchestrator("smoke", source),
        HarnessError::SmokeAssertion { .. } => ScenarioFailure {
            class: FailureClass::Assertion,
            stage: "smoke",
            code: "assertion_failed",
            message: error.to_string(),
        },
        HarnessError::ArtifactPersist { .. } => ScenarioFailure {
            class: FailureClass::Artifact,
            stage: "capture",
            code: "artifact_persist_failed",
            message: error.to_string(),
        },
        HarnessError::Pty(source) => classify_pty_failure(source),
        HarnessError::Shutdown(source) => {
            let mut failure = classify_orchestrator("shutdown", source);
            failure.class = FailureClass::Shutdown;
            failure
        }
    }
}

fn classify_pty_failure(error: &PtyScenarioError) -> ScenarioFailure {
    match error {
        PtyScenarioError::Open(source) => classify_orchestrator("pty", source),
        PtyScenarioError::ControlPlane(source) => classify_ssh_failure("pty", source),
        PtyScenarioError::Assertion { .. } => ScenarioFailure {
            class: FailureClass::Pty,
            stage: "pty",
            code: "pty_assertion_failed",
            message: error.to_string(),
        },
        PtyScenarioError::EmptyTranscript => ScenarioFailure {
            class: FailureClass::Pty,
            stage: "pty",
            code: "pty_transcript_empty",
            message: error.to_string(),
        },
        PtyScenarioError::IncompleteTranscript => ScenarioFailure {
            class: FailureClass::Pty,
            stage: "pty",
            code: "pty_transcript_incomplete",
            message: error.to_string(),
        },
    }
}

fn classify_orchestrator(stage: &'static str, error: &OrchestratorError) -> ScenarioFailure {
    match error {
        OrchestratorError::Spec(_) => ScenarioFailure {
            class: FailureClass::Config,
            stage,
            code: "spec_invalid",
            message: error.to_string(),
        },
        OrchestratorError::NetworkMode(_) => ScenarioFailure {
            class: FailureClass::Config,
            stage,
            code: "network_mode_invalid",
            message: error.to_string(),
        },
        OrchestratorError::Artifact(_) => ScenarioFailure {
            class: FailureClass::Artifact,
            stage,
            code: "artifact_render_failed",
            message: error.to_string(),
        },
        OrchestratorError::Backend(backend) => classify_backend_failure(stage, backend),
        OrchestratorError::Runtime(runtime) => classify_runtime_failure(stage, runtime),
        OrchestratorError::NetworkAllocation(_) => ScenarioFailure {
            class: FailureClass::Network,
            stage,
            code: "network_allocation_failed",
            message: error.to_string(),
        },
        OrchestratorError::MissingSshBridge { .. } => ScenarioFailure {
            class: FailureClass::Ssh,
            stage,
            code: "ssh_bridge_missing",
            message: error.to_string(),
        },
        OrchestratorError::StatePoisoned(_) => ScenarioFailure {
            class: FailureClass::Internal,
            stage,
            code: "state_poisoned",
            message: error.to_string(),
        },
        OrchestratorError::GuestExitedEarly { .. } => ScenarioFailure {
            class: FailureClass::Readiness,
            stage,
            code: "guest_exited_early",
            message: error.to_string(),
        },
        OrchestratorError::ReadinessTimeout { .. } => ScenarioFailure {
            class: FailureClass::Readiness,
            stage,
            code: "readiness_timeout",
            message: error.to_string(),
        },
    }
}

fn classify_runtime_failure(stage: &'static str, error: &RuntimeError) -> ScenarioFailure {
    match error {
        RuntimeError::Backend(backend) => classify_backend_failure(stage, backend),
        RuntimeError::GuestFs(GuestFsError::EmptyGuestId)
        | RuntimeError::GuestFs(GuestFsError::EmptySocketPath)
        | RuntimeError::GuestFs(GuestFsError::RemoveSocket { .. })
        | RuntimeError::GuestFs(GuestFsError::BindSocket { .. })
        | RuntimeError::GuestFs(GuestFsError::CreateMountPath { .. })
        | RuntimeError::GuestFs(GuestFsError::AddMount { .. })
        | RuntimeError::GuestFs(GuestFsError::WaitForMounts { .. })
        | RuntimeError::GuestFs(GuestFsError::CleanupSocket { .. }) => ScenarioFailure {
            class: FailureClass::Filesystem,
            stage,
            code: "guestfs_failed",
            message: error.to_string(),
        },
        RuntimeError::Ssh(ssh) => classify_ssh_failure(stage, ssh),
        RuntimeError::Vnet(_) | RuntimeError::VnetShutdown(_) => ScenarioFailure {
            class: FailureClass::Network,
            stage,
            code: "vnet_failed",
            message: error.to_string(),
        },
        RuntimeError::UnsupportedHypervisor => ScenarioFailure {
            class: FailureClass::Backend,
            stage,
            code: "unsupported_hypervisor",
            message: error.to_string(),
        },
    }
}

fn classify_backend_failure(stage: &'static str, error: &BackendError) -> ScenarioFailure {
    ScenarioFailure {
        class: FailureClass::Backend,
        stage,
        code: match error {
            BackendError::CreateRuntimeDir { .. } => "backend_runtime_dir_failed",
            BackendError::ChShell(_) => "backend_ch_shell_failed",
            BackendError::UnsupportedBackend(_) => "backend_unsupported",
            BackendError::HandleKindMismatch { .. } => "backend_handle_mismatch",
        },
        message: error.to_string(),
    }
}

fn classify_ssh_failure(stage: &'static str, error: &SshProxyError) -> ScenarioFailure {
    ScenarioFailure {
        class: FailureClass::Ssh,
        stage,
        code: match error {
            SshProxyError::GuestConnection { .. } => "ssh_guest_connection_failed",
            SshProxyError::ExecFailed { .. } => "ssh_exec_failed",
            SshProxyError::Ca(_) => "ssh_ca_failed",
            SshProxyError::Ssh(_) => "ssh_transport_failed",
            SshProxyError::ChannelClosed => "ssh_channel_closed",
            SshProxyError::MissingExitStatus { .. } => "ssh_missing_exit_status",
            SshProxyError::UnknownGuest(_) => "ssh_unknown_guest",
            SshProxyError::StatePoisoned(_) => "ssh_state_poisoned",
            SshProxyError::PtyTimeout { .. } => "pty_timeout",
            SshProxyError::Unsupported(_) => "ssh_unsupported",
        },
        message: error.to_string(),
    }
}

fn excerpt(output: &str) -> String {
    const LIMIT: usize = 160;
    let normalized = output.replace('\n', "\\n");
    if normalized.len() <= LIMIT {
        normalized
    } else {
        format!("{}...", &normalized[..LIMIT])
    }
}

fn print_usage() {
    println!("usage: harness_v1_4 [smoke|pty|shell] [--root <dir>] [--result-json <path>]");
}

fn new_harness_instance(root_dir: &Path) -> Result<HarnessInstance, DynError> {
    let namespace = RuntimeNamespace::for_process("motlie-vmm-v14", "h", root_dir)?;
    let demo_root = namespace
        .temp_root
        .join(format!("{}-demo", namespace.prefix));
    let socket_root = namespace
        .temp_root
        .join(format!("{}-sockets", namespace.prefix));
    let proxy_port = 32000 + port_offset(&namespace.prefix);
    Ok(HarnessInstance {
        namespace,
        demo_root,
        socket_root,
        proxy_port,
    })
}

fn print_instance_details(instance: &HarnessInstance, proxy_config: &SshProxyConfig) {
    println!("v1.4 harness instance: {}", instance.namespace.prefix);
    println!("  demo_root={}", instance.demo_root.display());
    println!("  socket_root={}", instance.socket_root.display());
    println!("  proxy=ssh://localhost:{}", proxy_config.listen.port());
}

fn port_offset(seed: &str) -> u16 {
    seed.bytes().fold(0u16, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u16::from(byte))
    }) % 10_000
}

fn demo_guest(
    guest_id: &str,
    artifacts_dir: &Path,
    demo_root: &Path,
    namespace: &RuntimeNamespace,
) -> GuestSpec {
    let (uid, gid) = demo_guest_ids(guest_id);
    GuestSpec {
        guest_id: guest_id.to_string(),
        hostname: format!("motlie-{guest_id}"),
        socket_path: namespace
            .guest_vsock_port_socket(guest_id, 5000)
            .expect("guest_id is validated by the harness")
            .display()
            .to_string(),
        user: GuestUser {
            name: guest_id.to_string(),
            uid,
            gid,
            home: PathBuf::from(format!("/home/{guest_id}")),
        },
        ssh: GuestSshAccess {
            principal: guest_id.to_string(),
            login_user: guest_id.to_string(),
        },
        mounts: vec![
            GuestMountSpec {
                tag: format!("{guest_id}-home"),
                guest_path: Some(PathBuf::from(format!("/home/{guest_id}"))),
                host_path: demo_root.join(format!("{guest_id}-home")),
            },
            GuestMountSpec {
                tag: format!("{guest_id}-workspace"),
                guest_path: Some(PathBuf::from("/workspace")),
                host_path: demo_root.join(format!("{guest_id}-workspace")),
            },
            GuestMountSpec {
                tag: format!("{guest_id}-agent-state"),
                guest_path: Some(PathBuf::from("/agent-state")),
                host_path: demo_root.join(format!("{guest_id}-agent-state")),
            },
        ],
        software: SoftwareProfile {
            packages: vec!["vim".to_string()],
        },
        resources: GuestResources::default(),
        storage: GuestStorage::default(),
        boot: BootArtifacts {
            kernel: artifacts_dir.join("Image"),
            initramfs: None,
            firmware: None,
            cmdline: None,
        },
    }
}

fn seed_host_mounts(guest: &GuestSpec) -> Result<(), DynError> {
    for mount in &guest.mounts {
        std::fs::create_dir_all(&mount.host_path)?;
    }
    let home = &guest.mounts[0].host_path;
    let ssh_dir = home.join(".ssh");
    std::fs::create_dir_all(home.join(".config"))?;
    std::fs::create_dir_all(&ssh_dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&ssh_dir, std::fs::Permissions::from_mode(0o700))?;
    }
    write_host_file_if_missing(
        &ssh_dir.join("authorized_keys"),
        &format!(
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample {}@dev\n",
            guest.guest_id
        ),
        0o600,
    )?;
    write_host_file_if_missing(
        &ssh_dir.join("config"),
        "Host github.com\n  User git\n",
        0o644,
    )?;
    write_host_file_if_missing(
        &home.join(".env"),
        &format!(
            "{}_API_KEY=demo-{}\n",
            guest.guest_id.to_uppercase(),
            guest.guest_id
        ),
        0o644,
    )?;
    write_host_file_if_missing(&home.join(".bashrc"), "# motlie v1.4 demo bashrc\n", 0o644)?;
    write_host_file_if_missing(
        &home.join(".profile"),
        "if [ -f \"$HOME/.bashrc\" ]; then\n  . \"$HOME/.bashrc\"\nfi\n",
        0o644,
    )?;
    write_host_file_if_missing(
        &guest.mounts[2].host_path.join("README.md"),
        "Dedicated read-write agent-state layer for Codex and Claude lives here.\n",
        0o644,
    )?;
    std::fs::write(
        guest.mounts[1].host_path.join("README.md"),
        format!(
            "{} workspace mounted from the host.\n",
            guest_display_name(&guest.guest_id)
        ),
    )?;
    Ok(())
}

fn write_host_file_if_missing(path: &Path, content: &str, mode: u32) -> Result<(), DynError> {
    if !path.exists() {
        std::fs::write(path, content)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
        }
    }
    Ok(())
}

fn demo_guest_ids(guest_id: &str) -> (u32, u32) {
    match guest_id {
        "alice" => (1000, 1000),
        "bob" => (1001, 1001),
        _ => (1000, 1000),
    }
}

fn guest_display_name(guest_id: &str) -> String {
    let mut chars = guest_id.chars();
    match chars.next() {
        Some(first) => {
            let mut out = first.to_uppercase().collect::<String>();
            out.push_str(chars.as_str());
            out
        }
        None => String::new(),
    }
}

async fn exec_until_success(
    handle: &VmHandle,
    command: &str,
    needle: &str,
    timeout: Duration,
) -> Result<ExecOutput, OrchestratorError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last_output: Option<ExecOutput> = None;
    let mut last_error: Option<OrchestratorError> = None;
    loop {
        match handle.exec(command, Duration::from_secs(10)).await {
            Ok(output) if output.exit_code == 0 && output.stdout.contains(needle) => {
                return Ok(output);
            }
            Ok(output) => last_output = Some(output),
            Err(err) => last_error = Some(err),
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(last_error.unwrap_or_else(|| match last_output {
                Some(output) => OrchestratorError::Runtime(RuntimeError::Ssh(SshProxyError::ExecFailed {
                    guest: handle.guest_id.clone(),
                    reason: format!(
                        "timed out waiting for command success: cmd={command} exit={} stdout={} stderr={}",
                        output.exit_code, output.stdout, output.stderr
                    ),
                })),
                None => OrchestratorError::Runtime(RuntimeError::Ssh(SshProxyError::ExecFailed {
                    guest: handle.guest_id.clone(),
                    reason: format!("timed out waiting for command success: cmd={command}"),
                })),
            }));
        }

        sleep(Duration::from_secs(1)).await;
    }
}
