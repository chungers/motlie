mod backend;
#[path = "../demo_support.rs"]
mod demo_support;
mod pty;
mod scenario;
mod shell;
mod terminal;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use motlie_vmm::backend::vz::egress::{
    run_vz_userspace_egress, VzHostForward, VzUserspaceEgressConfig,
};
use motlie_vmm::backend::BackendError;
use motlie_vmm::ca::SshCa;
use motlie_vmm::guestfs::GuestFsError;
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig, Ipv4Subnet};
use motlie_vmm::observability::VmObservability;
use motlie_vmm::orchestrator::{
    boot, prepare, LifecycleServices, OrchestratorError, PrepareRequest, ShutdownReport, VmHandle,
};
use motlie_vmm::provisioning::{
    GuestProvisioner, GuestProvisionerConfig, ProvisioningGuestRequest,
};
use motlie_vmm::runtime::{Runtime, RuntimeError};
use motlie_vmm::spec::{
    GuestMountSpec, GuestResources, GuestSpec, GuestSshAccess, GuestStorage, GuestUser,
    RuntimeNamespace, SoftwareProfile,
};
use motlie_vmm::ssh::{
    self, new_guest_registry, ExecOutput, PtyTranscriptEvent, SshProxyConfig, SshProxyError,
};
use pty::{PtyScenarioError, PtyScenarioResult};
use scenario::{ScenarioRunResult, ScenarioRunStatus};
use serde::Serialize;
use terminal::TerminalBackendKind;
use thiserror::Error;
use tokio::time::sleep;

use backend::HarnessBackend;
use demo_support::{cleanup_development_guest_disks, demo_guest_ids, demo_guest_socket_path};

type DynError = Box<dyn std::error::Error + Send + Sync>;

pub(crate) const PACKAGE_MANAGER_QUIESCENT_COMMAND: &str = "/bin/sh -lc 'if ! pgrep -x apt >/dev/null 2>&1 && ! pgrep -x apt-get >/dev/null 2>&1 && ! pgrep -x dpkg >/dev/null 2>&1 && ! pgrep -x unattended-upgr >/dev/null 2>&1 && ! pgrep -x unattended-upgrade >/dev/null 2>&1; then echo PKG_IDLE_OK; exit 0; fi; exit 1'";
pub(crate) const EGRESS_READY_COMMAND: &str = "/bin/sh -lc 'set -eu; getent hosts example.com >/dev/null; getent hosts www.google.com >/dev/null; code=$(curl -sS -o /dev/null -w \"%{http_code}\" --connect-timeout 5 --max-time 15 https://example.com); test \"$code\" = 200; code=$(curl -sS -o /dev/null -w \"%{http_code}\" --connect-timeout 5 --max-time 15 https://www.google.com/generate_204); test \"$code\" = 204; echo EGRESS_OK'";
pub(crate) const VFS_MEMFS_LAYER_COMMAND: &str = r#"/bin/sh -lc 'set -eu; user=$(id -un); home_dir=$(getent passwd "$user" | cut -d: -f6); for path in "$home_dir" /workspace /agent-state; do test -d "$path"; grep -E "^[^ ]+ ${path} fuse " /proc/mounts >/dev/null; test -w "$path"; done; test -f "$home_dir/.env"; test -f /workspace/README.md; test -f /agent-state/README.md; printf "vfs-home-%s\n" "$user" > "$home_dir/.motlie-vfs-write-test"; printf "vfs-workspace-%s\n" "$user" > /workspace/.motlie-vfs-write-test; printf "vfs-agent-state-%s\n" "$user" > /agent-state/.motlie-vfs-write-test; grep -q "vfs-home-$user" "$home_dir/.motlie-vfs-write-test"; grep -q "vfs-workspace-$user" /workspace/.motlie-vfs-write-test; grep -q "vfs-agent-state-$user" /agent-state/.motlie-vfs-write-test; echo VFS_MEMFS_OK'"#;
pub(crate) const APT_UPDATE_COMMAND: &str = r#"/bin/sh -lc 'set -eu; log=/tmp/motlie-vmm-apt-update.log; if sudo -n timeout --signal=TERM --kill-after=10 75s apt-get update -o Acquire::Retries=0 >"$log" 2>&1; then echo APT_OK; else cat "$log"; exit 1; fi'"#;
pub(crate) const AGENT_CLI_START_COMMAND: &str = r#"/bin/sh -lc 'set -eu; export TERM=dumb; export CI=1; command -v codex >/dev/null; command -v claude >/dev/null; codex_out="$(timeout 20s codex --version 2>&1)" || { printf "%s\n" "$codex_out"; exit 1; }; claude_out="$(timeout 20s claude --version 2>&1)" || { printf "%s\n" "$claude_out"; exit 1; }; combined="$(printf "%s\n%s\n" "$codex_out" "$claude_out")"; if printf "%s\n" "$combined" | grep -Eiq "os error|operation not permitted|permission denied|no such file|enoent|eacces"; then printf "%s\n" "$combined"; exit 1; fi; printf "%s\n" "$combined"; echo AGENT_CLI_OK'"#;
pub(crate) const SSH_PROXY_READY_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) fn resolved_native_source_dir(base_dir: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("MOTLIE_VZ_BASE_VM_DIR").map(PathBuf::from) {
        if path.join("disk.img").exists() && path.join("nvram.bin").exists() {
            return path;
        }
    }
    let local = base_dir.join("artifacts/source-base.vm");
    if local.join("disk.img").exists() && local.join("nvram.bin").exists() {
        local
    } else {
        base_dir.join("../v1.35/artifacts/source-base.vm")
    }
}

pub(crate) async fn wait_for_proxy_listener(
    listen: SocketAddr,
    timeout: Duration,
) -> Result<(), DynError> {
    let started = std::time::Instant::now();

    loop {
        match tokio::net::TcpStream::connect(listen).await {
            Ok(stream) => {
                drop(stream);
                return Ok(());
            }
            Err(err) => {
                if started.elapsed() >= timeout {
                    return Err(format!(
                        "SSH proxy on {listen} did not become ready within {}s: {}",
                        timeout.as_secs_f32(),
                        err
                    )
                    .into());
                }
                sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct HarnessInstance {
    namespace: RuntimeNamespace,
    demo_root: PathBuf,
    socket_root: PathBuf,
    proxy_port: u16,
}

#[derive(Clone)]
pub(crate) struct HarnessAllocatorOptions {
    first_cid: u32,
    max_guests: Option<u32>,
    admin_base: Ipv4Subnet,
    admin_guest_prefix: u8,
    egress_base: Ipv4Subnet,
    egress_guest_prefix: u8,
}

impl Default for HarnessAllocatorOptions {
    fn default() -> Self {
        let defaults = GuestNetAllocatorConfig::default();
        Self {
            first_cid: defaults.first_cid,
            max_guests: defaults.max_guests,
            admin_base: defaults.admin_pool.base,
            admin_guest_prefix: defaults.admin_pool.guest_prefix_len,
            egress_base: defaults.egress_pool.base,
            egress_guest_prefix: defaults.egress_pool.guest_prefix_len,
        }
    }
}

impl HarnessAllocatorOptions {
    fn build(&self, socket_root: &Path, socket_name_prefix: &str) -> GuestNetAllocatorConfig {
        let mut config = GuestNetAllocatorConfig {
            first_cid: self.first_cid,
            max_guests: self.max_guests,
            socket_dir: socket_root.to_path_buf(),
            socket_name_prefix: socket_name_prefix.to_string(),
            admin_pool: GuestNetAllocatorConfig::default().admin_pool,
            egress_pool: GuestNetAllocatorConfig::default().egress_pool,
        };
        config.admin_pool.base = self.admin_base;
        config.admin_pool.guest_prefix_len = self.admin_guest_prefix;
        config.egress_pool.base = self.egress_base;
        config.egress_pool.guest_prefix_len = self.egress_guest_prefix;
        config
    }
}

enum HarnessMode {
    Smoke,
    Pty,
    Shell,
    Scenario(PathBuf),
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
    backend: HarnessBackend,
    guest_id: String,
    pid: Option<u32>,
    proxy: String,
    terminal_backend: Option<TerminalBackendKind>,
    shutdown: Option<ShutdownReport>,
    shutdown_forced: Option<String>,
    observability: Option<VmObservability>,
    checks: Vec<ScenarioCheck>,
    pty: Option<PtyScenarioResult>,
    pty_transcript: Option<Vec<PtyTranscriptEvent>>,
    scenario_driver: Option<ScenarioRunResult>,
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
    #[error(transparent)]
    Pty(#[from] PtyScenarioError),
    #[error("shutdown failed: {0}")]
    Shutdown(#[source] OrchestratorError),
}

pub(crate) fn build_guest_provisioner(
    base_dir: &Path,
    artifacts_dir: &Path,
    backend: HarnessBackend,
    instance: &HarnessInstance,
    allocator_config: GuestNetAllocatorConfig,
    ca: &Arc<SshCa>,
    runtime: &Arc<Runtime>,
) -> Result<GuestProvisioner, DynError> {
    Ok(GuestProvisioner::new(GuestProvisionerConfig {
        namespace: instance.namespace.clone(),
        base_dir: base_dir.to_path_buf(),
        network_modes: backend.network_modes(),
        readiness_policy: backend.readiness_policy(),
        services: LifecycleServices {
            runtime: Arc::clone(runtime),
        },
        allocator: GuestNetAllocator::new(allocator_config)?,
        ssh_ca_pubkey: ca.public_key_openssh()?,
        guest_spec_factory: Arc::new({
            let artifacts_dir = artifacts_dir.to_path_buf();
            let demo_root = instance.demo_root.clone();
            move |request: &ProvisioningGuestRequest| {
                demo_guest(
                    &request.principal,
                    request.net_assignment.slot,
                    backend,
                    &artifacts_dir,
                    &demo_root,
                    &request.namespace,
                )
                .map_err(|err| err.to_string())
            }
        }),
        host_seed_hook: Some(Arc::new(|guest| {
            seed_host_mounts(guest).map_err(|err| err.to_string())
        })),
    }))
}

fn parse_vz_egress_ipv4(label: &str, value: &str) -> Result<Ipv4Addr, DynError> {
    value
        .parse::<Ipv4Addr>()
        .map_err(|err| format!("invalid {label}: {value}: {err}").into())
}

fn parse_vz_host_forward(value: &str) -> Result<VzHostForward, DynError> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 3 {
        return Err("host forward must be host_addr:host_port:guest_port".into());
    }
    let host_addr = parse_vz_egress_ipv4("host forward address", parts[0])?;
    let host_port = parts[1]
        .parse::<u16>()
        .map_err(|err| format!("invalid host port in {value}: {err}"))?;
    let guest_port = parts[2]
        .parse::<u16>()
        .map_err(|err| format!("invalid guest port in {value}: {err}"))?;
    Ok(VzHostForward::tcp(host_addr, host_port, guest_port))
}

fn print_vz_egress_usage() {
    eprintln!(
        "usage: harness_v1_5 vz-egress --socket-path <path> [--guest-ip <ipv4>] [--host-ip <ipv4>] [--netmask <ipv4>] [--dns-ip <ipv4>] [--host-forward-tcp <host_ip:host_port:guest_port>] [--log-frames]"
    );
}

fn run_vz_egress_subcommand(args: &[String]) -> Result<(), DynError> {
    let mut socket_path: Option<PathBuf> = None;
    let mut guest_ip = Ipv4Addr::new(10, 0, 2, 15);
    let mut host_ip = Ipv4Addr::new(10, 0, 2, 2);
    let mut netmask = Ipv4Addr::new(255, 255, 255, 0);
    let mut dns_ip = Ipv4Addr::new(10, 0, 2, 3);
    let mut forwards: Vec<VzHostForward> = Vec::new();
    let mut log_frames = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--socket-path" if i + 1 < args.len() => {
                socket_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--guest-ip" if i + 1 < args.len() => {
                guest_ip = parse_vz_egress_ipv4("guest ip", &args[i + 1])?;
                i += 2;
            }
            "--host-ip" if i + 1 < args.len() => {
                host_ip = parse_vz_egress_ipv4("host ip", &args[i + 1])?;
                i += 2;
            }
            "--netmask" if i + 1 < args.len() => {
                netmask = parse_vz_egress_ipv4("netmask", &args[i + 1])?;
                i += 2;
            }
            "--dns-ip" if i + 1 < args.len() => {
                dns_ip = parse_vz_egress_ipv4("dns ip", &args[i + 1])?;
                i += 2;
            }
            "--host-forward-tcp" if i + 1 < args.len() => {
                forwards.push(parse_vz_host_forward(&args[i + 1])?);
                i += 2;
            }
            "--log-frames" => {
                log_frames = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_vz_egress_usage();
                return Ok(());
            }
            other => return Err(format!("unknown vz-egress argument: {other}").into()),
        }
    }

    if log_frames {
        let _ = tracing_subscriber::fmt()
            .with_target(false)
            .with_writer(std::io::stderr)
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
            )
            .try_init();
    }

    let socket_path = socket_path.ok_or_else(|| "--socket-path is required".to_string())?;
    let config = VzUserspaceEgressConfig {
        socket_path,
        guest_ip,
        host_ip,
        netmask,
        dns_ip,
        host_forwards: forwards,
        log_frames,
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::clone(&shutdown);
    ctrlc::set_handler(move || {
        shutdown_flag.store(true, Ordering::SeqCst);
    })?;

    let stats = run_vz_userspace_egress(config, shutdown, None)?;
    eprintln!(
        "harness_v1_5 vz-egress summary: guest_to_host_frames={} host_to_guest_frames={}",
        stats.guest_to_host_frames, stats.host_to_guest_frames
    );
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), DynError> {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.first().is_some_and(|arg| arg == "vz-egress") {
        // v1.5 convergence contract: there is no separate VZ egress helper
        // binary. Image-building can still need a host-side egress process,
        // but it is hosted by the single harness binary through this explicit
        // operational subcommand. Normal guest launch uses the embedded VMM
        // egress handle instead.
        return run_vz_egress_subcommand(&raw_args[1..]);
    }

    let mut args = raw_args.into_iter();
    let mut mode = HarnessMode::Smoke;
    let mut root_override: Option<PathBuf> = None;
    let mut result_json_path: Option<PathBuf> = None;
    let mut terminal_backend = TerminalBackendKind::default();
    let mut allocator_options = HarnessAllocatorOptions::default();
    let mut harness_backend = HarnessBackend::default();
    let mut shell_auto_provision_default = false;
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
            "--terminal-backend" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--terminal-backend requires 'vt100' or 'shadow'".to_string())?;
                terminal_backend = value.parse()?;
            }
            "--backend" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--backend requires 'vz' or 'ch'".to_string())?;
                harness_backend = value.parse()?;
            }
            "--auto-provision" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--auto-provision requires 'on' or 'off'".to_string())?;
                shell_auto_provision_default = parse_auto_provision_mode(&value)?;
            }
            "--first-cid" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--first-cid requires a number".to_string())?;
                allocator_options.first_cid = value.parse()?;
            }
            "--max-guests" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-guests requires a number".to_string())?;
                allocator_options.max_guests = Some(value.parse()?);
            }
            "--admin-base" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--admin-base requires CIDR notation".to_string())?;
                allocator_options.admin_base = value.parse()?;
            }
            "--admin-guest-prefix" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--admin-guest-prefix requires a prefix length".to_string())?;
                allocator_options.admin_guest_prefix = value.parse()?;
            }
            "--egress-base" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--egress-base requires CIDR notation".to_string())?;
                allocator_options.egress_base = value.parse()?;
            }
            "--egress-guest-prefix" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--egress-guest-prefix requires a prefix length".to_string())?;
                allocator_options.egress_guest_prefix = value.parse()?;
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
            other if other.starts_with("--terminal-backend=") => {
                terminal_backend = other.trim_start_matches("--terminal-backend=").parse()?;
            }
            other if other.starts_with("--backend=") => {
                harness_backend = other.trim_start_matches("--backend=").parse()?;
            }
            other if other.starts_with("--auto-provision=") => {
                shell_auto_provision_default =
                    parse_auto_provision_mode(other.trim_start_matches("--auto-provision="))?;
            }
            other if other.starts_with("--first-cid=") => {
                allocator_options.first_cid = other.trim_start_matches("--first-cid=").parse()?;
            }
            other if other.starts_with("--max-guests=") => {
                allocator_options.max_guests =
                    Some(other.trim_start_matches("--max-guests=").parse()?);
            }
            other if other.starts_with("--admin-base=") => {
                allocator_options.admin_base = other.trim_start_matches("--admin-base=").parse()?;
            }
            other if other.starts_with("--admin-guest-prefix=") => {
                allocator_options.admin_guest_prefix =
                    other.trim_start_matches("--admin-guest-prefix=").parse()?;
            }
            other if other.starts_with("--egress-base=") => {
                allocator_options.egress_base =
                    other.trim_start_matches("--egress-base=").parse()?;
            }
            other if other.starts_with("--egress-guest-prefix=") => {
                allocator_options.egress_guest_prefix =
                    other.trim_start_matches("--egress-guest-prefix=").parse()?;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}").into());
            }
            "scenario" => {
                let value = args.next().ok_or_else(|| {
                    "scenario requires a path to a JSON scenario file".to_string()
                })?;
                mode = HarnessMode::Scenario(PathBuf::from(value));
            }
            "shell" => mode = HarnessMode::Shell,
            "pty" => mode = HarnessMode::Pty,
            "smoke" => mode = HarnessMode::Smoke,
            other => {
                return Err(format!("unsupported mode: {other}").into());
            }
        }
    }

    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/v1.5");
    let artifacts_dir = harness_backend.artifacts_dir(&base_dir);
    harness_backend.ensure_artifacts(&artifacts_dir)?;

    let root_dir = root_override.unwrap_or_else(RuntimeNamespace::root_from_env_or_temp);
    let instance = new_harness_instance(&root_dir)?;
    let allocator_config =
        allocator_options.build(&instance.socket_root, &instance.namespace.prefix);
    if matches!(mode, HarnessMode::Shell) {
        return shell::run_shell(
            &base_dir,
            &artifacts_dir,
            harness_backend,
            &instance,
            allocator_config,
            terminal_backend,
            shell_auto_provision_default,
        )
        .await;
    }
    if let HarnessMode::Scenario(path) = &mode {
        let result = scenario::run_scenario_file(scenario::ScenarioFileRequest {
            base_dir: &base_dir,
            artifacts_dir: &artifacts_dir,
            backend: harness_backend,
            instance: &instance,
            allocator_config,
            terminal_backend,
            path,
            result_json_path: result_json_path.as_deref(),
        })
        .await?;
        return scenario_exit(result);
    }

    let guest = demo_guest(
        "alice",
        0,
        harness_backend,
        &artifacts_dir,
        &instance.demo_root,
        &instance.namespace,
    )?;
    std::fs::create_dir_all(&instance.socket_root)?;

    seed_host_mounts(&guest)?;

    let ca = Arc::new(SshCa::new()?);
    let guest_registry = new_guest_registry();
    let proxy_config = SshProxyConfig {
        listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), instance.proxy_port),
        principal_resolver: None,
    };
    let proxy = format!("ssh://localhost:{}", proxy_config.listen.port());
    tokio::spawn(ssh::run_proxy(
        proxy_config.clone(),
        Arc::clone(&guest_registry),
    ));
    wait_for_proxy_listener(proxy_config.listen, SSH_PROXY_READY_TIMEOUT).await?;
    print_instance_details(&instance, &proxy_config);
    println!("  backend={harness_backend}");
    println!("  terminal_backend={terminal_backend}");
    let runtime = Arc::new(harness_backend.runtime(&ca, &guest_registry)?);

    let mut allocator = GuestNetAllocator::new(allocator_config)?;
    let prepared = prepare(
        PrepareRequest {
            guest,
            namespace: instance.namespace.clone(),
            backend_kind: runtime.hypervisor.kind(),
            network_modes: harness_backend.network_modes(),
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
                .ready(&harness_backend.readiness_policy())
                .await
                .map_err(HarnessError::Ready)?;
            observability = Some(active_handle.observability());

            match &mode {
                HarnessMode::Smoke => {
                    checks = run_smoke(active_handle).await?;
                }
                HarnessMode::Pty => {
                    let transcript_path = observability
                        .as_ref()
                        .expect("observability exists after readiness")
                        .run_bundle
                        .capture_paths
                        .pty_transcript_ndjson
                        .clone();
                    let screen_path = observability
                        .as_ref()
                        .expect("observability exists after readiness")
                        .run_bundle
                        .capture_paths
                        .pty_screen_json
                        .clone();
                    let asciicast_path = observability
                        .as_ref()
                        .expect("observability exists after readiness")
                        .run_bundle
                        .capture_paths
                        .pty_asciicast
                        .clone();
                    let screen_svg_path = observability
                        .as_ref()
                        .expect("observability exists after readiness")
                        .run_bundle
                        .capture_paths
                        .pty_screen_svg
                        .clone();
                    let pty_run = pty::run_pty_smoke(
                        active_handle,
                        terminal_backend,
                        transcript_path.clone(),
                        screen_path,
                        screen_svg_path,
                        asciicast_path,
                    )
                    .await?;
                    pty = Some(pty_run.result);
                    pty_transcript = Some(pty_run.transcript);
                    checks.push(ScenarioCheck {
                        name: "pty".to_string(),
                        detail: "PTY banner, prompt, resize, transcript, and terminal-close checks passed"
                            .to_string(),
                    });
                }
                HarnessMode::Shell | HarnessMode::Scenario(_) => {
                    return Err(HarnessError::UnsupportedScenario(
                        "interactive mode".to_string(),
                    ));
                }
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
    if harness_backend.cleanup_vz_disks() {
        cleanup_development_guest_disks(&instance.namespace, mode_name(&mode));
    }

    let result = ScenarioResult {
        status: if error.is_none() {
            ScenarioStatus::Passed
        } else {
            ScenarioStatus::Failed
        },
        scenario: mode_name(&mode).to_string(),
        backend: harness_backend,
        guest_id: "alice".to_string(),
        pid: shutdown
            .as_ref()
            .and_then(|report| report.pid)
            .or_else(|| handle.as_ref().and_then(|h| h.pid)),
        proxy: proxy.clone(),
        terminal_backend: matches!(mode, HarnessMode::Pty).then_some(terminal_backend),
        shutdown: shutdown.clone(),
        shutdown_forced: shutdown
            .as_ref()
            .and_then(|report| report.forced.map(str::to_string)),
        observability,
        checks,
        pty,
        pty_transcript,
        scenario_driver: None,
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
                "v1.5 harness {} passed: guest={} pid={:?} forced={:?} proxy=127.0.0.1:{}",
                mode_name(&mode),
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
    let expected_gateway = handle.net_assignment.egress_ipv4.host;
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
        .exec(VFS_MEMFS_LAYER_COMMAND, Duration::from_secs(20))
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "vfs-memfs-layer",
            source,
        })?;
    ensure_contains("vfs-memfs-layer", &vfs.stdout, "VFS_MEMFS_OK")?;
    checks.push(ScenarioCheck {
        name: "vfs-memfs-layer".to_string(),
        detail: "home, workspace, and agent-state VFS/FUSE memfs views are writable".to_string(),
    });

    let sudo = handle
        .exec(
            "/bin/sh -lc 'sudo -n true && echo SUDO_OK'",
            Duration::from_secs(10),
        )
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "sudo",
            source,
        })?;
    ensure_contains("sudo", &sudo.stdout, "SUDO_OK")?;
    checks.push(ScenarioCheck {
        name: "sudo".to_string(),
        detail: "passwordless sudo is available for the guest login user".to_string(),
    });

    let git = handle
        .exec(
            "/bin/sh -lc 'git --version | grep -q \"^git version \" && echo GIT_OK'",
            Duration::from_secs(10),
        )
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "git",
            source,
        })?;
    ensure_contains("git", &git.stdout, "GIT_OK")?;
    checks.push(ScenarioCheck {
        name: "git".to_string(),
        detail: "git is preinstalled in the guest image".to_string(),
    });

    let route = exec_until_success(
        handle,
        &format!(
            "/bin/sh -lc 'ip route | grep -q \"^default via {} \" && echo ROUTE_OK'",
            expected_gateway
        ),
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
        detail: "default route points at the Vz userspace egress gateway".to_string(),
    });

    let outbound = wait_for_egress_ready(handle, Duration::from_secs(30))
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "egress-ready",
            source,
        })?;
    ensure_contains("egress-ready", &outbound.stdout, "EGRESS_OK")?;
    checks.push(ScenarioCheck {
        name: "egress-ready".to_string(),
        detail: "DNS resolution and outbound HTTPS succeeded for the manual-certification targets"
            .to_string(),
    });

    let package_manager = exec_until_success(
        handle,
        PACKAGE_MANAGER_QUIESCENT_COMMAND,
        "PKG_IDLE_OK",
        Duration::from_secs(180),
    )
    .await
    .map_err(|source| HarnessError::SmokeExec {
        check: "package-manager-quiescent",
        source,
    })?;
    ensure_contains(
        "package-manager-quiescent",
        &package_manager.stdout,
        "PKG_IDLE_OK",
    )?;
    checks.push(ScenarioCheck {
        name: "package-manager-quiescent".to_string(),
        detail: "package manager background activity settled before apt validation".to_string(),
    });

    let apt_update = handle
        .exec(APT_UPDATE_COMMAND, Duration::from_secs(60))
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "apt-update",
            source,
        })?;
    ensure_contains("apt-update", &apt_update.stdout, "APT_OK")?;
    checks.push(ScenarioCheck {
        name: "apt-update".to_string(),
        detail: "Debian package index refresh succeeded over backend internet egress".to_string(),
    });

    let agent_cli = handle
        .exec(AGENT_CLI_START_COMMAND, Duration::from_secs(60))
        .await
        .map_err(|source| HarnessError::SmokeExec {
            check: "agent-cli-start",
            source,
        })?;
    ensure_contains("agent-cli-start", &agent_cli.stdout, "AGENT_CLI_OK")?;
    checks.push(ScenarioCheck {
        name: "agent-cli-start".to_string(),
        detail: "Codex and Claude CLIs start without OS-level execution errors".to_string(),
    });
    Ok(checks)
}

pub(crate) fn ensure_file_exists(path: &Path) -> Result<(), HarnessError> {
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

pub(crate) fn persist_json<T: Serialize>(path: &Path, value: &T) -> Result<(), DynError> {
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
        PtyScenarioError::Terminal(source) => match source {
            terminal::TerminalSessionError::ControlPlane(ssh) => classify_ssh_failure("pty", ssh),
            terminal::TerminalSessionError::StatePoisoned => ScenarioFailure {
                class: FailureClass::Internal,
                stage: "pty",
                code: "pty_state_poisoned",
                message: source.to_string(),
            },
            terminal::TerminalSessionError::Persist { .. } => ScenarioFailure {
                class: FailureClass::Artifact,
                stage: "pty",
                code: "pty_artifact_persist_failed",
                message: source.to_string(),
            },
            terminal::TerminalSessionError::Assertion { .. } => ScenarioFailure {
                class: FailureClass::Pty,
                stage: "pty",
                code: "pty_assertion_failed",
                message: source.to_string(),
            },
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
        OrchestratorError::ShutdownFailures { .. } => ScenarioFailure {
            class: FailureClass::Shutdown,
            stage,
            code: "shutdown_cleanup_failed",
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
        | RuntimeError::GuestFs(GuestFsError::CleanupSocket { .. })
        | RuntimeError::GuestFs(GuestFsError::TaskStatePoisoned) => ScenarioFailure {
            class: FailureClass::Filesystem,
            stage,
            code: "guestfs_failed",
            message: error.to_string(),
        },
        RuntimeError::Ssh(ssh) => classify_ssh_failure(stage, ssh),
        #[cfg(target_os = "linux")]
        RuntimeError::Vnet(_) | RuntimeError::VnetShutdown(_) => ScenarioFailure {
            class: FailureClass::Network,
            stage,
            code: "vnet_failed",
            message: error.to_string(),
        },
        RuntimeError::VzEgress(_) => ScenarioFailure {
            class: FailureClass::Network,
            stage,
            code: "vz_egress_failed",
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
            BackendError::VzShell(_) => "backend_vz_shell_failed",
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
            SshProxyError::GenerateServerKey { .. } => "ssh_server_key_failed",
            SshProxyError::ProxyBind { .. } => "ssh_proxy_bind_failed",
            SshProxyError::ProxyConnect { .. } => "ssh_proxy_connect_failed",
            SshProxyError::ProxyAuth { .. } => "ssh_proxy_auth_failed",
            SshProxyError::BindGuestBridgeSocket { .. } => "ssh_bridge_bind_failed",
            SshProxyError::CertAuth { .. } => "ssh_cert_auth_failed",
            SshProxyError::Russh { .. } => "ssh_transport_failed",
            SshProxyError::ChannelClosed => "ssh_channel_closed",
            SshProxyError::MissingExitStatus { .. } => "ssh_missing_exit_status",
            SshProxyError::UnknownGuest(_) => "ssh_unknown_guest",
            SshProxyError::ResolveGuest { .. } => "ssh_guest_resolution_failed",
            SshProxyError::StatePoisoned(_) => "ssh_state_poisoned",
            SshProxyError::CleanupGuestBridgeSocket { .. } => "ssh_bridge_cleanup_failed",
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
    println!(
        "usage: harness_v1_5 [smoke|pty|shell|scenario <file.json>] [--backend vz|ch] [--root <dir>] [--result-json <path>] [--terminal-backend vt100|shadow] [--auto-provision on|off] [--first-cid N] [--max-guests N] [--admin-base CIDR] [--admin-guest-prefix N] [--egress-base CIDR] [--egress-guest-prefix N]"
    );
    println!("       harness_v1_5 vz-egress --socket-path <path> [VZ image-build egress options]");
    println!(
        "note: --auto-provision only affects shell mode; scenario auto-provision remains scenario-owned"
    );
}

fn parse_auto_provision_mode(value: &str) -> Result<bool, DynError> {
    match value {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        other => Err(format!("unsupported auto-provision mode: {other}").into()),
    }
}

fn mode_name(mode: &HarnessMode) -> &'static str {
    match mode {
        HarnessMode::Smoke => "smoke",
        HarnessMode::Pty => "pty",
        HarnessMode::Shell => "shell",
        HarnessMode::Scenario(_) => "scenario",
    }
}

fn scenario_exit(result: ScenarioRunResult) -> Result<(), DynError> {
    match result.status {
        ScenarioRunStatus::Passed => {
            println!(
                "v1.5 harness scenario passed: scenario={} steps={} proxy={}",
                result.scenario,
                result.steps.len(),
                result.proxy
            );
            Ok(())
        }
        ScenarioRunStatus::Failed => Err(result
            .error
            .map(|failure| failure.message)
            .unwrap_or_else(|| "scenario failed".to_string())
            .into()),
    }
}

fn new_harness_instance(root_dir: &Path) -> Result<HarnessInstance, DynError> {
    // The harness can keep serving after stdin/SIGHUP detach, so delayed guest
    // boots must still fit every AF_UNIX socket under a user-supplied root.
    // Keep the namespace prefix and vnet socket directory intentionally short.
    let namespace = RuntimeNamespace::for_process("v15", "h", root_dir)?;
    let demo_root = namespace
        .temp_root
        .join(format!("{}-demo", namespace.prefix));
    let socket_root = namespace.temp_root.join("s");
    let proxy_port = 32000 + port_offset(&namespace.prefix);
    Ok(HarnessInstance {
        namespace,
        demo_root,
        socket_root,
        proxy_port,
    })
}

pub(crate) fn print_instance_details(instance: &HarnessInstance, proxy_config: &SshProxyConfig) {
    println!("v1.5 harness instance: {}", instance.namespace.prefix);
    println!("  demo_root={}", instance.demo_root.display());
    println!("  socket_root={}", instance.socket_root.display());
    println!("  proxy=ssh://localhost:{}", proxy_config.listen.port());
}

fn port_offset(seed: &str) -> u16 {
    seed.bytes().fold(0u16, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u16::from(byte))
    }) % 10_000
}

pub(crate) fn demo_guest(
    guest_id: &str,
    slot: u32,
    backend: HarnessBackend,
    artifacts_dir: &Path,
    demo_root: &Path,
    namespace: &RuntimeNamespace,
) -> Result<GuestSpec, DynError> {
    let (uid, gid) = demo_guest_ids(guest_id, slot)?;
    Ok(GuestSpec {
        guest_id: guest_id.to_string(),
        hostname: format!("motlie-{guest_id}"),
        socket_path: demo_guest_socket_path(namespace, guest_id)?,
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
        software: SoftwareProfile { packages: vec![] },
        resources: GuestResources::default(),
        storage: GuestStorage::default(),
        boot: backend.boot_artifacts(artifacts_dir),
    })
}

pub(crate) fn seed_host_mounts(guest: &GuestSpec) -> Result<(), DynError> {
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
    write_host_file_if_missing(&home.join(".bashrc"), "# motlie v1.5 demo bashrc\n", 0o644)?;
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
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let exec_timeout = remaining.min(Duration::from_secs(70));
        match handle.exec(command, exec_timeout).await {
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

pub(crate) async fn wait_for_egress_ready(
    handle: &VmHandle,
    timeout: Duration,
) -> Result<ExecOutput, OrchestratorError> {
    exec_until_success(handle, EGRESS_READY_COMMAND, "EGRESS_OK", timeout).await
}
