use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use motlie_vmm::backend::BackendError;
use motlie_vmm::ca::SshCa;
use motlie_vmm::guestfs::GuestFsError;
use motlie_vmm::network_alloc::GuestNetAllocatorConfig;
use motlie_vmm::orchestrator::{OrchestratorError, ShutdownReport};
use motlie_vmm::provisioning::{GuestProvisioner, ProvisioningError};
use motlie_vmm::runtime::RuntimeError;
use motlie_vmm::ssh::{
    self, exec_via_proxy, new_guest_registry, ExecOutput, PtyRead, PtyRequest, SshProxyConfig,
    SshProxyError,
};
use serde::{Deserialize, Serialize};

use crate::backend::HarnessBackend;
use crate::demo_support::cleanup_development_guest_disks;
use crate::terminal::{
    HarnessTerminalSession, TerminalBackendKind, TerminalSessionError, VteScreenSnapshot,
};
use crate::{
    build_guest_provisioner, persist_json, print_instance_details, wait_for_egress_ready, DynError,
    HarnessInstance, AGENT_CLI_START_COMMAND, APT_UPDATE_COMMAND,
    PACKAGE_MANAGER_QUIESCENT_COMMAND, VFS_MEMFS_LAYER_COMMAND,
};

#[derive(Debug, Deserialize)]
pub struct ScenarioDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub steps: Vec<ScenarioStep>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ScenarioStep {
    Boot {
        guest: String,
    },
    Ready {
        guest: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    Exec {
        guest: String,
        command: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
        #[serde(default)]
        expect: Option<ExecExpectation>,
    },
    ProxyExec {
        principal: String,
        command: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
        #[serde(default)]
        expect: Option<ExecExpectation>,
    },
    WaitPackageManagerQuiescent {
        guest: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    WaitEgressReady {
        guest: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    CheckVfsMemfs {
        guest: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    AptUpdate {
        guest: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    CheckAgentCli {
        guest: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    PtyOpen {
        guest: String,
        session: String,
        #[serde(default)]
        term: Option<String>,
        #[serde(default)]
        cols: Option<u32>,
        #[serde(default)]
        rows: Option<u32>,
        #[serde(default)]
        timeout_ms: Option<u64>,
        #[serde(default)]
        command: Option<String>,
    },
    PtySend {
        session: String,
        text: String,
    },
    PtySendLine {
        session: String,
        text: String,
    },
    PtyRead {
        session: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    PtyResize {
        session: String,
        cols: u32,
        rows: u32,
        #[serde(default)]
        pix_width: Option<u32>,
        #[serde(default)]
        pix_height: Option<u32>,
    },
    PtyExpect {
        session: String,
        contains: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    PtyExpectScreen {
        session: String,
        contains: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    PtyExpectTerminal {
        session: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    PtySnapshot {
        session: String,
        #[serde(default)]
        name: Option<String>,
    },
    Shutdown {
        guest: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ExecExpectation {
    #[serde(default)]
    pub exit_code: Option<u32>,
    #[serde(default)]
    pub stdout_contains: Option<String>,
    #[serde(default)]
    pub stderr_contains: Option<String>,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioRunStatus {
    Passed,
    Failed,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DriverFailureClass {
    Config,
    Artifact,
    Backend,
    Filesystem,
    Network,
    Ssh,
    Readiness,
    Pty,
    Shutdown,
    Internal,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct DriverFailure {
    pub class: DriverFailureClass,
    pub stage: &'static str,
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ScenarioSessionArtifacts {
    pub guest: String,
    pub terminal_backend: TerminalBackendKind,
    pub transcript_ndjson: PathBuf,
    pub screen_json: PathBuf,
    pub screen_svg: PathBuf,
    pub asciicast: PathBuf,
    pub final_screen: VteScreenSnapshot,
}

#[derive(Debug, Serialize)]
pub struct ScenarioStepResult {
    pub index: usize,
    pub action: &'static str,
    pub guest: Option<String>,
    pub session: Option<String>,
    pub detail: String,
    pub exec: Option<ExecOutput>,
    pub pty_read: Option<PtyRead>,
    pub screen: Option<VteScreenSnapshot>,
    pub shutdown: Option<ShutdownReport>,
}

#[derive(Debug, Serialize)]
pub struct ScenarioRunResult {
    pub status: ScenarioRunStatus,
    pub scenario: String,
    pub backend: HarnessBackend,
    pub description: Option<String>,
    pub artifact_root: PathBuf,
    pub proxy: String,
    pub terminal_backend: TerminalBackendKind,
    pub allocator_capacity: u32,
    pub steps: Vec<ScenarioStepResult>,
    pub sessions: HashMap<String, ScenarioSessionArtifacts>,
    pub error: Option<DriverFailure>,
    pub cleanup_error: Option<DriverFailure>,
}

#[derive(Debug, thiserror::Error)]
pub enum ScenarioDriverError {
    #[error("unknown guest '{0}'")]
    UnknownGuest(String),
    #[error("guest '{0}' is already booted")]
    GuestAlreadyBooted(String),
    #[error("unknown PTY session '{0}'")]
    UnknownSession(String),
    #[error("PTY session '{0}' already exists")]
    SessionAlreadyExists(String),
    #[error("exec failed: {0}")]
    Exec(#[source] OrchestratorError),
    #[error("proxy exec failed: {0}")]
    ProxyExec(#[source] SshProxyError),
    #[error("provisioning failed: {0}")]
    Provisioning(#[source] ProvisioningError),
    #[error(transparent)]
    Pty(#[from] TerminalSessionError),
}

impl From<ProvisioningError> for ScenarioDriverError {
    fn from(value: ProvisioningError) -> Self {
        Self::Provisioning(value)
    }
}

pub async fn run_scenario_file(
    base_dir: &Path,
    artifacts_dir: &Path,
    backend: HarnessBackend,
    instance: &HarnessInstance,
    allocator_config: GuestNetAllocatorConfig,
    terminal_backend: TerminalBackendKind,
    path: &Path,
    result_json_path: Option<&Path>,
) -> Result<ScenarioRunResult, DynError> {
    let bytes = std::fs::read(path)?;
    let definition: ScenarioDefinition = serde_json::from_slice(&bytes)?;
    let result = run_scenario_definition(
        base_dir,
        artifacts_dir,
        backend,
        instance,
        allocator_config,
        terminal_backend,
        definition,
    )
    .await?;
    if let Some(path) = result_json_path {
        persist_json(path, &result)?;
    }
    Ok(result)
}

pub async fn run_scenario_definition(
    base_dir: &Path,
    artifacts_dir: &Path,
    backend: HarnessBackend,
    instance: &HarnessInstance,
    allocator_config: GuestNetAllocatorConfig,
    terminal_backend: TerminalBackendKind,
    definition: ScenarioDefinition,
) -> Result<ScenarioRunResult, DynError> {
    let mut driver = ScenarioDriver::new(
        base_dir,
        artifacts_dir,
        backend,
        instance,
        allocator_config,
        terminal_backend,
    )?;
    let result = driver.run(definition).await;
    driver.shutdown_all().await;
    Ok(result)
}

struct ScenarioDriver {
    provisioner: GuestProvisioner,
    backend: HarnessBackend,
    proxy_config: SshProxyConfig,
    terminal_backend: TerminalBackendKind,
    terminals: HashMap<String, HarnessTerminalSession>,
    session_guests: HashMap<String, String>,
    artifact_root: PathBuf,
    namespace: motlie_vmm::spec::RuntimeNamespace,
}

impl ScenarioDriver {
    fn new(
        base_dir: &Path,
        _artifacts_dir: &Path,
        backend: HarnessBackend,
        instance: &HarnessInstance,
        allocator_config: GuestNetAllocatorConfig,
        terminal_backend: TerminalBackendKind,
    ) -> Result<Self, DynError> {
        let artifacts_dir = backend.artifacts_dir(base_dir);
        backend.ensure_artifacts(&artifacts_dir)?;
        std::fs::create_dir_all(&instance.socket_root)?;

        let ca = Arc::new(SshCa::new()?);
        let guest_registry = new_guest_registry();
        let runtime = Arc::new(backend.runtime(&ca, &guest_registry)?);
        let provisioner = build_guest_provisioner(
            base_dir,
            &artifacts_dir,
            backend,
            instance,
            allocator_config,
            &ca,
            &runtime,
        )?;
        let proxy_config = SshProxyConfig {
            listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), instance.proxy_port),
            principal_resolver: Some(provisioner.ssh_principal_resolver()),
        };
        tokio::spawn(ssh::run_proxy(
            proxy_config.clone(),
            Arc::clone(&guest_registry),
        ));
        print_instance_details(instance, &proxy_config);
        println!("  backend={backend}");
        println!("  terminal_backend={terminal_backend}");
        let artifact_root = instance
            .namespace
            .temp_root
            .join(format!("{}-scenarios", instance.namespace.prefix));
        std::fs::create_dir_all(&artifact_root)?;

        Ok(Self {
            provisioner,
            backend,
            proxy_config,
            terminal_backend,
            terminals: HashMap::new(),
            session_guests: HashMap::new(),
            artifact_root,
            namespace: instance.namespace.clone(),
        })
    }

    async fn run(&mut self, definition: ScenarioDefinition) -> ScenarioRunResult {
        let scenario_root = self
            .artifact_root
            .join(format!("{}-run", sanitize(&definition.name)));
        let _ = std::fs::create_dir_all(&scenario_root);
        let internal_result_path = scenario_root.join("scenario-result.json");

        let mut steps = Vec::new();
        let mut error = None;
        for (index, step) in definition.steps.iter().enumerate() {
            match self.execute_step(index, step, &scenario_root).await {
                Ok(result) => steps.push(result),
                Err(err) => {
                    error = Some(classify_driver_failure(&err));
                    steps.push(ScenarioStepResult {
                        index,
                        action: action_name(step),
                        guest: step_guest(step).map(ToString::to_string),
                        session: step_session(step).map(ToString::to_string),
                        detail: err.to_string(),
                        exec: None,
                        pty_read: None,
                        screen: None,
                        shutdown: None,
                    });
                    break;
                }
            }
        }

        let mut sessions = HashMap::new();
        let mut cleanup_error = None;
        for (name, session) in &self.terminals {
            match session.persist_artifacts() {
                Ok(()) => {
                    let guest = self
                        .session_guests
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    match session.snapshot() {
                        Ok(final_screen) => {
                            sessions.insert(
                                name.clone(),
                                ScenarioSessionArtifacts {
                                    guest,
                                    terminal_backend: session.backend(),
                                    transcript_ndjson: session.transcript_path().to_path_buf(),
                                    screen_json: session.screen_path().to_path_buf(),
                                    screen_svg: session.screen_svg_path().to_path_buf(),
                                    asciicast: session.asciicast_path().to_path_buf(),
                                    final_screen,
                                },
                            );
                        }
                        Err(err) => cleanup_error = Some(classify_driver_failure(&err.into())),
                    }
                }
                Err(err) => cleanup_error = Some(classify_driver_failure(&err.into())),
            }
        }

        let result = ScenarioRunResult {
            status: if error.is_none() {
                ScenarioRunStatus::Passed
            } else {
                ScenarioRunStatus::Failed
            },
            scenario: definition.name,
            backend: self.backend,
            description: definition.description,
            artifact_root: scenario_root,
            proxy: format!("ssh://localhost:{}", self.proxy_config.listen.port()),
            terminal_backend: self.terminal_backend,
            allocator_capacity: self.provisioner.capacity().unwrap_or_default(),
            steps,
            sessions,
            error,
            cleanup_error,
        };
        let _ = persist_json(&internal_result_path, &result);
        result
    }

    async fn execute_step(
        &mut self,
        index: usize,
        step: &ScenarioStep,
        scenario_root: &Path,
    ) -> Result<ScenarioStepResult, ScenarioDriverError> {
        match step {
            ScenarioStep::Boot { guest } => {
                if self
                    .provisioner
                    .snapshot(guest)
                    .map_err(ScenarioDriverError::from)?
                    .is_some_and(|guest| guest.active)
                {
                    return Err(ScenarioDriverError::GuestAlreadyBooted(guest.clone()));
                }
                let handle = self
                    .provisioner
                    .ensure_guest_for_principal(guest)
                    .await
                    .map_err(ScenarioDriverError::from)?;
                let detail = format!(
                    "booted {} pid={:?} api={} cid={} admin_subnet={} egress_subnet={}",
                    guest,
                    handle.handle.pid,
                    handle.handle.runtime_paths.api_socket.display(),
                    handle.net_assignment.cid,
                    handle.net_assignment.admin_subnet,
                    handle.net_assignment.egress_subnet,
                );
                Ok(ScenarioStepResult {
                    index,
                    action: "boot",
                    guest: Some(guest.clone()),
                    session: None,
                    detail,
                    exec: None,
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::Ready { guest, timeout_ms } => {
                let _ = timeout_ms;
                self.provisioner
                    .ready(guest)
                    .await
                    .map_err(ScenarioDriverError::from)?;
                Ok(ScenarioStepResult {
                    index,
                    action: "ready",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!("{} is exec-ready", guest),
                    exec: None,
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::Exec {
                guest,
                command,
                timeout_ms,
                expect,
            } => {
                let output = self
                    .provisioner
                    .exec(guest, command, duration_or_default(*timeout_ms, 20_000))
                    .await
                    .map_err(ScenarioDriverError::from)?;
                if let Some(expect) = expect {
                    check_exec_expectation(expect, &output)?;
                }
                Ok(ScenarioStepResult {
                    index,
                    action: "exec",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!("exec '{}'", command),
                    exec: Some(output),
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::ProxyExec {
                principal,
                command,
                timeout_ms,
                expect,
            } => {
                let output = exec_via_proxy(
                    self.proxy_config.listen,
                    principal,
                    command,
                    duration_or_default(*timeout_ms, 90_000),
                )
                .await
                .map_err(ScenarioDriverError::ProxyExec)?;
                if let Some(expect) = expect {
                    check_exec_expectation(expect, &output)?;
                }
                Ok(ScenarioStepResult {
                    index,
                    action: "proxy_exec",
                    guest: Some(principal.clone()),
                    session: None,
                    detail: format!("proxy exec '{}'", command),
                    exec: Some(output),
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::WaitPackageManagerQuiescent { guest, timeout_ms } => {
                let output = self
                    .provisioner
                    .exec(
                        guest,
                        PACKAGE_MANAGER_QUIESCENT_COMMAND,
                        duration_or_default(*timeout_ms, 65_000),
                    )
                    .await
                    .map_err(ScenarioDriverError::from)?;
                check_exec_expectation(
                    &ExecExpectation {
                        exit_code: Some(0),
                        stdout_contains: Some("PKG_IDLE_OK".to_string()),
                        stderr_contains: None,
                    },
                    &output,
                )?;
                Ok(ScenarioStepResult {
                    index,
                    action: "wait_package_manager_quiescent",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!("package manager background activity settled for {guest}"),
                    exec: Some(output),
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::WaitEgressReady { guest, timeout_ms } => {
                let handle = self
                    .provisioner
                    .active_vm_handle(guest)
                    .map_err(ScenarioDriverError::from)?;
                let output =
                    wait_for_egress_ready(&handle, duration_or_default(*timeout_ms, 30_000))
                        .await
                        .map_err(ScenarioDriverError::Exec)?;
                check_exec_expectation(
                    &ExecExpectation {
                        exit_code: Some(0),
                        stdout_contains: Some("EGRESS_OK".to_string()),
                        stderr_contains: None,
                    },
                    &output,
                )?;
                Ok(ScenarioStepResult {
                    index,
                    action: "wait_egress_ready",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!(
                        "DNS + HTTPS ready for manual-certification targets on {guest}"
                    ),
                    exec: Some(output),
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::CheckVfsMemfs { guest, timeout_ms } => {
                let output = self
                    .provisioner
                    .exec(
                        guest,
                        VFS_MEMFS_LAYER_COMMAND,
                        duration_or_default(*timeout_ms, 20_000),
                    )
                    .await
                    .map_err(ScenarioDriverError::from)?;
                check_exec_expectation(
                    &ExecExpectation {
                        exit_code: Some(0),
                        stdout_contains: Some("VFS_MEMFS_OK".to_string()),
                        stderr_contains: None,
                    },
                    &output,
                )?;
                Ok(ScenarioStepResult {
                    index,
                    action: "check_vfs_memfs",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!("VFS/FUSE memfs views are mounted and writable for {guest}"),
                    exec: Some(output),
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::AptUpdate { guest, timeout_ms } => {
                let output = self
                    .provisioner
                    .exec(
                        guest,
                        APT_UPDATE_COMMAND,
                        duration_or_default(*timeout_ms, 120_000),
                    )
                    .await
                    .map_err(ScenarioDriverError::from)?;
                check_exec_expectation(
                    &ExecExpectation {
                        exit_code: Some(0),
                        stdout_contains: Some("APT_OK".to_string()),
                        stderr_contains: None,
                    },
                    &output,
                )?;
                Ok(ScenarioStepResult {
                    index,
                    action: "apt_update",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!(
                        "apt-get update succeeded over backend internet egress for {guest}"
                    ),
                    exec: Some(output),
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::CheckAgentCli { guest, timeout_ms } => {
                let output = self
                    .provisioner
                    .exec(
                        guest,
                        AGENT_CLI_START_COMMAND,
                        duration_or_default(*timeout_ms, 60_000),
                    )
                    .await
                    .map_err(ScenarioDriverError::from)?;
                check_exec_expectation(
                    &ExecExpectation {
                        exit_code: Some(0),
                        stdout_contains: Some("AGENT_CLI_OK".to_string()),
                        stderr_contains: None,
                    },
                    &output,
                )?;
                Ok(ScenarioStepResult {
                    index,
                    action: "check_agent_cli",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!(
                        "Codex and Claude CLIs start without OS-level errors for {guest}"
                    ),
                    exec: Some(output),
                    pty_read: None,
                    screen: None,
                    shutdown: None,
                })
            }
            ScenarioStep::PtyOpen {
                guest,
                session,
                term,
                cols,
                rows,
                timeout_ms,
                command,
            } => {
                if self.terminals.contains_key(session) {
                    return Err(ScenarioDriverError::SessionAlreadyExists(session.clone()));
                }
                let request = PtyRequest {
                    term: term.clone().unwrap_or_else(|| "xterm-256color".to_string()),
                    col_width: cols.unwrap_or(80),
                    row_height: rows.unwrap_or(24),
                    pix_width: 0,
                    pix_height: 0,
                    command: command.clone(),
                };
                let guest_session = self
                    .provisioner
                    .open_pty(
                        guest,
                        request.clone(),
                        duration_or_default(*timeout_ms, 10_000),
                    )
                    .await
                    .map_err(ScenarioDriverError::from)?;
                let session_root = scenario_root.join("sessions").join(sanitize(session));
                let terminal = HarnessTerminalSession::new(
                    format!("{guest}:{session}"),
                    guest_session,
                    &request,
                    self.terminal_backend,
                    session_root.join("pty-transcript.ndjson"),
                    session_root.join("pty-screen.json"),
                    session_root.join("pty-screen.svg"),
                    session_root.join("pty.cast"),
                );
                let screen = terminal.snapshot()?;
                self.terminals.insert(session.clone(), terminal);
                self.session_guests.insert(session.clone(), guest.clone());
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_open",
                    guest: Some(guest.clone()),
                    session: Some(session.clone()),
                    detail: format!(
                        "opened PTY {} for {} at {}x{}",
                        session, guest, request.col_width, request.row_height
                    ),
                    exec: None,
                    pty_read: None,
                    screen: Some(screen),
                    shutdown: None,
                })
            }
            ScenarioStep::PtySend { session, text } => {
                let terminal = self.terminal(session)?;
                terminal.send(text.as_bytes()).await?;
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_send",
                    guest: None,
                    session: Some(session.clone()),
                    detail: format!("sent {} bytes", text.len()),
                    exec: None,
                    pty_read: None,
                    screen: Some(terminal.snapshot()?),
                    shutdown: None,
                })
            }
            ScenarioStep::PtySendLine { session, text } => {
                let terminal = self.terminal(session)?;
                terminal.send_line(text).await?;
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_send_line",
                    guest: None,
                    session: Some(session.clone()),
                    detail: format!("sent line '{}'", text),
                    exec: None,
                    pty_read: None,
                    screen: Some(terminal.snapshot()?),
                    shutdown: None,
                })
            }
            ScenarioStep::PtyRead {
                session,
                timeout_ms,
            } => {
                let terminal = self.terminal(session)?;
                let read = terminal
                    .read_for(duration_or_default(*timeout_ms, 1_000))
                    .await?;
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_read",
                    guest: None,
                    session: Some(session.clone()),
                    detail: format!("read {} bytes", read.bytes.len()),
                    exec: None,
                    pty_read: Some(read),
                    screen: Some(terminal.snapshot()?),
                    shutdown: None,
                })
            }
            ScenarioStep::PtyResize {
                session,
                cols,
                rows,
                pix_width,
                pix_height,
            } => {
                let terminal = self.terminal(session)?;
                terminal
                    .resize(
                        *cols,
                        *rows,
                        pix_width.unwrap_or(0),
                        pix_height.unwrap_or(0),
                    )
                    .await?;
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_resize",
                    guest: None,
                    session: Some(session.clone()),
                    detail: format!("resized PTY to {}x{}", cols, rows),
                    exec: None,
                    pty_read: None,
                    screen: Some(terminal.snapshot()?),
                    shutdown: None,
                })
            }
            ScenarioStep::PtyExpect {
                session,
                contains,
                timeout_ms,
            } => {
                let terminal = self.terminal(session)?;
                let read = terminal
                    .read_until_contains(
                        "pty_expect",
                        contains,
                        duration_or_default(*timeout_ms, 10_000),
                    )
                    .await?;
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_expect",
                    guest: None,
                    session: Some(session.clone()),
                    detail: format!("observed '{}'", contains),
                    exec: None,
                    pty_read: Some(read),
                    screen: Some(terminal.snapshot()?),
                    shutdown: None,
                })
            }
            ScenarioStep::PtyExpectScreen {
                session,
                contains,
                timeout_ms,
            } => {
                let terminal = self.terminal(session)?;
                let read = terminal
                    .read_until_screen_contains(
                        "pty_expect_screen",
                        contains,
                        duration_or_default(*timeout_ms, 10_000),
                    )
                    .await?;
                let screen = terminal.snapshot()?;
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_expect_screen",
                    guest: None,
                    session: Some(session.clone()),
                    detail: format!("rendered screen contains '{}'", contains),
                    exec: None,
                    pty_read: Some(read),
                    screen: Some(screen),
                    shutdown: None,
                })
            }
            ScenarioStep::PtyExpectTerminal {
                session,
                timeout_ms,
            } => {
                let terminal = self.terminal(session)?;
                let read = terminal
                    .read_until_terminal(duration_or_default(*timeout_ms, 5_000))
                    .await?;
                if !read.eof && !read.closed && read.exit_status.is_none() {
                    return Err(TerminalSessionError::Assertion {
                        step: "pty_expect_terminal",
                        expected: "EOF, close, or exit status".to_string(),
                        observed_excerpt: read.output.clone(),
                    }
                    .into());
                }
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_expect_terminal",
                    guest: None,
                    session: Some(session.clone()),
                    detail: "terminal closed or reported an exit status".to_string(),
                    exec: None,
                    pty_read: Some(read),
                    screen: Some(terminal.snapshot()?),
                    shutdown: None,
                })
            }
            ScenarioStep::PtySnapshot { session, name } => {
                let terminal = self.terminal(session)?;
                let screen = terminal.snapshot()?;
                let detail = format!(
                    "captured screen snapshot {}",
                    name.as_deref().unwrap_or("current")
                );
                Ok(ScenarioStepResult {
                    index,
                    action: "pty_snapshot",
                    guest: None,
                    session: Some(session.clone()),
                    detail,
                    exec: None,
                    pty_read: None,
                    screen: Some(screen),
                    shutdown: None,
                })
            }
            ScenarioStep::Shutdown { guest } => {
                let report = self
                    .provisioner
                    .shutdown_guest(guest)
                    .await
                    .map_err(ScenarioDriverError::from)?
                    .ok_or_else(|| ScenarioDriverError::UnknownGuest(guest.clone()))?;
                Ok(ScenarioStepResult {
                    index,
                    action: "shutdown",
                    guest: Some(guest.clone()),
                    session: None,
                    detail: format!("shutdown {} forced={:?}", guest, report.forced),
                    exec: None,
                    pty_read: None,
                    screen: None,
                    shutdown: Some(report),
                })
            }
        }
    }

    fn terminal(&self, session: &str) -> Result<&HarnessTerminalSession, ScenarioDriverError> {
        self.terminals
            .get(session)
            .ok_or_else(|| ScenarioDriverError::UnknownSession(session.to_string()))
    }

    async fn shutdown_all(&mut self) {
        for session in self.terminals.values() {
            let _ = session.persist_artifacts();
        }
        for guest in self
            .provisioner
            .guests()
            .unwrap_or_default()
            .into_iter()
            .filter(|guest| guest.active)
            .map(|guest| guest.principal)
        {
            let _ = self.provisioner.shutdown_guest(&guest).await;
        }
        if self.backend.cleanup_vz_disks() {
            cleanup_development_guest_disks(&self.namespace, "scenario");
        }
    }
}

fn duration_or_default(timeout_ms: Option<u64>, default_ms: u64) -> Duration {
    Duration::from_millis(timeout_ms.unwrap_or(default_ms))
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch,
            _ => '-',
        })
        .collect()
}

fn action_name(step: &ScenarioStep) -> &'static str {
    match step {
        ScenarioStep::Boot { .. } => "boot",
        ScenarioStep::Ready { .. } => "ready",
        ScenarioStep::Exec { .. } => "exec",
        ScenarioStep::ProxyExec { .. } => "proxy_exec",
        ScenarioStep::WaitPackageManagerQuiescent { .. } => "wait_package_manager_quiescent",
        ScenarioStep::WaitEgressReady { .. } => "wait_egress_ready",
        ScenarioStep::CheckVfsMemfs { .. } => "check_vfs_memfs",
        ScenarioStep::AptUpdate { .. } => "apt_update",
        ScenarioStep::CheckAgentCli { .. } => "check_agent_cli",
        ScenarioStep::PtyOpen { .. } => "pty_open",
        ScenarioStep::PtySend { .. } => "pty_send",
        ScenarioStep::PtySendLine { .. } => "pty_send_line",
        ScenarioStep::PtyRead { .. } => "pty_read",
        ScenarioStep::PtyResize { .. } => "pty_resize",
        ScenarioStep::PtyExpect { .. } => "pty_expect",
        ScenarioStep::PtyExpectScreen { .. } => "pty_expect_screen",
        ScenarioStep::PtyExpectTerminal { .. } => "pty_expect_terminal",
        ScenarioStep::PtySnapshot { .. } => "pty_snapshot",
        ScenarioStep::Shutdown { .. } => "shutdown",
    }
}

fn step_guest(step: &ScenarioStep) -> Option<&str> {
    match step {
        ScenarioStep::Boot { guest }
        | ScenarioStep::Ready { guest, .. }
        | ScenarioStep::Exec { guest, .. }
        | ScenarioStep::ProxyExec {
            principal: guest, ..
        }
        | ScenarioStep::WaitPackageManagerQuiescent { guest, .. }
        | ScenarioStep::WaitEgressReady { guest, .. }
        | ScenarioStep::CheckVfsMemfs { guest, .. }
        | ScenarioStep::AptUpdate { guest, .. }
        | ScenarioStep::CheckAgentCli { guest, .. }
        | ScenarioStep::PtyOpen { guest, .. }
        | ScenarioStep::Shutdown { guest } => Some(guest),
        _ => None,
    }
}

fn step_session(step: &ScenarioStep) -> Option<&str> {
    match step {
        ScenarioStep::PtyOpen { session, .. }
        | ScenarioStep::PtySend { session, .. }
        | ScenarioStep::PtySendLine { session, .. }
        | ScenarioStep::PtyRead { session, .. }
        | ScenarioStep::PtyResize { session, .. }
        | ScenarioStep::PtyExpect { session, .. }
        | ScenarioStep::PtyExpectScreen { session, .. }
        | ScenarioStep::PtyExpectTerminal { session, .. }
        | ScenarioStep::PtySnapshot { session, .. } => Some(session),
        _ => None,
    }
}

fn check_exec_expectation(
    expect: &ExecExpectation,
    output: &ExecOutput,
) -> Result<(), ScenarioDriverError> {
    if let Some(exit_code) = expect.exit_code {
        if output.exit_code != exit_code {
            return Err(TerminalSessionError::Assertion {
                step: "exec",
                expected: format!("exit code {exit_code}"),
                observed_excerpt: format!("actual exit code {}", output.exit_code),
            }
            .into());
        }
    }
    if let Some(needle) = &expect.stdout_contains {
        if !output.stdout.contains(needle) {
            return Err(TerminalSessionError::Assertion {
                step: "exec",
                expected: format!("stdout containing '{needle}'"),
                observed_excerpt: output.stdout.clone(),
            }
            .into());
        }
    }
    if let Some(needle) = &expect.stderr_contains {
        if !output.stderr.contains(needle) {
            return Err(TerminalSessionError::Assertion {
                step: "exec",
                expected: format!("stderr containing '{needle}'"),
                observed_excerpt: output.stderr.clone(),
            }
            .into());
        }
    }
    Ok(())
}

fn classify_driver_failure(error: &ScenarioDriverError) -> DriverFailure {
    match error {
        ScenarioDriverError::UnknownGuest(_) | ScenarioDriverError::UnknownSession(_) => {
            DriverFailure {
                class: DriverFailureClass::Config,
                stage: "scenario",
                code: "unknown_target",
                message: error.to_string(),
            }
        }
        ScenarioDriverError::GuestAlreadyBooted(_)
        | ScenarioDriverError::SessionAlreadyExists(_) => DriverFailure {
            class: DriverFailureClass::Config,
            stage: "scenario",
            code: "duplicate_target",
            message: error.to_string(),
        },
        ScenarioDriverError::Exec(source) => classify_orchestrator("exec", source),
        ScenarioDriverError::ProxyExec(source) => classify_ssh_failure("proxy_exec", source),
        ScenarioDriverError::Provisioning(source) => classify_provisioning_failure(source),
        ScenarioDriverError::Pty(source) => classify_terminal_failure(source),
    }
}

fn classify_provisioning_failure(error: &ProvisioningError) -> DriverFailure {
    match error {
        ProvisioningError::Orchestrator(source) => classify_orchestrator("provisioning", source),
        ProvisioningError::NetworkAllocation(_) => DriverFailure {
            class: DriverFailureClass::Network,
            stage: "provisioning",
            code: "network_allocation_failed",
            message: error.to_string(),
        },
        ProvisioningError::BuildGuestSpec { .. }
        | ProvisioningError::GuestIdMismatch { .. }
        | ProvisioningError::SeedHostState { .. }
        | ProvisioningError::EmptyPrincipal
        | ProvisioningError::UnknownPrincipal(_)
        | ProvisioningError::GuestNotBooted(_) => DriverFailure {
            class: DriverFailureClass::Config,
            stage: "provisioning",
            code: "guest_provisioning_failed",
            message: error.to_string(),
        },
        ProvisioningError::StatePoisoned(_) => DriverFailure {
            class: DriverFailureClass::Internal,
            stage: "provisioning",
            code: "provisioning_state_poisoned",
            message: error.to_string(),
        },
    }
}

fn classify_terminal_failure(error: &TerminalSessionError) -> DriverFailure {
    match error {
        TerminalSessionError::ControlPlane(source) => classify_ssh_failure("pty", source),
        TerminalSessionError::StatePoisoned => DriverFailure {
            class: DriverFailureClass::Internal,
            stage: "pty",
            code: "pty_state_poisoned",
            message: error.to_string(),
        },
        TerminalSessionError::Persist { .. } => DriverFailure {
            class: DriverFailureClass::Artifact,
            stage: "pty",
            code: "pty_artifact_persist_failed",
            message: error.to_string(),
        },
        TerminalSessionError::Assertion { .. } => DriverFailure {
            class: DriverFailureClass::Pty,
            stage: "pty",
            code: "pty_assertion_failed",
            message: error.to_string(),
        },
    }
}

fn classify_orchestrator(stage: &'static str, error: &OrchestratorError) -> DriverFailure {
    match error {
        OrchestratorError::Spec(_) => DriverFailure {
            class: DriverFailureClass::Config,
            stage,
            code: "spec_invalid",
            message: error.to_string(),
        },
        OrchestratorError::NetworkMode(_) => DriverFailure {
            class: DriverFailureClass::Config,
            stage,
            code: "network_mode_invalid",
            message: error.to_string(),
        },
        OrchestratorError::Artifact(_) => DriverFailure {
            class: DriverFailureClass::Artifact,
            stage,
            code: "artifact_render_failed",
            message: error.to_string(),
        },
        OrchestratorError::Backend(backend) => classify_backend_failure(stage, backend),
        OrchestratorError::Runtime(runtime) => classify_runtime_failure(stage, runtime),
        OrchestratorError::NetworkAllocation(_) => DriverFailure {
            class: DriverFailureClass::Network,
            stage,
            code: "network_allocation_failed",
            message: error.to_string(),
        },
        OrchestratorError::MissingSshBridge { .. } => DriverFailure {
            class: DriverFailureClass::Ssh,
            stage,
            code: "ssh_bridge_missing",
            message: error.to_string(),
        },
        OrchestratorError::StatePoisoned(_) => DriverFailure {
            class: DriverFailureClass::Internal,
            stage,
            code: "state_poisoned",
            message: error.to_string(),
        },
        OrchestratorError::ShutdownFailures { .. } => DriverFailure {
            class: DriverFailureClass::Shutdown,
            stage,
            code: "shutdown_cleanup_failed",
            message: error.to_string(),
        },
        OrchestratorError::GuestExitedEarly { .. } => DriverFailure {
            class: DriverFailureClass::Readiness,
            stage,
            code: "guest_exited_early",
            message: error.to_string(),
        },
        OrchestratorError::ReadinessTimeout { .. } => DriverFailure {
            class: DriverFailureClass::Readiness,
            stage,
            code: "readiness_timeout",
            message: error.to_string(),
        },
    }
}

fn classify_backend_failure(stage: &'static str, error: &BackendError) -> DriverFailure {
    DriverFailure {
        class: DriverFailureClass::Backend,
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

fn classify_runtime_failure(stage: &'static str, error: &RuntimeError) -> DriverFailure {
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
        | RuntimeError::GuestFs(GuestFsError::TaskStatePoisoned) => DriverFailure {
            class: DriverFailureClass::Filesystem,
            stage,
            code: "guestfs_failed",
            message: error.to_string(),
        },
        RuntimeError::Ssh(ssh) => classify_ssh_failure(stage, ssh),
        #[cfg(target_os = "linux")]
        RuntimeError::Vnet(_) | RuntimeError::VnetShutdown(_) => DriverFailure {
            class: DriverFailureClass::Network,
            stage,
            code: "vnet_failed",
            message: error.to_string(),
        },
        RuntimeError::VzEgress(_) => DriverFailure {
            class: DriverFailureClass::Network,
            stage,
            code: "vz_egress_failed",
            message: error.to_string(),
        },
        RuntimeError::UnsupportedHypervisor => DriverFailure {
            class: DriverFailureClass::Backend,
            stage,
            code: "unsupported_hypervisor",
            message: error.to_string(),
        },
    }
}

fn classify_ssh_failure(stage: &'static str, error: &SshProxyError) -> DriverFailure {
    DriverFailure {
        class: DriverFailureClass::Ssh,
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
