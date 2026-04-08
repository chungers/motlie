use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use motlie_vmm::backend::BackendError;
use motlie_vmm::ca::SshCa;
use motlie_vmm::guestfs::GuestFsError;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::orchestrator::{
    LifecycleServices, OrchestratorError, PrepareRequest, ReadinessPolicy, ShutdownReport,
    VmHandle, boot, prepare,
};
use motlie_vmm::runtime::{
    ControlPlaneBacking, FilesystemBacking, HypervisorBacking, NetworkBacking, Runtime,
    RuntimeError,
};
use motlie_vmm::ssh::{
    self, ExecOutput, PtyRead, PtyRequest, SshProxyConfig, SshProxyError, new_guest_registry,
};
use serde::{Deserialize, Serialize};

use crate::terminal::{HarnessTerminalSession, TerminalSessionError, VteScreenSnapshot};
use crate::{
    DynError, HarnessInstance, demo_guest, ensure_file_exists, persist_json,
    print_instance_details, seed_host_mounts,
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
    pub transcript_ndjson: PathBuf,
    pub screen_json: PathBuf,
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
    pub description: Option<String>,
    pub artifact_root: PathBuf,
    pub proxy: String,
    pub allocator_capacity: u32,
    pub steps: Vec<ScenarioStepResult>,
    pub sessions: HashMap<String, ScenarioSessionArtifacts>,
    pub error: Option<DriverFailure>,
    pub cleanup_error: Option<DriverFailure>,
}

#[derive(Debug, thiserror::Error)]
pub enum ScenarioDriverError {
    #[error("scenario setup failed: {0}")]
    Setup(String),
    #[error("SSH CA setup failed: {0}")]
    Ca(String),
    #[error("unknown guest '{0}'")]
    UnknownGuest(String),
    #[error("guest '{0}' is already booted")]
    GuestAlreadyBooted(String),
    #[error("unknown PTY session '{0}'")]
    UnknownSession(String),
    #[error("PTY session '{0}' already exists")]
    SessionAlreadyExists(String),
    #[error("prepare failed: {0}")]
    Prepare(#[source] OrchestratorError),
    #[error("boot failed: {0}")]
    Boot(#[source] OrchestratorError),
    #[error("readiness failed: {0}")]
    Ready(#[source] OrchestratorError),
    #[error("exec failed: {0}")]
    Exec(#[source] OrchestratorError),
    #[error("shutdown failed: {0}")]
    Shutdown(#[source] OrchestratorError),
    #[error(transparent)]
    Pty(#[from] TerminalSessionError),
}

pub async fn run_scenario_file(
    base_dir: &Path,
    artifacts_dir: &Path,
    instance: &HarnessInstance,
    allocator_config: GuestNetAllocatorConfig,
    path: &Path,
    result_json_path: Option<&Path>,
) -> Result<ScenarioRunResult, DynError> {
    let bytes = std::fs::read(path)?;
    let definition: ScenarioDefinition = serde_json::from_slice(&bytes)?;
    let result = run_scenario_definition(
        base_dir,
        artifacts_dir,
        instance,
        allocator_config,
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
    instance: &HarnessInstance,
    allocator_config: GuestNetAllocatorConfig,
    definition: ScenarioDefinition,
) -> Result<ScenarioRunResult, DynError> {
    let mut driver = ScenarioDriver::new(base_dir, artifacts_dir, instance, allocator_config)?;
    let result = driver.run(definition).await;
    driver.shutdown_all().await;
    Ok(result)
}

struct ScenarioDriver {
    base_dir: PathBuf,
    artifacts_dir: PathBuf,
    instance: HarnessInstance,
    runtime: Arc<Runtime>,
    ca: Arc<SshCa>,
    allocator: GuestNetAllocator,
    proxy_config: SshProxyConfig,
    handles: HashMap<String, VmHandle>,
    terminals: HashMap<String, HarnessTerminalSession>,
    session_guests: HashMap<String, String>,
    artifact_root: PathBuf,
}

impl ScenarioDriver {
    fn new(
        base_dir: &Path,
        artifacts_dir: &Path,
        instance: &HarnessInstance,
        allocator_config: GuestNetAllocatorConfig,
    ) -> Result<Self, DynError> {
        ensure_file_exists(&artifacts_dir.join("rootfs.squashfs"))?;
        ensure_file_exists(&artifacts_dir.join("Image"))?;
        std::fs::create_dir_all(&instance.socket_root)?;

        let ca = Arc::new(SshCa::new()?);
        let guest_registry = new_guest_registry();
        let proxy_config = SshProxyConfig {
            listen: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), instance.proxy_port),
        };
        tokio::spawn(ssh::run_proxy(
            proxy_config.clone(),
            Arc::clone(&guest_registry),
        ));
        print_instance_details(instance, &proxy_config);

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
        let allocator = GuestNetAllocator::new(allocator_config)?;
        let artifact_root = instance
            .namespace
            .temp_root
            .join(format!("{}-scenarios", instance.namespace.prefix));
        std::fs::create_dir_all(&artifact_root)?;

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            artifacts_dir: artifacts_dir.to_path_buf(),
            instance: HarnessInstance {
                namespace: instance.namespace.clone(),
                demo_root: instance.demo_root.clone(),
                socket_root: instance.socket_root.clone(),
                proxy_port: instance.proxy_port,
            },
            runtime,
            ca,
            allocator,
            proxy_config,
            handles: HashMap::new(),
            terminals: HashMap::new(),
            session_guests: HashMap::new(),
            artifact_root,
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
                                    transcript_ndjson: session.transcript_path().to_path_buf(),
                                    screen_json: session.screen_path().to_path_buf(),
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
            description: definition.description,
            artifact_root: scenario_root,
            proxy: format!("ssh://localhost:{}", self.proxy_config.listen.port()),
            allocator_capacity: self.allocator.capacity().unwrap_or_default(),
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
                if self.handles.contains_key(guest) {
                    return Err(ScenarioDriverError::GuestAlreadyBooted(guest.clone()));
                }
                let spec = demo_guest(
                    guest,
                    &self.artifacts_dir,
                    &self.instance.demo_root,
                    &self.instance.namespace,
                );
                seed_host_mounts(&spec)
                    .map_err(|source| ScenarioDriverError::Setup(source.to_string()))?;
                let prepared = prepare(
                    PrepareRequest {
                        guest: spec,
                        namespace: self.instance.namespace.clone(),
                        network_modes: NetworkModes {
                            admin: AdminNetMode::None,
                            egress: EgressNetMode::VhostUser,
                        },
                        base_dir: self.base_dir.clone(),
                        ssh_ca_pubkey: Some(
                            self.ca
                                .public_key_openssh()
                                .map_err(|source| ScenarioDriverError::Ca(source.to_string()))?,
                        ),
                    },
                    &mut self.allocator,
                )
                .map_err(ScenarioDriverError::Prepare)?;

                let handle = boot(
                    prepared,
                    LifecycleServices {
                        runtime: Arc::clone(&self.runtime),
                    },
                )
                .await
                .map_err(ScenarioDriverError::Boot)?;
                let detail = format!(
                    "booted {} pid={:?} api={} cid={} admin_subnet={} egress_subnet={}",
                    guest,
                    handle.pid,
                    handle.runtime_paths.api_socket.display(),
                    handle.net_assignment.cid,
                    handle.net_assignment.admin_subnet,
                    handle.net_assignment.egress_subnet,
                );
                self.handles.insert(guest.clone(), handle);
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
                let handle = self.handle(guest)?;
                handle
                    .ready(&ReadinessPolicy {
                        api_socket_timeout: duration_or_default(*timeout_ms, 10_000),
                        guestfs_timeout: duration_or_default(*timeout_ms, 15_000),
                        ssh_bridge_timeout: duration_or_default(*timeout_ms, 15_000),
                        exec_ready_timeout: duration_or_default(*timeout_ms, 20_000),
                    })
                    .await
                    .map_err(ScenarioDriverError::Ready)?;
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
                let handle = self.handle(guest)?;
                let output = handle
                    .exec(command, duration_or_default(*timeout_ms, 20_000))
                    .await
                    .map_err(ScenarioDriverError::Exec)?;
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
                let handle = self.handle(guest)?;
                let request = PtyRequest {
                    term: term.clone().unwrap_or_else(|| "xterm-256color".to_string()),
                    col_width: cols.unwrap_or(80),
                    row_height: rows.unwrap_or(24),
                    pix_width: 0,
                    pix_height: 0,
                    command: command.clone(),
                };
                let guest_session = handle
                    .open_pty(request.clone(), duration_or_default(*timeout_ms, 10_000))
                    .await
                    .map_err(ScenarioDriverError::Exec)?;
                let session_root = scenario_root.join("sessions").join(sanitize(session));
                let terminal = HarnessTerminalSession::new(
                    format!("{guest}:{session}"),
                    guest_session,
                    &request,
                    session_root.join("pty-transcript.ndjson"),
                    session_root.join("pty-screen.json"),
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
                let handle = self
                    .handles
                    .remove(guest)
                    .ok_or_else(|| ScenarioDriverError::UnknownGuest(guest.clone()))?;
                let report = handle
                    .shutdown()
                    .await
                    .map_err(ScenarioDriverError::Shutdown)?;
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

    fn handle(&self, guest: &str) -> Result<&VmHandle, ScenarioDriverError> {
        self.handles
            .get(guest)
            .ok_or_else(|| ScenarioDriverError::UnknownGuest(guest.to_string()))
    }

    fn terminal(&self, session: &str) -> Result<&HarnessTerminalSession, ScenarioDriverError> {
        self.terminals
            .get(session)
            .ok_or_else(|| ScenarioDriverError::UnknownSession(session.to_string()))
    }

    async fn shutdown_all(&mut self) {
        for (_name, handle) in self.handles.drain() {
            let _ = handle.shutdown().await;
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
        ScenarioDriverError::Setup(_) => DriverFailure {
            class: DriverFailureClass::Artifact,
            stage: "setup",
            code: "scenario_setup_failed",
            message: error.to_string(),
        },
        ScenarioDriverError::Ca(_) => DriverFailure {
            class: DriverFailureClass::Ssh,
            stage: "setup",
            code: "scenario_ca_failed",
            message: error.to_string(),
        },
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
        ScenarioDriverError::Prepare(source) => classify_orchestrator("prepare", source),
        ScenarioDriverError::Boot(source) => classify_orchestrator("boot", source),
        ScenarioDriverError::Ready(source) => classify_orchestrator("ready", source),
        ScenarioDriverError::Exec(source) => classify_orchestrator("exec", source),
        ScenarioDriverError::Shutdown(source) => {
            let mut failure = classify_orchestrator("shutdown", source);
            failure.class = DriverFailureClass::Shutdown;
            failure
        }
        ScenarioDriverError::Pty(source) => classify_terminal_failure(source),
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
        | RuntimeError::GuestFs(GuestFsError::CleanupSocket { .. }) => DriverFailure {
            class: DriverFailureClass::Filesystem,
            stage,
            code: "guestfs_failed",
            message: error.to_string(),
        },
        RuntimeError::Ssh(ssh) => classify_ssh_failure(stage, ssh),
        RuntimeError::Vnet(_) | RuntimeError::VnetShutdown(_) => DriverFailure {
            class: DriverFailureClass::Network,
            stage,
            code: "vnet_failed",
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
