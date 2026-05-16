use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use motlie_vmm::provisioning::{GuestProvisioner, ProvisionedGuestSnapshot};
use motlie_vmm::spec::RuntimeNamespace;
use motlie_vmm::ssh::{self, SshProxyConfig, SshProxyError};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub const DEMO_GUEST_VSOCK_PORT: u32 = 5000;
const PROXY_RESTART_RESET_WINDOW: Duration = Duration::from_secs(30);
const PROXY_RESTART_MAX_BACKOFF_SECS: u64 = 30;
const PROXY_RESTART_FAILURE_CAP: u32 = 8;

#[derive(Debug)]
pub enum HostEvent {
    StdinLine(Result<String, io::Error>),
    StdinClosed,
    Terminate(&'static str),
    Hangup,
}

pub struct HostEvents {
    rx: mpsc::UnboundedReceiver<HostEvent>,
}

impl HostEvents {
    pub async fn recv(&mut self) -> Option<HostEvent> {
        self.rx.recv().await
    }
}

pub struct ProxyRestartState {
    last_started_at: Instant,
    consecutive_failures: u32,
}

impl ProxyRestartState {
    pub fn new() -> Self {
        Self {
            last_started_at: Instant::now(),
            consecutive_failures: 0,
        }
    }

    pub fn mark_started(&mut self) {
        self.last_started_at = Instant::now();
    }

    pub fn next_delay(&mut self) -> Result<Duration, String> {
        if self.last_started_at.elapsed() >= PROXY_RESTART_RESET_WINDOW {
            self.consecutive_failures = 0;
        }

        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        if self.consecutive_failures > PROXY_RESTART_FAILURE_CAP {
            return Err(format!(
                "SSH proxy exited {} consecutive times within {:?}",
                self.consecutive_failures, PROXY_RESTART_RESET_WINDOW
            ));
        }

        let exponent = self.consecutive_failures.saturating_sub(1).min(5);
        let backoff_secs = (1u64 << exponent).min(PROXY_RESTART_MAX_BACKOFF_SECS);
        Ok(Duration::from_secs(backoff_secs))
    }
}

pub fn spawn_host_events() -> (HostEvents, mpsc::UnboundedSender<HostEvent>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let stdin_tx = tx.clone();
    thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            if stdin_tx.send(HostEvent::StdinLine(line)).is_err() {
                return;
            }
        }
        let _ = stdin_tx.send(HostEvent::StdinClosed);
    });
    (HostEvents { rx }, tx)
}

pub fn install_signal_watchers(tx: mpsc::UnboundedSender<HostEvent>) -> io::Result<()> {
    let ctrlc_tx = tx.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = ctrlc_tx.send(HostEvent::Terminate("SIGINT"));
        }
    });

    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = signal(SignalKind::terminate())?;
        let sigterm_tx = tx.clone();
        tokio::spawn(async move {
            while sigterm.recv().await.is_some() {
                let _ = sigterm_tx.send(HostEvent::Terminate("SIGTERM"));
            }
        });

        let mut sighup = signal(SignalKind::hangup())?;
        tokio::spawn(async move {
            while sighup.recv().await.is_some() {
                let _ = tx.send(HostEvent::Hangup);
            }
        });
    }

    Ok(())
}

pub fn spawn_proxy_task(
    config: SshProxyConfig,
    registry: ssh::GuestRegistry,
) -> JoinHandle<Result<(), SshProxyError>> {
    tokio::spawn(ssh::run_proxy(config, registry))
}

pub fn prompt(stdout: &mut io::Stdout, prompt: &str, headless: &mut bool) {
    if *headless {
        return;
    }

    print!("{prompt}");
    if let Err(err) = stdout.flush() {
        *headless = true;
        eprintln!("notice: stdout I/O error ({err}); continuing headless");
    }
}

pub fn stdin_line_or_detach(
    line_result: Result<String, io::Error>,
    headless: &mut bool,
) -> Option<String> {
    match line_result {
        Ok(line) => Some(line),
        Err(err) => {
            if !*headless {
                *headless = true;
                eprintln!("notice: stdin I/O error ({err}); continuing headless");
            }
            None
        }
    }
}

pub fn guest_runtime_paths(
    guest: &ProvisionedGuestSnapshot,
) -> Option<&motlie_vmm::spec::GuestRuntimePaths> {
    guest.runtime_paths.as_ref()
}

pub async fn shutdown_active_guests(provisioner: &GuestProvisioner, label: &str) {
    for guest in provisioner
        .guests()
        .unwrap_or_default()
        .into_iter()
        .filter(|guest| guest.active)
        .map(|guest| guest.principal)
    {
        if let Err(err) = provisioner.shutdown_guest(&guest).await {
            eprintln!("warning: failed to shutdown {label} guest '{guest}': {err}");
        }
    }
}

pub fn cleanup_development_guest_disks(namespace: &RuntimeNamespace, label: &str) {
    if std::env::var_os("MOTLIE_VZ_KEEP_GUEST_DISKS").is_some() {
        return;
    }

    let runtime_root = namespace
        .temp_root
        .join(format!("{}-runtime", namespace.prefix));
    match cleanup_development_guest_disks_under(&runtime_root) {
        Ok(0) => {}
        Ok(count) => eprintln!("notice: removed {count} {label} VZ development disk artifact(s)"),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => eprintln!(
            "warning: failed to clean {label} VZ development disks under {}: {err}",
            runtime_root.display()
        ),
    }
}

fn cleanup_development_guest_disks_under(runtime_root: &Path) -> io::Result<usize> {
    let mut removed = 0usize;
    let mut pending = vec![runtime_root.to_path_buf()];

    while let Some(dir) = pending.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err),
        };

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if path.extension().and_then(|value| value.to_str()) == Some("vm") {
                    fs::remove_dir_all(&path)?;
                    removed += 1;
                } else {
                    pending.push(path);
                }
                continue;
            }

            if !file_type.is_file() {
                continue;
            }
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if file_name == "seed.dmg" || file_name.ends_with(".tmp-copy") {
                fs::remove_file(&path)?;
                removed += 1;
            }
        }
    }

    Ok(removed)
}

pub fn demo_guest_ids(_guest_id: &str, slot: u32) -> Result<(u32, u32), String> {
    let uid = 3000u32
        .checked_add(slot)
        .ok_or_else(|| format!("guest slot {slot} exceeds supported uid/gid range"))?;
    Ok((uid, uid))
}

pub fn demo_guest_socket_path(
    namespace: &RuntimeNamespace,
    guest_id: &str,
) -> Result<PathBuf, String> {
    namespace
        .guest_vsock_port_socket(guest_id, DEMO_GUEST_VSOCK_PORT)
        .map_err(|err| err.to_string())
}
