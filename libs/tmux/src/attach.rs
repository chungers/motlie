use crate::control::shell_escape;
use crate::error::{Error, Result};
use crate::transport::SshConfig;
use crate::types::{HostKeyPolicy, TmuxSocket};
use std::ffi::OsString;
use std::process::{ExitStatus, Stdio};

/// Result of attaching the user's current PTY to a tmux target.
#[derive(Debug)]
pub struct AttachExit {
    pub status: ExitStatus,
}

impl AttachExit {
    /// Shell-compatible exit code.
    ///
    /// On Unix, signal-terminated children map to `128 + signal`.
    pub fn shell_status(&self) -> i32 {
        shell_status_code(&self.status)
    }

    pub fn success(&self) -> bool {
        self.status.success()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttachCommand {
    program: OsString,
    args: Vec<OsString>,
}

impl AttachCommand {
    fn new(program: impl Into<OsString>, args: Vec<OsString>) -> Self {
        Self {
            program: program.into(),
            args,
        }
    }

    fn display(&self) -> String {
        let mut parts = vec![self.program.to_string_lossy().into_owned()];
        parts.extend(
            self.args
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned()),
        );
        parts.join(" ")
    }
}

pub(crate) fn local_attach_command(
    tmux_bin: &str,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> AttachCommand {
    let mut args = tmux_socket_args(socket);
    args.push(OsString::from("attach-session"));
    args.push(OsString::from("-t"));
    args.push(OsString::from(target));
    AttachCommand::new(tmux_bin, args)
}

pub(crate) fn ssh_attach_command(
    config: &SshConfig,
    tmux_bin: &str,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> AttachCommand {
    let mut args = vec![OsString::from("-t")];

    if config.port() != 22 {
        args.push(OsString::from("-p"));
        args.push(OsString::from(config.port().to_string()));
    }

    match config.host_key_policy() {
        HostKeyPolicy::Verify => {}
        HostKeyPolicy::TrustFirstUse => {
            args.push(OsString::from("-o"));
            args.push(OsString::from("StrictHostKeyChecking=accept-new"));
        }
        HostKeyPolicy::Insecure => {
            args.push(OsString::from("-o"));
            args.push(OsString::from("StrictHostKeyChecking=no"));
            args.push(OsString::from("-o"));
            args.push(OsString::from("UserKnownHostsFile=/dev/null"));
        }
    }

    if let Some(identity_file) = config.identity_file() {
        args.push(OsString::from("-i"));
        args.push(identity_file.as_os_str().to_owned());
    }

    args.push(OsString::from(ssh_destination(config)));
    args.push(OsString::from(remote_tmux_attach_command(
        tmux_bin, socket, target,
    )));

    AttachCommand::new("ssh", args)
}

pub(crate) fn run_attach_command(command: AttachCommand) -> Result<AttachExit> {
    run_attach_command_unix(command)
}

fn tmux_socket_args(socket: Option<&TmuxSocket>) -> Vec<OsString> {
    match socket {
        None => Vec::new(),
        Some(TmuxSocket::Name(name)) => vec![OsString::from("-L"), OsString::from(name)],
        Some(TmuxSocket::Path(path)) => vec![OsString::from("-S"), OsString::from(path)],
    }
}

fn remote_tmux_attach_command(tmux_bin: &str, socket: Option<&TmuxSocket>, target: &str) -> String {
    let mut parts = vec![shell_escape(tmux_bin)];
    match socket {
        None => {}
        Some(TmuxSocket::Name(name)) => {
            parts.push("-L".to_string());
            parts.push(shell_escape(name));
        }
        Some(TmuxSocket::Path(path)) => {
            parts.push("-S".to_string());
            parts.push(shell_escape(path));
        }
    }
    parts.push("attach-session".to_string());
    parts.push("-t".to_string());
    parts.push(shell_escape(target));
    parts.join(" ")
}

fn ssh_destination(config: &SshConfig) -> String {
    let host = ssh_host(config.host());
    if config.user().is_empty() {
        host
    } else {
        format!("{}@{}", config.user(), host)
    }
}

fn ssh_host(host: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{}]", host)
    } else {
        host.to_string()
    }
}

#[cfg(unix)]
fn run_attach_command_unix(command: AttachCommand) -> Result<AttachExit> {
    use std::os::unix::process::CommandExt;

    let mut child = std::process::Command::new(&command.program)
        .args(&command.args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .process_group(0)
        .spawn()
        .map_err(|e| {
            Error::Transport(format!(
                "failed to spawn attach command '{}': {}",
                command.display(),
                e
            ))
        })?;

    let _terminal_guard = match ForegroundProcessGroup::enter(child.id()) {
        Ok(guard) => guard,
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
    };
    let status = child.wait().map_err(|e| {
        Error::Transport(format!(
            "failed to wait for attach command '{}': {}",
            command.display(),
            e
        ))
    })?;

    Ok(AttachExit { status })
}

#[cfg(not(unix))]
fn run_attach_command_unix(command: AttachCommand) -> Result<AttachExit> {
    let status = std::process::Command::new(&command.program)
        .args(&command.args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| {
            Error::Transport(format!(
                "failed to run attach command '{}': {}",
                command.display(),
                e
            ))
        })?;
    Ok(AttachExit { status })
}

#[cfg(unix)]
struct ForegroundProcessGroup {
    fd: libc::c_int,
    original_pgrp: libc::pid_t,
    active: bool,
}

#[cfg(unix)]
impl ForegroundProcessGroup {
    fn enter(child_pid: u32) -> Result<Option<Self>> {
        let fd = libc::STDIN_FILENO;
        let is_tty = unsafe { libc::isatty(fd) };
        if is_tty != 1 {
            return Ok(None);
        }

        let original_pgrp = unsafe { libc::tcgetpgrp(fd) };
        if original_pgrp == -1 {
            return Err(Error::Transport(format!(
                "failed to read foreground process group: {}",
                std::io::Error::last_os_error()
            )));
        }

        let child_pgrp: libc::pid_t = child_pid
            .try_into()
            .map_err(|_| Error::Transport(format!("child pid {} does not fit pid_t", child_pid)))?;

        tcsetpgrp_ignoring_sigttou(fd, child_pgrp, "transfer terminal foreground process group")?;

        Ok(Some(Self {
            fd,
            original_pgrp,
            active: true,
        }))
    }
}

#[cfg(unix)]
impl Drop for ForegroundProcessGroup {
    fn drop(&mut self) {
        if self.active {
            if let Err(err) = tcsetpgrp_ignoring_sigttou(
                self.fd,
                self.original_pgrp,
                "restore terminal foreground process group after attach",
            ) {
                tracing::warn!(error = %err, "failed to restore terminal foreground process group");
            }
        }
        self.active = false;
    }
}

#[cfg(unix)]
fn tcsetpgrp_ignoring_sigttou(fd: libc::c_int, pgrp: libc::pid_t, action: &str) -> Result<()> {
    let _guard = SignalDispositionGuard::ignore(libc::SIGTTOU)?;
    if unsafe { libc::tcsetpgrp(fd, pgrp) } == -1 {
        return Err(Error::Transport(format!(
            "failed to {action}: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

#[cfg(unix)]
struct SignalDispositionGuard {
    signal: libc::c_int,
    previous: libc::sigaction,
    active: bool,
}

#[cfg(unix)]
impl SignalDispositionGuard {
    fn ignore(signal: libc::c_int) -> Result<Self> {
        let mut next: libc::sigaction = unsafe { std::mem::zeroed() };
        next.sa_sigaction = libc::SIG_IGN;
        if unsafe { libc::sigemptyset(&mut next.sa_mask) } == -1 {
            return Err(Error::Transport(format!(
                "failed to initialize signal mask: {}",
                std::io::Error::last_os_error()
            )));
        }

        let mut previous: libc::sigaction = unsafe { std::mem::zeroed() };
        if unsafe { libc::sigaction(signal, &next, &mut previous) } == -1 {
            return Err(Error::Transport(format!(
                "failed to ignore signal {signal}: {}",
                std::io::Error::last_os_error()
            )));
        }

        Ok(Self {
            signal,
            previous,
            active: true,
        })
    }
}

#[cfg(unix)]
impl Drop for SignalDispositionGuard {
    fn drop(&mut self) {
        if self.active
            && unsafe { libc::sigaction(self.signal, &self.previous, std::ptr::null_mut()) } == -1
        {
            tracing::warn!(
                signal = self.signal,
                error = %std::io::Error::last_os_error(),
                "failed to restore signal disposition"
            );
        }
        self.active = false;
    }
}

#[cfg(unix)]
fn shell_status_code(status: &ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;

    if let Some(code) = status.code() {
        code
    } else if let Some(signal) = status.signal() {
        128 + signal
    } else {
        1
    }
}

#[cfg(not(unix))]
fn shell_status_code(status: &ExitStatus) -> i32 {
    status.code().unwrap_or(1)
}

#[cfg(test)]
impl AttachCommand {
    fn program_str(&self) -> String {
        self.program.to_string_lossy().into_owned()
    }

    fn args_str(&self) -> Vec<String> {
        self.args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;

    #[test]
    fn local_attach_command_preserves_socket_and_target() {
        let command = local_attach_command(
            "/opt/bin/tmux",
            Some(&TmuxSocket::Name("motlie".into())),
            "$3",
        );

        assert_eq!(command.program_str(), "/opt/bin/tmux");
        assert_eq!(
            command.args_str(),
            vec!["-L", "motlie", "attach-session", "-t", "$3"]
        );
    }

    #[test]
    fn ssh_attach_command_uses_config_and_remote_shell_escaping() {
        let config = SshConfig::new("example.com", "deploy")
            .with_port(2222)
            .with_host_key_policy(HostKeyPolicy::Insecure)
            .with_identity_file("/keys/deploy")
            .expect("identity file should be accepted");
        let command = ssh_attach_command(
            &config,
            "/opt/tmux bin/tmux",
            Some(&TmuxSocket::Path("/tmp/tmux socket".into())),
            "$7",
        );

        assert_eq!(command.program_str(), "ssh");
        assert_eq!(
            command.args_str(),
            vec![
                "-t",
                "-p",
                "2222",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-i",
                "/keys/deploy",
                "deploy@example.com",
                "'/opt/tmux bin/tmux' -S '/tmp/tmux socket' attach-session -t '$7'",
            ]
        );
    }

    #[test]
    fn ssh_attach_command_brackets_ipv6_host() {
        let config = SshConfig::new("2001:db8::1", "deploy");
        let command = ssh_attach_command(&config, "tmux", None, "$1");

        assert_eq!(command.args_str()[1], "deploy@[2001:db8::1]");
    }

    #[cfg(unix)]
    #[test]
    fn attach_exit_shell_status_maps_exit_codes_and_signals() {
        assert_eq!(shell_status_code(&ExitStatus::from_raw(0)), 0);
        assert_eq!(shell_status_code(&ExitStatus::from_raw(7 << 8)), 7);
        assert_eq!(shell_status_code(&ExitStatus::from_raw(libc::SIGTERM)), 143);
    }

    #[cfg(unix)]
    #[test]
    fn tcsetpgrp_helper_uses_sigttou_safe_path() {
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);

        let result =
            tcsetpgrp_ignoring_sigttou(fds[0], unsafe { libc::getpgrp() }, "test tcsetpgrp");

        let _ = unsafe { libc::close(fds[0]) };
        let _ = unsafe { libc::close(fds[1]) };
        assert!(result.is_err());
        let message = result.err().map(|err| err.to_string()).unwrap_or_default();
        assert!(message.contains("failed to test tcsetpgrp"));
    }

    #[test]
    fn remote_tmux_attach_command_escapes_quotes() {
        let command = remote_tmux_attach_command("tmux", None, "it'is");

        assert_eq!(command, "'tmux' attach-session -t 'it'\\''is'");
    }

    #[test]
    fn display_includes_program_and_args() {
        let command = AttachCommand::new("tmux", vec![OsString::from("attach-session")]);

        assert_eq!(command.display(), "tmux attach-session");
    }
}
