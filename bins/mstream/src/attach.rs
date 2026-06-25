use std::env;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, bail, Context};
use motlie_tmux::{run_attach_command_with_options, AttachCommand, AttachOptions};
use serde_json::Value;
use tokio::time::{sleep, Duration};

use crate::cli::AttachArgs;
use crate::client;
use crate::protocol::{
    AttachCommandRecord, AttachResolveRecord, AttachResolveRequest, ClientRequest,
};

const ATTACH_TAG: &str = "@mstream/attach";
const ATTACH_TARGET_TAG: &str = "@mstream/attach-target";
const ATTACH_SPAWNED_BY_TAG: &str = "@mstream/attach-spawned-by";
const ATTACH_CREATED_AT_TAG: &str = "@mstream/attach-created-at";
const ATTACH_WINDOW_NAME: &str = "mstream-attach";
const RETURN_POLL_MS: u64 = 500;

pub async fn run(socket: &Path, args: AttachArgs) -> anyhow::Result<i32> {
    let in_tmux = env::var_os("TMUX").is_some();
    let here = args.here || in_tmux;

    if args.sweep {
        if !in_tmux {
            bail!("--sweep requires the caller to be inside tmux ($TMUX is not set)");
        }
        let _ = sweep_attach_windows()?;
    }

    let target = match args.target {
        Some(target) => target,
        None if args.sweep => return Ok(0),
        None => bail!("<target> is required unless --sweep is used"),
    };
    let resolved = resolve_attach_command(socket, target).await?;

    if args.print {
        println!("{}", resolved.command.shell);
        return Ok(0);
    }

    if here {
        if !in_tmux {
            bail!("--here requires the caller to be inside tmux ($TMUX is not set)");
        }
        attach_here(&resolved).await?;
        return Ok(0);
    }

    let command = attach_command_from_record(resolved.command);
    let exit = tokio::task::spawn_blocking(move || {
        run_attach_command_with_options(command, AttachOptions::default())
    })
    .await??;
    Ok(exit.shell_status())
}

async fn resolve_attach_command(
    socket: &Path,
    target: String,
) -> anyhow::Result<AttachResolveRecord> {
    let request = ClientRequest::ResolveAttach(AttachResolveRequest { target });
    let records = client::send_request(socket, &request).await?;
    let record = records
        .into_iter()
        .next()
        .context("daemon returned no attach resolution record")?;
    if record.get("type").and_then(Value::as_str) == Some("error") {
        let message = record
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("attach resolution failed");
        bail!(message.to_string());
    }
    let resolved: AttachResolveRecord = serde_json::from_value(record)?;
    if resolved.record_type != "ok" || resolved.op != "attach_resolve" {
        bail!("daemon returned unexpected attach resolution record");
    }
    Ok(resolved)
}

fn attach_command_from_record(record: AttachCommandRecord) -> AttachCommand {
    AttachCommand::new(
        OsString::from(record.program),
        record.args.into_iter().map(OsString::from).collect(),
    )
}

async fn attach_here(resolved: &AttachResolveRecord) -> anyhow::Result<()> {
    let window_id = create_attach_window(resolved)?;
    if let Err(err) = tmux_status(["switch-client", "-t", window_id.as_str()]) {
        let _ = kill_window(&window_id);
        return Err(err);
    }
    wait_for_return_and_reap(&window_id).await
}

fn create_attach_window(resolved: &AttachResolveRecord) -> anyhow::Result<String> {
    let shell = attach_window_shell(&resolved.command.shell);
    let window_id = tmux_output([
        "new-window",
        "-d",
        "-P",
        "-F",
        "#{window_id}",
        "-n",
        ATTACH_WINDOW_NAME,
        shell.as_str(),
    ])?
    .trim()
    .to_string();
    if window_id.is_empty() {
        bail!("tmux new-window did not return a window id");
    }

    if let Err(err) = tag_attach_window(&window_id, resolved) {
        let _ = kill_window(&window_id);
        return Err(err);
    }
    Ok(window_id)
}

fn tag_attach_window(window_id: &str, resolved: &AttachResolveRecord) -> anyhow::Result<()> {
    let spawned_by = tmux_output(["display-message", "-p", "#{pane_id}"])
        .unwrap_or_else(|_| "unknown".to_string());
    let created_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    set_window_option(window_id, ATTACH_TAG, "true")?;
    set_window_option(window_id, ATTACH_TARGET_TAG, &resolved.target)?;
    set_window_option(window_id, ATTACH_SPAWNED_BY_TAG, spawned_by.trim())?;
    set_window_option(window_id, ATTACH_CREATED_AT_TAG, &created_at)?;
    Ok(())
}

fn set_window_option(window_id: &str, key: &str, value: &str) -> anyhow::Result<()> {
    tmux_status(["set-option", "-w", "-t", window_id, key, value])
}

fn attach_window_shell(attach_shell: &str) -> String {
    format!(
        "status=0; {attach_shell} || status=$?; tmux kill-window -t \"$TMUX_PANE\" >/dev/null 2>&1 || true; exit $status"
    )
}

async fn wait_for_return_and_reap(window_id: &str) -> anyhow::Result<()> {
    let mut armed = false;
    loop {
        sleep(Duration::from_millis(RETURN_POLL_MS)).await;
        match window_is_active(window_id) {
            Ok(active) if reap_should_kill(&mut armed, active) => {
                kill_window(window_id)?;
                return Ok(());
            }
            Ok(_) => {}
            Err(_) => return Ok(()),
        }
    }
}

fn reap_should_kill(armed: &mut bool, active: bool) -> bool {
    if active {
        *armed = true;
        false
    } else {
        *armed
    }
}

fn window_is_active(window_id: &str) -> anyhow::Result<bool> {
    let active = tmux_output(["display-message", "-p", "-t", window_id, "#{window_active}"])?;
    Ok(active.trim() == "1")
}

fn sweep_attach_windows() -> anyhow::Result<usize> {
    let output = tmux_output([
        "list-windows",
        "-a",
        "-F",
        "#{window_id}\t#{window_active}\t#{@mstream/attach}",
    ])?;
    let mut killed = 0usize;
    for window_id in inactive_attach_windows(&output) {
        kill_window(&window_id)?;
        killed += 1;
    }
    Ok(killed)
}

fn inactive_attach_windows(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(3, '\t');
            let window_id = fields.next()?.trim();
            let active = fields.next()?.trim();
            let tag = fields.next().unwrap_or_default().trim();
            if !window_id.is_empty() && active != "1" && !tag.is_empty() && tag != "0" {
                Some(window_id.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn kill_window(window_id: &str) -> anyhow::Result<()> {
    tmux_status(["kill-window", "-t", window_id])
}

fn tmux_status<I, S>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args = args
        .into_iter()
        .map(|arg| arg.as_ref().to_owned())
        .collect::<Vec<_>>();
    let output = Command::new("tmux")
        .args(&args)
        .output()
        .with_context(|| format!("failed to run tmux {}", display_args(&args)))?;
    if !output.status.success() {
        bail!(tmux_error(&args, &output));
    }
    Ok(())
}

fn tmux_output<I, S>(args: I) -> anyhow::Result<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args = args
        .into_iter()
        .map(|arg| arg.as_ref().to_owned())
        .collect::<Vec<_>>();
    let output = Command::new("tmux")
        .args(&args)
        .output()
        .with_context(|| format!("failed to run tmux {}", display_args(&args)))?;
    if !output.status.success() {
        bail!(tmux_error(&args, &output));
    }
    String::from_utf8(output.stdout).context("tmux output was not valid UTF-8")
}

fn tmux_error(args: &[OsString], output: &std::process::Output) -> anyhow::Error {
    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow!("tmux {} failed: {}", display_args(args), stderr.trim())
}

fn display_args(args: &[OsString]) -> String {
    args.iter()
        .map(|arg| arg.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attach_window_shell_self_kills_after_attach_exits() {
        let shell = attach_window_shell("ssh -t host 'tmux attach-session -t '\''$1'\'''");

        assert!(shell.contains("ssh -t host"));
        assert!(shell.contains("tmux kill-window -t \"$TMUX_PANE\""));
        assert!(shell.contains("exit $status"));
    }

    #[test]
    fn inactive_attach_windows_selects_only_tagged_inactive_windows() {
        let windows = inactive_attach_windows("@1\t1\ttrue\n@2\t0\ttrue\n@3\t0\t\n@4\t0\t0\n");

        assert_eq!(windows, vec!["@2"]);
    }

    #[test]
    fn reap_waits_until_visit_window_was_observed_active() {
        let mut armed = false;

        assert!(!reap_should_kill(&mut armed, false));
        assert!(!armed);
        assert!(!reap_should_kill(&mut armed, true));
        assert!(armed);
        assert!(reap_should_kill(&mut armed, false));
    }

    #[test]
    fn attach_command_from_record_preserves_argv() {
        let command = attach_command_from_record(AttachCommandRecord {
            program: "ssh".to_string(),
            args: vec![
                "-t".to_string(),
                "host".to_string(),
                "tmux attach".to_string(),
            ],
            shell: "ssh -t host 'tmux attach'".to_string(),
        });

        assert_eq!(command.program().to_string_lossy(), "ssh");
        assert_eq!(command.args().len(), 3);
        assert_eq!(command.args()[2].to_string_lossy(), "tmux attach");
    }
}
