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
    AttachCommandRecord, AttachResolveMode, AttachResolveRecord, AttachResolveRequest,
    ClientRequest,
};

const ATTACH_TAG: &str = "@mstream/attach";
const ATTACH_TARGET_TAG: &str = "@mstream/attach-target";
const ATTACH_SPAWNED_BY_TAG: &str = "@mstream/attach-spawned-by";
const ATTACH_CREATED_AT_TAG: &str = "@mstream/attach-created-at";
const ATTACH_READY_TAG: &str = "@mstream/attach-ready";
const ATTACH_WINDOW_NAME: &str = "mstream-attach";
const ATTACH_FAILURE_CAPTURE_LINES: &str = "-120";
const RETURN_POLL_MS: u64 = 500;

#[derive(Debug, Clone, PartialEq, Eq)]
struct AttachWindow {
    window_id: String,
    pane_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VisitPaneState {
    Running,
    Dead { exit_status: Option<i32> },
    Gone,
}

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
    let mode = if here {
        AttachResolveMode::WindowInjection
    } else {
        AttachResolveMode::Pty
    };
    let resolved = resolve_attach_command(socket, target, mode).await?;

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
    mode: AttachResolveMode,
) -> anyhow::Result<AttachResolveRecord> {
    let request = ClientRequest::ResolveAttach(AttachResolveRequest { target, mode });
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
    let (visit, client_ttys) = prepare_attach_here_visit(
        resolved,
        sweep_attach_windows,
        caller_session_id,
        attached_client_ttys,
        create_attach_window,
    )?;
    if let Err(err) = switch_clients_to_window(&client_ttys, &visit.window_id) {
        let _ = kill_window(&visit.window_id);
        return Err(err);
    }
    wait_for_visit_pane_exit_and_reap(&visit, &resolved.target).await
}

fn prepare_attach_here_visit(
    resolved: &AttachResolveRecord,
    sweep: impl FnOnce() -> anyhow::Result<usize>,
    caller_session_id: impl FnOnce() -> anyhow::Result<String>,
    attached_client_ttys: impl FnOnce(&str) -> anyhow::Result<Vec<String>>,
    create_attach_window: impl FnOnce(&AttachResolveRecord, &str) -> anyhow::Result<AttachWindow>,
) -> anyhow::Result<(AttachWindow, Vec<String>)> {
    sweep()?;
    let caller_session_id = caller_session_id()?;
    let client_ttys = attached_client_ttys(&caller_session_id)?;
    if client_ttys.is_empty() {
        bail!("no attached tmux clients found for caller session {caller_session_id}");
    }
    let visit = create_attach_window(resolved, &caller_session_id)?;
    Ok((visit, client_ttys))
}

fn caller_session_id() -> anyhow::Result<String> {
    let output = match env::var("TMUX_PANE") {
        Ok(pane_id) if !pane_id.is_empty() => tmux_output([
            "display-message",
            "-p",
            "-t",
            pane_id.as_str(),
            "#{session_id}",
        ])?,
        _ => tmux_output(["display-message", "-p", "#{session_id}"])?,
    };
    let session_id = output.trim();
    if session_id.is_empty() {
        bail!("tmux did not return a caller session id");
    }
    Ok(session_id.to_string())
}

fn attached_client_ttys(session_id: &str) -> anyhow::Result<Vec<String>> {
    let output = match tmux_output(["list-clients", "-F", "#{client_tty}\t#{session_id}"]) {
        Ok(output) => output,
        Err(err) if tmux_error_contains(&err, "no clients") => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    Ok(attached_client_ttys_for_session(&output, session_id))
}

fn attached_client_ttys_for_session(output: &str, session_id: &str) -> Vec<String> {
    let mut ttys = Vec::new();
    for line in output.lines() {
        let Some((tty, client_session_id)) = line.split_once('\t') else {
            continue;
        };
        let tty = tty.trim();
        if client_session_id.trim() == session_id
            && !tty.is_empty()
            && !ttys.iter().any(|existing| existing == tty)
        {
            ttys.push(tty.to_string());
        }
    }
    ttys
}

fn switch_clients_to_window(client_ttys: &[String], window_id: &str) -> anyhow::Result<()> {
    let mut switched = 0usize;
    let mut errors = Vec::new();
    for client_tty in client_ttys {
        match tmux_status(["switch-client", "-c", client_tty.as_str(), "-t", window_id]) {
            Ok(()) => switched += 1,
            Err(err) if tmux_error_contains(&err, "can't find client") => {}
            Err(err) => errors.push(err.to_string()),
        }
    }

    if switched > 0 {
        Ok(())
    } else if errors.is_empty() {
        bail!("no attached tmux clients found for caller session")
    } else {
        bail!(
            "failed to switch attached tmux clients: {}",
            errors.join("; ")
        )
    }
}

fn create_attach_window(
    resolved: &AttachResolveRecord,
    caller_session_id: &str,
) -> anyhow::Result<AttachWindow> {
    let shell = attach_window_shell(&resolved.command.shell);
    let output = tmux_output([
        "new-window",
        "-d",
        "-P",
        "-F",
        "#{window_id}\t#{pane_id}",
        "-t",
        caller_session_id,
        "-n",
        ATTACH_WINDOW_NAME,
        shell.as_str(),
    ])?;
    let visit = parse_attach_window(&output)?;

    if let Err(err) = prepare_attach_window_for_release(&visit, resolved) {
        let _ = kill_window(&visit.window_id);
        return Err(err);
    }
    Ok(visit)
}

fn prepare_attach_window_for_release(
    visit: &AttachWindow,
    resolved: &AttachResolveRecord,
) -> anyhow::Result<()> {
    set_visit_pane_remain_on_exit(&visit.pane_id)?;
    tag_attach_window(&visit.window_id, resolved)?;
    set_window_option(&visit.window_id, ATTACH_READY_TAG, "true")
}

fn set_visit_pane_remain_on_exit(pane_id: &str) -> anyhow::Result<()> {
    tmux_status(["set-option", "-p", "-t", pane_id, "remain-on-exit", "on"])
}

fn parse_attach_window(output: &str) -> anyhow::Result<AttachWindow> {
    let line = output.trim();
    let Some((window_id, pane_id)) = line.split_once('\t') else {
        bail!("tmux new-window did not return window and pane ids");
    };
    let window_id = window_id.trim();
    let pane_id = pane_id.trim();
    if window_id.is_empty() || pane_id.is_empty() {
        bail!("tmux new-window returned an empty window or pane id");
    }
    Ok(AttachWindow {
        window_id: window_id.to_string(),
        pane_id: pane_id.to_string(),
    })
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
        "ready_tries=600; while [ \"$ready_tries\" -gt 0 ]; do if [ \"$(tmux display-message -p -t \"$TMUX_PANE\" '#{{{ATTACH_READY_TAG}}}' 2>/dev/null)\" = \"true\" ]; then break; fi; ready_tries=$((ready_tries - 1)); sleep 0.05; done; if [ \"$ready_tries\" -le 0 ]; then tmux kill-window -t \"$TMUX_PANE\" >/dev/null 2>&1 || true; exit 124; fi; status=0; {{ {attach_shell}; }} || status=$?; if [ \"$status\" -eq 0 ]; then tmux kill-window -t \"$TMUX_PANE\" >/dev/null 2>&1 || true; fi; exit \"$status\""
    )
}

async fn wait_for_visit_pane_exit_and_reap(
    visit: &AttachWindow,
    target: &str,
) -> anyhow::Result<()> {
    loop {
        sleep(Duration::from_millis(RETURN_POLL_MS)).await;
        match visit_pane_state(&visit.pane_id)? {
            VisitPaneState::Running => {}
            VisitPaneState::Gone => return Ok(()),
            VisitPaneState::Dead { exit_status } => {
                let pane_output = capture_visit_pane_output(&visit.pane_id);
                return reap_dead_visit(visit, target, exit_status, &pane_output, kill_window);
            }
        }
    }
}

fn visit_pane_state(pane_id: &str) -> anyhow::Result<VisitPaneState> {
    match tmux_output([
        "display-message",
        "-p",
        "-t",
        pane_id,
        "#{pane_dead}\t#{pane_dead_status}",
    ]) {
        Ok(output) => parse_visit_pane_state(&output),
        Err(err) if tmux_error_contains(&err, "can't find pane") => Ok(VisitPaneState::Gone),
        Err(err) if tmux_error_contains(&err, "can't find window") => Ok(VisitPaneState::Gone),
        Err(err) => Err(err),
    }
}

fn parse_visit_pane_state(output: &str) -> anyhow::Result<VisitPaneState> {
    let line = output.trim_end();
    let (dead, status) = line.split_once('\t').unwrap_or((line, ""));
    match dead.trim() {
        "0" => Ok(VisitPaneState::Running),
        "1" => {
            let status = status.trim();
            let exit_status = if status.is_empty() {
                None
            } else {
                Some(
                    status
                        .parse::<i32>()
                        .with_context(|| format!("invalid tmux pane_dead_status: {status}"))?,
                )
            };
            Ok(VisitPaneState::Dead { exit_status })
        }
        value => bail!("tmux pane_dead returned unexpected value: {value}"),
    }
}

fn capture_visit_pane_output(pane_id: &str) -> String {
    tmux_output([
        "capture-pane",
        "-p",
        "-t",
        pane_id,
        "-S",
        ATTACH_FAILURE_CAPTURE_LINES,
    ])
    .unwrap_or_else(|err| format!("<failed to capture pane output: {err}>"))
}

fn reap_dead_visit(
    visit: &AttachWindow,
    target: &str,
    exit_status: Option<i32>,
    pane_output: &str,
    kill: impl FnOnce(&str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    kill(&visit.window_id)?;
    if exit_status == Some(0) {
        return Ok(());
    }
    Err(attach_failure_error(target, exit_status, pane_output))
}

fn attach_failure_error(
    target: &str,
    exit_status: Option<i32>,
    pane_output: &str,
) -> anyhow::Error {
    let status = exit_status
        .map(|status| status.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let pane_output = pane_output.trim_end();
    if pane_output.is_empty() {
        anyhow!("attach {target} failed in visit window with exit status {status}; no pane output captured")
    } else {
        anyhow!(
            "attach {target} failed in visit window with exit status {status}; pane output:\n{pane_output}"
        )
    }
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
    match tmux_status(["kill-window", "-t", window_id]) {
        Ok(()) => Ok(()),
        Err(err) if tmux_error_contains(&err, "can't find window") => Ok(()),
        Err(err) => Err(err),
    }
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

fn tmux_error_contains(err: &anyhow::Error, needle: &str) -> bool {
    err.to_string().contains(needle)
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
    fn attach_window_shell_waits_for_ready_tag_and_self_kills_success_only() {
        let shell = attach_window_shell("ssh -t host 'tmux attach-session -t '''$1''''");

        assert!(shell.contains("#{@mstream/attach-ready}"));
        assert!(shell.contains("sleep 0.05"));
        assert!(shell.contains("exit 124"));
        assert!(shell.contains("ssh -t host"));
        assert!(shell.contains(
            "} || status=$?; if [ \"$status\" -eq 0 ]; then tmux kill-window -t \"$TMUX_PANE\""
        ));
        assert!(shell.contains("exit \"$status\""));
    }

    #[test]
    fn parse_visit_pane_state_reads_running_dead_and_status() {
        assert_eq!(
            parse_visit_pane_state("0\t\n").unwrap(),
            VisitPaneState::Running
        );
        assert_eq!(
            parse_visit_pane_state("1\t255\n").unwrap(),
            VisitPaneState::Dead {
                exit_status: Some(255)
            }
        );
        assert_eq!(
            parse_visit_pane_state("1\t\n").unwrap(),
            VisitPaneState::Dead { exit_status: None }
        );
        assert!(parse_visit_pane_state("unexpected\t0\n")
            .unwrap_err()
            .to_string()
            .contains("unexpected value"));
    }

    #[test]
    fn failed_visit_is_reaped_and_reports_captured_output() {
        let visit = AttachWindow {
            window_id: "@7".to_string(),
            pane_id: "%13".to_string(),
        };
        let killed = std::cell::RefCell::new(Vec::new());

        let err = reap_dead_visit(
            &visit,
            "local::$203",
            Some(255),
            "can't find session: $203\n",
            |window_id| {
                killed.borrow_mut().push(window_id.to_string());
                Ok(())
            },
        )
        .expect_err("failed attach should return a clear error");

        assert_eq!(killed.into_inner(), vec!["@7"]);
        let message = err.to_string();
        assert!(message.contains("local::$203"));
        assert!(message.contains("exit status 255"));
        assert!(message.contains("can't find session"));
    }

    #[test]
    fn inactive_attach_windows_selects_only_tagged_inactive_windows() {
        let windows = inactive_attach_windows("@1\t1\ttrue\n@2\t0\ttrue\n@3\t0\t\n@4\t0\t0\n");

        assert_eq!(windows, vec!["@2"]);
    }

    #[test]
    fn attach_here_sweeps_stale_windows_before_creating_new_visit() {
        let events = std::cell::RefCell::new(Vec::new());
        let resolved = AttachResolveRecord {
            record_type: "ok".to_string(),
            op: "attach_resolve".to_string(),
            target: "local::$1".to_string(),
            command: AttachCommandRecord {
                program: "tmux".to_string(),
                args: vec![
                    "attach-session".to_string(),
                    "-t".to_string(),
                    "$1".to_string(),
                ],
                shell: "tmux attach-session -t '$1'".to_string(),
            },
        };

        let (visit, client_ttys) = prepare_attach_here_visit(
            &resolved,
            || {
                events.borrow_mut().push("sweep");
                Ok(1)
            },
            || {
                events.borrow_mut().push("session");
                Ok("$2".to_string())
            },
            |session_id| {
                events.borrow_mut().push("clients");
                assert_eq!(session_id, "$2");
                Ok(vec!["/dev/pts/4".to_string()])
            },
            |record, session_id| {
                events.borrow_mut().push("create");
                assert_eq!(record.target, "local::$1");
                assert_eq!(session_id, "$2");
                Ok(AttachWindow {
                    window_id: "@7".to_string(),
                    pane_id: "%13".to_string(),
                })
            },
        )
        .expect("attach here setup succeeds");

        assert_eq!(
            events.into_inner(),
            vec!["sweep", "session", "clients", "create"]
        );
        assert_eq!(visit.window_id, "@7");
        assert_eq!(client_ttys, vec!["/dev/pts/4"]);
    }

    #[test]
    fn attached_client_ttys_selects_real_clients_in_caller_session() {
        let clients = attached_client_ttys_for_session(
            "/dev/pts/4\t$2\n\t$2\n/dev/pts/38\t$2\n/dev/pts/9\t$3\n/dev/pts/4\t$2\n",
            "$2",
        );

        assert_eq!(clients, vec!["/dev/pts/4", "/dev/pts/38"]);
    }

    #[test]
    fn parse_attach_window_requires_window_and_pane_ids() {
        let visit = parse_attach_window("@7\t%13\n").expect("attach window ids parse");

        assert_eq!(
            visit,
            AttachWindow {
                window_id: "@7".to_string(),
                pane_id: "%13".to_string(),
            }
        );
        assert!(parse_attach_window("@7\n").is_err());
        assert!(parse_attach_window("@7\t\n").is_err());
    }

    #[test]
    fn tmux_error_contains_matches_missing_window_race() {
        let err = anyhow!("tmux kill-window -t @262 failed: can't find window: @262");

        assert!(tmux_error_contains(&err, "can't find window"));
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
