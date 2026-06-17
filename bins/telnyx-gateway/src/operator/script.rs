use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use motlie_driver::{CommandEffect, CommandEngine, CommandOutput};

use crate::operator::commands::{GatewayCommand, GatewayContext};

pub async fn run_operator_line(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
    line: &str,
) -> anyhow::Result<CommandOutput> {
    if let Some(path) = parse_repl_file_command(line)? {
        return run_repl_file(engine, &path).await;
    }

    run_non_repl_operator_line(engine, line).await
}

pub async fn run_repl_file(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
    path: &Path,
) -> anyhow::Result<CommandOutput> {
    let path = expand_user_path(path);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read repl file {}", path.display()))?;
    let mut output = CommandOutput::default();

    for (index, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        output
            .lines
            .push(format!("{}:{} > {}", path.display(), index + 1, trimmed));
        let command_output = run_non_repl_operator_line(engine, trimmed)
            .await
            .map_err(|error| anyhow!("{}:{}: {error}", path.display(), index + 1))?;
        let should_exit = command_output.effects.contains(&CommandEffect::ExitShell);
        output.lines.extend(command_output.lines);
        output.effects.extend(command_output.effects);
        if should_exit {
            break;
        }
    }

    Ok(output)
}

async fn run_non_repl_operator_line(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
    line: &str,
) -> anyhow::Result<CommandOutput> {
    let argv = shlex::split(line).ok_or_else(|| anyhow!("invalid shell quoting"))?;
    if argv.first().is_some_and(|command| command == "quit") {
        return run_quit_command(engine, &argv).await;
    }

    engine
        .run_argv(&argv)
        .await
        .map_err(|error| anyhow!("{error}"))
}

async fn run_quit_command(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
    argv: &[String],
) -> anyhow::Result<CommandOutput> {
    if argv.len() > 2 {
        bail!("usage: quit [dump_path]");
    }

    let mut shutdown_argv = vec!["shutdown".to_string()];
    if let Some(path) = argv.get(1) {
        shutdown_argv.push(path.clone());
    }
    let mut output = engine
        .run_argv(&shutdown_argv)
        .await
        .map_err(|error| anyhow!("{error}"))?;
    for line in &mut output.lines {
        if line == "shutdown requested" {
            *line = "quit requested".to_string();
        }
    }
    Ok(output)
}

pub fn parse_repl_file_command(line: &str) -> anyhow::Result<Option<PathBuf>> {
    let argv = shlex::split(line).ok_or_else(|| anyhow!("invalid shell quoting"))?;
    let Some(command) = argv.first() else {
        return Ok(None);
    };
    if command != "source" {
        return Ok(None);
    }
    if argv.len() != 2 {
        bail!("usage: {command} <path>");
    }
    Ok(argv.get(1).map(PathBuf::from))
}

pub fn expand_user_path(path: &Path) -> PathBuf {
    let Some(raw) = path.to_str() else {
        return path.to_path_buf();
    };
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return path.to_path_buf();
    };

    if raw == "~" || raw == "$HOME" {
        return home;
    }
    raw.strip_prefix("~/")
        .or_else(|| raw.strip_prefix("$HOME/"))
        .map(|suffix| home.join(suffix))
        .unwrap_or_else(|| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call_control::TelnyxClient;
    use crate::operator::state::shared_state;

    #[test]
    fn parse_repl_file_command_accepts_source_only() {
        assert_eq!(
            parse_repl_file_command("source ~/telnyx/config.repl").expect("parse source command"),
            Some(PathBuf::from("~/telnyx/config.repl"))
        );
        assert!(parse_repl_file_command("load /tmp/config.repl")
            .expect("parse load command")
            .is_none());
        assert!(parse_repl_file_command("status")
            .expect("parse status command")
            .is_none());
    }

    #[test]
    fn expand_user_path_accepts_home_env_prefix() {
        let expanded = expand_user_path(Path::new("$HOME/artifacts/hf-cache"));
        assert!(expanded.ends_with("artifacts/hf-cache"));
        assert!(!expanded.to_string_lossy().contains("$HOME"));
    }

    #[tokio::test]
    async fn run_repl_file_replays_commands() {
        let path = std::env::temp_dir().join(format!(
            "motlie-telnyx-gateway-test-{}.repl",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            "# ignored\ninbound enable --manual\nasr use kroko-2025\n",
        )
        .expect("write replay file");

        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = run_repl_file(&mut engine, &path)
            .await
            .expect("run replay file");

        assert!(output
            .lines
            .iter()
            .any(|line| line.ends_with("inbound enable --manual")));
        assert!(output
            .lines
            .iter()
            .any(|line| line == "inbound enabled manual"));
        assert!(output
            .lines
            .iter()
            .any(|line| line.starts_with("asr backend for next calls: kroko-2025")));
        assert_eq!(
            engine.context().session.next_asr_backend,
            crate::adapter::LiveAsrBackend::Kroko2025
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn quit_requests_gateway_shutdown() {
        let state = shared_state("127.0.0.1:0".parse().expect("valid addr"));
        let telnyx = TelnyxClient::new("https://api.example.test".to_string(), None, true);
        let context = GatewayContext::new(state.clone(), telnyx);
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);

        let output = run_operator_line(&mut engine, "quit")
            .await
            .expect("quit should shutdown gateway");

        assert_eq!(output.lines, vec!["quit requested"]);
        assert!(output.effects.contains(&CommandEffect::ExitShell));
        assert!(state.read().await.shutdown_requested);
    }
}
