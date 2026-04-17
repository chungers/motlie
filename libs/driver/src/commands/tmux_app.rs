use std::collections::BTreeMap;

use async_trait::async_trait;
use clap::{Arg, Command};

use crate::commands::tmux::{
    execute_tmux_command, tmux_complete, tmux_help, TargetsCommand, TmuxCommand,
    TmuxCompletionContext, TmuxFrontendState, TmuxMirrorSnapshot, TmuxState,
};
use crate::completion::{CompletionCandidate, CompletionRequest};
use crate::engine::{CommandOutput, CommandSet};
use crate::error::{DriverError, DriverResult};
use crate::naming::{validate_qualified_name, ResolveName, ResolvedName};

#[derive(Default)]
pub struct TmuxAppState {
    connections: BTreeMap<String, TmuxState>,
    current: Option<String>,
}

impl TmuxAppState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn current_alias(&self) -> Option<&str> {
        self.current.as_deref()
    }

    pub fn connection_aliases(&self) -> impl Iterator<Item = &str> {
        self.connections.keys().map(String::as_str)
    }

    pub fn connections(&self) -> &BTreeMap<String, TmuxState> {
        &self.connections
    }

    pub async fn connect_alias(&mut self, uri: &str, alias: &str) -> DriverResult<()> {
        if self.connections.contains_key(alias) {
            return Err(DriverError::invalid_argument(
                "alias",
                format!("'{alias}' is already connected"),
            ));
        }

        let state = TmuxState::connect(uri).await?;
        self.connections.insert(alias.to_string(), state);
        Ok(())
    }

    pub async fn disconnect_alias(&mut self, alias: &str) -> DriverResult<()> {
        let Some(mut state) = self.connections.remove(alias) else {
            return Err(DriverError::unknown_scope(alias));
        };

        state.shutdown_managed_state().await?;
        if self.current.as_deref() == Some(alias) {
            self.current = None;
        }
        Ok(())
    }

    pub fn set_current(&mut self, alias: &str) -> DriverResult<()> {
        if !self.connections.contains_key(alias) {
            return Err(DriverError::unknown_scope(alias));
        }
        self.current = Some(alias.to_string());
        Ok(())
    }

    fn connection_mut(&mut self, alias: &str) -> DriverResult<&mut TmuxState> {
        self.connections
            .get_mut(alias)
            .ok_or_else(|| DriverError::unknown_scope(alias))
    }

    fn resolve_scope(&self, raw: &str) -> DriverResult<ResolvedName> {
        self.resolve_name(TmuxNameKind::ScopedEntity, raw)
    }

    fn resolve_command_scope(&self) -> DriverResult<String> {
        self.current.clone().ok_or(DriverError::MissingCurrentScope)
    }

    fn targets_completion_contexts(&self) -> BTreeMap<String, TmuxCompletionContext> {
        self.connections
            .iter()
            .map(|(alias, state)| (alias.clone(), TmuxCommand::completion_context(state)))
            .collect()
    }
}

#[async_trait]
impl TmuxFrontendState for TmuxAppState {
    fn frontend_host_uri(&self) -> String {
        match self.current.as_deref() {
            Some(alias) => match self.connections.get(alias) {
                Some(state) => format!("{alias} ({})", state.host_uri),
                None => "(current connection missing)".to_string(),
            },
            None => "(no current connection selected)".to_string(),
        }
    }

    fn mirror_snapshot(&self) -> TmuxMirrorSnapshot {
        match self.current.as_deref() {
            Some(alias) => self
                .connections
                .get(alias)
                .map(TmuxState::mirror_snapshot)
                .unwrap_or_else(|| TmuxMirrorSnapshot {
                    text: String::new(),
                    label: format!("{alias}: unavailable"),
                    ansi: false,
                    watch_health: None,
                }),
            None => TmuxMirrorSnapshot {
                text: String::new(),
                label: "no current connection selected".to_string(),
                ansi: false,
                watch_health: None,
            },
        }
    }

    fn mirror_history_page(
        &self,
        after: Option<u64>,
        limit: usize,
    ) -> crate::HistoryPage<crate::commands::tmux::TmuxHistoryEntry> {
        match self.current.as_deref() {
            Some(alias) => self
                .connections
                .get(alias)
                .map(|state| state.mirror_history_page(after, limit))
                .unwrap_or(crate::HistoryPage {
                    items: Vec::new(),
                    next_after: None,
                    oldest_available: None,
                    newest_available: None,
                }),
            None => crate::HistoryPage {
                items: Vec::new(),
                next_after: None,
                oldest_available: None,
                newest_available: None,
            },
        }
    }

    fn has_live_follow(&self) -> bool {
        self.current
            .as_deref()
            .and_then(|alias| self.connections.get(alias))
            .is_some_and(TmuxState::has_live_follow)
    }

    async fn refresh_mirror(&mut self) -> DriverResult<()> {
        if let Some(alias) = self.current.clone() {
            self.connection_mut(&alias)?.refresh_mirror().await?;
        }
        Ok(())
    }

    async fn shutdown_managed_state(&mut self) -> DriverResult<()> {
        if let Some(alias) = self.current.clone() {
            self.connection_mut(&alias)?
                .shutdown_managed_state()
                .await?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TmuxNameKind {
    ScopedEntity,
}

impl ResolveName<TmuxNameKind> for TmuxAppState {
    type Resolved = ResolvedName;

    fn resolve_name(&self, _kind: TmuxNameKind, raw: &str) -> DriverResult<Self::Resolved> {
        let qualified = validate_qualified_name(raw)?;
        let scope = match qualified.scope {
            Some(scope) => {
                if !self.connections.contains_key(scope) {
                    return Err(DriverError::unknown_scope(scope));
                }
                scope.to_string()
            }
            None => self
                .current
                .clone()
                .ok_or(DriverError::MissingCurrentScope)?,
        };

        Ok(ResolvedName::new(scope, qualified.value))
    }
}

#[derive(Debug, Clone)]
pub struct ConnectCommand {
    pub uri: String,
    pub alias: String,
}

#[derive(Debug, Clone)]
pub struct DisconnectCommand {
    pub alias: String,
}

#[derive(Debug, Clone)]
pub struct UseCommand {
    pub alias: String,
}

pub enum TmuxAppCommand {
    Connect(ConnectCommand),
    Disconnect(DisconnectCommand),
    Use(UseCommand),
    Connections,
    Tmux(TmuxCommand),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTmuxScope {
    pub alias: String,
}

pub enum TmuxAppResolved {
    Connect(ConnectCommand),
    Disconnect {
        alias: String,
    },
    Use {
        alias: String,
    },
    Connections,
    TargetsAll,
    Tmux {
        scope: ResolvedTmuxScope,
        command: TmuxCommand,
    },
}

#[derive(Debug, Clone, Default)]
pub struct TmuxAppCompletionContext {
    pub aliases: Vec<String>,
    pub current: Option<String>,
    pub per_scope: BTreeMap<String, TmuxCompletionContext>,
}

#[async_trait]
impl CommandSet<TmuxAppState> for TmuxAppCommand {
    type CompletionContext = TmuxAppCompletionContext;
    type Resolved = TmuxAppResolved;

    fn root_command() -> Command {
        let mut root = Command::new("tmux")
            .subcommand(connect_subcommand())
            .subcommand(disconnect_subcommand())
            .subcommand(use_subcommand())
            .subcommand(connections_subcommand());

        for subcommand in TmuxCommand::root_command().get_subcommands() {
            root = root.subcommand(subcommand.clone());
        }

        root
    }

    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self> {
        let Some((name, submatches)) = matches.subcommand() else {
            return Err(DriverError::message("missing tmux command"));
        };

        match name {
            "connect" => Ok(Self::Connect(ConnectCommand {
                uri: required_string(submatches, "uri")?,
                alias: required_string(submatches, "alias")?,
            })),
            "disconnect" => Ok(Self::Disconnect(DisconnectCommand {
                alias: required_string(submatches, "alias")?,
            })),
            "use" => Ok(Self::Use(UseCommand {
                alias: required_string(submatches, "alias")?,
            })),
            "connections" => Ok(Self::Connections),
            _ => Ok(Self::Tmux(
                <TmuxCommand as CommandSet<TmuxState>>::from_matches(matches)?,
            )),
        }
    }

    fn completion_context(context: &TmuxAppState) -> Self::CompletionContext {
        let mut aliases = context
            .connection_aliases()
            .map(str::to_string)
            .collect::<Vec<_>>();
        aliases.sort();

        Self::CompletionContext {
            aliases,
            current: context.current.clone(),
            per_scope: context.targets_completion_contexts(),
        }
    }

    fn help(topic: &[String]) -> Option<String> {
        match topic {
            [name] if name == "connect" => Some(connect_help()),
            [name] if name == "disconnect" => Some(disconnect_help()),
            [name] if name == "use" => Some(use_help()),
            [name] if name == "connections" => Some(connections_help()),
            _ => tmux_help(topic),
        }
    }

    fn complete(
        request: CompletionRequest<'_>,
        context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> {
        match (request.command_path, request.arg_id) {
            (["disconnect"], Some("alias")) | (["use"], Some("alias")) => context
                .aliases
                .iter()
                .filter(|alias| alias.starts_with(request.prefix))
                .cloned()
                .map(CompletionCandidate::new)
                .collect(),
            _ => complete_tmux_app_scoped(request, context),
        }
    }

    fn resolve_command(self, context: &TmuxAppState) -> DriverResult<Self::Resolved> {
        match self {
            Self::Connect(cmd) => Ok(TmuxAppResolved::Connect(cmd)),
            Self::Disconnect(cmd) => {
                if !context.connections.contains_key(&cmd.alias) {
                    return Err(DriverError::unknown_scope(cmd.alias));
                }
                Ok(TmuxAppResolved::Disconnect { alias: cmd.alias })
            }
            Self::Use(cmd) => {
                if !context.connections.contains_key(&cmd.alias) {
                    return Err(DriverError::unknown_scope(cmd.alias));
                }
                Ok(TmuxAppResolved::Use { alias: cmd.alias })
            }
            Self::Connections => Ok(TmuxAppResolved::Connections),
            Self::Tmux(command) => resolve_tmux_command(context, command),
        }
    }

    async fn execute(
        resolved: Self::Resolved,
        context: &mut TmuxAppState,
    ) -> DriverResult<CommandOutput> {
        match resolved {
            TmuxAppResolved::Connect(cmd) => execute_connect(context, cmd).await,
            TmuxAppResolved::Disconnect { alias } => execute_disconnect(context, &alias).await,
            TmuxAppResolved::Use { alias } => execute_use(context, &alias),
            TmuxAppResolved::Connections => execute_connections(context),
            TmuxAppResolved::TargetsAll => execute_targets_all(context).await,
            TmuxAppResolved::Tmux { scope, command } => {
                let state = context.connection_mut(&scope.alias)?;
                execute_tmux_command(state, command).await
            }
        }
    }
}

fn connect_subcommand() -> Command {
    Command::new("connect")
        .about("Connect to an SSH host and register it under an alias")
        .arg(Arg::new("uri").required(true))
        .arg(
            Arg::new("as_kw")
                .required(true)
                .value_parser(["as"])
                .hide(true),
        )
        .arg(Arg::new("alias").required(true))
}

fn disconnect_subcommand() -> Command {
    Command::new("disconnect")
        .about("Disconnect a previously registered alias")
        .arg(Arg::new("alias").required(true))
}

fn use_subcommand() -> Command {
    Command::new("use")
        .about("Select the current tmux connection alias")
        .arg(Arg::new("alias").required(true))
}

fn connections_subcommand() -> Command {
    Command::new("connections").about("List registered tmux connection aliases")
}

fn required_string(matches: &clap::ArgMatches, name: &str) -> DriverResult<String> {
    matches
        .get_one::<String>(name)
        .cloned()
        .ok_or_else(|| DriverError::message(format!("missing required argument '{name}'")))
}

fn connect_help() -> String {
    [
        "connect <ssh-uri> as <alias>",
        "",
        "Connect to a tmux host and register it under an alias for namespaced commands.",
        "",
        "Example:",
        "  connect ssh://dchung@motliehost?identity-file=/home/dchung/.ssh/motliehost as prod",
    ]
    .join("\n")
}

fn disconnect_help() -> String {
    [
        "disconnect <alias>",
        "",
        "Shutdown any driver-managed monitor/stream state for the alias and remove it from the session.",
    ]
    .join("\n")
}

fn use_help() -> String {
    [
        "use <alias>",
        "",
        "Select the current tmux connection. Bare tmux targets/sessions resolve against this alias.",
    ]
    .join("\n")
}

fn connections_help() -> String {
    [
        "connections",
        "",
        "List the registered aliases in this tmux driver session.",
    ]
    .join("\n")
}

fn resolve_tmux_command(
    context: &TmuxAppState,
    command: TmuxCommand,
) -> DriverResult<TmuxAppResolved> {
    match command {
        TmuxCommand::Create(cmd) => {
            let alias = context.resolve_command_scope()?;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope { alias },
                command: TmuxCommand::Create(cmd),
            })
        }
        TmuxCommand::NewWindow(mut cmd) => {
            let resolved = context.resolve_scope(&cmd.session)?;
            cmd.session = resolved.value;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved.scope,
                },
                command: TmuxCommand::NewWindow(cmd),
            })
        }
        TmuxCommand::SplitPane(mut cmd) => {
            let resolved = context.resolve_scope(&cmd.target)?;
            cmd.target = resolved.value;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved.scope,
                },
                command: TmuxCommand::SplitPane(cmd),
            })
        }
        TmuxCommand::Kill(mut cmd) => {
            let resolved = context.resolve_scope(&cmd.target)?;
            cmd.target = resolved.value;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved.scope,
                },
                command: TmuxCommand::Kill(cmd),
            })
        }
        TmuxCommand::Mirror(cmd) => {
            let alias = context.resolve_command_scope()?;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope { alias },
                command: TmuxCommand::Mirror(cmd),
            })
        }
        TmuxCommand::Tui(cmd) => {
            let alias = context.resolve_command_scope()?;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope { alias },
                command: TmuxCommand::Tui(cmd),
            })
        }
        TmuxCommand::Targets(cmd) => {
            if let Some(alias) = context.current_alias() {
                Ok(TmuxAppResolved::Tmux {
                    scope: ResolvedTmuxScope {
                        alias: alias.to_string(),
                    },
                    command: TmuxCommand::Targets(cmd),
                })
            } else {
                Ok(TmuxAppResolved::TargetsAll)
            }
        }
        TmuxCommand::Send(mut cmd) => {
            let resolved = context.resolve_scope(&cmd.target)?;
            cmd.target = resolved.value;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved.scope,
                },
                command: TmuxCommand::Send(cmd),
            })
        }
        TmuxCommand::Keys(mut cmd) => {
            let resolved = context.resolve_scope(&cmd.target)?;
            cmd.target = resolved.value;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved.scope,
                },
                command: TmuxCommand::Keys(cmd),
            })
        }
        TmuxCommand::Capture(mut cmd) => {
            let resolved = context.resolve_scope(&cmd.target)?;
            cmd.target = resolved.value;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved.scope,
                },
                command: TmuxCommand::Capture(cmd),
            })
        }
        TmuxCommand::Monitor(mut cmd) => match cmd.action {
            crate::commands::tmux::MonitorActionCommand::Start(mut start) => {
                let resolved = context.resolve_scope(&start.session)?;
                start.session = resolved.value;
                cmd.action = crate::commands::tmux::MonitorActionCommand::Start(start);
                Ok(TmuxAppResolved::Tmux {
                    scope: ResolvedTmuxScope {
                        alias: resolved.scope,
                    },
                    command: TmuxCommand::Monitor(cmd),
                })
            }
            crate::commands::tmux::MonitorActionCommand::Stop => {
                let alias = context.resolve_command_scope()?;
                Ok(TmuxAppResolved::Tmux {
                    scope: ResolvedTmuxScope { alias },
                    command: TmuxCommand::Monitor(cmd),
                })
            }
        },
        TmuxCommand::History(mut cmd) => {
            let mut resolved_scope = None;
            let mut sessions = Vec::with_capacity(cmd.sessions.len());
            for session in &cmd.sessions {
                let resolved = context.resolve_scope(session)?;
                if let Some(scope) = resolved_scope.as_ref() {
                    if scope != &resolved.scope {
                        return Err(DriverError::invalid_argument(
                            "sessions",
                            "all sessions in one history command must belong to the same alias",
                        ));
                    }
                } else {
                    resolved_scope = Some(resolved.scope.clone());
                }
                sessions.push(resolved.value);
            }
            cmd.sessions = sessions;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved_scope.ok_or(DriverError::MissingCurrentScope)?,
                },
                command: TmuxCommand::History(cmd),
            })
        }
        TmuxCommand::Stream(mut cmd) => {
            let resolved = context.resolve_scope(&cmd.target)?;
            cmd.target = resolved.value;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope {
                    alias: resolved.scope,
                },
                command: TmuxCommand::Stream(cmd),
            })
        }
        TmuxCommand::Upload(cmd) => {
            let alias = context.resolve_command_scope()?;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope { alias },
                command: TmuxCommand::Upload(cmd),
            })
        }
        TmuxCommand::Download(cmd) => {
            let alias = context.resolve_command_scope()?;
            Ok(TmuxAppResolved::Tmux {
                scope: ResolvedTmuxScope { alias },
                command: TmuxCommand::Download(cmd),
            })
        }
    }
}

async fn execute_connect(
    context: &mut TmuxAppState,
    cmd: ConnectCommand,
) -> DriverResult<CommandOutput> {
    context.connect_alias(&cmd.uri, &cmd.alias).await?;
    Ok(CommandOutput::line(format!(
        "Connected {} as {}",
        cmd.uri, cmd.alias
    )))
}

async fn execute_disconnect(
    context: &mut TmuxAppState,
    alias: &str,
) -> DriverResult<CommandOutput> {
    context.disconnect_alias(alias).await?;
    Ok(CommandOutput::line(format!("Disconnected {alias}")))
}

fn execute_use(context: &mut TmuxAppState, alias: &str) -> DriverResult<CommandOutput> {
    context.set_current(alias)?;
    Ok(CommandOutput::line(format!("Current connection: {alias}")))
}

fn execute_connections(context: &TmuxAppState) -> DriverResult<CommandOutput> {
    if context.connections.is_empty() {
        return Ok(CommandOutput::line("No connected tmux hosts"));
    }

    let mut lines = Vec::new();
    for (alias, state) in &context.connections {
        let marker = if context.current.as_deref() == Some(alias.as_str()) {
            "*"
        } else {
            " "
        };
        lines.push(format!("{marker} {alias} -> {}", state.host_uri));
    }

    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

async fn execute_targets_all(context: &mut TmuxAppState) -> DriverResult<CommandOutput> {
    if context.connections.is_empty() {
        return Ok(CommandOutput::line("No connected tmux hosts"));
    }

    let mut lines = Vec::new();
    for (alias, state) in &mut context.connections {
        lines.push(format!("[{alias}] {}", state.host_uri));
        let output = execute_tmux_command(state, TmuxCommand::Targets(TargetsCommand)).await?;
        for line in output.lines {
            lines.push(line);
        }
    }

    Ok(CommandOutput {
        lines,
        effects: Vec::new(),
    })
}

fn complete_tmux_app_scoped(
    request: CompletionRequest<'_>,
    context: &TmuxAppCompletionContext,
) -> Vec<CompletionCandidate> {
    if !is_scoped_tmux_request(request.command_path, request.arg_id) {
        return Vec::new();
    }

    if let Some((alias_prefix, inner_prefix)) = request.prefix.split_once('/') {
        let mut out = Vec::new();
        for alias in &context.aliases {
            if !alias.starts_with(alias_prefix) {
                continue;
            }
            if let Some(scope_context) = context.per_scope.get(alias) {
                for candidate in tmux_complete(
                    CompletionRequest {
                        command_path: request.command_path,
                        arg_id: request.arg_id,
                        prefix: inner_prefix,
                    },
                    scope_context,
                ) {
                    out.push(CompletionCandidate {
                        value: format!("{alias}/{}", candidate.value),
                        help: candidate.help,
                    });
                }
            }
        }
        return out;
    }

    if let Some(current) = context.current.as_deref() {
        if let Some(scope_context) = context.per_scope.get(current) {
            return tmux_complete(request, scope_context);
        }
    }

    let mut out = Vec::new();
    for alias in &context.aliases {
        if let Some(scope_context) = context.per_scope.get(alias) {
            for candidate in tmux_complete(
                CompletionRequest {
                    command_path: request.command_path,
                    arg_id: request.arg_id,
                    prefix: "",
                },
                scope_context,
            ) {
                let qualified = format!("{alias}/{}", candidate.value);
                if qualified.starts_with(request.prefix) {
                    out.push(CompletionCandidate {
                        value: qualified,
                        help: candidate.help,
                    });
                }
            }
        }
    }
    out
}

fn is_scoped_tmux_request(command_path: &[&str], arg_id: Option<&str>) -> bool {
    matches!(
        (command_path, arg_id),
        (["new-window"], Some("session"))
            | (["monitor", "start"], Some("session"))
            | (["history"], Some("sessions"))
            | (["split-pane"], Some("target"))
            | (["kill"], Some("target"))
            | (["send"], Some("target"))
            | (["keys"], Some("target"))
            | (["capture"], Some("target"))
            | (["stream"], Some("target"))
    )
}

#[cfg(test)]
mod tests {
    use super::{TmuxAppCommand, TmuxAppState};
    use crate::engine::CommandEngine;
    use crate::error::DriverError;

    use motlie_tmux::HostHandle;

    fn test_state() -> TmuxAppState {
        let mut state = TmuxAppState::new();
        let mut alpha = crate::commands::tmux::TmuxState::new("ssh://alpha", HostHandle::local());
        alpha.known_sessions = vec!["demo".to_string()];
        alpha.known_targets = vec!["demo".to_string(), "demo:0.0".to_string()];
        let mut beta = crate::commands::tmux::TmuxState::new("ssh://beta", HostHandle::local());
        beta.known_sessions = vec!["build".to_string()];
        beta.known_targets = vec!["build".to_string(), "build:0.0".to_string()];
        state.connections.insert("alpha".to_string(), alpha);
        state.connections.insert("beta".to_string(), beta);
        state
    }

    #[tokio::test]
    async fn multi_host_requires_current_scope_for_bare_tmux_command() {
        let state = test_state();
        let mut engine = CommandEngine::<TmuxAppState, TmuxAppCommand>::new(state);

        let error = engine
            .run_line("create demo")
            .await
            .expect_err("current scope required");
        assert!(matches!(error, DriverError::MissingCurrentScope));
    }

    #[tokio::test]
    async fn multi_host_use_sets_current_scope() {
        let state = test_state();
        let mut engine = CommandEngine::<TmuxAppState, TmuxAppCommand>::new(state);

        let output = engine.run_line("use alpha").await.expect("use output");
        assert_eq!(output.lines, vec!["Current connection: alpha"]);
        assert_eq!(engine.context().current_alias(), Some("alpha"));
    }

    #[test]
    fn multi_host_completion_qualifies_targets_without_current_scope() {
        let state = test_state();
        let engine = CommandEngine::<TmuxAppState, TmuxAppCommand>::new(state);
        let values = engine
            .complete("send a", "send a".len())
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();

        assert!(values.contains(&"alpha/demo".to_string()));
        assert!(values.contains(&"alpha/demo:0.0".to_string()));
    }

    #[test]
    fn multi_host_completion_uses_current_scope_for_bare_targets() {
        let mut state = test_state();
        state.current = Some("beta".to_string());
        let engine = CommandEngine::<TmuxAppState, TmuxAppCommand>::new(state);
        let values = engine
            .complete("send bu", "send bu".len())
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();

        assert!(values.contains(&"build".to_string()));
        assert!(!values.iter().any(|value| value.starts_with("alpha/")));
    }

    #[test]
    fn multi_host_resolve_rewrites_qualified_target() {
        let state = test_state();
        let resolved = super::resolve_tmux_command(
            &state,
            crate::commands::tmux::TmuxCommand::Send(crate::commands::tmux::SendCommand {
                target: "beta/build:0.0".to_string(),
                text: vec!["echo".to_string(), "hi".to_string()],
            }),
        )
        .expect("qualified target resolves");

        match resolved {
            super::TmuxAppResolved::Tmux { scope, command } => {
                assert_eq!(scope.alias, "beta");
                match command {
                    crate::commands::tmux::TmuxCommand::Send(cmd) => {
                        assert_eq!(cmd.target, "build:0.0");
                        assert_eq!(cmd.text, vec!["echo", "hi"]);
                    }
                    _ => panic!("expected send command"),
                }
            }
            _ => panic!("expected scoped tmux resolution"),
        }
    }

    #[test]
    fn multi_host_targets_without_current_resolve_to_all_connections() {
        let state = test_state();
        let resolved = super::resolve_tmux_command(
            &state,
            crate::commands::tmux::TmuxCommand::Targets(crate::commands::tmux::TargetsCommand),
        )
        .expect("targets resolve");

        assert!(matches!(resolved, super::TmuxAppResolved::TargetsAll));
    }

    #[tokio::test]
    async fn multi_host_connections_lists_aliases() {
        let mut state = test_state();
        state.set_current("beta").expect("known scope");
        let mut engine = CommandEngine::<TmuxAppState, TmuxAppCommand>::new(state);
        let output = engine
            .run_line("connections")
            .await
            .expect("connections output");
        let rendered = output.lines.join("\n");

        assert!(rendered.contains("  alpha -> ssh://alpha"));
        assert!(rendered.contains("* beta -> ssh://beta"));
    }
}
