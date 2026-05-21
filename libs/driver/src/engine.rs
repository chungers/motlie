use async_trait::async_trait;

use crate::clap::{analyze_completion, render_help};
use crate::completion::dedup_sorted;
use crate::completion::{CompletionCandidate, CompletionRequest};
use crate::error::{DriverError, DriverResult};

pub(crate) const BUILTIN_COMMANDS: &[&str] = &["help", "quit", "exit"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandEffect {
    ExitShell,
    EnterTui,
    ExitTui,
}

#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub lines: Vec<String>,
    pub effects: Vec<CommandEffect>,
}

impl CommandOutput {
    pub fn line(line: impl Into<String>) -> Self {
        Self {
            lines: vec![line.into()],
            effects: Vec::new(),
        }
    }

    pub fn text(text: impl AsRef<str>) -> Self {
        Self {
            lines: text.as_ref().lines().map(ToOwned::to_owned).collect(),
            effects: Vec::new(),
        }
    }

    pub fn with_effect(mut self, effect: CommandEffect) -> Self {
        self.effects.push(effect);
        self
    }
}

#[async_trait]
pub trait CommandSet<C>: Sized {
    type CompletionContext: Send + 'static;
    type Resolved;

    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self>;
    fn completion_context(context: &C) -> Self::CompletionContext;
    fn help(_topic: &[String]) -> Option<String> {
        None
    }
    fn complete(
        _request: CompletionRequest<'_>,
        _context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> {
        Vec::new()
    }
    fn resolve_command(self, context: &C) -> DriverResult<Self::Resolved>;
    async fn execute(resolved: Self::Resolved, context: &mut C) -> DriverResult<CommandOutput>;
}

#[derive(Debug)]
pub struct CommandEngine<C, S> {
    context: C,
    _commands: std::marker::PhantomData<S>,
}

impl<C, S> CommandEngine<C, S>
where
    S: CommandSet<C>,
{
    pub fn new(context: C) -> Self {
        Self {
            context,
            _commands: std::marker::PhantomData,
        }
    }

    pub fn context(&self) -> &C {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut C {
        &mut self.context
    }

    pub fn completion_context(&self) -> S::CompletionContext {
        S::completion_context(&self.context)
    }

    pub async fn run_line(&mut self, line: &str) -> DriverResult<CommandOutput> {
        let argv = shlex::split(line).ok_or(DriverError::InvalidShellQuoting)?;
        self.run_argv(&argv).await
    }

    pub async fn run_argv(&mut self, argv: &[String]) -> DriverResult<CommandOutput> {
        let root = S::root_command();

        if let Some(first) = argv.first() {
            if first == "help" {
                let topic = argv.iter().skip(1).cloned().collect::<Vec<_>>();
                if let Some(text) = S::help(&topic) {
                    return Ok(CommandOutput::text(text));
                }
                let text = render_help(&root, &topic)?;
                return Ok(CommandOutput::text(text));
            }

            if first == "quit" || first == "exit" {
                return Ok(CommandOutput::default().with_effect(CommandEffect::ExitShell));
            }
        }

        let argv = if argv
            .first()
            .map(|arg| arg == root.get_name())
            .unwrap_or(false)
        {
            argv.to_vec()
        } else {
            let mut prefixed = Vec::with_capacity(argv.len() + 1);
            prefixed.push(root.get_name().to_string());
            prefixed.extend(argv.iter().cloned());
            prefixed
        };

        let matches = root.try_get_matches_from(argv)?;
        let command = S::from_matches(&matches)?;
        let resolved = command.resolve_command(&self.context)?;
        S::execute(resolved, &mut self.context).await
    }

    pub fn complete(&self, line: &str, cursor: usize) -> Vec<CompletionCandidate> {
        complete_with_context::<C, S>(line, cursor, &self.completion_context())
    }
}

pub(crate) fn complete_with_context<C, S>(
    line: &str,
    cursor: usize,
    completion_context: &S::CompletionContext,
) -> Vec<CompletionCandidate>
where
    S: CommandSet<C>,
{
    let root = S::root_command();
    let completion = analyze_completion(&root, line, cursor);
    let path_refs = completion
        .command_path
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    let mut out = completion.static_candidates;

    if path_refs.is_empty() {
        for builtin in BUILTIN_COMMANDS {
            if builtin.starts_with(&completion.prefix) {
                out.push(CompletionCandidate::new(*builtin));
            }
        }
    }

    out.extend(S::complete(
        CompletionRequest {
            command_path: &path_refs,
            arg_id: completion.arg_id.as_deref(),
            prefix: &completion.prefix,
        },
        completion_context,
    ));

    dedup_sorted(out)
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};

    use super::{CommandEffect, CommandEngine, CommandOutput, CommandSet};
    use crate::completion::{CompletionCandidate, CompletionRequest};
    use crate::error::DriverResult;

    #[derive(Default)]
    struct DemoContext {
        counter: usize,
    }

    #[derive(Parser)]
    struct DemoRoot {
        #[command(subcommand)]
        command: DemoCommand,
    }

    #[derive(Subcommand)]
    enum DemoCommand {
        Echo(EchoCommand),
        Count(CountCommand),
        Tui,
    }

    #[derive(Args)]
    struct EchoCommand {
        value: String,
    }

    #[derive(Args)]
    struct CountCommand {
        value: usize,
    }

    #[async_trait]
    impl CommandSet<DemoContext> for DemoCommand {
        type CompletionContext = usize;
        type Resolved = Self;

        fn root_command() -> clap::Command {
            DemoRoot::command().name("demo")
        }

        fn from_matches(matches: &clap::ArgMatches) -> DriverResult<Self> {
            Ok(DemoRoot::from_arg_matches(matches)?.command)
        }

        fn completion_context(context: &DemoContext) -> Self::CompletionContext {
            context.counter
        }

        fn help(topic: &[String]) -> Option<String> {
            if topic == [String::from("count")] {
                return Some("count help".to_string());
            }
            None
        }

        fn complete(
            request: CompletionRequest<'_>,
            context: &Self::CompletionContext,
        ) -> Vec<CompletionCandidate> {
            if request.command_path == ["echo"] && request.arg_id == Some("value") {
                return vec![CompletionCandidate::new(format!("seen-{context}"))];
            }
            Vec::new()
        }

        fn resolve_command(self, _context: &DemoContext) -> DriverResult<Self::Resolved> {
            Ok(self)
        }

        async fn execute(
            resolved: Self::Resolved,
            context: &mut DemoContext,
        ) -> DriverResult<CommandOutput> {
            match resolved {
                DemoCommand::Echo(cmd) => Ok(CommandOutput::line(cmd.value)),
                DemoCommand::Count(cmd) => {
                    context.counter += cmd.value;
                    Ok(CommandOutput::line(format!("count={}", context.counter)))
                }
                DemoCommand::Tui => {
                    Ok(CommandOutput::line("entering tui").with_effect(CommandEffect::EnterTui))
                }
            }
        }
    }

    #[tokio::test]
    async fn run_line_prefixes_root_and_executes_command() {
        let mut engine = CommandEngine::<DemoContext, DemoCommand>::new(DemoContext::default());
        let output = engine.run_line("echo hello").await.expect("echo output");
        assert_eq!(output.lines, vec!["hello"]);
    }

    #[tokio::test]
    async fn run_argv_updates_context() {
        let mut engine = CommandEngine::<DemoContext, DemoCommand>::new(DemoContext::default());
        let argv = vec!["demo".to_string(), "count".to_string(), "2".to_string()];
        let output = engine.run_argv(&argv).await.expect("count output");
        assert_eq!(output.lines, vec!["count=2"]);
        assert_eq!(engine.context().counter, 2);
    }

    #[tokio::test]
    async fn builtin_help_prefers_command_set_help() {
        let mut engine = CommandEngine::<DemoContext, DemoCommand>::new(DemoContext::default());
        let output = engine.run_line("help count").await.expect("help output");
        assert_eq!(output.lines, vec!["count help"]);
    }

    #[tokio::test]
    async fn builtin_quit_returns_exit_effect() {
        let mut engine = CommandEngine::<DemoContext, DemoCommand>::new(DemoContext::default());
        let output = engine.run_line("quit").await.expect("quit output");
        assert!(output.effects.contains(&CommandEffect::ExitShell));
    }

    #[test]
    fn completion_merges_builtin_static_and_dynamic_candidates() {
        let mut engine = CommandEngine::<DemoContext, DemoCommand>::new(DemoContext::default());
        engine.context_mut().counter = 3;

        let root = engine.complete("e", 1);
        let values = root
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();
        assert!(values.contains(&"echo".to_string()));

        let dynamic = engine.complete("echo s", "echo s".len());
        let values = dynamic
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();
        assert!(values.contains(&"seen-3".to_string()));

        let builtin = engine.complete("he", 2);
        let values = builtin
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();
        assert!(values.contains(&"help".to_string()));
    }
}
