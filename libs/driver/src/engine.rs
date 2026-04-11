use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::clap::{analyze_completion, render_help};
use crate::completion::dedup_sorted;
use crate::completion::{CompletionCandidate, CompletionRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandEffect {
    ExitShell,
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

    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> Result<Self>;
    fn completion_context(context: &C) -> Self::CompletionContext;
    fn complete(
        _request: CompletionRequest<'_>,
        _context: &Self::CompletionContext,
    ) -> Vec<CompletionCandidate> {
        Vec::new()
    }
    async fn execute(self, context: &mut C) -> Result<CommandOutput>;
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

    pub async fn run_line(&mut self, line: &str) -> Result<CommandOutput> {
        let argv = shlex::split(line).ok_or_else(|| anyhow::anyhow!("invalid shell quoting"))?;
        self.run_argv(&argv).await
    }

    pub async fn run_argv(&mut self, argv: &[String]) -> Result<CommandOutput> {
        let root = S::root_command();

        if let Some(first) = argv.first() {
            if first == "help" {
                let topic = argv.iter().skip(1).cloned().collect::<Vec<_>>();
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
        command.execute(&mut self.context).await
    }

    pub fn complete(&self, line: &str, cursor: usize) -> Vec<CompletionCandidate> {
        let root = S::root_command();
        let completion = analyze_completion(&root, line, cursor);
        let path_refs = completion
            .command_path
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();

        let mut out = completion.static_candidates;

        if path_refs.is_empty() {
            for builtin in ["help", "quit"] {
                if builtin.starts_with(&completion.prefix) {
                    out.push(CompletionCandidate::new(builtin));
                }
            }
        }

        out.extend(S::complete(
            CompletionRequest {
                command_path: &path_refs,
                arg_id: completion.arg_id.as_deref(),
                prefix: &completion.prefix,
            },
            &self.completion_context(),
        ));

        dedup_sorted(out)
    }
}

pub fn not_implemented(feature: &str) -> Result<CommandOutput> {
    bail!("{feature} is not implemented in the driver scaffold yet")
}
