use anyhow::{bail, Result};
use async_trait::async_trait;

use crate::completion::{CompletionCandidate, CompletionRequest};

#[derive(Debug, Default, Clone)]
pub struct CommandOutput {
    pub lines: Vec<String>,
}

impl CommandOutput {
    pub fn line(line: impl Into<String>) -> Self {
        Self {
            lines: vec![line.into()],
        }
    }
}

#[async_trait]
pub trait CommandSet<C>: Sized {
    fn root_command() -> clap::Command;
    fn from_matches(matches: &clap::ArgMatches) -> Result<Self>;
    fn complete(_request: CompletionRequest<'_>, _context: &C) -> Vec<CompletionCandidate> {
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

    pub async fn run_line(&mut self, line: &str) -> Result<CommandOutput> {
        let argv = shlex::split(line).ok_or_else(|| anyhow::anyhow!("invalid shell quoting"))?;
        self.run_argv(&argv).await
    }

    pub async fn run_argv(&mut self, argv: &[String]) -> Result<CommandOutput> {
        let root = S::root_command();
        let matches = root.try_get_matches_from(argv)?;
        let command = S::from_matches(&matches)?;
        command.execute(&mut self.context).await
    }

    pub fn complete(&self, line: &str, cursor: usize) -> Vec<CompletionCandidate> {
        let partial = line.get(..cursor).unwrap_or(line);
        let prefix = partial
            .split_whitespace()
            .last()
            .unwrap_or_default();
        S::complete(
            CompletionRequest {
                command_path: &[],
                arg_id: None,
                prefix,
            },
            &self.context,
        )
    }
}

pub fn not_implemented(feature: &str) -> Result<CommandOutput> {
    bail!("{feature} is not implemented in the driver scaffold yet")
}
