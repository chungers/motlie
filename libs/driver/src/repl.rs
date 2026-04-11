#[cfg(feature = "repl")]
use anyhow::Result;

#[cfg(feature = "repl")]
use crate::engine::{CommandEngine, CommandSet};

#[cfg(feature = "repl")]
pub struct ReplFrontend<C, S> {
    engine: CommandEngine<C, S>,
    prompt: String,
    name: String,
}

#[cfg(feature = "repl")]
impl<C, S> ReplFrontend<C, S>
where
    S: CommandSet<C>,
{
    pub fn new(engine: CommandEngine<C, S>) -> Self {
        Self {
            engine,
            prompt: "> ".to_string(),
            name: "motlie".to_string(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    pub async fn run(&mut self) -> Result<()> {
        let _ = &self.engine;
        let _ = &self.prompt;
        let _ = &self.name;
        Ok(())
    }
}
