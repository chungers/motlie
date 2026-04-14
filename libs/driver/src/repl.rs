#[cfg(feature = "repl")]
use std::marker::PhantomData;
#[cfg(feature = "repl")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "repl")]
use reedline::{
    Completer, DefaultPrompt, DefaultPromptSegment, Reedline, Signal, Span, Suggestion,
};

#[cfg(feature = "repl")]
use crate::engine::complete_with_context;
#[cfg(feature = "repl")]
use crate::engine::{CommandEffect, CommandEngine, CommandSet};
#[cfg(feature = "repl")]
use crate::error::{DriverError, DriverResult};

#[cfg(feature = "repl")]
struct EngineCompleter<C, S>
where
    S: CommandSet<C>,
{
    context: Arc<Mutex<S::CompletionContext>>,
    _commands: PhantomData<S>,
}

#[cfg(feature = "repl")]
impl<C, S> EngineCompleter<C, S>
where
    S: CommandSet<C>,
{
    fn new(context: Arc<Mutex<S::CompletionContext>>) -> Self {
        Self {
            context,
            _commands: PhantomData,
        }
    }
}

#[cfg(feature = "repl")]
impl<C, S> Completer for EngineCompleter<C, S>
where
    C: 'static,
    S: CommandSet<C> + Send + 'static,
{
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let context = match self.context.lock() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };
        let candidates = complete_with_context::<C, S>(line, pos, &context);
        let start = line
            .get(..pos)
            .and_then(|prefix| {
                prefix
                    .rmatch_indices(char::is_whitespace)
                    .next()
                    .map(|(index, matched)| index + matched.len())
            })
            .unwrap_or(0);

        candidates
            .into_iter()
            .map(|candidate| Suggestion {
                value: candidate.value,
                description: candidate.help,
                style: None,
                extra: None,
                span: Span::new(start, pos),
                append_whitespace: true,
            })
            .collect()
    }
}

#[cfg(feature = "repl")]
pub struct ReplFrontend<C, S> {
    engine: CommandEngine<C, S>,
    prompt: String,
    name: String,
}

#[cfg(feature = "repl")]
impl<C, S> ReplFrontend<C, S>
where
    C: 'static,
    S: CommandSet<C> + Send + 'static,
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

    pub fn engine(&self) -> &CommandEngine<C, S> {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut CommandEngine<C, S> {
        &mut self.engine
    }

    pub async fn run(&mut self) -> DriverResult<Option<CommandEffect>> {
        let completion_context = Arc::new(Mutex::new(self.engine.completion_context()));
        let completer = Box::new(EngineCompleter::<C, S>::new(completion_context.clone()));
        let mut line_editor = Reedline::create().with_completer(completer);
        let prompt = DefaultPrompt::new(
            DefaultPromptSegment::Basic(format!("{}{}", self.name, self.prompt)),
            DefaultPromptSegment::Empty,
        );

        loop {
            match line_editor.read_line(&prompt)? {
                Signal::Success(line) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let output = match self.engine.run_line(trimmed).await {
                        Ok(output) => output,
                        Err(error) => {
                            print_driver_error(&error);
                            continue;
                        }
                    };
                    for line in &output.lines {
                        println!("{line}");
                    }

                    if let Ok(mut guard) = completion_context.lock() {
                        *guard = self.engine.completion_context();
                    }

                    if let Some(effect) = output.effects.first() {
                        return Ok(Some(effect.clone()));
                    }
                }
                Signal::CtrlC | Signal::CtrlD => return Ok(None),
            }
        }
    }
}

#[cfg(feature = "repl")]
fn print_driver_error(error: &DriverError) {
    match error {
        DriverError::Clap(clap_error) => {
            let rendered = clap_error.render().to_string();
            print!("{rendered}");
            if !rendered.ends_with('\n') {
                println!();
            }
        }
        _ => eprintln!("{error}"),
    }
}
