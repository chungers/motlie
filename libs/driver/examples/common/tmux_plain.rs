use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use motlie_driver::clap::analyze_completion;
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};
use motlie_driver::completion::{CompletionCandidate, CompletionRequest, dedup_sorted};
use motlie_driver::{CommandEffect, CommandEngine, CommandSet};
use reedline::{
    Completer, DefaultPrompt, DefaultPromptSegment, Reedline, Signal, Span, Suggestion,
};

use crate::tmux_ui;

struct TmuxCompleter {
    context: Arc<Mutex<<TmuxCommand as CommandSet<TmuxState>>::CompletionContext>>,
}

impl TmuxCompleter {
    fn new(context: Arc<Mutex<<TmuxCommand as CommandSet<TmuxState>>::CompletionContext>>) -> Self {
        Self { context }
    }
}

impl Completer for TmuxCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let context = match self.context.lock() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };

        let root = TmuxCommand::root_command();
        let completion = analyze_completion(&root, line, pos);
        let path_refs = completion
            .command_path
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let mut candidates = completion.static_candidates;

        if path_refs.is_empty() {
            for builtin in ["help", "quit"] {
                if builtin.starts_with(&completion.prefix) {
                    candidates.push(CompletionCandidate::new(builtin));
                }
            }
        }

        candidates.extend(TmuxCommand::complete(
            CompletionRequest {
                command_path: &path_refs,
                arg_id: completion.arg_id.as_deref(),
                prefix: &completion.prefix,
            },
            &context,
        ));

        let candidates = dedup_sorted(candidates);
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

pub async fn run(engine: &mut CommandEngine<TmuxState, TmuxCommand>) -> Result<()> {
    let completion_context = Arc::new(Mutex::new(engine.completion_context()));
    let completer = Box::new(TmuxCompleter::new(completion_context.clone()));
    let mut line_editor = Reedline::create().with_completer(completer);
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("tmux> ".to_string()),
        DefaultPromptSegment::Empty,
    );

    loop {
        match line_editor.read_line(&prompt)? {
            Signal::Success(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let plain_monitor = parse_plain_monitor(trimmed);
                let output = engine.run_line(trimmed).await?;
                for line in &output.lines {
                    println!("{line}");
                }

                if let Ok(mut guard) = completion_context.lock() {
                    *guard = engine.completion_context();
                }

                if let Some(session_name) = plain_monitor {
                    if engine.context().mirror_snapshot().watch_health.is_some() {
                        run_plain_monitor(engine, &session_name, Duration::from_secs(3)).await?;
                        if let Ok(mut guard) = completion_context.lock() {
                            *guard = engine.completion_context();
                        }
                    }
                }

                if let Some(effect) = output.effects.first() {
                    match effect {
                        CommandEffect::EnterTui => match tmux_ui::run(engine).await? {
                            tmux_ui::TuiAction::Quit => return Ok(()),
                            tmux_ui::TuiAction::ReturnToRepl => {}
                        },
                        CommandEffect::ExitShell => return Ok(()),
                        CommandEffect::ExitTui => {}
                    }
                }
            }
            Signal::CtrlC | Signal::CtrlD => return Ok(()),
        }
    }
}

fn parse_plain_monitor(line: &str) -> Option<String> {
    let argv = shlex::split(line)?;
    if argv.len() == 2 && argv.first().map(String::as_str) == Some("monitor") {
        return argv.get(1).cloned();
    }
    None
}

async fn run_plain_monitor(
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    session_name: &str,
    duration: Duration,
) -> Result<()> {
    println!("Monitoring {session_name} for {}s...", duration.as_secs());
    let deadline = tokio::time::Instant::now() + duration;
    let mut previous = String::new();

    loop {
        if tokio::time::Instant::now() >= deadline {
            break;
        }

        tokio::time::sleep(Duration::from_millis(150)).await;
        engine.context_mut().refresh_mirror().await?;
        let snapshot = engine.context().mirror_snapshot();
        render_incremental(&mut previous, &snapshot.text);

        if snapshot.watch_health.is_none() {
            break;
        }
    }

    engine.context_mut().shutdown_managed_state().await?;
    println!("Monitor stopped.");
    Ok(())
}

fn render_incremental(previous: &mut String, current: &str) {
    if current.is_empty() || current == previous {
        return;
    }

    let delta = if current.starts_with(previous.as_str()) {
        &current[previous.len()..]
    } else {
        current
    };

    if !delta.trim().is_empty() {
        print!("{delta}");
        if !delta.ends_with('\n') {
            println!();
        }
    }

    previous.clear();
    previous.push_str(current);
}
