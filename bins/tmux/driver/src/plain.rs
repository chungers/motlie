use std::sync::{Arc, Mutex};
use anyhow::Result;
use motlie_driver::clap::analyze_completion;
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};
use motlie_driver::completion::{CompletionCandidate, CompletionRequest, dedup_sorted};
use motlie_driver::term::asciicast::AsciicastRecorder;
use motlie_driver::{CommandEffect, CommandEngine, CommandSet};
use reedline::{
    Completer, DefaultPrompt, DefaultPromptSegment, Reedline, Signal, Span, Suggestion,
};

use crate::ui;

const HISTORY_PAGE_SIZE: usize = 32;

pub struct TmuxCompleter {
    context: Arc<Mutex<<TmuxCommand as CommandSet<TmuxState>>::CompletionContext>>,
}

impl TmuxCompleter {
    pub fn new(
        context: Arc<Mutex<<TmuxCommand as CommandSet<TmuxState>>::CompletionContext>>,
    ) -> Self {
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

pub async fn run(
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    mut recorder: Option<AsciicastRecorder>,
    host_uri: &str,
) -> Result<()> {
    let completion_context = Arc::new(Mutex::new(engine.completion_context()));
    let completer = Box::new(TmuxCompleter::new(completion_context.clone()));
    let mut line_editor = Reedline::create().with_completer(completer);
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("tmux> ".to_string()),
        DefaultPromptSegment::Empty,
    );
    record_output(&mut recorder, &format!("Connected to {host_uri}\n"))?;

    loop {
        match line_editor.read_line(&prompt)? {
            Signal::Success(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                record_output(&mut recorder, &format!("tmux> {trimmed}\n"))?;
                record_input(&mut recorder, &format!("{trimmed}\n"))?;
                let output = engine.run_line(trimmed).await?;
                for line in &output.lines {
                    println!("{line}");
                    record_output(&mut recorder, &format!("{line}\n"))?;
                }

                if let Ok(mut guard) = completion_context.lock() {
                    *guard = engine.completion_context();
                }

                if engine.context().has_live_follow() {
                    run_live_follow(engine, &mut recorder).await?;
                    if let Ok(mut guard) = completion_context.lock() {
                        *guard = engine.completion_context();
                    }
                }

                if let Some(effect) = output.effects.first() {
                    match effect {
                        CommandEffect::EnterTui => match ui::run(engine).await? {
                            ui::TuiAction::Quit => return Ok(()),
                            ui::TuiAction::ReturnToRepl => {}
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

async fn run_live_follow(
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    recorder: &mut Option<AsciicastRecorder>,
) -> Result<()> {
    println!("Live follow attached. Press Ctrl-C to stop and return to the prompt.");
    record_output(
        recorder,
        "Live follow attached. Press Ctrl-C to stop and return to the prompt.\n",
    )?;
    let mut previous = String::new();
    let mut cursor = None;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                engine.context_mut().shutdown_managed_state().await?;
                println!();
                println!("Live follow stopped.");
                record_output(recorder, "^C\nLive follow stopped.\n")?;
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(150)) => {}
        }

        engine.context_mut().refresh_mirror().await?;
        let page = engine
            .context()
            .mirror_history_page(cursor, HISTORY_PAGE_SIZE);
        for record in page.items {
            render_incremental(&mut previous, &record.item.text);
            record_output(recorder, &record.item.text)?;
            if !record.item.text.ends_with('\n') {
                record_output(recorder, "\n")?;
            }
            cursor = Some(record.seq);
        }

        if !engine.context().has_live_follow() {
            break;
        }
    }
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

fn record_output(recorder: &mut Option<AsciicastRecorder>, text: &str) -> Result<()> {
    if let Some(recorder) = recorder.as_mut() {
        recorder.record_output(text)?;
    }
    Ok(())
}

fn record_input(recorder: &mut Option<AsciicastRecorder>, text: &str) -> Result<()> {
    if let Some(recorder) = recorder.as_mut() {
        recorder.record_input(text)?;
    }
    Ok(())
}
