#[path = "common/tmux_ui.rs"]
mod tmux_ui;

use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};
use motlie_driver::{CommandEffect, CommandEngine, ReplFrontend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut use_tui = false;
    let mut uri = "ssh://localhost".to_string();

    for arg in std::env::args().skip(1) {
        if arg == "--tui" {
            use_tui = true;
        } else {
            uri = arg;
        }
    }

    let state = TmuxState::connect(&uri).await?;
    println!("Connected to {uri}");

    let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
    if use_tui {
        match tmux_ui::run(&mut engine).await? {
            tmux_ui::TuiAction::Quit => return Ok(()),
            tmux_ui::TuiAction::ReturnToRepl => {}
        }
    }

    let mut repl = ReplFrontend::new(engine)
        .with_name("tmux")
        .with_prompt("tmux> ");

    loop {
        match repl.run().await? {
            Some(CommandEffect::EnterTui) => match tmux_ui::run(repl.engine_mut()).await? {
                tmux_ui::TuiAction::Quit => return Ok(()),
                tmux_ui::TuiAction::ReturnToRepl => continue,
            },
            Some(CommandEffect::ExitShell) | None => return Ok(()),
            Some(CommandEffect::ExitTui) => continue,
        }
    }
}
