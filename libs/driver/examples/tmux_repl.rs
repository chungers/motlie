#[path = "common/tmux_plain.rs"]
mod tmux_plain;
#[path = "common/tmux_ui.rs"]
mod tmux_ui;

use motlie_driver::CommandEngine;
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};

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

    tmux_plain::run(&mut engine).await
}
