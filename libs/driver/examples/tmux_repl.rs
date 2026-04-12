use motlie_driver::CommandEngine;
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};
use motlie_driver::tmux_frontend::{TuiAction, run_tmux_repl, run_tmux_tui};

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
        let mut recorder = None;
        match run_tmux_tui(&mut engine, &mut recorder).await? {
            TuiAction::Quit => return Ok(()),
            TuiAction::ReturnToRepl => {}
        }
    }

    let mut recorder = None;
    run_tmux_repl(&mut engine, &mut recorder, &uri).await?;
    Ok(())
}
