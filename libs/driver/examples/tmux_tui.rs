use motlie_driver::CommandEngine;
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};
use motlie_driver::tmux_frontend::run_tmux_tui;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let state = TmuxState::connect(&uri).await?;
    let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
    let mut recorder = None;
    let _ = run_tmux_tui(&mut engine, &mut recorder).await?;
    Ok(())
}
