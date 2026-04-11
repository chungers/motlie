#[path = "common/tmux_ui.rs"]
mod tmux_ui;

use motlie_driver::CommandEngine;
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let state = TmuxState::connect(&uri).await?;
    let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
    let _ = tmux_ui::run(&mut engine).await?;
    Ok(())
}
