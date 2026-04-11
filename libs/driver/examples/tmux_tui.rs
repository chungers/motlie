use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};
use motlie_driver::{CommandEngine, TuiFrontend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let state = TmuxState::connect(&uri).await?;
    let engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
    let mut frontend = TuiFrontend::new(engine)
        .with_title(format!("tmux @ {uri}"))
        .with_prompt("tmux> ");

    frontend.run().await
}
