use motlie_driver::{CommandEngine, ReplFrontend};

#[cfg(feature = "commands-tmux")]
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let uri = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ssh://localhost".to_string());

    let state = TmuxState::connect(&uri).await?;
    println!("Connected to {uri}");

    let engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
    let mut repl = ReplFrontend::new(engine)
        .with_name("tmux")
        .with_prompt("tmux> ");

    repl.run().await
}
