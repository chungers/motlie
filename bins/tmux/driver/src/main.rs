#[cfg(not(feature = "commands-tmux"))]
compile_error!("motlie-tmux-driver requires the `commands-tmux` feature");

#[cfg(feature = "repl")]
mod plain;

#[cfg(feature = "tui")]
mod ui;

#[cfg(all(feature = "repl", not(feature = "tui")))]
mod ui {
    use anyhow::bail;
    use motlie_driver::CommandEngine;
    use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};

    #[allow(dead_code)]
    pub enum TuiAction {
        ReturnToRepl,
        Quit,
    }

    pub async fn run(
        _engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    ) -> anyhow::Result<TuiAction> {
        bail!("this binary was built without the `tui` feature")
    }
}

use anyhow::Result;
use motlie_driver::CommandEngine;
use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};

#[tokio::main]
async fn main() -> Result<()> {
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
        return run_tui(&mut engine).await;
    }

    run_repl(&mut engine).await
}

#[cfg(feature = "repl")]
async fn run_repl(engine: &mut CommandEngine<TmuxState, TmuxCommand>) -> Result<()> {
    plain::run(engine).await
}

#[cfg(not(feature = "repl"))]
async fn run_repl(_engine: &mut CommandEngine<TmuxState, TmuxCommand>) -> Result<()> {
    anyhow::bail!("this binary was built without the `repl` feature; pass --tui or rebuild with `repl`")
}

#[cfg(feature = "tui")]
async fn run_tui(engine: &mut CommandEngine<TmuxState, TmuxCommand>) -> Result<()> {
    let _ = ui::run(engine).await?;
    Ok(())
}

#[cfg(not(feature = "tui"))]
async fn run_tui(_engine: &mut CommandEngine<TmuxState, TmuxCommand>) -> Result<()> {
    anyhow::bail!("this binary was built without the `tui` feature; rebuild with `tui`")
}
