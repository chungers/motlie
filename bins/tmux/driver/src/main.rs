use std::path::PathBuf;

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
use motlie_driver::term::asciicast::AsciicastMetadata;

struct DriverOptions {
    use_tui: bool,
    uri: String,
    record_asciicast: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = parse_args()?;

    let state = TmuxState::connect(&options.uri).await?;
    println!("Connected to {}", options.uri);

    let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);

    if options.use_tui {
        return run_tui(&mut engine).await;
    }

    run_repl(&mut engine, &options).await
}

#[cfg(feature = "repl")]
async fn run_repl(engine: &mut CommandEngine<TmuxState, TmuxCommand>, options: &DriverOptions) -> Result<()> {
    let recorder = options
        .record_asciicast
        .as_ref()
        .map(|path| {
            let meta = AsciicastMetadata {
                title: "motlie-tmux-driver".to_string(),
                command: Some(std::env::args().collect::<Vec<_>>().join(" ")),
                term_type: std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".to_string()),
                cols: std::env::var("COLUMNS")
                    .ok()
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(80),
                rows: std::env::var("LINES")
                    .ok()
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(24),
            };
            motlie_driver::term::asciicast::AsciicastRecorder::create(path, &meta)
        })
        .transpose()?;

    plain::run(engine, recorder, &options.uri).await
}

#[cfg(not(feature = "repl"))]
async fn run_repl(_engine: &mut CommandEngine<TmuxState, TmuxCommand>, _options: &DriverOptions) -> Result<()> {
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

fn parse_args() -> Result<DriverOptions> {
    let mut use_tui = false;
    let mut uri = "ssh://localhost".to_string();
    let mut record_asciicast = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tui" => use_tui = true,
            "--record-asciicast" => {
                let path = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--record-asciicast requires a path"))?;
                record_asciicast = Some(PathBuf::from(path));
            }
            _ => uri = arg,
        }
    }

    Ok(DriverOptions {
        use_tui,
        uri,
        record_asciicast,
    })
}
