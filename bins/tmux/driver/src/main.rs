use std::path::PathBuf;

#[cfg(not(feature = "commands-tmux"))]
compile_error!("motlie-tmux-driver requires the `commands-tmux` feature");

use motlie_driver::commands::tmux::{TmuxCommand, TmuxState};
use motlie_driver::commands::tmux_app::{TmuxAppCommand, TmuxAppState};
#[cfg(any(feature = "repl", feature = "tui"))]
use motlie_driver::term::asciicast::AsciicastMetadata;
#[cfg(feature = "repl")]
use motlie_driver::tmux_frontend::run_tmux_repl;
#[cfg(feature = "tui")]
use motlie_driver::tmux_frontend::run_tmux_tui;
use motlie_driver::CommandEngine;
use motlie_driver::{DriverError, DriverResult};

struct DriverOptions {
    use_tui: bool,
    multi_host: bool,
    uri: Option<String>,
    #[cfg_attr(not(any(feature = "repl", feature = "tui")), allow(dead_code))]
    record_asciicast: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> DriverResult<()> {
    let options = parse_args()?;

    if options.multi_host {
        let state = TmuxAppState::new();
        let mut engine = CommandEngine::<TmuxAppState, TmuxAppCommand>::new(state);
        let result = if options.use_tui {
            run_tui(&mut engine, &options).await
        } else {
            run_repl(&mut engine, &options, "multi-host tmux session").await
        };
        let shutdown_result = engine.context_mut().shutdown_all_managed_state().await;
        return result.and(shutdown_result);
    }

    let uri = options
        .uri
        .clone()
        .unwrap_or_else(|| "ssh://localhost".to_string());
    let state = TmuxState::connect(&uri).await?;
    println!("Connected to {uri}");

    let mut engine = CommandEngine::<TmuxState, TmuxCommand>::new(state);
    if options.use_tui {
        return run_tui(&mut engine, &options).await;
    }

    run_repl(&mut engine, &options, &uri).await
}

#[cfg(feature = "repl")]
async fn run_repl<C, S>(
    engine: &mut CommandEngine<C, S>,
    options: &DriverOptions,
    banner: &str,
) -> DriverResult<()>
where
    C: motlie_driver::commands::tmux::TmuxFrontendState + Send + 'static,
    S: motlie_driver::CommandSet<C> + Send + 'static,
{
    let mut recorder = options
        .record_asciicast
        .as_ref()
        .map(|path| {
            motlie_driver::term::asciicast::AsciicastRecorder::create(path, &asciicast_metadata())
        })
        .transpose()?;

    run_tmux_repl(engine, &mut recorder, banner).await
}

#[cfg(not(feature = "repl"))]
async fn run_repl<C, S>(
    _engine: &mut CommandEngine<C, S>,
    _options: &DriverOptions,
    _banner: &str,
) -> DriverResult<()>
where
    C: motlie_driver::commands::tmux::TmuxFrontendState + Send + 'static,
    S: motlie_driver::CommandSet<C> + Send + 'static,
{
    Err(DriverError::message(
        "this binary was built without the `repl` feature; pass --tui or rebuild with `repl`",
    ))
}

#[cfg(feature = "tui")]
async fn run_tui<C, S>(
    engine: &mut CommandEngine<C, S>,
    options: &DriverOptions,
) -> DriverResult<()>
where
    C: motlie_driver::commands::tmux::TmuxFrontendState + Send + 'static,
    S: motlie_driver::CommandSet<C> + Send + 'static,
{
    let mut recorder = options
        .record_asciicast
        .as_ref()
        .map(|path| {
            motlie_driver::term::asciicast::AsciicastRecorder::create(path, &asciicast_metadata())
        })
        .transpose()?;
    let _ = run_tmux_tui(engine, &mut recorder).await?;
    Ok(())
}

#[cfg(not(feature = "tui"))]
async fn run_tui<C, S>(
    _engine: &mut CommandEngine<C, S>,
    _options: &DriverOptions,
) -> DriverResult<()>
where
    C: motlie_driver::commands::tmux::TmuxFrontendState + Send + 'static,
    S: motlie_driver::CommandSet<C> + Send + 'static,
{
    Err(DriverError::message(
        "this binary was built without the `tui` feature; rebuild with `tui`",
    ))
}

fn parse_args() -> DriverResult<DriverOptions> {
    let mut use_tui = false;
    let mut multi_host = false;
    let mut uri = None;
    let mut record_asciicast = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tui" => use_tui = true,
            "--multi-host" => multi_host = true,
            "--record-asciicast" => {
                let path = args.next().ok_or_else(|| {
                    DriverError::invalid_argument("record-asciicast", "requires a path")
                })?;
                record_asciicast = Some(PathBuf::from(path));
            }
            _ => {
                if uri.is_some() {
                    return Err(DriverError::invalid_argument(
                        "uri",
                        "only one positional ssh uri is allowed",
                    ));
                }
                uri = Some(arg);
            }
        }
    }

    if multi_host && uri.is_some() {
        return Err(DriverError::invalid_argument(
            "uri",
            "--multi-host starts without a preconnected host; use 'connect <ssh-uri> as <alias>' inside the driver session",
        ));
    }

    Ok(DriverOptions {
        use_tui,
        multi_host,
        uri,
        record_asciicast,
    })
}

#[cfg(any(feature = "repl", feature = "tui"))]
fn asciicast_metadata() -> AsciicastMetadata {
    AsciicastMetadata {
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
    }
}
