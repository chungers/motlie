//! Example: Event-driven monitoring via OutputBus -> Subscription -> pipe().
//!
//! This example covers the terminal-consumer side of Track A:
//! - `HostHandle::start_monitoring_session()`
//! - `OutputBus::subscribe()`
//! - `Subscription::pipe()`
//! - `StdioSink` and `CallbackSink`
//! - `PipeHandle::id()` + `join()`
//!
//! Usage:
//!   monitor_pipe <uri> <session> [--seconds N] [--sink raw|prefixed|json|callback]
//!
//! Examples:
//!   ./target/debug/examples/monitor_pipe ssh://localhost build
//!   ./target/debug/examples/monitor_pipe ssh://localhost build --sink json
//!   ./target/debug/examples/monitor_pipe ssh://localhost build --sink callback --seconds 5
//!   ./target/debug/examples/monitor_pipe 'ssh://deploy@prod?identity-file=/path/to/key' build

use anyhow::{anyhow, Result};
use motlie_tmux::{
    CallbackSink, SinkEvent, SinkFilter, SinkKind, SshConfig, StdioFormat, StdioSink,
};
use std::any::Any;
use std::future::Future;
use std::io::Write;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const HELP: &str = "\
monitor_pipe — event-driven monitoring through the sink pipeline

USAGE:
    monitor_pipe <uri> <session> [OPTIONS]

ARGS:
    <uri>       SSH URI to connect (e.g. ssh://localhost, ssh://deploy@prod,
                ssh://host?identity-file=/path/to/key)
    <session>   tmux session name to monitor

OPTIONS:
    --seconds N       Run time before stopping [default: 3]
    --sink MODE       Sink type: raw | prefixed | json | callback [default: prefixed]
    -h, --help        Print this help

SINKS:
    raw        StdioSink::Raw — content only, no labels
    prefixed   StdioSink::Prefixed — [host] source_key | content
    json       StdioSink::Json — JSON line per event
    callback   CallbackSink — custom callback + flush summary

EXAMPLES:
    monitor_pipe ssh://localhost build
    monitor_pipe ssh://localhost build --sink json
    monitor_pipe ssh://localhost build --sink callback --seconds 5
    monitor_pipe 'ssh://deploy@prod?identity-file=/path/to/key' build";

#[derive(Clone, Copy)]
enum SinkMode {
    Raw,
    Prefixed,
    Json,
    Callback,
}

struct Args {
    uri: String,
    session: String,
    seconds: u64,
    sink: SinkMode,
}

#[derive(Clone, Copy, Default)]
struct CallbackStats {
    data_events: usize,
    gap_events: usize,
    dropped_events: usize,
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", HELP);
        std::process::exit(0);
    }

    if args.len() < 3 {
        return Err(anyhow!(
            "usage: {} <uri> <session> [--seconds N] [--sink raw|prefixed|json|callback]\n\nTry -h for detailed help.",
            args[0]
        ));
    }

    let mut seconds = 3u64;
    let mut sink = SinkMode::Prefixed;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--seconds" => {
                i += 1;
                seconds = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--seconds requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--seconds must be a positive integer"))?;
                if seconds == 0 {
                    return Err(anyhow!("--seconds must be > 0"));
                }
            }
            "--sink" => {
                i += 1;
                sink = match args
                    .get(i)
                    .ok_or_else(|| anyhow!("--sink requires a value"))?
                    .as_str()
                {
                    "raw" => SinkMode::Raw,
                    "prefixed" => SinkMode::Prefixed,
                    "json" => SinkMode::Json,
                    "callback" => SinkMode::Callback,
                    other => {
                        return Err(anyhow!(
                            "unknown sink '{}' (raw|prefixed|json|callback)",
                            other
                        ));
                    }
                };
            }
            other => return Err(anyhow!("unknown flag: {}", other)),
        }
        i += 1;
    }

    Ok(Args {
        uri: args[1].clone(),
        session: args[2].clone(),
        seconds,
        sink,
    })
}

fn callback_on_output(state: &Arc<dyn Any + Send + Sync>, event: SinkEvent) -> Result<()> {
    let stats = state
        .downcast_ref::<Mutex<CallbackStats>>()
        .ok_or_else(|| anyhow!("callback state type mismatch"))?;

    match event {
        SinkEvent::Data(output) => {
            {
                let mut stats = stats.lock().expect("callback stats lock poisoned");
                stats.data_events += 1;
            }

            let mut stdout = std::io::stdout().lock();
            writeln!(
                stdout,
                "[callback {} {}] {}",
                output.host,
                output.source_key(),
                output.content.trim_end_matches('\n')
            )?;
            stdout.flush()?;
        }
        SinkEvent::Gap { dropped, .. } => {
            {
                let mut stats = stats.lock().expect("callback stats lock poisoned");
                stats.gap_events += 1;
                stats.dropped_events += dropped;
            }

            let mut stderr = std::io::stderr().lock();
            writeln!(stderr, "[callback gap] dropped={} event(s)", dropped)?;
            stderr.flush()?;
        }
    }

    Ok(())
}

fn callback_on_flush(
    state: &Arc<dyn Any + Send + Sync>,
) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    let snapshot = state
        .downcast_ref::<Mutex<CallbackStats>>()
        .map(|stats| *stats.lock().expect("callback stats lock poisoned"))
        .unwrap_or_default();

    Box::pin(async move {
        let mut stderr = std::io::stderr().lock();
        writeln!(
            stderr,
            "Callback summary: data_events={}, gap_events={}, dropped_events={}",
            snapshot.data_events, snapshot.gap_events, snapshot.dropped_events
        )?;
        stderr.flush()?;
        Ok(())
    })
}

fn sink_name(mode: SinkMode) -> &'static str {
    match mode {
        SinkMode::Raw => "raw",
        SinkMode::Prefixed => "prefixed",
        SinkMode::Json => "json",
        SinkMode::Callback => "callback",
    }
}

fn build_sink(mode: SinkMode) -> SinkKind {
    match mode {
        SinkMode::Raw => SinkKind::Stdio(StdioSink::new(StdioFormat::Raw)),
        SinkMode::Prefixed => SinkKind::Stdio(StdioSink::new(StdioFormat::Prefixed)),
        SinkMode::Json => SinkKind::Stdio(StdioSink::new(StdioFormat::Json)),
        SinkMode::Callback => SinkKind::Callback(CallbackSink {
            name: "example-callback".into(),
            state: Arc::new(Mutex::new(CallbackStats::default())),
            on_output: callback_on_output,
            on_flush: Some(callback_on_flush),
        }),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;

    let host = SshConfig::parse(&args.uri)?.connect().await?;
    let monitor = host.start_monitoring_session(&args.session).await?;

    let bus = host.output_bus();
    let sub = bus.subscribe(vec![SinkFilter::for_session(&args.session)], 64)?;
    let pipe = sub.pipe(build_sink(args.sink));

    eprintln!(
        "Monitoring {} for {}s using {} sink. Ctrl-C to stop early.",
        args.session,
        args.seconds,
        sink_name(args.sink)
    );
    eprintln!(
        "Flow: start_monitoring_session -> output_bus.subscribe -> pipe -> unsubscribe -> join"
    );

    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(args.seconds)) => {}
        _ = tokio::signal::ctrl_c() => {}
    }

    monitor.shutdown().await?;
    bus.unsubscribe(pipe.id())?;
    pipe.join().await?;

    eprintln!("Stopped.");
    Ok(())
}
