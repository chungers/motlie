use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use motlie_driver::CommandEngine;
#[cfg(feature = "sherpa")]
use motlie_telnyx_gateway::adapter::default_artifact_root;
#[cfg(not(feature = "sherpa"))]
use motlie_telnyx_gateway::adapter::UnavailableAsrFactory;
use motlie_telnyx_gateway::adapter::{EchoAsrFactory, SharedAsrFactory};
use motlie_telnyx_gateway::call_control::TelnyxClient;
use motlie_telnyx_gateway::cli::{Cli, CliCommand};
use motlie_telnyx_gateway::operator::commands::{GatewayCommand, GatewayContext};
use motlie_telnyx_gateway::operator::state::{shared_state, LogLevel};
use motlie_telnyx_gateway::serve::{serve, AppServices};
use tokio::time::{self, Duration};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let default_tui_log = PathBuf::from("telnyx-gateway.log");
    let log_file = cli
        .log_file
        .as_deref()
        .or_else(|| cli.tui.then_some(default_tui_log.as_path()));
    let _logging_guard = motlie_telnyx_gateway::logging::init(log_file)?;

    if let Some(CliCommand::ReplayCapture(args)) = cli.command.as_ref() {
        let report =
            motlie_telnyx_gateway::replay::replay_capture(args, build_asr_factory(&cli)).await?;
        print_replay_report(&report);
        return Ok(());
    }

    let state = shared_state(cli.bind);
    {
        let mut guard = state.write().await;
        guard.log(
            LogLevel::Info,
            format!("listener configured on {}", cli.bind),
        );
        guard.config.capture_dir = cli.capture_dir.clone();
    }

    let api_key = std::env::var(&cli.telnyx_api_key_env).ok();
    let telnyx = TelnyxClient::new(cli.telnyx_api_base.clone(), api_key, cli.dry_run_telnyx);
    let asr = build_asr_factory(&cli);
    let services = AppServices {
        state: state.clone(),
        telnyx: telnyx.clone(),
        asr,
    };

    let server = tokio::spawn(serve(cli.bind, services));
    let context = GatewayContext::new(state.clone(), telnyx);
    let mut replay_engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context.clone());

    if let Some(path) = &cli.load {
        replay_commands(&mut replay_engine, path).await?;
    }

    let socket_task = if let Some(path) = cli.socket.clone() {
        let socket_context = Arc::new(context.for_new_source());
        {
            let mut guard = state.write().await;
            guard.log(
                LogLevel::Info,
                format!("operator socket listening on {}", path.display()),
            );
        }
        Some(tokio::spawn(
            motlie_telnyx_gateway::operator::socket::run_command_socket(path, socket_context),
        ))
    } else {
        None
    };

    if cli.tui {
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context);
        motlie_telnyx_gateway::operator::tui::run_tui(&mut engine).await?;
    } else {
        println!("telnyx-gateway listening on {}", cli.bind);
        if cli.socket.is_some() {
            println!(
                "inbound is disabled by default; use the command socket for M1 operator control"
            );
            wait_for_shutdown(state.clone()).await?;
        } else {
            println!(
                "inbound is disabled by default; use --tui or --socket for M1 operator control"
            );
            tokio::signal::ctrl_c().await?;
        }
    }

    if let Some(task) = socket_task {
        task.abort();
    }

    server.abort();
    Ok(())
}

fn print_replay_report(report: &motlie_telnyx_gateway::replay::ReplayReport) {
    println!("capture_dir: {}", report.capture_dir);
    println!("wav: {}", report.wav_path);
    println!("samples: {}", report.sample_count);
    println!("transcript:");
    println!("{}", report.transcript);
    if let Some(wer) = &report.wer {
        println!("wer: {:.1}%", wer.rate() * 100.0);
        println!("reference_words: {}", wer.reference_words);
        println!("hypothesis_words: {}", wer.hypothesis_words);
        println!("substitutions: {}", wer.substitutions);
        println!("deletions: {}", wer.deletions);
        println!("insertions: {}", wer.insertions);
        println!("errors: {}", wer.errors);
        if !wer.errors_by_token.is_empty() {
            println!("token_errors:");
            for error in &wer.errors_by_token {
                match error {
                    motlie_telnyx_gateway::replay::WerTokenError::Substitution {
                        reference,
                        hypothesis,
                    } => println!("  S {reference} -> {hypothesis}"),
                    motlie_telnyx_gateway::replay::WerTokenError::Deletion { reference } => {
                        println!("  D {reference}");
                    }
                    motlie_telnyx_gateway::replay::WerTokenError::Insertion { hypothesis } => {
                        println!("  I {hypothesis}");
                    }
                }
            }
        }
    }
}

async fn wait_for_shutdown(
    state: motlie_telnyx_gateway::operator::state::SharedState,
) -> anyhow::Result<()> {
    let mut tick = time::interval(Duration::from_millis(200));
    loop {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                return Ok(());
            }
            _ = tick.tick() => {
                if state.read().await.shutdown_requested {
                    return Ok(());
                }
            }
        }
    }
}

fn build_asr_factory(_cli: &Cli) -> SharedAsrFactory {
    if std::env::var_os("MOTLIE_TELNYX_ECHO_ASR").is_some() {
        return Arc::new(EchoAsrFactory);
    }

    #[cfg(feature = "sherpa")]
    {
        let artifact_root = default_artifact_root(_cli.asr_artifact_root.clone());
        Arc::new(motlie_telnyx_gateway::adapter::SherpaAsrFactory::new(
            artifact_root,
            !_cli.no_asr_download,
        ))
    }

    #[cfg(not(feature = "sherpa"))]
    {
        Arc::new(UnavailableAsrFactory::new(
            "gateway was built without a live ASR backend; rebuild with --features sherpa or set MOTLIE_TELNYX_ECHO_ASR=1 for protocol testing",
        ))
    }
}

async fn replay_commands(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let raw = fs::read_to_string(path)?;
    for (index, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        engine
            .run_line(trimmed)
            .await
            .map_err(|err| anyhow::anyhow!("{}:{}: {err}", path.display(), index + 1))?;
    }
    Ok(())
}
