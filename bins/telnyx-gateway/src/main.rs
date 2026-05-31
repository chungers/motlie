use std::fs;
use std::sync::Arc;

use clap::Parser;
use motlie_driver::CommandEngine;
#[cfg(feature = "sherpa")]
use motlie_telnyx_gateway::adapter::default_artifact_root;
#[cfg(not(feature = "sherpa"))]
use motlie_telnyx_gateway::adapter::UnavailableAsrFactory;
use motlie_telnyx_gateway::adapter::{EchoAsrFactory, SharedAsrFactory};
use motlie_telnyx_gateway::call_control::TelnyxClient;
use motlie_telnyx_gateway::cli::Cli;
use motlie_telnyx_gateway::operator::commands::{GatewayCommand, GatewayContext};
use motlie_telnyx_gateway::operator::socket::describe_socket_deferral;
use motlie_telnyx_gateway::operator::state::{shared_state, LogLevel};
use motlie_telnyx_gateway::serve::{serve, AppServices};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    motlie_telnyx_gateway::logging::init();
    let cli = Cli::parse();

    let state = shared_state(cli.bind);
    {
        let mut guard = state.write().await;
        guard.log(
            LogLevel::Info,
            format!("listener configured on {}", cli.bind),
        );
        if let Some(socket) = &cli.socket {
            guard.log(LogLevel::Warn, describe_socket_deferral(socket));
        }
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
    let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(GatewayContext::new(
        state.clone(),
        telnyx,
    ));

    if let Some(path) = &cli.load {
        replay_commands(&mut engine, path).await?;
    }

    if cli.tui {
        motlie_telnyx_gateway::operator::tui::run_tui(&mut engine).await?;
    } else {
        println!("telnyx-gateway listening on {}", cli.bind);
        println!("inbound is disabled by default; use --tui for M1 operator control");
        tokio::signal::ctrl_c().await?;
    }

    server.abort();
    Ok(())
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
