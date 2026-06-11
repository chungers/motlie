mod config;
mod gateway_client;
mod http;
mod socket;
mod text_ws;
mod tmux_bridge;

use clap::Parser;
use config::{Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Daemon(args) => http::run_daemon(args).await,
        Command::Call(args) => socket::run_call_client(args).await,
    }
}
