use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "telnyx-agent")]
#[command(about = "Motlie Telnyx text-call tmux bridge daemon")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Daemon(DaemonArgs),
    Call(CallArgs),
}

#[derive(Clone, Debug, Args)]
pub struct DaemonArgs {
    #[arg(long)]
    pub gateway_url: String,
    #[arg(long)]
    pub public_url: String,
    #[arg(long = "subscribe-number")]
    pub subscribe_numbers: Vec<String>,
    #[arg(long, default_value = "env:MOTLIE_APP_CALLBACK_SECRET")]
    pub callback_secret_ref: String,
    #[arg(long)]
    pub tmux_target: String,
    #[arg(long)]
    pub socket: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8181")]
    pub bind: SocketAddr,
    #[arg(long)]
    pub gateway_token: Option<String>,
    #[arg(long, default_value_t = 45_000)]
    pub outbound_timeout_ms: u64,
    #[arg(long, default_value_t = 120_000)]
    pub reply_timeout_ms: u64,
}

#[derive(Clone, Debug, Args)]
pub struct CallArgs {
    #[arg(long)]
    pub socket: PathBuf,
    #[arg(long)]
    pub to: String,
}
