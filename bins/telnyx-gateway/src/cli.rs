use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;

#[derive(Clone, Debug, Parser)]
#[command(author, version, about = "Operator-driven Telnyx voice gateway")]
pub struct Cli {
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: SocketAddr,

    #[arg(long)]
    pub tui: bool,

    #[arg(long)]
    pub socket: Option<PathBuf>,

    #[arg(long)]
    pub load: Option<PathBuf>,

    #[arg(long, default_value = "https://api.telnyx.com/v2")]
    pub telnyx_api_base: String,

    #[arg(long, default_value = "TELNYX_API_KEY")]
    pub telnyx_api_key_env: String,

    #[arg(long)]
    pub dry_run_telnyx: bool,

    #[arg(long)]
    pub asr_artifact_root: Option<PathBuf>,

    #[arg(long)]
    pub no_asr_download: bool,

    #[arg(long)]
    pub log_file: Option<PathBuf>,

    #[arg(long)]
    pub capture_dir: Option<PathBuf>,
}
