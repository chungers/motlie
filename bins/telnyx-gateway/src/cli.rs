use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

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

    #[command(subcommand)]
    pub command: Option<CliCommand>,
}

#[derive(Clone, Debug, Subcommand)]
pub enum CliCommand {
    /// Replay a captured ASR-input WAV and optionally compute WER.
    ReplayCapture(ReplayCaptureArgs),

    /// Replay a golden ASR corpus across one or more backends.
    ReplayCorpus(ReplayCorpusArgs),
}

#[derive(Clone, Debug, Args)]
pub struct ReplayCaptureArgs {
    /// Capture directory containing asr-input-16khz.wav.
    pub capture_dir: PathBuf,

    /// Reference transcript text for WER.
    #[arg(long, conflicts_with = "reference_file")]
    pub reference: Option<String>,

    /// File containing the reference transcript text for WER.
    #[arg(long)]
    pub reference_file: Option<PathBuf>,

    /// Audio chunk size to feed into the streaming recognizer.
    #[arg(long, default_value_t = 20)]
    pub chunk_ms: u32,

    /// ASR backend to use for this replay.
    #[arg(long, value_enum, default_value_t = ReplayBackendArg::Auto)]
    pub backend: ReplayBackendArg,
}

#[derive(Clone, Debug, Args)]
pub struct ReplayCorpusArgs {
    /// Golden corpus manifest JSON file.
    pub manifest: PathBuf,

    /// ASR backend to run. Repeat to compare multiple backends.
    #[arg(long, value_enum)]
    pub backend: Vec<ReplayBackendArg>,

    /// Audio chunk size to feed into each streaming recognizer.
    #[arg(long, default_value_t = 20)]
    pub chunk_ms: u32,
}

impl ReplayCorpusArgs {
    pub fn selected_backends(&self) -> Vec<ReplayBackendArg> {
        if self.backend.is_empty() {
            vec![ReplayBackendArg::Auto]
        } else {
            self.backend.clone()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ReplayBackendArg {
    Auto,
    Echo,
    Sherpa,
}

impl ReplayBackendArg {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Echo => "echo",
            Self::Sherpa => "sherpa",
        }
    }
}
