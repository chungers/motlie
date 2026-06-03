use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::adapter::LiveAsrBackend;
use crate::replay::DEFAULT_TRAILING_SILENCE_PAD_MS;

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

    #[arg(long, value_enum, default_value_t = LiveAsrBackend::default())]
    pub asr_backend: LiveAsrBackend,

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

    /// Generate Qwen3-TTS WAVs for the offline call-center golden ASR corpus.
    GoldenTts(GoldenTtsArgs),

    /// Run the offline call-center ASR A/B matrix without the gateway.
    AsrGoldenAb(AsrGoldenAbArgs),
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

    /// Trailing silence to feed before finish so streaming decoders flush final tokens.
    #[arg(long, default_value_t = DEFAULT_TRAILING_SILENCE_PAD_MS)]
    pub trailing_silence_pad_ms: u32,

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

    /// Trailing silence to feed before finish so streaming decoders flush final tokens.
    #[arg(long, default_value_t = DEFAULT_TRAILING_SILENCE_PAD_MS)]
    pub trailing_silence_pad_ms: u32,
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

#[derive(Clone, Debug, Args)]
pub struct GoldenTtsArgs {
    /// Qwen3 call-center corpus manifest JSON file.
    pub manifest: PathBuf,

    /// Directory where one 16 kHz mono PCM WAV is written per sample.
    #[arg(
        long,
        default_value = "bins/telnyx-gateway/corpus/generated/qwen3-call-center"
    )]
    pub output_dir: PathBuf,

    /// Regenerate WAVs that already exist.
    #[arg(long)]
    pub force: bool,

    /// Generate only the first N samples for smoke tests.
    #[arg(long)]
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, Args)]
pub struct AsrGoldenAbArgs {
    /// Qwen3 call-center corpus manifest JSON file.
    pub manifest: PathBuf,

    /// Directory containing source WAVs from golden-tts.
    #[arg(
        long,
        default_value = "bins/telnyx-gateway/corpus/generated/qwen3-call-center"
    )]
    pub audio_dir: PathBuf,

    /// ASR backend to run. Repeat to compare multiple backends.
    #[arg(long, value_enum)]
    pub backend: Vec<ReplayBackendArg>,

    /// Telnyx audio spec to score. Repeat to compare multiple specs.
    #[arg(long, value_enum)]
    pub codec: Vec<GoldenCodecArg>,

    /// Audio chunk size to feed into each recognizer.
    #[arg(long, default_value_t = 20)]
    pub chunk_ms: u32,

    /// Trailing silence to feed before finish so streaming decoders flush final tokens.
    #[arg(long, default_value_t = DEFAULT_TRAILING_SILENCE_PAD_MS)]
    pub trailing_silence_pad_ms: u32,

    /// Score only the first N samples for smoke tests.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Optional JSON report path for the full matrix.
    #[arg(long)]
    pub output_json: Option<PathBuf>,
}

impl AsrGoldenAbArgs {
    pub fn selected_backends(&self) -> Vec<ReplayBackendArg> {
        if self.backend.is_empty() {
            vec![
                ReplayBackendArg::SherpaZipformer2023,
                ReplayBackendArg::SherpaZipformerKroko2025,
                ReplayBackendArg::Moonshine,
                ReplayBackendArg::Whisper,
            ]
        } else {
            self.backend.clone()
        }
    }

    pub fn selected_codecs(&self) -> Vec<GoldenCodecArg> {
        if self.codec.is_empty() {
            vec![GoldenCodecArg::L16_16k, GoldenCodecArg::Pcmu8k]
        } else {
            self.codec.clone()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum GoldenCodecArg {
    #[value(name = "l16-16k")]
    L16_16k,
    #[value(name = "pcmu-8k")]
    Pcmu8k,
}

impl GoldenCodecArg {
    pub fn label(self) -> &'static str {
        match self {
            Self::L16_16k => "L16-16k",
            Self::Pcmu8k => "PCMU-8k",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ReplayBackendArg {
    Auto,
    Echo,
    Sherpa,
    #[value(name = "sherpa-zipformer-2023")]
    SherpaZipformer2023,
    #[value(name = "sherpa-zipformer-kroko-2025")]
    SherpaZipformerKroko2025,
    Moonshine,
    Whisper,
}

impl ReplayBackendArg {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Echo => "echo",
            Self::Sherpa => "sherpa",
            Self::SherpaZipformer2023 => "sherpa-zipformer-en-2023-06-26",
            Self::SherpaZipformerKroko2025 => "sherpa-zipformer-en-kroko-2025-08-06",
            Self::Moonshine => "moonshine-streaming-en",
            Self::Whisper => "whisper-base-en",
        }
    }
}
