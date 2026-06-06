use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use motlie_driver::CommandEngine;
use motlie_telnyx_gateway::adapter::{
    default_artifact_root, AsrRegistry, EchoAsrFactory, SharedAsrFactory, SharedAsrRegistry,
    SherpaAsrArtifact,
};
use motlie_telnyx_gateway::call_control::TelnyxClient;
use motlie_telnyx_gateway::cli::{Cli, CliCommand, ReplayBackendArg};
use motlie_telnyx_gateway::conversation::{default_conversation_handler, ConversationRuntime};
use motlie_telnyx_gateway::media::SharedMediaRegistry;
use motlie_telnyx_gateway::operator::commands::{GatewayCommand, GatewayContext};
use motlie_telnyx_gateway::operator::script::run_repl_file;
use motlie_telnyx_gateway::operator::state::{shared_state, LogLevel};
use motlie_telnyx_gateway::replay::ReplayBackend;
use motlie_telnyx_gateway::serve::{serve, AppServices};
#[cfg(not(feature = "piper"))]
use motlie_telnyx_gateway::tts::unavailable_registry;
use motlie_telnyx_gateway::tts::SharedTtsRegistry;
#[cfg(feature = "piper")]
use motlie_telnyx_gateway::tts::{SharedTtsFactory, TtsRegistry};
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

    match cli.command.as_ref() {
        Some(CliCommand::ReplayCapture(args)) => {
            let backend = args.backend;
            let report = motlie_telnyx_gateway::replay::replay_capture(
                args,
                backend.label(),
                build_asr_factory(&cli, backend),
            )
            .await?;
            print_replay_report(&report);
            return Ok(());
        }
        Some(CliCommand::ReplayCorpus(args)) => {
            let backends = args
                .selected_backends()
                .into_iter()
                .map(|backend| {
                    ReplayBackend::new(backend.label(), build_asr_factory(&cli, backend))
                })
                .collect();
            let report = motlie_telnyx_gateway::replay::replay_corpus(args, backends).await?;
            print_corpus_report(&report);
            return Ok(());
        }
        Some(CliCommand::GoldenTts(args)) => {
            let artifact_root = default_artifact_root(cli.asr_artifact_root.clone());
            let report = motlie_telnyx_gateway::golden_ab::generate_tts_wavs(
                args,
                artifact_root,
                !cli.no_asr_download,
            )
            .await?;
            print_golden_tts_report(&report);
            return Ok(());
        }
        Some(CliCommand::AsrGoldenAb(args)) => {
            let backends = args
                .selected_backends()
                .into_iter()
                .map(|backend| {
                    ReplayBackend::new(backend.label(), build_asr_factory(&cli, backend))
                })
                .collect();
            let report = motlie_telnyx_gateway::golden_ab::run_golden_ab(args, backends).await?;
            print_golden_ab_report(&report);
            return Ok(());
        }
        None => {}
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
    let asr = build_live_asr_registry(&cli);
    let media = SharedMediaRegistry::default();
    let tts = build_tts_registry(&cli);
    let services = AppServices {
        state: state.clone(),
        telnyx: telnyx.clone(),
        asr,
        media: media.clone(),
        conversation: ConversationRuntime::new(
            telnyx.clone(),
            tts.clone(),
            default_conversation_handler(),
        ),
    };

    let server = tokio::spawn(serve(cli.bind, services));
    let context = GatewayContext::with_services(
        state.clone(),
        telnyx,
        media,
        tts,
        motlie_telnyx_gateway::adapter::LiveAsrBackend::default(),
    );
    let mut replay_engine = CommandEngine::<GatewayContext, GatewayCommand>::new(context.clone());

    if let Some(path) = &cli.load {
        let _ = run_repl_file(&mut replay_engine, path).await?;
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
        let tui_context = context.for_new_source();
        let mut engine = CommandEngine::<GatewayContext, GatewayCommand>::new(tui_context);
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

fn print_corpus_report(report: &motlie_telnyx_gateway::replay::CorpusReplayReport) {
    println!("manifest: {}", report.manifest_path);
    for entry in &report.entries {
        println!("entry: {}", entry.id);
        if let Some(description) = &entry.description {
            println!("  description: {description}");
        }
        println!("  capture_dir: {}", entry.capture_dir);
        println!("  wav: {}", entry.wav_path);
        println!("  codec: {}", entry.codec);
        println!("  media_sample_rate_hz: {}", entry.media_sample_rate_hz);
        println!("  direction: {}", entry.direction);
        if let Some(baseline) = &entry.baseline {
            println!("  baseline_backend: {}", baseline.backend);
            println!("  baseline_wer: {:.1}%", baseline.wer_percent);
            println!("  baseline_errors: {}", baseline.errors);
            println!("  baseline_reference_words: {}", baseline.reference_words);
            if let Some(source) = &baseline.source {
                println!("  baseline_source: {source}");
            }
        }
        if let Some(notes) = &entry.notes {
            println!("  notes: {notes}");
        }
        for replay in &entry.reports {
            print_replay_report_with_indent(replay, "  ");
        }
    }
}

fn build_tts_registry(cli: &Cli) -> SharedTtsRegistry {
    #[cfg(feature = "piper")]
    {
        let artifact_root = default_artifact_root(cli.asr_artifact_root.clone());
        let piper: SharedTtsFactory = Arc::new(motlie_telnyx_gateway::tts::PiperTtsFactory::new(
            artifact_root,
            !cli.no_asr_download,
        ));
        Arc::new(TtsRegistry::new(piper))
    }

    #[cfg(not(feature = "piper"))]
    {
        let _ = cli;
        unavailable_registry()
    }
}

fn print_replay_report(report: &motlie_telnyx_gateway::replay::ReplayReport) {
    print_replay_report_with_indent(report, "");
}

fn print_replay_report_with_indent(
    report: &motlie_telnyx_gateway::replay::ReplayReport,
    indent: &str,
) {
    println!("{indent}backend: {}", report.backend);
    println!("{indent}capture_dir: {}", report.capture_dir);
    println!("{indent}wav: {}", report.wav_path);
    println!("{indent}wav_sample_rate_hz: {}", report.wav_sample_rate_hz);
    println!("{indent}wav_channels: {}", report.wav_channels);
    println!("{indent}samples: {}", report.sample_count);
    println!("{indent}chunk_ms: {}", report.chunk_ms);
    println!("{indent}latency:");
    println!("{indent}  audio_ms: {}", report.latency.audio_ms);
    println!("{indent}  chunk_count: {}", report.latency.chunk_count);
    println!(
        "{indent}  trailing_silence_pad_ms: {}",
        report.latency.trailing_silence_pad_ms
    );
    println!(
        "{indent}  trailing_silence_pad_chunks: {}",
        report.latency.trailing_silence_pad_chunks
    );
    println!(
        "{indent}  ingest_total_ms: {}",
        report.latency.ingest_total_ms
    );
    println!(
        "{indent}  ingest_avg_ms: {:.1}",
        report.latency.ingest_avg_ms()
    );
    println!("{indent}  ingest_max_ms: {}", report.latency.ingest_max_ms);
    println!("{indent}  finish_ms: {}", report.latency.finish_ms);
    println!("{indent}  wall_ms: {}", report.latency.wall_ms);
    println!("{indent}transcript:");
    println!("{indent}{}", report.transcript);
    if let Some(wer) = &report.wer {
        println!("{indent}wer: {:.1}%", wer.rate() * 100.0);
        println!("{indent}reference_words: {}", wer.reference_words);
        println!("{indent}hypothesis_words: {}", wer.hypothesis_words);
        println!("{indent}substitutions: {}", wer.substitutions);
        println!("{indent}deletions: {}", wer.deletions);
        println!("{indent}insertions: {}", wer.insertions);
        println!("{indent}errors: {}", wer.errors);
        if !wer.errors_by_token.is_empty() {
            println!("{indent}token_errors:");
            for error in &wer.errors_by_token {
                match error {
                    motlie_telnyx_gateway::replay::WerTokenError::Substitution {
                        reference,
                        hypothesis,
                    } => println!("{indent}  S {reference} -> {hypothesis}"),
                    motlie_telnyx_gateway::replay::WerTokenError::Deletion { reference } => {
                        println!("{indent}  D {reference}");
                    }
                    motlie_telnyx_gateway::replay::WerTokenError::Insertion { hypothesis } => {
                        println!("{indent}  I {hypothesis}");
                    }
                }
            }
        }
    }
}

fn print_golden_tts_report(report: &motlie_telnyx_gateway::golden_ab::GoldenTtsReport) {
    println!("manifest: {}", report.manifest_path);
    println!("output_dir: {}", report.output_dir);
    println!("generated: {}", report.generated);
    println!("skipped: {}", report.skipped);
    for sample in &report.samples {
        println!(
            "sample: {} category={} status={} samples={} elapsed_ms={} wav={}",
            sample.id,
            sample.category,
            sample.status,
            sample.sample_count,
            sample.elapsed_ms,
            sample.wav_path
        );
    }
}

fn print_golden_ab_report(report: &motlie_telnyx_gateway::golden_ab::GoldenAbReport) {
    println!("manifest: {}", report.manifest_path);
    println!("audio_dir: {}", report.audio_dir);
    println!("chunk_ms: {}", report.chunk_ms);
    println!(
        "trailing_silence_pad_ms: {}",
        report.trailing_silence_pad_ms
    );
    println!("entries: {}", report.entries.len());
    for summary in &report.summaries {
        println!(
            "summary: backend={} codec={} category={} samples={} wer={:.1}% errors={}/{} tail_pad_avg_ms={:.1} ingest_avg_ms={:.1} finish_avg_ms={:.1} wall_avg_ms={:.1}",
            summary.backend,
            summary.codec,
            summary.category,
            summary.sample_count,
            summary.wer_percent,
            summary.errors,
            summary.reference_words,
            summary.trailing_silence_pad_avg_ms,
            summary.ingest_avg_ms,
            summary.finish_avg_ms,
            summary.wall_avg_ms
        );
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

fn build_live_asr_registry(cli: &Cli) -> SharedAsrRegistry {
    if std::env::var_os("MOTLIE_TELNYX_ECHO_ASR").is_some() {
        let echo: SharedAsrFactory = Arc::new(EchoAsrFactory);
        return Arc::new(AsrRegistry::new(echo.clone(), echo));
    }

    Arc::new(AsrRegistry::new(
        build_sherpa_asr_factory(cli, SherpaAsrArtifact::ZipformerEn20230626),
        build_sherpa_asr_factory(cli, SherpaAsrArtifact::ZipformerEnKroko20250806),
    ))
}

fn build_asr_factory(cli: &Cli, backend: ReplayBackendArg) -> SharedAsrFactory {
    match backend {
        ReplayBackendArg::Auto => build_auto_asr_factory(cli),
        ReplayBackendArg::Echo => Arc::new(EchoAsrFactory),
        ReplayBackendArg::Sherpa | ReplayBackendArg::SherpaZipformer2023 => {
            build_sherpa_asr_factory(cli, SherpaAsrArtifact::ZipformerEn20230626)
        }
        ReplayBackendArg::SherpaZipformerKroko2025 => {
            build_sherpa_asr_factory(cli, SherpaAsrArtifact::ZipformerEnKroko20250806)
        }
        ReplayBackendArg::Moonshine => build_moonshine_asr_factory(cli),
        ReplayBackendArg::Whisper => build_whisper_asr_factory(cli),
    }
}

fn build_auto_asr_factory(cli: &Cli) -> SharedAsrFactory {
    if std::env::var_os("MOTLIE_TELNYX_ECHO_ASR").is_some() {
        return Arc::new(EchoAsrFactory);
    }
    build_sherpa_asr_factory(cli, SherpaAsrArtifact::default())
}

#[cfg(feature = "sherpa")]
fn build_sherpa_asr_factory(cli: &Cli, artifact: SherpaAsrArtifact) -> SharedAsrFactory {
    let artifact_root = default_artifact_root(cli.asr_artifact_root.clone());
    Arc::new(
        motlie_telnyx_gateway::adapter::SherpaAsrFactory::with_artifact(
            artifact_root,
            !cli.no_asr_download,
            artifact,
        ),
    )
}

#[cfg(not(feature = "sherpa"))]
fn build_sherpa_asr_factory(_cli: &Cli, _artifact: SherpaAsrArtifact) -> SharedAsrFactory {
    Arc::new(motlie_telnyx_gateway::adapter::UnavailableAsrFactory::new(
        "gateway was built without a live ASR backend; rebuild with --features sherpa or use --backend echo for replay protocol testing",
    ))
}

#[cfg(feature = "moonshine")]
fn build_moonshine_asr_factory(cli: &Cli) -> SharedAsrFactory {
    let artifact_root = default_artifact_root(cli.asr_artifact_root.clone());
    Arc::new(motlie_telnyx_gateway::adapter::MoonshineAsrFactory::new(
        artifact_root,
        !cli.no_asr_download,
    ))
}

#[cfg(not(feature = "moonshine"))]
fn build_moonshine_asr_factory(_cli: &Cli) -> SharedAsrFactory {
    Arc::new(motlie_telnyx_gateway::adapter::UnavailableAsrFactory::new(
        "gateway was built without Moonshine ASR; rebuild with --features moonshine or --features golden-ab",
    ))
}

#[cfg(feature = "whisper")]
fn build_whisper_asr_factory(cli: &Cli) -> SharedAsrFactory {
    let artifact_root = default_artifact_root(cli.asr_artifact_root.clone());
    Arc::new(motlie_telnyx_gateway::adapter::WhisperAsrFactory::new(
        artifact_root,
        !cli.no_asr_download,
    ))
}

#[cfg(not(feature = "whisper"))]
fn build_whisper_asr_factory(_cli: &Cli) -> SharedAsrFactory {
    Arc::new(motlie_telnyx_gateway::adapter::UnavailableAsrFactory::new(
        "gateway was built without Whisper ASR; rebuild with --features whisper or --features golden-ab",
    ))
}
