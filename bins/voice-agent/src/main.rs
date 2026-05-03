use std::ffi::OsStr;
use std::fs::{self, File};
use std::future::Future;
use std::io::{self, Cursor, Read};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use motlie_model::typed::{
    AudioBuf, BatchTranscriber, BufferedSpeechSynthesizer, BufferedVoiceCloneSynthesizer,
    CloneReference, Mono, StreamingTranscriber, SynthesisRequest, TranscriptionSession,
};
use motlie_model::{
    ArtifactPolicy, BundleHandle, Capabilities, CapabilityDescriptor, CapabilityKind, ModelError,
    SpeechParams, StartOptions, TranscriptSegment, TranscriptionParams,
};
use motlie_model_moonshine::MoonshineHandle;
use motlie_model_piper::PiperHandle;
use motlie_model_qwen3_tts_cpp::Qwen3TtsCppHandle;
use motlie_model_sherpa_onnx::SherpaOnnxHandle;
use motlie_model_whisper_cpp::WhisperCppHandle;
use motlie_models::asr::{moonshine_streaming_en, sherpa_onnx_streaming_en, whisper_base_en};
use motlie_models::tts::{piper_en_us_ljspeech_medium, qwen3_tts_cpp};
use motlie_models::{
    AsrModels, Catalog, LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX, TtsModels,
    download_bundle_artifacts,
};
use motlie_voice::pipeline::convert::{decode_samples_to_f32, downmix_to_mono, f32_to_i16_clamped};
use motlie_voice::pipeline::resample::{LinearInterpolator, Resampler};
use motlie_voice::wav::{StreamingWavWriter, WavSample, decode_streaming_wav_to_f32};

const ASR_TARGET_SAMPLE_RATE_HZ: u32 = 16_000;
const STREAMING_ASR_CHUNK_SAMPLES: usize = 3_200;
const QWEN_REFERENCE_SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Parser)]
#[command(
    author = "@codex-tts",
    version,
    about = "Typed voice-agent orchestration for Motlie voice skills"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Speak(SpeakArgs),
    Listen(ListenArgs),
    Turn(TurnArgs),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, ValueEnum)]
enum TtsBackend {
    Piper,
    Qwen3cpp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, ValueEnum)]
enum AsrBackend {
    Whisper,
    Sherpa,
    Moonshine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, ValueEnum)]
enum ListenInputFormat {
    Wav,
    RawS16le,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TtsExecutionMode {
    Buffered,
    Streaming,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AsrExecutionMode {
    Batch,
    StreamingFinalOnly,
    StreamingWithPartials,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EndpointKind {
    Local,
    Ssh,
}

#[derive(Debug, Args)]
struct SpeakArgs {
    #[arg(long, value_enum, default_value = "piper")]
    backend: TtsBackend,
    #[arg(long)]
    endpoint: Option<String>,
    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    wav: Option<PathBuf>,
    #[arg(long)]
    voice: Option<String>,
    #[arg(long)]
    reference_audio: Option<PathBuf>,
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Args)]
struct ListenArgs {
    #[arg(long, value_enum, default_value = "whisper")]
    backend: AsrBackend,
    #[arg(long)]
    endpoint: Option<String>,
    #[arg(long)]
    seconds: Option<u32>,
    #[arg(long)]
    wav: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "wav")]
    input_format: ListenInputFormat,
    #[arg(long)]
    partials: bool,
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Args)]
struct TurnArgs {
    #[arg(long, value_enum, default_value = "piper")]
    tts_backend: TtsBackend,
    #[arg(long, value_enum, default_value = "whisper")]
    asr_backend: AsrBackend,
    #[arg(long)]
    playback_endpoint: Option<String>,
    #[arg(long)]
    capture_endpoint: Option<String>,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 8)]
    seconds: u32,
    #[arg(long)]
    voice: Option<String>,
    #[arg(long)]
    reference_audio: Option<PathBuf>,
    #[arg(long)]
    quiet: bool,
}

#[derive(Clone, Debug)]
struct EndpointConfig {
    kind: EndpointKind,
    ssh_target: Option<String>,
    play_cmd: String,
    record_cmd: RecordCommand,
}

#[derive(Debug)]
struct VoiceConfig {
    skill_root: PathBuf,
}

#[derive(Clone, Debug)]
enum RecordCommand {
    SoxCapture { program: String },
    RemoteSoxProbe,
}

impl RecordCommand {
    fn render(&self, seconds: Option<u32>) -> String {
        let trim_suffix = seconds
            .map(|value| format!(" trim 0 {value}"))
            .unwrap_or_default();
        match self {
            Self::SoxCapture { program } => format!("{program} -q -t wav -{trim_suffix}"),
            Self::RemoteSoxProbe => remote_record_command(&trim_suffix),
        }
    }
}

enum OutputTarget {
    WavFile(PathBuf),
    Playback(EndpointConfig),
}

enum OutputSink<S: WavSample> {
    File {
        writer: hound::WavWriter<std::io::BufWriter<File>>,
        _sample: std::marker::PhantomData<S>,
    },
    Playback {
        writer: StreamingWavWriter<ChildStdin, S>,
        child: Child,
    },
}

struct QuietStderrGuard {
    saved_stderr_fd: i32,
    _devnull: File,
}

enum StartedTtsHandle {
    Buffered(BufferedTtsHandle),
}

enum BufferedTtsHandle {
    Piper(PiperHandle),
    Qwen3cpp(Qwen3TtsCppHandle),
}

enum StartedAsrHandle {
    Batch(BatchAsrHandle),
    Streaming(StreamingAsrHandle),
}

enum BatchAsrHandle {
    Whisper(WhisperCppHandle),
}

enum StreamingAsrHandle {
    Sherpa(SherpaOnnxHandle),
    Moonshine(MoonshineHandle),
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = VoiceConfig::new(resolve_skill_root()?);

    match cli.command {
        Commands::Speak(args) => run_speak(&config, &args).await,
        Commands::Listen(args) => run_listen(&config, &args).await.map(|transcript| {
            if !transcript.is_empty() {
                println!("{transcript}");
            }
        }),
        Commands::Turn(args) => run_turn(&config, &args).await,
    }
}

fn resolve_skill_root() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MOTLIE_VOICE_SKILL_ROOT") {
        let root = PathBuf::from(path);
        if root.is_dir() {
            return Ok(root);
        }
        bail!(
            "MOTLIE_VOICE_SKILL_ROOT points to missing directory '{}'",
            root.display()
        );
    }

    let exe = std::env::current_exe().context("resolve current voice-agent executable path")?;
    if let Some(root) = skill_root_from_path(&exe) {
        return Ok(root);
    }

    bail!(
        "failed to resolve .agents/skills/voice from current executable '{}' and no MOTLIE_VOICE_SKILL_ROOT override was set",
        exe.display()
    );
}

fn skill_root_from_path(path: &Path) -> Option<PathBuf> {
    path.ancestors().find_map(|ancestor| {
        let is_voice = ancestor.file_name() == Some(OsStr::new("voice"));
        let is_skills = ancestor.parent().and_then(Path::file_name) == Some(OsStr::new("skills"));
        let is_agents = ancestor
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            == Some(OsStr::new(".agents"));
        if is_voice && is_skills && is_agents {
            Some(ancestor.to_path_buf())
        } else {
            None
        }
    })
}

impl VoiceConfig {
    fn new(skill_root: PathBuf) -> Self {
        Self { skill_root }
    }

    fn artifact_root(&self) -> PathBuf {
        self.skill_root.join("artifacts/hf-cache")
    }

    fn reference_root(&self) -> PathBuf {
        self.skill_root.join("speak/references/voices")
    }

    fn endpoint(&self, endpoint_name: Option<&str>) -> Result<EndpointConfig> {
        match endpoint_name {
            None | Some("local") => self.local_endpoint(),
            Some(endpoint) => self.remote_endpoint(endpoint),
        }
    }

    fn local_endpoint(&self) -> Result<EndpointConfig> {
        Ok(EndpointConfig {
            kind: EndpointKind::Local,
            ssh_target: None,
            play_cmd: local_play_command()?,
            record_cmd: local_record_command()?,
        })
    }

    fn remote_endpoint(&self, endpoint_name: &str) -> Result<EndpointConfig> {
        let ssh_target = endpoint_name
            .strip_prefix("ssh:")
            .unwrap_or(endpoint_name)
            .trim();
        if ssh_target.is_empty() {
            bail!("remote endpoint must be provided as ssh:<host> or <host>");
        }
        Ok(EndpointConfig {
            kind: EndpointKind::Ssh,
            ssh_target: Some(ssh_target.to_string()),
            play_cmd: remote_play_command(),
            record_cmd: RecordCommand::RemoteSoxProbe,
        })
    }
}

impl<S: WavSample> OutputSink<S> {
    fn new(target: OutputTarget, sample_rate_hz: u32) -> Result<Self> {
        match target {
            OutputTarget::WavFile(path) => {
                let writer = hound::WavWriter::create(
                    &path,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: sample_rate_hz,
                        bits_per_sample: S::BITS_PER_SAMPLE,
                        sample_format: S::SAMPLE_FORMAT,
                    },
                )
                .with_context(|| format!("failed to create wav file '{}'", path.display()))?;
                Ok(Self::File {
                    writer,
                    _sample: std::marker::PhantomData,
                })
            }
            OutputTarget::Playback(endpoint) => {
                let mut child = playback_command(&endpoint)?
                    .spawn()
                    .context("spawn playback command")?;
                let stdin = child.stdin.take().context("take playback stdin")?;
                let writer = StreamingWavWriter::new(stdin, sample_rate_hz, 1)
                    .context("start playback wav stream")?;
                Ok(Self::Playback { writer, child })
            }
        }
    }

    fn write_samples(&mut self, samples: &[S]) -> Result<()> {
        match self {
            Self::File { writer, .. } => {
                for &sample in samples {
                    S::write_to_hound(writer, sample).context("write wav file sample")?;
                }
            }
            Self::Playback { writer, .. } => {
                writer
                    .write_chunk(samples)
                    .context("write playback wav chunk")?;
            }
        }
        Ok(())
    }

    fn finalize(self) -> Result<()> {
        match self {
            Self::File { writer, .. } => writer.finalize().context("finalize wav file"),
            Self::Playback { writer, mut child } => {
                writer.finalize().context("finalize playback wav stream")?;
                let status = child.wait().context("wait for playback command")?;
                ensure_success(status, "playback command")
            }
        }
    }
}

impl QuietStderrGuard {
    fn maybe_enable(quiet: bool) -> io::Result<Option<Self>> {
        if quiet {
            Self::enable().map(Some)
        } else {
            Ok(None)
        }
    }

    fn enable() -> io::Result<Self> {
        let devnull = File::options().write(true).open("/dev/null")?;
        let stderr_fd = std::io::stderr().as_raw_fd();

        let saved_stderr_fd = unsafe { libc::dup(stderr_fd) };
        if saved_stderr_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        if unsafe { libc::dup2(devnull.as_raw_fd(), stderr_fd) } < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(saved_stderr_fd);
            }
            return Err(error);
        }

        Ok(Self {
            saved_stderr_fd,
            _devnull: devnull,
        })
    }
}

impl Drop for QuietStderrGuard {
    fn drop(&mut self) {
        let stderr_fd = std::io::stderr().as_raw_fd();
        unsafe {
            libc::dup2(self.saved_stderr_fd, stderr_fd);
            libc::close(self.saved_stderr_fd);
        }
    }
}

async fn run_speak(config: &VoiceConfig, args: &SpeakArgs) -> Result<()> {
    let target = match &args.wav {
        Some(path) => OutputTarget::WavFile(path.clone()),
        None => OutputTarget::Playback(config.endpoint(args.endpoint.as_deref())?),
    };
    let text = resolve_text(args.text.clone())?;
    let requested_model = selected_tts_model(args.backend);
    let reference = resolve_reference_audio(
        config,
        args.voice.as_deref(),
        args.reference_audio.as_deref(),
    )?
    .map(load_clone_reference)
    .transpose()?;
    if reference.is_some()
        && !requested_model
            .descriptor()
            .capabilities
            .supports(CapabilityKind::VoiceClone)
    {
        bail!(
            "selected TTS model '{}' does not advertise voice-clone support; omit --voice/--reference-audio or choose a clone-capable model",
            requested_model.as_str()
        );
    }

    let _quiet_stderr =
        QuietStderrGuard::maybe_enable(args.quiet).context("failed to enable quiet stderr mode")?;
    match start_selected_tts(config, args.quiet, requested_model).await? {
        StartedTtsHandle::Buffered(handle) => {
            speak_with_buffered_tts(handle, text, target, reference).await
        }
    }
}

async fn speak_with_buffered_tts(
    handle: BufferedTtsHandle,
    text: String,
    target: OutputTarget,
    reference: Option<CloneReference<QWEN_REFERENCE_SAMPLE_RATE_HZ, Mono>>,
) -> Result<()> {
    match handle {
        BufferedTtsHandle::Piper(handle) => {
            if reference.is_some() {
                bail!("Piper does not implement buffered voice-clone synthesis");
            }
            let body_result: Result<()> = async {
                let audio = handle
                    .synthesize_buffered(SynthesisRequest {
                        text,
                        params: SpeechParams::default(),
                    })
                    .await
                    .context("run Piper buffered synthesis")?;
                let mut sink = OutputSink::<i16>::new(target, 22_050)?;
                sink.write_samples(audio.samples())?;
                sink.finalize()
            }
            .await;
            finish_with_shutdown(body_result, handle.shutdown().await)
        }
        BufferedTtsHandle::Qwen3cpp(handle) => {
            let body_result: Result<()> = async {
                let request = SynthesisRequest {
                    text,
                    params: SpeechParams::default(),
                };
                let audio = match reference {
                    Some(reference) => handle
                        .synthesize_with_reference_buffered(request, reference)
                        .await
                        .context("run qwen3-tts.cpp buffered voice-clone synthesis")?,
                    None => handle
                        .synthesize_buffered(request)
                        .await
                        .context("run qwen3-tts.cpp buffered synthesis")?,
                };

                let mut sink = OutputSink::<i16>::new(target, 24_000)?;
                let samples = f32_to_i16_clamped(audio.samples());
                sink.write_samples(&samples)?;
                sink.finalize()
            }
            .await;
            finish_with_shutdown(body_result, handle.shutdown().await)
        }
    }
}

async fn run_listen(config: &VoiceConfig, args: &ListenArgs) -> Result<String> {
    let requested_model = selected_asr_model(args.backend);
    let execution_mode = asr_execution_mode(&requested_model.descriptor().capabilities)
        .with_context(|| {
            format!(
                "resolve ASR execution mode for '{}'",
                requested_model.as_str()
            )
        })?;
    if args.partials && execution_mode != AsrExecutionMode::StreamingWithPartials {
        bail!(
            "--partials requires a streaming ASR model with partial-update support; selected '{}' does not advertise that contract",
            requested_model.as_str()
        );
    }

    let _quiet_stderr =
        QuietStderrGuard::maybe_enable(args.quiet).context("failed to enable quiet stderr mode")?;
    let started = start_selected_asr(config, args.quiet, requested_model).await?;

    if args.input_format == ListenInputFormat::RawS16le {
        let path = args
            .wav
            .as_ref()
            .context("--input-format raw-s16le requires --wav <path-or-fifo>")?;
        return match started {
            StartedAsrHandle::Batch(_) => bail!(
                "--input-format raw-s16le requires a streaming ASR model; selected '{}' is batch-only",
                requested_model.as_str()
            ),
            StartedAsrHandle::Streaming(StreamingAsrHandle::Sherpa(handle)) => {
                transcribe_streaming_live_raw(handle, path, args.partials).await
            }
            StartedAsrHandle::Streaming(StreamingAsrHandle::Moonshine(handle)) => {
                transcribe_streaming_live_raw(handle, path, args.partials).await
            }
        };
    }

    let bytes = match &args.wav {
        Some(path) => fs::read(path).with_context(|| format!("read wav '{}'", path.display()))?,
        None => {
            let endpoint = config.endpoint(args.endpoint.as_deref())?;
            capture_wav_bytes(&endpoint, args.seconds)?
        }
    };

    match started {
        StartedAsrHandle::Batch(BatchAsrHandle::Whisper(handle)) => {
            let audio = decode_wav_bytes_to_f32_mono16k(&bytes)?;
            transcribe_batch(handle, audio).await
        }
        StartedAsrHandle::Streaming(StreamingAsrHandle::Sherpa(handle)) => {
            let audio = decode_wav_bytes_to_i16_mono16k(&bytes)?;
            transcribe_streaming(handle, audio, args.partials).await
        }
        StartedAsrHandle::Streaming(StreamingAsrHandle::Moonshine(handle)) => {
            let audio = decode_wav_bytes_to_i16_mono16k(&bytes)?;
            transcribe_streaming(handle, audio, args.partials).await
        }
    }
}

async fn run_turn(config: &VoiceConfig, args: &TurnArgs) -> Result<()> {
    let speak_args = SpeakArgs {
        backend: args.tts_backend,
        endpoint: args.playback_endpoint.clone(),
        text: Some(args.prompt.clone()),
        wav: None,
        voice: args.voice.clone(),
        reference_audio: args.reference_audio.clone(),
        quiet: args.quiet,
    };
    run_speak(config, &speak_args).await?;

    let listen_args = ListenArgs {
        backend: args.asr_backend,
        endpoint: args.capture_endpoint.clone(),
        seconds: Some(args.seconds),
        wav: None,
        input_format: ListenInputFormat::Wav,
        partials: false,
        quiet: args.quiet,
    };
    let transcript = run_listen(config, &listen_args).await?;
    if !transcript.is_empty() {
        println!("{transcript}");
    }
    Ok(())
}

async fn transcribe_batch<H>(
    handle: H,
    audio: AudioBuf<f32, ASR_TARGET_SAMPLE_RATE_HZ, Mono>,
) -> Result<String>
where
    H: BundleHandle + BatchTranscriber<Input = AudioBuf<f32, ASR_TARGET_SAMPLE_RATE_HZ, Mono>>,
{
    let body_result: Result<String> = async {
        let update = handle
            .transcribe(
                audio,
                TranscriptionParams {
                    language: Some("en".into()),
                    emit_partials: false,
                },
            )
            .await
            .context("transcribe captured audio with batch ASR")?;
        Ok(render_plain_transcript(&update.segments).unwrap_or_default())
    }
    .await;
    finish_with_shutdown(body_result, handle.shutdown().await)
}

async fn transcribe_streaming<H>(
    handle: H,
    audio: AudioBuf<i16, ASR_TARGET_SAMPLE_RATE_HZ, Mono>,
    partials: bool,
) -> Result<String>
where
    H: BundleHandle + StreamingTranscriber<Input = AudioBuf<i16, ASR_TARGET_SAMPLE_RATE_HZ, Mono>>,
{
    let body_result: Result<String> = async {
        let mut session = handle
            .open_session(TranscriptionParams {
                language: Some("en".into()),
                emit_partials: partials,
            })
            .await
            .context("open streaming ASR session")?;

        let mut final_segments = Vec::new();
        for chunk in audio.into_samples().chunks(STREAMING_ASR_CHUNK_SAMPLES) {
            if let Some(update) = session
                .ingest(AudioBuf::new(chunk.to_vec()))
                .await
                .context("ingest audio chunk into streaming ASR")?
            {
                if partials {
                    print_segment_events(&update.segments);
                } else {
                    final_segments.extend(
                        update
                            .segments
                            .into_iter()
                            .filter(|segment| segment.final_segment),
                    );
                }
            }
        }

        let final_update = session
            .finish()
            .await
            .context("finish streaming ASR session")?;
        if partials {
            print_segment_events(&final_update.segments);
        } else {
            final_segments.extend(final_update.segments);
        }

        Ok(render_plain_transcript(&final_segments).unwrap_or_default())
    }
    .await;
    finish_with_shutdown(body_result, handle.shutdown().await)
}

async fn transcribe_streaming_live_raw<H>(handle: H, path: &Path, partials: bool) -> Result<String>
where
    H: BundleHandle + StreamingTranscriber<Input = AudioBuf<i16, ASR_TARGET_SAMPLE_RATE_HZ, Mono>>,
{
    let path = path.to_path_buf();
    let body_result: Result<String> = async {
        let mut session = handle
            .open_session(TranscriptionParams {
                language: Some("en".into()),
                emit_partials: partials,
            })
            .await
            .context("open streaming ASR session")?;

        let file = File::open(&path)
            .with_context(|| format!("open raw PCM stream '{}'", path.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let mut pending = Vec::new();
        let mut raw_buf = vec![0_u8; STREAMING_ASR_CHUNK_SAMPLES * 2];
        let mut final_segments = Vec::new();

        loop {
            let read = reader
                .read(&mut raw_buf)
                .with_context(|| format!("read raw PCM stream '{}'", path.display()))?;
            if read == 0 {
                break;
            }
            pending.extend_from_slice(&raw_buf[..read]);

            let complete_bytes = pending.len() - (pending.len() % 2);
            if complete_bytes == 0 {
                continue;
            }

            let samples = pending[..complete_bytes]
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect::<Vec<_>>();
            pending.drain(..complete_bytes);

            if let Some(update) = session
                .ingest(AudioBuf::new(samples))
                .await
                .context("ingest raw PCM chunk into streaming ASR")?
            {
                if partials {
                    print_segment_events(&update.segments);
                } else {
                    final_segments.extend(
                        update
                            .segments
                            .into_iter()
                            .filter(|segment| segment.final_segment),
                    );
                }
            }
        }

        if !pending.is_empty() {
            bail!(
                "raw PCM stream '{}' ended with {} trailing byte(s); expected 16-bit samples",
                path.display(),
                pending.len()
            );
        }

        let final_update = session
            .finish()
            .await
            .context("finish streaming ASR session")?;
        if partials {
            print_segment_events(&final_update.segments);
        } else {
            final_segments.extend(final_update.segments);
        }

        Ok(render_plain_transcript(&final_segments).unwrap_or_default())
    }
    .await;
    finish_with_shutdown(body_result, handle.shutdown().await)
}

fn selected_tts_model(backend: TtsBackend) -> TtsModels {
    match backend {
        TtsBackend::Piper => TtsModels::PiperEnUsLjspeechMedium,
        TtsBackend::Qwen3cpp => TtsModels::Qwen3TtsCpp0_6B,
    }
}

fn selected_asr_model(backend: AsrBackend) -> AsrModels {
    match backend {
        AsrBackend::Whisper => AsrModels::WhisperBaseEn,
        AsrBackend::Sherpa => AsrModels::SherpaOnnxStreamingEn,
        AsrBackend::Moonshine => AsrModels::MoonshineStreamingEn,
    }
}

fn tts_execution_mode(capabilities: &Capabilities) -> Result<TtsExecutionMode> {
    let descriptor = capabilities
        .descriptor_for(CapabilityKind::Speech)
        .context("selected model does not advertise speech synthesis capability")?;
    if descriptor == &CapabilityDescriptor::speech_buffered() {
        Ok(TtsExecutionMode::Buffered)
    } else if descriptor == &CapabilityDescriptor::speech_stream() {
        Ok(TtsExecutionMode::Streaming)
    } else {
        bail!(
            "unsupported speech capability descriptor for voice-agent: {:?}",
            descriptor
        );
    }
}

fn asr_execution_mode(capabilities: &Capabilities) -> Result<AsrExecutionMode> {
    let descriptor = capabilities
        .descriptor_for(CapabilityKind::Transcription)
        .context("selected model does not advertise transcription capability")?;
    if descriptor == &CapabilityDescriptor::transcription_batch() {
        Ok(AsrExecutionMode::Batch)
    } else if descriptor == &CapabilityDescriptor::transcription_stream_final_only() {
        Ok(AsrExecutionMode::StreamingFinalOnly)
    } else if descriptor == &CapabilityDescriptor::transcription_stream_partial() {
        Ok(AsrExecutionMode::StreamingWithPartials)
    } else {
        bail!(
            "unsupported transcription capability descriptor for voice-agent: {:?}",
            descriptor
        );
    }
}

async fn start_selected_tts(
    config: &VoiceConfig,
    quiet: bool,
    model: TtsModels,
) -> Result<StartedTtsHandle> {
    match tts_execution_mode(&model.descriptor().capabilities)? {
        TtsExecutionMode::Buffered => match model {
            TtsModels::PiperEnUsLjspeechMedium => Ok(StartedTtsHandle::Buffered(
                BufferedTtsHandle::Piper(start_piper(config, quiet).await?),
            )),
            TtsModels::Qwen3TtsCpp0_6B => Ok(StartedTtsHandle::Buffered(
                BufferedTtsHandle::Qwen3cpp(start_qwen(config, quiet).await?),
            )),
            _ => bail!(
                "selected TTS model '{}' advertises buffered speech but has no buffered voice-agent adapter",
                model.as_str()
            ),
        },
        TtsExecutionMode::Streaming => bail!(
            "selected TTS model '{}' advertises streaming speech; voice-agent currently supports only buffered TTS composition",
            model.as_str()
        ),
    }
}

async fn start_selected_asr(
    config: &VoiceConfig,
    quiet: bool,
    model: AsrModels,
) -> Result<StartedAsrHandle> {
    match asr_execution_mode(&model.descriptor().capabilities)? {
        AsrExecutionMode::Batch => match model {
            AsrModels::WhisperBaseEn => Ok(StartedAsrHandle::Batch(BatchAsrHandle::Whisper(
                start_whisper(config, quiet).await?,
            ))),
            _ => bail!(
                "selected ASR model '{}' advertises batch transcription but has no batch voice-agent adapter",
                model.as_str()
            ),
        },
        AsrExecutionMode::StreamingFinalOnly | AsrExecutionMode::StreamingWithPartials => {
            match model {
                AsrModels::SherpaOnnxStreamingEn => Ok(StartedAsrHandle::Streaming(
                    StreamingAsrHandle::Sherpa(start_sherpa(config, quiet).await?),
                )),
                AsrModels::MoonshineStreamingEn => Ok(StartedAsrHandle::Streaming(
                    StreamingAsrHandle::Moonshine(start_moonshine(config, quiet).await?),
                )),
                _ => bail!(
                    "selected ASR model '{}' advertises streaming transcription but has no streaming voice-agent adapter",
                    model.as_str()
                ),
            }
        }
    }
}

async fn start_piper(config: &VoiceConfig, quiet: bool) -> Result<PiperHandle> {
    start_with_bootstrap(
        config,
        quiet,
        TtsModels::PiperEnUsLjspeechMedium.bundle_id(),
        "Piper",
        piper_en_us_ljspeech_medium::start_typed,
    )
    .await
}

async fn start_qwen(config: &VoiceConfig, quiet: bool) -> Result<Qwen3TtsCppHandle> {
    start_with_bootstrap(
        config,
        quiet,
        TtsModels::Qwen3TtsCpp0_6B.bundle_id(),
        "qwen3-tts.cpp",
        qwen3_tts_cpp::start_typed,
    )
    .await
}

async fn start_whisper(config: &VoiceConfig, quiet: bool) -> Result<WhisperCppHandle> {
    start_with_bootstrap(
        config,
        quiet,
        AsrModels::WhisperBaseEn.bundle_id(),
        "Whisper",
        whisper_base_en::start_typed,
    )
    .await
}

async fn start_sherpa(config: &VoiceConfig, quiet: bool) -> Result<SherpaOnnxHandle> {
    start_with_bootstrap(
        config,
        quiet,
        AsrModels::SherpaOnnxStreamingEn.bundle_id(),
        "Sherpa",
        sherpa_onnx_streaming_en::start_typed,
    )
    .await
}

async fn start_moonshine(config: &VoiceConfig, quiet: bool) -> Result<MoonshineHandle> {
    start_with_bootstrap(
        config,
        quiet,
        AsrModels::MoonshineStreamingEn.bundle_id(),
        "Moonshine",
        moonshine_streaming_en::start_typed,
    )
    .await
}

async fn start_with_bootstrap<H, Start, Fut>(
    config: &VoiceConfig,
    quiet: bool,
    bundle_id: motlie_model::BundleId,
    label: &str,
    start: Start,
) -> Result<H>
where
    Start: Fn(StartOptions) -> Fut,
    Fut: Future<Output = std::result::Result<H, ModelError>>,
{
    let artifact_root = config.artifact_root();
    match start(local_only_options(&artifact_root)).await {
        Ok(handle) => Ok(handle),
        Err(err) if missing_local_artifacts(&err) => {
            log_status(
                quiet,
                &format!(
                    "[voice-agent] downloading {label} artifacts into '{}'; please wait...",
                    artifact_root.display()
                ),
            );
            let catalog = Catalog::with_defaults();
            download_bundle_artifacts(&catalog, &bundle_id, &artifact_root)
                .map_err(anyhow::Error::from)
                .with_context(|| format!("download curated artifacts for bundle '{bundle_id}'"))?;
            start(local_only_options(&artifact_root))
                .await
                .map_err(anyhow::Error::from)
                .with_context(|| format!("start {label} after downloading artifacts"))
        }
        Err(err) => Err(anyhow::Error::from(err)).with_context(|| format!("start {label}")),
    }
}

fn local_only_options(artifact_root: &Path) -> StartOptions {
    StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root.to_path_buf(),
        }),
        ..Default::default()
    }
}

fn missing_local_artifacts(error: &ModelError) -> bool {
    match error {
        // @codex-tts 2026-04-24 -- Keep this predicate in sync with the curated
        // LocalOnly error prefix exported by motlie-models. Voice-skill bootstrap
        // depends on that shared prefix to decide when a missing local artifact
        // should trigger an explicit download step.
        ModelError::InvalidConfiguration(message) => {
            message.contains(LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX)
        }
        _ => false,
    }
}

fn resolve_text(text: Option<String>) -> Result<String> {
    match text {
        Some(text) => normalize_text(text),
        None => {
            let mut buffer = String::new();
            io::stdin()
                .read_to_string(&mut buffer)
                .context("failed to read synthesis text from stdin")?;
            normalize_text(buffer)
        }
    }
}

fn normalize_text(text: String) -> Result<String> {
    let trimmed = text.trim_end_matches(['\r', '\n']);
    if trimmed.is_empty() {
        bail!("synthesis text is empty");
    }
    Ok(trimmed.to_owned())
}

fn resolve_reference_audio(
    config: &VoiceConfig,
    voice_alias: Option<&str>,
    reference_audio: Option<&Path>,
) -> Result<Option<PathBuf>> {
    if voice_alias.is_none() && reference_audio.is_none() {
        return Ok(None);
    }
    if let Some(path) = reference_audio {
        if !path.is_file() {
            bail!("reference audio '{}' does not exist", path.display());
        }
        return Ok(Some(path.to_path_buf()));
    }

    let alias = voice_alias.context("missing voice alias")?;
    let normalized = normalize_voice_alias(alias);
    let candidate = config.reference_root().join(format!("{normalized}.wav"));
    let alt = config
        .reference_root()
        .join(format!("{}.wav", normalized.replace('-', "_")));
    if candidate.is_file() {
        return Ok(Some(candidate));
    }
    if alt.is_file() {
        return Ok(Some(alt));
    }
    bail!(
        "voice alias '{}' resolved to '{}' but no reference WAV exists",
        alias,
        candidate.display()
    )
}

fn normalize_voice_alias(alias: &str) -> String {
    let trimmed = alias.trim().to_ascii_lowercase();
    let without_prefix = trimmed.strip_prefix("voice of ").unwrap_or(&trimmed);
    without_prefix
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
}

fn load_clone_reference(
    path: PathBuf,
) -> Result<CloneReference<QWEN_REFERENCE_SAMPLE_RATE_HZ, Mono>> {
    let reader = hound::WavReader::open(&path)
        .with_context(|| format!("open reference wav '{}'", path.display()))?;
    let (spec, samples) =
        decode_samples_to_f32(reader).context("decode reference wav samples to f32")?;
    let mono = downmix_to_mono(&samples, spec.channels).context("downmix reference wav to mono")?;
    let resampled = LinearInterpolator
        .resample_f32(&mono, spec.sample_rate, QWEN_REFERENCE_SAMPLE_RATE_HZ)
        .context("resample reference wav to 16 kHz")?;
    Ok(CloneReference {
        audio: AudioBuf::new(resampled),
        transcript: None,
    })
}

fn decode_wav_bytes_to_f32_mono16k(
    bytes: &[u8],
) -> Result<AudioBuf<f32, ASR_TARGET_SAMPLE_RATE_HZ, Mono>> {
    let (spec, samples) =
        decode_streaming_wav_to_f32(Cursor::new(bytes)).context("failed to decode wav samples")?;
    let mono = downmix_to_mono(&samples, spec.channels).context("failed to downmix wav to mono")?;
    // @codex-tts 2026-04-24 -- This file-input path still uses linear interpolation.
    // Live remote-push capture already runs at 16 kHz raw PCM and avoids this resampler.
    // Keep the caveat documented in DESIGN/PLAN until motlie-voice grows an anti-aliased
    // resampler suitable for the batch file-input path.
    let resampled = LinearInterpolator
        .resample_f32(&mono, spec.sample_rate, ASR_TARGET_SAMPLE_RATE_HZ)
        .context("failed to resample wav to 16 kHz")?;
    ensure_audio_not_silent(&resampled)?;
    Ok(AudioBuf::new(resampled))
}

fn ensure_audio_not_silent(samples: &[f32]) -> Result<()> {
    if samples.is_empty() {
        bail!("captured audio was empty");
    }

    let mean_square = samples
        .iter()
        .map(|sample| {
            let sample = f64::from(*sample);
            sample * sample
        })
        .sum::<f64>()
        / samples.len() as f64;
    let rms = mean_square.sqrt();

    if rms <= 1.0e-4 {
        bail!(
            "captured audio was silent (RMS={rms:.6e}); likely the wrong input device or missing microphone permission on the capture host"
        );
    }

    Ok(())
}

fn decode_wav_bytes_to_i16_mono16k(
    bytes: &[u8],
) -> Result<AudioBuf<i16, ASR_TARGET_SAMPLE_RATE_HZ, Mono>> {
    let audio = decode_wav_bytes_to_f32_mono16k(bytes)?;
    Ok(AudioBuf::new(f32_to_i16_clamped(audio.samples())))
}

fn render_plain_transcript(segments: &[TranscriptSegment]) -> Option<String> {
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() { None } else { Some(text) }
}

fn print_segment_events(segments: &[TranscriptSegment]) {
    for segment in segments {
        let marker = if segment.final_segment {
            "[final]"
        } else {
            "[partial]"
        };
        println!(
            "{marker} [{:.2}s - {:.2}s] {}",
            segment.start_ms as f64 / 1000.0,
            segment.end_ms as f64 / 1000.0,
            segment.text
        );
    }
}

fn finish_with_shutdown<T>(
    body_result: Result<T>,
    shutdown_result: std::result::Result<(), ModelError>,
) -> Result<T> {
    let shutdown_result = shutdown_result.context("shutdown failed");
    match (body_result, shutdown_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Ok(())) => Err(error),
        (Err(body_error), Err(shutdown_error)) => {
            Err(body_error.context(format!("additionally, {shutdown_error:#}")))
        }
    }
}

fn capture_wav_bytes(endpoint: &EndpointConfig, seconds: Option<u32>) -> Result<Vec<u8>> {
    let mut capture_cmd = match endpoint.kind {
        EndpointKind::Local => {
            let mut cmd = Command::new("bash");
            cmd.arg("-lc").arg(endpoint.record_cmd.render(seconds));
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::inherit());
            cmd
        }
        EndpointKind::Ssh => {
            let ssh_target = endpoint
                .ssh_target
                .as_deref()
                .context("missing ssh target for ssh endpoint")?;
            let mut cmd = Command::new("ssh");
            cmd.arg(ssh_target).arg(endpoint.record_cmd.render(seconds));
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::inherit());
            cmd
        }
    };
    let output = capture_cmd.output().context("run capture command")?;
    ensure_success(output.status, "capture command")?;
    if output.stdout.is_empty() {
        bail!("capture command returned no audio");
    }
    Ok(output.stdout)
}

fn playback_command(endpoint: &EndpointConfig) -> Result<Command> {
    let mut cmd = match endpoint.kind {
        EndpointKind::Local => {
            let mut cmd = Command::new("bash");
            cmd.arg("-lc").arg(&endpoint.play_cmd);
            cmd
        }
        EndpointKind::Ssh => {
            let ssh_target = endpoint
                .ssh_target
                .as_deref()
                .context("missing ssh target for ssh endpoint")?;
            let mut cmd = Command::new("ssh");
            cmd.arg(ssh_target).arg(&endpoint.play_cmd);
            cmd
        }
    };
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::inherit());
    Ok(cmd)
}

fn local_play_command() -> Result<String> {
    if let Some(play) = resolve_preferred_command(
        "play",
        &[
            "/opt/homebrew/bin/play",
            "/usr/local/bin/play",
            "/opt/local/bin/play",
        ],
    ) {
        return Ok(format!("{play} -t wav -"));
    }
    if let Some(ffplay) = resolve_preferred_command(
        "ffplay",
        &[
            "/opt/homebrew/bin/ffplay",
            "/usr/local/bin/ffplay",
            "/opt/local/bin/ffplay",
        ],
    ) {
        return Ok(format!("{ffplay} -autoexit -nodisp -i pipe:0"));
    }
    bail!(
        "no local playback command found; install sox (`play`) or ffplay, or use --endpoint ssh:<host>"
    )
}

fn local_record_command() -> Result<RecordCommand> {
    let program = resolve_preferred_command(
        "rec",
        &[
            "/opt/homebrew/bin/rec",
            "/usr/local/bin/rec",
            "/opt/local/bin/rec",
        ],
    )
    .context(
        "no local recording command found; install sox (`rec`) or use --endpoint ssh:<host>",
    )?;
    Ok(RecordCommand::SoxCapture { program })
}

fn remote_play_command() -> String {
    remote_command_with_fallbacks(
        &[
            "/opt/homebrew/bin/play",
            "/usr/local/bin/play",
            "/opt/local/bin/play",
        ],
        "play",
        "-t wav -",
        Some((
            &[
                "/opt/homebrew/bin/ffplay",
                "/usr/local/bin/ffplay",
                "/opt/local/bin/ffplay",
            ],
            "ffplay",
            "-autoexit -nodisp -i pipe:0",
        )),
        "no remote playback command found (expected play or ffplay)",
    )
}

fn remote_record_command(trim_suffix: &str) -> String {
    remote_command_with_fallbacks(
        &[
            "/opt/homebrew/bin/rec",
            "/usr/local/bin/rec",
            "/opt/local/bin/rec",
        ],
        "rec",
        &format!("-q -t wav -{trim_suffix}"),
        None,
        "no remote recording command found (expected rec)",
    )
}

fn resolve_preferred_command(command: &str, known_paths: &[&str]) -> Option<String> {
    known_paths
        .iter()
        .find(|path| Path::new(path).is_file())
        .map(|path| (*path).to_string())
        .or_else(|| command_path(command))
}

fn command_path(command: &str) -> Option<String> {
    let output = Command::new("bash")
        .arg("-lc")
        .arg(format!("command -v {command} 2>/dev/null"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let resolved = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if resolved.is_empty() {
        None
    } else {
        Some(resolved)
    }
}

fn remote_command_with_fallbacks(
    preferred_paths: &[&str],
    fallback_name: &str,
    args: &str,
    alternate: Option<(&[&str], &str, &str)>,
    error_message: &str,
) -> String {
    let mut clauses = Vec::new();
    for path in preferred_paths {
        clauses.push((format!("[ -x {path} ]"), format!("exec {path} {args}")));
    }
    clauses.push((
        format!("command -v {fallback_name} >/dev/null 2>&1"),
        format!("exec {fallback_name} {args}"),
    ));

    if let Some((alternate_paths, alternate_name, alternate_args)) = alternate {
        for path in alternate_paths {
            clauses.push((
                format!("[ -x {path} ]"),
                format!("exec {path} {alternate_args}"),
            ));
        }
        clauses.push((
            format!("command -v {alternate_name} >/dev/null 2>&1"),
            format!("exec {alternate_name} {alternate_args}"),
        ));
    }

    let mut clauses = clauses.into_iter();
    let Some((first_condition, first_action)) = clauses.next() else {
        return format!("echo '{error_message}' >&2; exit 127");
    };

    let mut command = format!("if {first_condition}; then {first_action}");
    for (condition, action) in clauses {
        command.push_str(&format!("; elif {condition}; then {action}"));
    }
    command.push_str(&format!("; else echo '{error_message}' >&2; exit 127; fi"));
    command
}

fn ensure_success(status: std::process::ExitStatus, label: &str) -> Result<()> {
    if status.success() {
        Ok(())
    } else {
        bail!("{label} failed with status {status}")
    }
}

fn log_status(quiet: bool, message: &str) {
    if !quiet {
        eprintln!("{message}");
    }
}
