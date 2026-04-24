use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(author = "@codex-tts", version, about = "Typed voice-agent orchestration for Motlie speech examples")]
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
    record_cmd: String,
}

#[derive(Debug)]
struct VoiceConfig {
    repo_root: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct ExampleSpec {
    example_name: &'static str,
    feature_name: &'static str,
    cuda_feature: Option<&'static str>,
}

#[derive(Clone, Copy, Debug)]
enum BackendKind {
    Tts(TtsBackend),
    Asr(AsrBackend),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = repo_root()?;
    let config = VoiceConfig::new(repo_root);

    match cli.command {
        Commands::Speak(args) => run_speak(&config, &args),
        Commands::Listen(args) => run_listen(&config, &args).map(|transcript| {
            if !transcript.is_empty() {
                println!("{transcript}");
            }
        }),
        Commands::Turn(args) => run_turn(&config, &args),
    }
}

fn repo_root() -> Result<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .context("resolve repo root from bins/voice-agent")?;
    Ok(root)
}

impl VoiceConfig {
    fn new(repo_root: PathBuf) -> Self {
        Self { repo_root }
    }

    fn piper_artifact_root(&self) -> PathBuf {
        self.preferred_existing_root(&[
            self.default_repo_artifact_root(),
            self.sibling_motlie_artifact_root(),
            default_home_cache_root(),
        ])
    }

    fn qwen_artifact_root(&self) -> PathBuf {
        self.preferred_existing_root(&[
            PathBuf::from("/tmp/qwen3-tts-models"),
            self.repo_root.join("artifacts/models/qwen3-tts-models"),
            self.repo_root
                .parent()
                .unwrap_or(&self.repo_root)
                .join("motlie")
                .join("artifacts/models/qwen3-tts-models"),
        ])
    }

    fn whisper_artifact_root(&self) -> PathBuf {
        self.preferred_existing_root(&[
            self.default_repo_artifact_root(),
            self.sibling_motlie_artifact_root(),
            default_home_cache_root(),
        ])
    }

    fn sherpa_artifact_root(&self) -> PathBuf {
        self.preferred_existing_root(&[
            self.default_repo_artifact_root(),
            self.sibling_motlie_artifact_root(),
            default_home_cache_root(),
        ])
    }

    fn moonshine_artifact_root(&self) -> PathBuf {
        self.preferred_existing_root(&[
            self.default_repo_artifact_root(),
            self.sibling_motlie_artifact_root(),
            default_home_cache_root(),
        ])
    }

    fn reference_root(&self) -> PathBuf {
        self.repo_root
            .join(".agents/skills/voice/speak/references/voices")
    }

    fn default_repo_artifact_root(&self) -> PathBuf {
        self.repo_root.join("artifacts/models/hf-cache")
    }

    fn sibling_motlie_artifact_root(&self) -> PathBuf {
        self.repo_root
            .parent()
            .unwrap_or(&self.repo_root)
            .join("motlie")
            .join("artifacts/models/hf-cache")
    }

    fn preferred_existing_root(&self, candidates: &[PathBuf]) -> PathBuf {
        candidates
            .iter()
            .find(|path| path.exists())
            .cloned()
            .unwrap_or_else(|| candidates[0].clone())
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
            record_cmd: remote_record_command(),
        })
    }
}

fn run_speak(config: &VoiceConfig, args: &SpeakArgs) -> Result<()> {
    ensure_examples(config, &[BackendKind::Tts(args.backend)])?;

    let endpoint = config.endpoint(args.endpoint.as_deref())?;
    let example = tts_spec(args.backend);
    let example_binary = example_binary_path(config, example.example_name);
    let artifact_root = tts_artifact_root(config, args.backend);
    if !artifact_root.exists() {
        bail!("artifact root '{}' does not exist", artifact_root.display());
    }

    let mut cmd = Command::new(example_binary);
    cmd.arg("--artifact-root").arg(&artifact_root);
    if args.quiet {
        cmd.arg("--quiet");
    }
    if let Some(wav) = &args.wav {
        cmd.arg("--wav").arg(wav);
    }
    if let Some(reference_audio) = resolve_reference_audio(config, args.backend, args.voice.as_deref(), args.reference_audio.as_deref())? {
        cmd.arg("--reference-audio").arg(reference_audio);
    }
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());

    let mut child = cmd.spawn().context("spawn TTS example")?;
    write_text_input(child.stdin.take(), args.text.as_deref())?;

    if args.wav.is_some() {
        let status = child.wait()?;
        ensure_success(status, "tts example")?;
        return Ok(());
    }

    let tts_stdout = child.stdout.take().context("take TTS stdout")?;
    let mut sink = playback_command(&endpoint)?.spawn().context("spawn playback command")?;
    sink.stdin
        .take()
        .context("take playback stdin")?
        .write_all(&read_all(tts_stdout)?)?;
    let playback_status = sink.wait()?;
    ensure_success(playback_status, "playback command")?;
    let tts_status = child.wait()?;
    ensure_success(tts_status, "tts example")?;
    Ok(())
}

fn run_listen(config: &VoiceConfig, args: &ListenArgs) -> Result<String> {
    ensure_examples(config, &[BackendKind::Asr(args.backend)])?;

    let example = asr_spec(args.backend);
    let example_binary = example_binary_path(config, example.example_name);
    let artifact_root = asr_artifact_root(config, args.backend);
    if !artifact_root.exists() {
        bail!("artifact root '{}' does not exist", artifact_root.display());
    }

    let mut asr = Command::new(example_binary);
    asr.arg("--artifact-root").arg(&artifact_root);
    if args.quiet {
      asr.arg("--quiet");
    }
    if args.partials && args.backend != AsrBackend::Whisper {
      asr.arg("--partials");
    }

    if let Some(wav) = &args.wav {
        let bytes = fs::read(wav).with_context(|| format!("read wav {}", wav.display()))?;
        return run_asr_with_bytes(asr, &bytes);
    }

    let endpoint = config.endpoint(args.endpoint.as_deref())?;
    let capture_bytes = capture_wav_bytes(&endpoint, args.seconds)?;
    run_asr_with_bytes(asr, &capture_bytes)
}

fn run_turn(config: &VoiceConfig, args: &TurnArgs) -> Result<()> {
    let speak_args = SpeakArgs {
        backend: args.tts_backend,
        endpoint: args.playback_endpoint.clone(),
        text: Some(args.prompt.clone()),
        wav: None,
        voice: args.voice.clone(),
        reference_audio: args.reference_audio.clone(),
        quiet: args.quiet,
    };
    run_speak(config, &speak_args)?;

    let listen_args = ListenArgs {
        backend: args.asr_backend,
        endpoint: args.capture_endpoint.clone(),
        seconds: Some(args.seconds),
        wav: None,
        partials: false,
        quiet: args.quiet,
    };
    let transcript = run_listen(config, &listen_args)?;
    if !transcript.is_empty() {
        println!("{transcript}");
    }
    Ok(())
}

fn run_asr_with_bytes(mut asr: Command, bytes: &[u8]) -> Result<String> {
    asr.stdin(Stdio::piped());
    asr.stdout(Stdio::piped());
    let mut child = asr.spawn().context("spawn ASR example")?;
    let mut stdin = child.stdin.take().context("take ASR stdin")?;
    stdin.write_all(bytes)?;
    drop(stdin);
    let output = child.wait_with_output()?;
    ensure_success(output.status, "asr example")?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn capture_wav_bytes(endpoint: &EndpointConfig, seconds: Option<u32>) -> Result<Vec<u8>> {
    let mut capture_cmd = match endpoint.kind {
        EndpointKind::Local => {
            let mut cmd = Command::new("bash");
            cmd.arg("-lc").arg(with_trim(&endpoint.record_cmd, seconds));
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::inherit());
            cmd
        }
        EndpointKind::Ssh => {
            let ssh_target = endpoint.ssh_target.as_deref().context("missing ssh target for ssh endpoint")?;
            let mut cmd = Command::new("ssh");
            cmd.arg(ssh_target).arg(with_trim(&endpoint.record_cmd, seconds));
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::inherit());
            cmd
        }
    };
    let output = capture_cmd.output().context("run capture command")?;
    ensure_success(output.status, "capture command")?;
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
            let ssh_target = endpoint.ssh_target.as_deref().context("missing ssh target for ssh endpoint")?;
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
    if Path::new("/opt/homebrew/bin/play").is_file() {
        return Ok("/opt/homebrew/bin/play -t wav -".to_string());
    }
    if command_in_path("play") {
        return Ok("play -t wav -".to_string());
    }
    if command_in_path("ffplay") {
        return Ok("ffplay -autoexit -nodisp -i pipe:0".to_string());
    }
    bail!(
        "no local playback command found; install sox (`play`) or ffplay, or use --endpoint ssh:<host>"
    )
}

fn local_record_command() -> Result<String> {
    if Path::new("/opt/homebrew/bin/rec").is_file() {
        return Ok("/opt/homebrew/bin/rec -q -t wav -".to_string());
    }
    if command_in_path("rec") {
        return Ok("rec -q -t wav -".to_string());
    }
    bail!(
        "no local recording command found; install sox (`rec`) or use --endpoint ssh:<host>"
    )
}

fn remote_play_command() -> String {
    "if [ -x /opt/homebrew/bin/play ]; then exec /opt/homebrew/bin/play -t wav -; elif command -v play >/dev/null 2>&1; then exec play -t wav -; elif command -v ffplay >/dev/null 2>&1; then exec ffplay -autoexit -nodisp -i pipe:0; else echo 'no remote playback command found (expected play or ffplay)' >&2; exit 127; fi".to_string()
}

fn remote_record_command() -> String {
    "if [ -x /opt/homebrew/bin/rec ]; then exec /opt/homebrew/bin/rec -q -t wav -__MOTLIE_TRIM__; elif command -v rec >/dev/null 2>&1; then exec rec -q -t wav -__MOTLIE_TRIM__; else echo 'no remote recording command found (expected rec)' >&2; exit 127; fi".to_string()
}

fn command_in_path(command: &str) -> bool {
    Command::new("bash")
        .arg("-lc")
        .arg(format!("command -v {command} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn with_trim(record_cmd: &str, seconds: Option<u32>) -> String {
    let suffix = match seconds {
        Some(value) => format!(" trim 0 {value}"),
        None => String::new(),
    };
    if record_cmd.contains("__MOTLIE_TRIM__") {
        record_cmd.replace("__MOTLIE_TRIM__", &suffix)
    } else {
        format!("{record_cmd}{suffix}")
    }
}

fn ensure_examples(config: &VoiceConfig, backends: &[BackendKind]) -> Result<()> {
    let mut examples = BTreeSet::new();
    let mut features = BTreeSet::new();
    let mut cuda_features = BTreeSet::new();
    let mut needs_qwen = false;

    for backend in backends {
        let spec = match backend {
            BackendKind::Tts(tts) => {
                if *tts == TtsBackend::Qwen3cpp {
                    needs_qwen = true;
                }
                tts_spec(*tts)
            }
            BackendKind::Asr(asr) => asr_spec(*asr),
        };
        examples.insert(spec.example_name);
        features.insert(spec.feature_name);
        if let Some(cuda_feature) = spec.cuda_feature {
            if wants_cuda(config)? {
                cuda_features.insert(cuda_feature);
            }
        }
    }

    if needs_qwen {
        initialize_qwen_submodule(&config.repo_root)?;
    }

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&config.repo_root);
    cmd.arg("build").arg("-p").arg("motlie-models");
    cmd.arg("--release");
    for example in examples {
        cmd.arg("--example").arg(example);
    }
    cmd.arg("--no-default-features");

    let feature_list = features
        .into_iter()
        .chain(cuda_features)
        .collect::<Vec<_>>()
        .join(",");
    cmd.arg("--features").arg(feature_list);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let status = cmd.status().context("run cargo build for example set")?;
    ensure_success(status, "cargo build")?;
    Ok(())
}

fn wants_cuda(_config: &VoiceConfig) -> Result<bool> {
    Ok(Command::new("nvidia-smi")
        .arg("-L")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false))
}

fn initialize_qwen_submodule(repo_root: &Path) -> Result<()> {
    let qwen_root = repo_root.join("libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp");
    if qwen_root.join("CMakeLists.txt").is_file() {
        return Ok(());
    }
    let status = Command::new("git")
        .current_dir(repo_root)
        .args([
            "submodule",
            "update",
            "--init",
            "--recursive",
            "libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp",
        ])
        .status()
        .context("initialize qwen3-tts.cpp submodule")?;
    ensure_success(status, "git submodule update")
}

fn tts_spec(backend: TtsBackend) -> ExampleSpec {
    match backend {
        TtsBackend::Piper => ExampleSpec {
            example_name: "tts_piper",
            feature_name: "model-piper-en-us-ljspeech-medium",
            cuda_feature: Some("piper-cuda"),
        },
        TtsBackend::Qwen3cpp => ExampleSpec {
            example_name: "tts_qwen3_tts_cpp",
            feature_name: "model-qwen3-tts-cpp",
            cuda_feature: Some("qwen3-tts-cpp-cuda"),
        },
    }
}

fn asr_spec(backend: AsrBackend) -> ExampleSpec {
    match backend {
        AsrBackend::Whisper => ExampleSpec {
            example_name: "asr_whisper",
            feature_name: "model-whisper-base-en",
            cuda_feature: Some("whisper-cpp-cuda"),
        },
        AsrBackend::Sherpa => ExampleSpec {
            example_name: "asr_sherpa_onnx",
            feature_name: "model-sherpa-onnx-streaming",
            cuda_feature: Some("sherpa-onnx-cuda"),
        },
        AsrBackend::Moonshine => ExampleSpec {
            example_name: "asr_moonshine",
            feature_name: "model-moonshine-streaming",
            cuda_feature: None,
        },
    }
}

fn example_binary_path(config: &VoiceConfig, example_name: &str) -> PathBuf {
    config
        .repo_root
        .join("target")
        .join("release")
        .join("examples")
        .join(example_name)
}

fn tts_artifact_root(config: &VoiceConfig, backend: TtsBackend) -> PathBuf {
    match backend {
        TtsBackend::Piper => config.piper_artifact_root(),
        TtsBackend::Qwen3cpp => config.qwen_artifact_root(),
    }
}

fn asr_artifact_root(config: &VoiceConfig, backend: AsrBackend) -> PathBuf {
    match backend {
        AsrBackend::Whisper => config.whisper_artifact_root(),
        AsrBackend::Sherpa => config.sherpa_artifact_root(),
        AsrBackend::Moonshine => config.moonshine_artifact_root(),
    }
}

fn resolve_reference_audio(
    config: &VoiceConfig,
    backend: TtsBackend,
    voice_alias: Option<&str>,
    reference_audio: Option<&Path>,
) -> Result<Option<PathBuf>> {
    if voice_alias.is_none() && reference_audio.is_none() {
        return Ok(None);
    }
    if backend != TtsBackend::Qwen3cpp {
        bail!("--voice and --reference-audio are only supported with backend 'qwen3cpp'");
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
    without_prefix.split_whitespace().collect::<Vec<_>>().join("-")
}

fn write_text_input(stdin: Option<std::process::ChildStdin>, text: Option<&str>) -> Result<()> {
    let mut stdin = stdin.context("take child stdin")?;
    if let Some(text) = text {
        stdin.write_all(text.as_bytes())?;
        stdin.write_all(b"\n")?;
    } else {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        stdin.write_all(buffer.as_bytes())?;
    }
    drop(stdin);
    Ok(())
}

fn read_all<R: Read>(mut reader: R) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn ensure_success(status: std::process::ExitStatus, label: &str) -> Result<()> {
    if status.success() {
        Ok(())
    } else {
        bail!("{label} failed with status {status}")
    }
}

fn default_home_cache_root() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join(".cache/huggingface/hub")
}
