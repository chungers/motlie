use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
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
    Setup,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuildProfile {
    Debug,
    Release,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Acceleration {
    Auto,
    Cpu,
    Cuda,
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
    env_path: PathBuf,
    file_vars: HashMap<String, String>,
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
    let mut config = VoiceConfig::load(repo_root)?;

    match cli.command {
        Commands::Speak(args) => run_speak(&mut config, &args),
        Commands::Listen(args) => run_listen(&mut config, &args).map(|transcript| {
            if !transcript.is_empty() {
                println!("{transcript}");
            }
        }),
        Commands::Turn(args) => run_turn(&mut config, &args),
        Commands::Setup => run_setup(&mut config),
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
    fn load(repo_root: PathBuf) -> Result<Self> {
        let env_path = repo_root.join(".agents/voice/voice.env");
        let file_vars = if env_path.is_file() {
            load_shell_env_file(&env_path, &repo_root)?
        } else {
            HashMap::new()
        };

        Ok(Self {
            repo_root,
            env_path,
            file_vars,
        })
    }

    fn var(&self, key: &str) -> Option<String> {
        std::env::var(key)
            .ok()
            .or_else(|| self.file_vars.get(key).cloned())
    }

    fn build_profile(&self) -> Result<BuildProfile> {
        match self
            .var("MOTLIE_VOICE_BUILD_PROFILE")
            .unwrap_or_else(|| "release".to_string())
            .as_str()
        {
            "release" => Ok(BuildProfile::Release),
            "debug" => Ok(BuildProfile::Debug),
            other => bail!("unsupported MOTLIE_VOICE_BUILD_PROFILE='{other}'"),
        }
    }

    fn acceleration(&self) -> Result<Acceleration> {
        match self
            .var("MOTLIE_VOICE_ACCELERATION")
            .unwrap_or_else(|| "auto".to_string())
            .as_str()
        {
            "auto" => Ok(Acceleration::Auto),
            "cpu" => Ok(Acceleration::Cpu),
            "cuda" => Ok(Acceleration::Cuda),
            other => bail!("unsupported MOTLIE_VOICE_ACCELERATION='{other}'"),
        }
    }

    fn piper_artifact_root(&self) -> PathBuf {
        self.var("PIPER_ARTIFACT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(default_home_cache_root)
    }

    fn qwen_artifact_root(&self) -> PathBuf {
        self.var("QWEN3_TTS_CPP_ARTIFACT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/qwen3-tts-models"))
    }

    fn whisper_artifact_root(&self) -> PathBuf {
        self.var("WHISPER_ARTIFACT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(default_home_cache_root)
    }

    fn sherpa_artifact_root(&self) -> PathBuf {
        self.var("SHERPA_ARTIFACT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(default_home_cache_root)
    }

    fn moonshine_artifact_root(&self) -> PathBuf {
        self.var("MOONSHINE_ARTIFACT_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(default_home_cache_root)
    }

    fn reference_root(&self) -> PathBuf {
        self.var("MOTLIE_VOICE_REFERENCE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.repo_root.join("artifacts/voice-references"))
    }

    fn playback_endpoint_name(&self) -> String {
        self.var("MOTLIE_VOICE_PLAYBACK_ENDPOINT")
            .unwrap_or_else(|| "motliehost".to_string())
    }

    fn capture_endpoint_name(&self) -> String {
        self.var("MOTLIE_VOICE_CAPTURE_ENDPOINT")
            .unwrap_or_else(|| "motliehost".to_string())
    }

    fn endpoint(&mut self, endpoint_name: &str) -> Result<EndpointConfig> {
        let kind = match self.ensure_endpoint_field(endpoint_name, "KIND", None)?.as_str() {
            "local" => EndpointKind::Local,
            "ssh" => EndpointKind::Ssh,
            other => bail!("unsupported endpoint kind '{other}' for endpoint '{endpoint_name}'"),
        };

        let play_cmd = self.ensure_endpoint_field(endpoint_name, "PLAY_CMD", Some(kind))?;
        let record_cmd = self.ensure_endpoint_field(endpoint_name, "RECORD_CMD", Some(kind))?;
        let ssh_target = if kind == EndpointKind::Ssh {
            Some(self.ensure_endpoint_field(endpoint_name, "SSH_TARGET", Some(kind))?)
        } else {
            None
        };

        Ok(EndpointConfig {
            kind,
            ssh_target,
            play_cmd,
            record_cmd,
        })
    }

    fn ensure_endpoint_field(
        &mut self,
        endpoint_name: &str,
        field_name: &str,
        known_kind: Option<EndpointKind>,
    ) -> Result<String> {
        let key = format!("MOTLIE_ENDPOINT_{}_{}", endpoint_key(endpoint_name), field_name);
        if let Some(value) = self.var(&key) {
            return Ok(value);
        }
        if !interactive_tty() {
            bail!(
                "missing endpoint field {field_name} for endpoint '{endpoint_name}'; set it in {} or run `cargo run -p voice-agent -- setup`",
                self.env_path.display()
            );
        }

        let default_value = endpoint_field_default(endpoint_name, field_name, known_kind);
        let prompt = format!("Missing {field_name} for endpoint '{endpoint_name}'");
        let value = prompt_with_default(&prompt, default_value.as_deref())?;
        if value.trim().is_empty() {
            bail!("empty value provided for endpoint field {field_name}");
        }
        self.upsert_env_var(&key, &value)?;
        self.file_vars.insert(key, value.clone());
        eprintln!("[voice-agent] stored {field_name} for endpoint '{endpoint_name}' in {}", self.env_path.display());
        Ok(value)
    }

    fn upsert_env_var(&self, key: &str, value: &str) -> Result<()> {
        if let Some(parent) = self.env_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut lines = if self.env_path.is_file() {
            fs::read_to_string(&self.env_path)?
                .lines()
                .map(|line| line.to_string())
                .collect::<Vec<_>>()
        } else {
            vec!["# @codex-tts 2026-04-23 -- Generated by voice-agent.".to_string()]
        };

        let replacement = format!("{key}={}", shell_quote(value));
        let mut replaced = false;
        for line in &mut lines {
            if line.starts_with(&format!("{key}=")) {
                *line = replacement.clone();
                replaced = true;
                break;
            }
        }
        if !replaced {
            lines.push(replacement);
        }

        fs::write(&self.env_path, format!("{}\n", lines.join("\n")))?;
        Ok(())
    }
}

fn run_setup(config: &mut VoiceConfig) -> Result<()> {
    if !interactive_tty() {
        bail!("voice-agent setup requires an interactive terminal");
    }

    let playback_default = config.playback_endpoint_name();
    let capture_default = config.capture_endpoint_name();
    let playback = prompt_with_default("Default playback endpoint name", Some(&playback_default))?;
    let capture = prompt_with_default("Default capture endpoint name", Some(&capture_default))?;
    config.upsert_env_var("MOTLIE_VOICE_PLAYBACK_ENDPOINT", &playback)?;
    config.file_vars.insert("MOTLIE_VOICE_PLAYBACK_ENDPOINT".into(), playback.clone());
    config.upsert_env_var("MOTLIE_VOICE_CAPTURE_ENDPOINT", &capture)?;
    config.file_vars.insert("MOTLIE_VOICE_CAPTURE_ENDPOINT".into(), capture.clone());

    let _ = config.endpoint(&playback)?;
    if capture != playback {
        let _ = config.endpoint(&capture)?;
    }

    let reference_root = config.reference_root();
    config.upsert_env_var(
        "MOTLIE_VOICE_REFERENCE_ROOT",
        reference_root.to_string_lossy().as_ref(),
    )?;
    eprintln!("[voice-agent] wrote configuration to {}", config.env_path.display());
    Ok(())
}

fn run_speak(config: &mut VoiceConfig, args: &SpeakArgs) -> Result<()> {
    ensure_examples(config, &[BackendKind::Tts(args.backend)])?;

    let endpoint_name = args
        .endpoint
        .clone()
        .unwrap_or_else(|| config.playback_endpoint_name());
    let endpoint = config.endpoint(&endpoint_name)?;
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

fn run_listen(config: &mut VoiceConfig, args: &ListenArgs) -> Result<String> {
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

    let endpoint_name = args
        .endpoint
        .clone()
        .unwrap_or_else(|| config.capture_endpoint_name());
    let endpoint = config.endpoint(&endpoint_name)?;
    let capture_bytes = capture_wav_bytes(&endpoint, args.seconds)?;
    run_asr_with_bytes(asr, &capture_bytes)
}

fn run_turn(config: &mut VoiceConfig, args: &TurnArgs) -> Result<()> {
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

fn with_trim(record_cmd: &str, seconds: Option<u32>) -> String {
    match seconds {
        Some(value) => format!("{record_cmd} trim 0 {value}"),
        None => record_cmd.to_string(),
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
    if matches!(config.build_profile()?, BuildProfile::Release) {
        cmd.arg("--release");
    }
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

fn wants_cuda(config: &VoiceConfig) -> Result<bool> {
    match config.acceleration()? {
        Acceleration::Cpu => Ok(false),
        Acceleration::Auto => Ok(
            Command::new("nvidia-smi")
                .arg("-L")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false),
        ),
        Acceleration::Cuda => {
            let ok = Command::new("nvidia-smi")
                .arg("-L")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false);
            if !ok {
                bail!("MOTLIE_VOICE_ACCELERATION=cuda but no usable NVIDIA device is visible");
            }
            Ok(true)
        }
    }
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
    let profile = match config.build_profile().unwrap_or(BuildProfile::Release) {
        BuildProfile::Release => "release",
        BuildProfile::Debug => "debug",
    };
    config
        .repo_root
        .join("target")
        .join(profile)
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

fn load_shell_env_file(path: &Path, repo_root: &Path) -> Result<HashMap<String, String>> {
    let output = Command::new("bash")
        .arg("-lc")
        .arg("set -a; source \"$1\"; env -0")
        .arg("_")
        .arg(path)
        .env("REPO_ROOT", repo_root)
        .output()
        .with_context(|| format!("source {}", path.display()))?;
    ensure_success(output.status, "source voice.env")?;

    let mut vars = HashMap::new();
    for entry in output.stdout.split(|byte| *byte == 0) {
        if entry.is_empty() {
            continue;
        }
        if let Some(position) = entry.iter().position(|byte| *byte == b'=') {
            let key = &entry[..position];
            let value = &entry[position + 1..];
            vars.insert(
                String::from_utf8_lossy(key).to_string(),
                String::from_utf8_lossy(value).to_string(),
            );
        }
    }
    Ok(vars)
}

fn interactive_tty() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

fn prompt_with_default(prompt: &str, default: Option<&str>) -> Result<String> {
    let mut stdout = io::stdout();
    match default {
        Some(default) => write!(stdout, "{prompt} [{default}]: ")?,
        None => write!(stdout, "{prompt}: ")?,
    }
    stdout.flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_string();
    Ok(if input.is_empty() {
        default.unwrap_or_default().to_string()
    } else {
        input
    })
}

fn endpoint_key(name: &str) -> String {
    name.to_ascii_uppercase().replace('-', "_")
}

fn endpoint_field_default(endpoint_name: &str, field_name: &str, known_kind: Option<EndpointKind>) -> Option<String> {
    let kind = known_kind.unwrap_or_else(|| {
        if endpoint_name == "local-linux" {
            EndpointKind::Local
        } else {
            EndpointKind::Ssh
        }
    });

    match field_name {
        "KIND" => Some(match kind {
            EndpointKind::Local => "local".to_string(),
            EndpointKind::Ssh => "ssh".to_string(),
        }),
        "SSH_TARGET" => Some(endpoint_name.to_string()),
        "PLAY_CMD" => Some(match kind {
            EndpointKind::Local => "play -t wav -".to_string(),
            EndpointKind::Ssh => "/opt/homebrew/bin/play -t wav -".to_string(),
        }),
        "RECORD_CMD" => Some(match kind {
            EndpointKind::Local => "rec -q -t wav -".to_string(),
            EndpointKind::Ssh => "/opt/homebrew/bin/rec -q -t wav -".to_string(),
        }),
        _ => None,
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn default_home_cache_root() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join(".cache/huggingface/hub")
}
