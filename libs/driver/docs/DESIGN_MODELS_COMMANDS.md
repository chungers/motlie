# motlie-driver — Models Command Bindings Design

## Status

Proposal. No code lands with this doc. The `models` `CommandSet<C>` is
greenfield at the product level: there is no pre-existing models driver, no
existing command surface to migrate, and no backward-compatibility constraint
on the proposed grammar. It is brownfield at the crate level in exactly the
same way the tmux adapter is: it adds a new sibling module under
`libs/driver/src/commands/` and a new binary under `bins/models/driver/`,
without changing `libs/driver`'s core engine or the tmux adapter.

## Changelog

- `(2026-04-17, @claude-opus-47, initial proposal for a models CommandSet that exposes load/unload/chat/session/transcribe/synthesize/pipeline/info across the full libs/models surface)`
- `(2026-04-22, @claude-opus-47, rewritten against post-typed-contracts libs/model(s): BundleHandle is now Sized with associated types so Box<dyn BundleHandle> is no longer possible; TranscriptionModel/SpeechModel::open_stream replaced by BatchTranscriber/StreamingTranscriber/SpeechSynthesizer/VoiceCloneSynthesizer from libs/model/src/typed.rs; load dispatch retargeted at namespaced start_typed per curated bundle; SpeechRequest→SynthesisRequest; VoiceConditioning replaced by CloneReference<RATE_HZ, Mono>; AudioSpec is gone — sample rate + layout are carried by AudioBuf type parameters; pipeline tts-asr delegates to the existing stream_speech_into_asr helper)`

## Problem

`libs/models` owns a `Catalog` of curated ASR, TTS, chat, and embedding
bundles. Each bundle is instantiated through a namespaced `start_typed`
entry point (`motlie_models::{tts,asr,chat,embeddings}::{bundle}::start_typed`)
which returns a concrete `Sized` typed handle implementing the capability
traits from `libs/model/src/typed.rs`. Today each capability is exercised by a
dedicated example binary under `libs/models/examples/` (for example
`tts_qwen3_tts_cpp/`, `asr_whisper_base_en/`) — one `main.rs` per slice,
each argv-parsed separately, with no shared surface for interactive
experimentation.

We want the same interactive experience for models that we already have for
tmux: a single REPL/TUI driver that loads any subset of enabled bundles
simultaneously, calls their capabilities, and lets a human validate behavior
end-to-end (including TTS→ASR round-trip) without rebuilding per-capability
binaries.

The existing `CommandEngine<C, S>` + `CommandSet<C>` contract
(`libs/driver/src/engine.rs:44-63`) is the right execution substrate: tmux is
one adapter, models becomes another. The work is entirely in designing the
`ModelsCommand` enum, its `ModelsState`, and how state maps to
`motlie_models`'s public API.

## Goals

- One `CommandSet<C>` implementation (`ModelsCommand`) that covers every
  capability `libs/models` exposes today: ASR, TTS, chat, embeddings, plus
  lifecycle (load/unload/list/info) and a pipeline sub-command for
  cross-capability round-trips.
- Multiple models loaded simultaneously, addressed by alias
  (`--alias chatter`, `--alias my-asr`, …), mirroring the tmux multi-host
  `connect <uri> as <alias>` pattern from `tmux_app.rs:38-50`.
- Multi-turn chat via driver-owned conversation history (`session new`,
  `session say`, `session show`, `session clear`) — `motlie_model::ChatModel`
  is stateless, so history lives in the driver.
- Discoverable completion: loaded aliases, available selectors, session ids,
  capability-typed argument validation.
- Drop-in binary `bins/models/driver` that reuses `run_repl` / `run_tui` the
  same way `bins/tmux/driver/src/main.rs:25-54` does.

## Non-goals

- No changes to `libs/models` / `libs/model` public APIs. The driver
  consumes the post-refactor surface as-is: namespaced `start_typed` entry
  points per curated bundle, the `Sized` `BundleHandle` trait, and the
  typed `BatchTranscriber` / `StreamingTranscriber` / `SpeechSynthesizer` /
  `VoiceCloneSynthesizer` / `SpeechStream` traits from
  `libs/model/src/typed.rs`. Any dyn/Sized impedance is absorbed by a
  driver-local adapter layer.
- No artifact-download orchestration beyond forwarding the existing
  `ArtifactPolicy` / `--allow-fetch` switch. Acquisition remains the catalog's
  job.
- No streaming partial-transcript UI in v1. Partials are emitted to the log
  only when `--emit-partials` is set; the mirror pane is not wired up yet.
- No CUDA/CPU selection UI. Quantization is exposed; execution substrate is
  controlled by build features (e.g., `MOTLIE_MODEL_FORCE_CPU`) as it is
  today.

## High-Level System Design

```
+---------------------------+
| bins/models/driver/main   |   parse CLI flags, construct ModelsState,
| (tokio main)              |   dispatch run_repl / run_tui
+------------+--------------+
             |
             v
+---------------------------+
| CommandEngine<            |   generic engine from libs/driver/src/engine.rs
|   ModelsState,            |
|   ModelsCommand>          |
+------------+--------------+
             |
             v
+---------------------------+   execute_models_command() matches on
| ModelsCommand             |   ModelsCommand variant, resolves alias,
|  (clap Subcommand)        |   calls capability on BundleHandle.
+------------+--------------+
             |
             v
+---------------------------+
| ModelsState               |   Catalog + BTreeMap<alias, LoadedModel>
|  - catalog: Catalog       |   + current: Option<String>
|  - loaded: BTreeMap<...>  |   + sessions per chat model
|  - current: Option<String>|
+------------+--------------+
             |
             v
+---------------------------+
| motlie_models             |   ModelSelector → namespaced start_typed →
|   (libs/models)           |   typed handle implementing
|                           |   {ChatModel, BatchTranscriber,
|                           |    StreamingTranscriber, SpeechSynthesizer,
|                           |    VoiceCloneSynthesizer, EmbeddingModel}
+---------------------------+
```

`BundleHandle` is `Sized` with associated types (`type Chat`, `type Completion`,
`type Embeddings`) in the post-refactor `libs/model/src/lib.rs`, so the driver
cannot store a uniform `Box<dyn BundleHandle>`. ASR and TTS capabilities live
in `libs/model/src/typed.rs` as independent traits, not as methods on the
handle. The driver therefore captures per-capability typed trait objects
during `load` and attaches them to the `LoadedModel`; the original
`BundleHandle` is retained only for lifecycle + descriptor + metrics access.

Data flow for a one-shot chat:

1. User types `chat say "hi there"` at the REPL.
2. Engine parses via `ModelsCommand::from_matches`
   (`CommandSet::from_matches`, `libs/driver/src/engine.rs:50`).
3. `resolve_command` looks up `state.current` to obtain the alias if none is
   given; errors out with `DriverError::MissingCurrentScope` otherwise
   (same error arm already used by `tmux_app.rs`).
4. `execute` calls `handle.chat()?.generate(ChatRequest { … })` on the
   resolved `LoadedModel`.
5. `ChatResponse.content` is rendered as `CommandOutput::text(...)`.

Data flow for TTS→ASR pipeline:

1. `pipeline tts-asr "the quick brown fox" --tts-alias voice --asr-alias ears`
2. Delegate to `motlie_model::typed::stream_speech_into_asr(tts, asr,
   SynthesisRequest { text, .. }, TranscriptionParams::default())` — it already
   owns the streaming bridge, including any TTS→ASR rate/layout adaptation via
   an `AudioTransform`.
3. Collect the final `TranscriptionUpdate`.
4. Print: input text, input length, output transcript, a simple
   case-insensitive word-overlap ratio, and optionally save the intermediate
   wav to `--keep-wav <path>` (captured by tapping `SpeechStream::next_chunk`
   before handing the stream to the helper — or by using a thin local clone of
   the helper when `--keep-wav` is set).

## CommandSet

```rust
// libs/driver/src/commands/models.rs

use motlie_models::{
    Catalog, ModelSelector, BundleDescriptor, BackendKind, CapabilityKind,
};
use motlie_model::{
    ChatMessage, ChatRequest, ChatRole, ChatResponse,
    LoadedBundleDescriptor, QuantizationBits, StartOptions, ArtifactPolicy,
    SpeechParams, TranscriptionParams, TranscriptionUpdate,
};
use motlie_model::typed::{
    AudioBuf, BatchTranscriber, CloneReference, Mono,
    SpeechStream, SpeechSynthesizer, StreamingTranscriber, SynthesisRequest,
    TranscriptionSession, VoiceCloneSynthesizer,
};

pub struct ModelsState {
    pub(crate) catalog: Catalog,
    pub(crate) artifact_root: PathBuf,
    pub(crate) loaded: BTreeMap<String, LoadedModel>,
    pub(crate) current: Option<String>,
}

/// Capability-erased view the driver keeps per loaded alias.
///
/// The post-refactor typed traits in `libs/model/src/typed.rs` use
/// `-> impl Future<Output = …>` return types, which are **not** trait-object
/// safe. `BundleHandle` is additionally `Sized` with associated types. Neither
/// `Box<dyn BundleHandle>` nor `Box<dyn SpeechSynthesizer>` compiles today.
///
/// The driver therefore introduces a thin local adapter layer with
/// `BoxFuture`-returning methods and implements it for each concrete typed
/// handle at `load` time. The original bundle handle is consumed into these
/// adapters; shutdown is captured as a `FnOnce` closure to preserve the
/// `shutdown(self)` contract without exposing the concrete handle type.
pub(crate) struct LoadedModel {
    pub alias: String,
    pub selector: ModelSelector,
    pub descriptor: LoadedBundleDescriptor,
    pub backend: BackendKind,
    pub capabilities: LoadedCapabilities,
    pub shutdown: ShutdownFn,
    pub sessions: BTreeMap<String, ChatSession>,
    pub current_session: Option<String>,
}

type ShutdownFn =
    Box<dyn FnOnce() -> futures::future::BoxFuture<'static, Result<(), ModelError>> + Send>;

/// Dyn-safe capability shims owned by the driver. Each field is `None` when
/// the bundle does not expose the capability (compile-time feature-gated
/// absence is normalized to `None` at load time).
#[derive(Default)]
pub(crate) struct LoadedCapabilities {
    pub chat: Option<Box<dyn DynChat>>,
    pub embedding: Option<Box<dyn DynEmbedding>>,
    pub batch_asr: Option<Box<dyn DynBatchAsr>>,
    pub streaming_asr: Option<Box<dyn DynStreamingAsr>>,
    pub tts: Option<Box<dyn DynTts>>,
    /// Voice cloning's reference contract (`CloneReference<RATE_HZ, C>`) is
    /// fixed per bundle. `DynVoiceClone16Mono` hard-codes the 16 kHz mono
    /// contract used by today's only cloning-capable bundle
    /// (`qwen3-tts.cpp`); additional arms go here when new contracts ship.
    pub voice_clone_16k_mono: Option<Box<dyn DynVoiceClone16Mono>>,
}

// Representative adapter trait (others follow the same pattern):
pub(crate) trait DynTts: Send + Sync {
    fn synthesize<'a>(
        &'a self,
        request: SynthesisRequest,
    ) -> futures::future::BoxFuture<'a, Result<Box<dyn DynSpeechStream>, ModelError>>;
}

pub(crate) struct ChatSession {
    pub id: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Parser)]
struct ModelsRoot { #[command(subcommand)] command: ModelsCommand }

#[derive(Subcommand)]
pub enum ModelsCommand {
    Load(LoadCommand),
    Unload(UnloadCommand),
    List(ListCommand),
    Use(UseCommand),
    Info(InfoCommand),

    Chat(ChatCommand),          // one-shot
    Session(SessionCommand),    // multi-turn

    Transcribe(TranscribeCommand),
    Synthesize(SynthesizeCommand),
    Embed(EmbedCommand),

    Pipeline(PipelineCommand),
}
```

`ModelsCommand` implements `CommandSet<ModelsState>` exactly the way
`TmuxCommand` does (`libs/driver/src/commands/tmux.rs:655-696`):

```rust
#[async_trait]
impl CommandSet<ModelsState> for ModelsCommand {
    type CompletionContext = ModelsCompletionContext;
    type Resolved = Self;

    fn root_command() -> clap::Command { ModelsRoot::command().name("models") }
    fn from_matches(m: &clap::ArgMatches) -> DriverResult<Self> {
        Ok(ModelsRoot::from_arg_matches(m)?.command)
    }
    fn completion_context(ctx: &ModelsState) -> Self::CompletionContext { … }
    fn help(topic: &[String]) -> Option<String> { models_help(topic) }
    fn complete(req: CompletionRequest<'_>, ctx: &Self::CompletionContext)
        -> Vec<CompletionCandidate> { models_complete(req, ctx) }
    fn resolve_command(self, _ctx: &ModelsState) -> DriverResult<Self::Resolved> { Ok(self) }
    async fn execute(resolved: Self, ctx: &mut ModelsState)
        -> DriverResult<CommandOutput> { execute_models_command(ctx, resolved).await }
}
```

## Command Reference

All commands take `--alias NAME` as an optional argument that falls back to
`state.current` when omitted. Commands that have no sensible "current"
fallback (e.g. `pipeline tts-asr`) require explicit aliases per capability.

### Lifecycle

#### `load <selector> [--alias NAME] [--allow-fetch] [--artifact-root PATH] [--quantization q4|q8|f32] [--use]`

Instantiate a bundle from a `ModelSelector` string (`tts:…`, `asr:…`,
`chat:…`, `embedding:…`) and retain the resulting `BundleHandle` in
`state.loaded` under `alias` (defaults to the selector's last path segment).

```rust
#[derive(Args)]
pub struct LoadCommand {
    pub selector: String,                // e.g. "chat:qwen/qwen3_4b"
    #[arg(long)] pub alias: Option<String>,
    #[arg(long = "allow-fetch")] pub allow_fetch: bool,
    #[arg(long = "artifact-root")] pub artifact_root: Option<PathBuf>,
    #[arg(long, value_enum)] pub quantization: Option<QuantBitsArg>,
    #[arg(long)] pub r#use: bool,        // also set as current
}
```

Maps to a `match` over the parsed `ModelSelector`. There is no uniform
`selector.bundle()?.start()` path post-refactor: each curated bundle has its
own namespaced `start_typed` entry point under `motlie_models::{cap}::{bundle}`,
returning a concrete `Sized` typed handle. The driver's `load` dispatch fans
out once per feature-gated variant and wraps the handle into the
`LoadedCapabilities` adapter shims:

```rust
let selector: ModelSelector = cmd.selector.parse()?;
let options = StartOptions {
    artifact_policy: Some(match cmd.allow_fetch {
        true  => ArtifactPolicy::AllowFetch { root: cmd.artifact_root },
        false => ArtifactPolicy::LocalOnly   {
            root: cmd.artifact_root.unwrap_or_else(|| state.artifact_root.clone())
        },
    }),
    quantization: cmd.quantization.map(Into::into),
    ..Default::default()
};
let loaded = match selector {
    #[cfg(feature = "model-qwen3-tts-cpp")]
    ModelSelector::Tts(Tts::Qwen3TtsCpp0_6B) => {
        let handle = motlie_models::tts::qwen3_tts_cpp::start_typed(options).await?;
        LoadedModel::from_qwen3_tts_cpp(alias, selector, handle)
    }
    #[cfg(feature = "model-piper")]
    ModelSelector::Tts(Tts::PiperEnUsLjspeechMedium) => {
        let handle = motlie_models::tts::piper_en_us_ljspeech_medium::start_typed(options).await?;
        LoadedModel::from_piper(alias, selector, handle)
    }
    // … asr, chat, embedding branches mirror this shape …
};
state.loaded.insert(loaded.alias.clone(), loaded);
```

Because `ModelSelector` is `#[non_exhaustive]` with feature-gated variants,
the match must either be exhaustive over the compile-time enabled set or end
in `_ => Err(DriverError::unknown_scope("selector not compiled in"))`.
Completion follows the same visibility: `available_selectors` in the
completion context is whatever the current binary was built with.

Output: `loaded '<alias>' = <selector>  backend=<BackendKind>  q=<resolved_quantization>`.

#### `unload <alias>`

Invokes the captured `LoadedModel::shutdown` closure (which owns and consumes
the original `BundleHandle::shutdown(self)` call). Removes from
`state.loaded`. Clears `state.current` if it matched.

#### `list [--loaded] [--available] [--capability chat|asr|tts|embedding]`

- `list --loaded` (default): one line per loaded model with alias, selector,
  backend, resolved quantization, active session count.
- `list --available`: enumerate `catalog.bundles()` — show `BundleDescriptor`
  id, family, backend, capabilities, whether artifacts are cached locally.
- `list --capability X`: filter either view by `CapabilityKind`.

#### `use <alias>`

Set `state.current`. Validates alias is loaded.

#### `info [alias]`

Dump on the chosen loaded model:

- alias + selector + bundle_id
- display_name
- `BackendKind` (e.g. `LlamaCpp`, `MistralRs`, `Ort`, `Qwen3TtsCpp`,
  `SherpaOnnx`, `WhisperCpp`) — this is what the task calls *BackendMode*;
  `libs/models` does not have a `BackendMode` enum today. The closest public
  type is `BackendKind` on `LoadedBundleDescriptor` and `BundleDescriptor`,
  so we surface it as `backend: <BackendKind>` and leave space for a future
  `BackendMode` (CPU vs CUDA vs Metal) when the models layer introduces one.
- `resolved_quantization` (from `LoadedBundleDescriptor.resolved_quantization`)
- capability list + `InteractionStyle` per capability
- Sample rate + channel layout for TTS models, reported as
  `output: {sample_rate_hz} Hz / {layout}` where layout is `Mono`/`Stereo`.
  The values come from the first `SpeechStream` chunk (`chunk.sample_rate_hz()`)
  and are cached on `LoadedModel` at `load` time via a one-shot probe
  synthesis. ASR models have no intrinsic rate: each `transcribe` call supplies
  its own `AudioBuf<S, RATE_HZ, C>`, so `info` for ASR reports
  `input: typed AudioBuf per call`.
- `metric_snapshot()` summary if any

### Chat — one-shot

#### `chat <prompt...> [--alias NAME] [--system TEXT] [--max-tokens N] [--temperature F]`

Ephemeral, no history. Builds a single-turn `ChatRequest`:

```rust
chat_model.generate(ChatRequest {
    messages: vec![
        ChatMessage::new(ChatRole::System, system.unwrap_or("Be concise.")),
        ChatMessage::new(ChatRole::User, prompt.join(" ")),
    ],
    params: GenerationParams { max_tokens, temperature, .. },
    ..Default::default()
}).await?
```

Prints `response.content`.

### Chat — multi-turn

Sessions are owned by a single loaded chat model (so a session always runs
against one backend). Stored on `LoadedModel.sessions`.

```text
session new [--name ID]         # default ID = "s{N}"
session say <text...>           # append user turn, generate, append assistant
session show                    # print transcript
session clear                   # empty message list; keep session
session drop <id>               # delete session
session list                    # list sessions for current alias
session use <id>                # set LoadedModel.current_session
```

`session say` equivalent:

```rust
let session = loaded.session_mut(required_current)?;
session.messages.push(ChatMessage::new(ChatRole::User, text));
let response = chat_model.generate(ChatRequest {
    messages: session.messages.clone(),
    ..Default::default()
}).await?;
session.messages.push(ChatMessage::new(ChatRole::Assistant, response.content.clone()));
CommandOutput::text(response.content)
```

### ASR

#### `transcribe <wav-path> [--alias NAME] [--language LANG] [--emit-partials] [--batch] [--chunk-frames N]`

The post-refactor ASR surface has two independent traits, and the driver
exposes both:

- `--batch` (default when the loaded bundle's `BatchTranscriber` is present):
  decode the wav into an `AudioBuf<i16, RATE_HZ, Mono>` sized to the bundle's
  required input rate (resample with a proper anti-alias filter if needed),
  then call `BatchTranscriber::transcribe(audio, TranscriptionParams { language })`.
- Streaming (default when only `StreamingTranscriber` is available, or when
  `--emit-partials` is set): open a session, chunk the buffer by frame count
  (default 3200 frames ≈ 200 ms at 16 kHz), `session.ingest(chunk)` and
  surface any non-empty `TranscriptionUpdate` as `[partial]` lines, then
  `session.finish()` for the final transcript.

```rust
// --batch branch
let audio = decode_wav_to_typed_audio::<16_000, Mono>(wav_path)?;
let transcript = asr.capabilities.batch_asr.as_ref()
    .ok_or_else(|| DriverError::invalid_argument("alias", "no BatchTranscriber"))?
    .transcribe(audio, TranscriptionParams { language, emit_partials: false })
    .await?;

// streaming branch
let mut session = asr.capabilities.streaming_asr.as_ref()
    .ok_or_else(|| DriverError::invalid_argument("alias", "no StreamingTranscriber"))?
    .open_session(TranscriptionParams { language, emit_partials })
    .await?;
for chunk in chunks_of(&audio, chunk_frames) {
    if let Some(update) = session.ingest(chunk).await? {
        if emit_partials { print_partial(&update); }
    }
}
let final_update = session.finish().await?;
```

Note: `AudioSpec { sample_rate_hz, channels, encoding }` is gone. Sample
rate and channel layout are carried as type parameters on `AudioBuf`, and
encoding is always `f32` or `i16` in the buffer itself. The driver's wav
decoder is responsible for converting to the bundle's required shape at
`transcribe` time.

### TTS

#### `synthesize <text...> --out <wav-path> [--alias NAME] [--speaking-rate F] [--reference-audio PATH] [--reference-text TXT] [--seed N]`

Calls either `SpeechSynthesizer::synthesize(SynthesisRequest { text, params })`
or — when `--reference-audio` is set and the bundle's `voice_clone_16k_mono`
capability is present —
`VoiceCloneSynthesizer::<16_000, Mono>::synthesize_with_reference(request,
CloneReference { audio, transcript })`. The returned `SpeechStream` is drained
via `next_chunk()`; sample rate comes from the first chunk
(`chunk.sample_rate_hz()`). A `hound::WavWriter` sized for that rate receives
every chunk and `finish()` is called on the stream.

The 16 kHz mono `CloneReference` contract is load-bearing for
`qwen3-tts.cpp` — see the in-repo memory at
`libs/models/examples/tts_qwen3_tts_cpp/README.md` about the speaker encoder
being trained on 16-kHz-bandlimited content. The driver must preserve that
contract when decoding the `--reference-audio` file (downmix to mono + proper
anti-alias downsample to 16 kHz; do not pass raw 24 kHz or 44.1 kHz through).

```rust
let request = SynthesisRequest { text: text.join(" "), params: SpeechParams { speaking_rate, seed, .. } };
let mut stream: Box<dyn DynSpeechStream> = if let Some(ref_path) = reference_audio {
    let reference = decode_wav_to_reference::<16_000, Mono>(ref_path)?;
    tts.capabilities.voice_clone_16k_mono.as_ref()
        .ok_or_else(|| DriverError::invalid_argument("alias", "no VoiceCloneSynthesizer"))?
        .synthesize_with_reference(request, CloneReference { audio: reference, transcript: reference_text })
        .await?
} else {
    tts.capabilities.tts.as_ref()
        .ok_or_else(|| DriverError::invalid_argument("alias", "no SpeechSynthesizer"))?
        .synthesize(request).await?
};
let first = stream.next_chunk().await?.ok_or(empty_stream)?;
let mut writer = hound::WavWriter::create(out_path, wav_spec(first.sample_rate_hz()))?;
write_chunk(&mut writer, first)?;
while let Some(chunk) = stream.next_chunk().await? { write_chunk(&mut writer, chunk)?; }
stream.finish().await?;
writer.finalize()?;
```

Returns `CommandOutput` with sample count, sample rate, layout, and output
path.

### Embeddings

#### `embed <text...> [--alias NAME] [--out-file PATH] [--json]`

Calls `handle.embeddings()?.embed(EmbeddingRequest { inputs })`. Each
positional is one input string; whitespace-heavy inputs should be quoted.
Default output is one line per vector as `dim=<N> norm=<F>
preview=[v0, v1, v2, …]`. `--json` dumps full vectors to stdout (or
`--out-file`).

### Pipeline

#### `pipeline tts-asr <text...> --tts-alias NAME --asr-alias NAME [--keep-wav PATH]`

Delegates to the existing helper `motlie_model::typed::stream_speech_into_asr`
in `libs/model/src/typed.rs`, which already bridges a `SpeechSynthesizer` to a
`StreamingTranscriber` with an optional `AudioTransform` for rate/layout
adaptation (e.g. `I16MonoResampler`). The driver's role is only to thread
loaded-state lookups and to tee the intermediate wav out when `--keep-wav`
is set.

```rust
let tts = state.loaded_ref(&cmd.tts_alias)?;
let asr = state.loaded_ref(&cmd.asr_alias)?;

let transcript = if let Some(path) = cmd.keep_wav {
    // Local fork of stream_speech_into_asr that also writes to a WavWriter
    // sized from the first chunk. Same semantics otherwise.
    bridge_with_wav_tap(tts, asr, SynthesisRequest { text: cmd.text.join(" "), .. Default::default() }, &path).await?
} else {
    stream_speech_into_asr(
        tts.tts_handle()?,                // &impl SpeechSynthesizer
        identity_transform_or_i16_mono_resampler(tts, asr),
        asr.streaming_asr_handle()?,      // &impl StreamingTranscriber
        SynthesisRequest { text: cmd.text.join(" "), .. Default::default() },
        TranscriptionParams::default(),
    ).await?
};

CommandOutput {
    lines: vec![
        format!("input:  {input}"),
        format!("output: {}", transcript.segments.iter().map(|s| &s.text).join(" ")),
        format!("overlap: {:.1}%", word_overlap(&input, &output) * 100.0),
    ],
    effects: vec![],
}
```

The helper assumes mono throughout; cross-layout pipelines are deferred.
`stream_speech_into_asr` is a free function taking borrowed typed handles,
which means the driver needs a narrow accessor (`tts_handle()`, etc.) that
yields the concrete `impl` type. In practice that means each `load`-dispatch
arm either stores a concrete bundle handle alongside the dyn-adapters **or**
the pipeline is implemented per-bundle-pair in a small match. The concrete
arm is cleaner; dyn adapters cover single-capability commands only.

Other pipelines (`pipeline chat-tts-asr`, `pipeline asr-tts`) follow the same
shape; v1 ships only `tts-asr` to validate the engine, the rest track as
PLAN items.

## Completion

```rust
pub struct ModelsCompletionContext {
    pub loaded_aliases: Vec<String>,
    pub available_selectors: Vec<String>,  // catalog.bundles()
    pub sessions_for_current: Vec<String>,
    pub quantization_variants: &'static [&'static str], // ["q4","q8","f32"]
}

fn models_complete(req: CompletionRequest<'_>, ctx: &ModelsCompletionContext)
    -> Vec<CompletionCandidate>
{
    match (req.command_path, req.arg_id) {
        (["load"],        Some("selector"))  => filter(&ctx.available_selectors, req.prefix),
        (["load"],        Some("quantization"))
            => filter(&["q4","q8","f32"], req.prefix),
        (["unload"],      Some("alias"))
        | (["use"],       Some("alias"))
        | (["info"],      Some("alias"))
        | (["chat"],      Some("alias"))
        | (["transcribe"],Some("alias"))
        | (["synthesize"],Some("alias"))
        | (["embed"],     Some("alias"))
            => filter(&ctx.loaded_aliases, req.prefix),
        (["pipeline","tts-asr"], Some("tts_alias" | "asr_alias"))
            => filter(&ctx.loaded_aliases, req.prefix),
        (["session","use" | "drop"], Some("id"))
            => filter(&ctx.sessions_for_current, req.prefix),
        _ => Vec::new(),
    }
}
```

The pattern mirrors `tmux_complete()` in `libs/driver/src/commands/tmux.rs:711-749`.

## Error surface

Reuses `DriverError` (`libs/driver/src/error.rs`) without any new arms:

- unknown alias → `DriverError::unknown_scope(alias)` (same as tmux multi-host)
- no alias supplied and `state.current` is `None` → `DriverError::MissingCurrentScope`
- bundle selector parse failure → `DriverError::invalid_argument("selector", ModelsError::to_string())`
- wrong capability (e.g. `chat` on an ASR alias) → `DriverError::invalid_argument("alias", "loaded model does not expose chat capability")`
- `ModelError` from the models layer → wrap via `DriverError::message(err.to_string())`

(If we find that we want typed discrimination in the TUI later, we add one
arm `DriverError::Model(ModelError)` and `From<ModelError>`. Not needed for v1.)

## Binary wiring

A thin `bins/models/driver/src/main.rs` copied from the tmux pattern:

```rust
#[tokio::main]
async fn main() -> DriverResult<()> {
    let options = parse_args()?;
    let state = ModelsState::new(Catalog::with_defaults(), motlie_models::default_artifact_root());
    let mut engine = CommandEngine::<ModelsState, ModelsCommand>::new(state);
    let result = if options.use_tui {
        run_models_tui(&mut engine, &options).await  // can start as alias of run_repl for v1
    } else {
        run_models_repl(&mut engine, &options).await
    };
    let shutdown_result = engine.context_mut().shutdown_all_loaded().await;
    result.and(shutdown_result)
}
```

For v1 the TUI is omitted — the tmux TUI is tied to the mirror/history model
of tmux scrollback, which has no analog in models. REPL-only plus structured
text output is sufficient and keeps the slice small.

## Mapping summary

| Command             | libs/models entry                                                  | libs/model capability                                       |
|---------------------|--------------------------------------------------------------------|-------------------------------------------------------------|
| `load`              | namespaced `motlie_models::{tts,asr,chat,embeddings}::{bundle}::start_typed` | —                                                           |
| `unload`            | `BundleHandle::shutdown` (captured in `ShutdownFn`)                | —                                                           |
| `list --available`  | `Catalog::bundles`                                                 | —                                                           |
| `list --loaded`     | cached `LoadedBundleDescriptor`                                    | —                                                           |
| `info`              | cached `LoadedBundleDescriptor` + `metric_snapshot`                | —                                                           |
| `chat`, `session …` | typed chat handle from `start_typed`                               | `ChatModel::generate`                                       |
| `transcribe`        | typed ASR handle from `start_typed`                                | `BatchTranscriber::transcribe` / `StreamingTranscriber::open_session` |
| `synthesize`        | typed TTS handle from `start_typed`                                | `SpeechSynthesizer::synthesize` (+ `VoiceCloneSynthesizer::synthesize_with_reference` when `--reference-audio`) |
| `embed`             | typed embedding handle from `start_typed`                          | `EmbeddingModel::embed`                                     |
| `pipeline tts-asr`  | both TTS + streaming ASR handles                                   | `motlie_model::typed::stream_speech_into_asr`               |

## Alternatives Considered

### A1. Per-capability `CommandSet<C>` (one per ASR/TTS/Chat)

Instead of one `ModelsCommand`, ship `AsrCommand`, `TtsCommand`, `ChatCommand`
as sibling `CommandSet<C>` implementations with their own `CommandEngine`.

- **Pro:** tighter scope per engine, smaller `clap` tree each.
- **Con:** impossible to express `pipeline tts-asr` without introducing a
  cross-engine coordinator (which is just `ModelsState` by another name).
- **Con:** forces the user to run three REPLs and coordinate artifacts by
  hand; defeats the stated goal of "exercise all models interactively."
- **Rejected.**

### A2. Reuse `TmuxAppState` multi-scope scaffold via naming traits

The `ResolveName` trait in `libs/driver/src/naming.rs` supports `alias/target`
qualified names. We could make every models command require
`alias/session-id` syntax everywhere, mirroring tmux multi-host.

- **Pro:** uniform grammar with tmux multi-host.
- **Con:** most models commands have no sub-entity (only chat sessions do),
  so qualified names add friction without payoff.
- **Decision:** use `--alias NAME` flags for model selection and reserve
  `alias/session-id` only for the chat `session` subtree, where it actually
  pays off. If chat session multi-alias workflows become common, promote
  sessions to a `ResolveName` kind later.

### A3. In-engine history mirror (à la tmux)

`TmuxState` retains `mirror_history` for watch/stream replay
(`libs/driver/src/commands/tmux.rs:36`). We could do the same for model
outputs — keep the last N chat responses, last N transcripts, etc.

- **Pro:** consistent UX with tmux.
- **Con:** tmux history exists because tmux output is ephemeral and
  streaming; model outputs are already printed to stdout and, for wav/json,
  persisted by the caller via `--out`. A separate mirror buffer duplicates
  what's already on disk or in the scrollback.
- **Decision:** skip for v1. Revisit if a models TUI is built.

## Appendix — PLAN outline

Concrete PLAN tasks (to be written up in `libs/driver/docs/PLAN.md` after
this DESIGN is accepted):

- P1: scaffold `libs/driver/src/commands/models.rs` behind
      `commands-models` feature; wire empty `ModelsCommand`/`ModelsState`
      stubs + `CommandSet` impl that returns `CommandOutput::line("todo")`.
      Define the `DynChat`/`DynTts`/`DynBatchAsr`/`DynStreamingAsr`/
      `DynEmbedding`/`DynVoiceClone16Mono` adapter traits with blanket impls
      forwarding to the concrete typed handles (boxing the `impl Future`
      returns into `BoxFuture`). This is the only place the driver pays for
      the dyn/Sized mismatch.
- P2: implement `load`/`unload`/`list`/`use`/`info` as a `match`
      dispatch over feature-gated `ModelSelector` variants, each arm
      calling the matching `motlie_models::<cap>::<bundle>::start_typed` and
      constructing a `LoadedModel` via a per-bundle `from_*` helper that
      populates `LoadedCapabilities` and the `ShutdownFn`.
- P3: implement `chat` one-shot + `session` multi-turn.
- P4: implement `transcribe` over both `BatchTranscriber` and
      `StreamingTranscriber` (flag-selectable); wav decoder resamples to the
      bundle's required rate via an anti-alias-filtered resampler (not
      `LinearInterpolator`) before constructing the typed `AudioBuf`.
- P5: implement `synthesize` over `SpeechSynthesizer` and the 16 kHz mono
      `VoiceCloneSynthesizer` branch; `--reference-audio` path reuses the
      example-layer `decode_wav_to_reference` shape but pre-filters downsamples
      so the 16 kHz contract receives bandlimited content.
- P6: implement `embed` over the typed embedding handle returned by
      `start_typed`.
- P7: implement `pipeline tts-asr` on top of
      `motlie_model::typed::stream_speech_into_asr`, with a small
      `--keep-wav` fork that tees first-chunk-sized WAV bytes to disk.
- P8: `bins/models/driver` with REPL frontend + asciicast recording option.
- P9: tests (analog of `tmux.rs` tests — completion, help, error arms).
      Integration harness uses already-cached artifacts under
      `default_artifact_root()` and enables per-model features behind
      `--ignored` flags for CI.

Each PLAN task references a section of this DESIGN.

## Changelog link-back

When implementation proceeds, each commit should update this doc's Changelog
with `(date, identity, summary)` and reference any sibling changes in
`libs/driver/docs/DESIGN.md`.
