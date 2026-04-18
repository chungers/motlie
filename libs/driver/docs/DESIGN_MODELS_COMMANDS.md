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

## Problem

`libs/models` now owns a Catalog of curated ASR, TTS, chat, and embedding
bundles (`libs/models/src/lib.rs:910-1145`). Today each capability is
exercised by a hand-rolled example binary in `libs/models/examples/v0.*` and
`tts_v0.*` — one `main.rs` per slice, each argv-parsed separately, with no
shared surface for interactive experimentation.

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
  is stateless (`libs/model/src/lib.rs:598-601`), so history lives in the
  driver.
- Discoverable completion: loaded aliases, available selectors, session ids,
  capability-typed argument validation.
- Drop-in binary `bins/models/driver` that reuses `run_repl` / `run_tui` the
  same way `bins/tmux/driver/src/main.rs:25-54` does.

## Non-goals

- No changes to `libs/models` public API (no new factory functions, no
  reshaping of `BundleHandle`). The driver consumes what's there.
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
| motlie_models             |   ModelSelector::from_str → Bundle → start →
|   (libs/models)           |   BundleHandle → {chat,speech,transcription,...}
+---------------------------+
```

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
2. Open `SpeechStream` via the TTS alias, drain all chunks into an in-memory
   PCM buffer tagged with the stream's `AudioSpec`.
3. Open `TranscriptionStream` via the ASR alias with that same `AudioSpec`,
   push the buffer in ~6400-byte chunks (matching `v0.7/main.rs:95`), finish,
   collect transcript.
4. Print: input text, input hash/length, output transcript, a simple
   case-insensitive word-overlap ratio, and optionally save the intermediate
   wav to `--keep-wav <path>`.

## CommandSet

```rust
// libs/driver/src/commands/models.rs

use motlie_models::{
    Catalog, ModelSelector, BundleDescriptor, BackendKind, CapabilityKind,
};
use motlie_model::{
    BundleHandle, ChatMessage, ChatRequest, ChatRole, ChatResponse,
    LoadedBundleDescriptor, QuantizationBits, StartOptions, ArtifactPolicy,
    AudioSpec, SpeechRequest, SpeechParams, VoiceConditioning,
    TranscriptionParams,
};

pub struct ModelsState {
    pub(crate) catalog: Catalog,
    pub(crate) artifact_root: PathBuf,
    pub(crate) loaded: BTreeMap<String, LoadedModel>,
    pub(crate) current: Option<String>,
}

pub(crate) struct LoadedModel {
    pub alias: String,
    pub selector: ModelSelector,
    pub descriptor: LoadedBundleDescriptor,
    pub backend: BackendKind,
    pub handle: Box<dyn BundleHandle>,
    pub sessions: BTreeMap<String, ChatSession>,
    pub current_session: Option<String>,
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

Maps directly to:

```rust
let selector: ModelSelector = cmd.selector.parse()?;
let bundle = selector.bundle();
let handle = bundle.start(StartOptions {
    artifact_policy: Some(match cmd.allow_fetch {
        true  => ArtifactPolicy::AllowFetch { root: cmd.artifact_root },
        false => ArtifactPolicy::LocalOnly   {
            root: cmd.artifact_root.unwrap_or_else(|| state.artifact_root.clone())
        },
    }),
    quantization: cmd.quantization.map(Into::into),
    ..Default::default()
}).await?;
```

Output: `loaded '<alias>' = <selector>  backend=<BackendKind>  q=<resolved_quantization>`.

#### `unload <alias>`

Calls `handle.shutdown().await`. Removes from `state.loaded`. Clears
`state.current` if it matched.

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
- `audio_spec` for TTS models (opened lazily: `synthesize --probe` is an
  alternative, but `info` is clearer — we start a zero-length `SpeechStream`
  once, read `audio_spec()`, drop it, and cache the spec on `LoadedModel`).
  For ASR models there is no intrinsic audio_spec; it is set per-stream. The
  `info` block for ASR reports "audio_spec: caller-supplied per stream".
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

#### `transcribe <wav-path> [--alias NAME] [--language LANG] [--emit-partials] [--chunk-bytes N]`

Opens `TranscriptionModel::open_stream`, pushes chunks (default 6400 bytes,
same as `v0.7/main.rs:95`), collects final segments, prints timestamped
transcript. `--emit-partials` plus the stream returning non-empty
`TranscriptionUpdate` during `push_chunk` triggers `[partial]` prefixed log
lines; final segments are always printed.

Maps to (paraphrased from `v0.7/main.rs:77-127`):

```rust
let (pcm_bytes, encoding) = decode_wav(wav_path)?;
let stream = asr.transcription()?.open_stream(
    AudioSpec { sample_rate_hz, channels, encoding },
    TranscriptionParams { language, emit_partials },
).await?;
push_chunks(&mut stream, pcm_bytes, chunk_bytes).await?;
let final_update = stream.finish().await?;
```

### TTS

#### `synthesize <text...> --out <wav-path> [--alias NAME] [--speaking-rate F] [--reference-audio PATH] [--reference-text TXT] [--seed N]`

Opens `SpeechModel::open_stream`, pulls chunks, writes a `.wav` file using
`hound` exactly like `tts_v0.4/main.rs:77-152`. `VoiceConditioning` is built
from `--reference-audio` + `--reference-text` if provided, else `None`.

Returns `CommandOutput` with bytes written, sample rate, encoding, output
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

Sketch:

```rust
let tts = state.loaded_mut(&cmd.tts_alias)?;
let asr = state.loaded_ref(&cmd.asr_alias)?;

let speech = tts.handle.speech()?;
let mut sstream = speech.open_stream(SpeechRequest {
    text: cmd.text.join(" "), .. Default::default()
}).await?;
let audio_spec = sstream.audio_spec().clone();
let mut pcm = Vec::new();
while let Some(chunk) = sstream.next_chunk().await? {
    pcm.extend_from_slice(&chunk.data);
    if chunk.end_of_stream { break; }
}
sstream.finish().await?;
if let Some(path) = cmd.keep_wav { write_wav(path, &audio_spec, &pcm)?; }

let mut tstream = asr.handle.transcription()?
    .open_stream(audio_spec, TranscriptionParams::default()).await?;
push_chunks(&mut tstream, pcm, 6400).await?;
let final_update = tstream.finish().await?;

CommandOutput {
    lines: vec![
        format!("input:  {input}"),
        format!("output: {}", final_update.segments.iter().map(|s| &s.text).join(" ")),
        format!("overlap: {:.1}%", word_overlap(&input, &output) * 100.0),
    ],
    effects: vec![],
}
```

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

| Command             | libs/models entry                                | libs/model capability trait               |
|---------------------|--------------------------------------------------|-------------------------------------------|
| `load`              | `ModelSelector::from_str` → `bundle()` → `start` | —                                         |
| `unload`            | `BundleHandle::shutdown`                         | —                                         |
| `list --available`  | `Catalog::bundles`                               | —                                         |
| `list --loaded`     | `LoadedBundleDescriptor` on each `BundleHandle`  | —                                         |
| `info`              | `BundleHandle::descriptor` + `metric_snapshot`   | —                                         |
| `chat`, `session …` | `handle.chat()`                                  | `ChatModel::generate`                     |
| `transcribe`        | `handle.transcription()`                         | `TranscriptionModel::open_stream`, stream |
| `synthesize`        | `handle.speech()`                                | `SpeechModel::open_stream`, stream        |
| `embed`             | `handle.embeddings()`                            | `EmbeddingModel::embed`                   |
| `pipeline tts-asr`  | both `speech` + `transcription`                  | both                                      |

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
- P2: implement `load`/`unload`/`list`/`use`/`info` over
      `Catalog::with_defaults()` + `ModelSelector::from_str`.
- P3: implement `chat` one-shot + `session` multi-turn.
- P4: implement `transcribe`.
- P5: implement `synthesize`.
- P6: implement `embed`.
- P7: implement `pipeline tts-asr`.
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
