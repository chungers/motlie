# Telnyx Gateway API and Pipeline Controls

## Status

Current API snapshot for the Telnyx gateway operator/TUI/socket control surface.

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-15 PDT | @codex-m6-ds-rv | Replaced split startup config with one durable `--config <gateway.toml>` file. `state dump` now emits readable TOML with full `[voice_quality.*]`; `.repl` command files are only sourced interactively with `source <path>`. |
| 2026-06-15 PDT | @codex-m6-ds-rv | Added the current ASR -> optional aggregator -> static processor -> TTS pipeline contract and the operator/config control-knob inventory. |

## End-to-End Conversation Pipeline

The gateway-local conversation path has one processor contract and one speech queue contract:

```text
ASR -> optional early-response aggregator -> ConversationProcessorKind -> SpeechQueueRequest -> TTS/media
```

`early_response.enabled` only controls whether ASR partials enter an aggregation stage before the processor. It does not select a different response processor.

### Early Response Enabled

```text
Telnyx inbound media
  -> codec decode / resample / speech gate
  -> streaming ASR session
  -> ASR partials with utterance_id, confidence, stability, speech_state
  -> aggregate_early_resp_partials(policy)
  -> ConversationProcessorKind::process_stream(EarlyResponse(event))
  -> EarlyResponseIntent::{StartOrUpdate, Cancel, Commit}
  -> SpeechQueueRequest(source_label = "early response")
  -> tts.generation_mode buffered|streaming
  -> Telnyx outbound media frames
```

### Early Response Disabled

```text
Telnyx inbound media
  -> codec decode / resample / speech gate
  -> streaming ASR session
  -> local endpoint + ASR finish + final settle
  -> ConversationProcessorKind::process_stream(CommittedTurn(turn))
  -> ConversationCommand::Say { text }
  -> SpeechQueueRequest(source_label = "conversation say")
  -> tts.generation_mode buffered|streaming
  -> Telnyx outbound media frames
```

When the aggregator is disabled, the processor sees only `CommittedTurn`. It does not receive ASR partial text, provisional turn IDs, confidence/stability updates, or early-response cancel/commit events.

### Processor Contract

```rust
pub enum ConversationProcessorInput {
    EarlyResponse(EarlyResponseEvent),
    CommittedTurn(ConversationCommittedTurn),
}

pub enum ConversationProcessorOutput {
    EarlyResponse(EarlyResponseIntent),
    Command(ConversationCommand),
    Error(String),
}

pub enum ConversationProcessorKind {
    Identity,
}
```

Current implementation rule: processors are statically selected by enum. Do not add `Box<dyn ...>` dispatch for gateway-local processors. The only current processor is `Identity`, which repeats accepted caller text exactly and does not add an `I heard:` prefix.

## Operator Call Sequence

All commands are available through the TUI and line-oriented command socket.

Typical live identity test setup:

```text
asr use kroko-2025
tts use kokoro-82m
quality tts generation-mode streaming
quality tts chunking on
quality tts prebuffer-chunks 1
quality early-response on
quality early-response boundary none
quality early-response start-timing endpoint-candidate-only
conversation smoke-test on
conversation barge-in on
warm all
dial <+E164-number>
```

Notes:

- `conversation processor identity [call-id]` selects the per-call static processor kind, but does not by itself enable local auto-replies.
- `conversation smoke-test on` is the current command that enables gateway-local identity/repeat replies. It is not a separate processor pipeline. It turns barge-in off for deterministic echo testing, so live interruptibility tests should run `conversation barge-in on` after it.
- `quality early-response boundary none` is useful for identity latency tests with stable unpunctuated partials. Real agent tests usually prefer `clause` or `sentence`.
- `quality early-response start-timing endpoint-candidate-only` avoids starting provisional speech while the caller is still actively speaking; `while-speaking` is more aggressive and can be canceled by barge-in.

## Control Knobs

### Startup CLI vs Runtime Controls

Startup uses one declarative gateway TOML file:

```bash
telnyx-gateway --config ./gateway.toml
```

The file owns durable process/Telnyx/media/conversation/startup state plus full `[voice_quality.*]`. `state dump <path>` writes the same readable TOML shape. If `[voice_quality.logging] enabled = true`, the same file must include `[quality_logging] path = ...` so restart can recreate the JSONL sink. The CLI keeps narrow process overrides (`--bind`, `--tui`, `--socket`, `--artifact-root`, `--log-file`) for local launches, but `--quality-config`, `--quality-profile`, `--load`, `--turn-log-jsonl`, and `--conversation-smoke-test` are not part of the startup surface. Runtime tuning remains available through TUI/socket `quality ...` commands. Ad hoc command scripts can be replayed inside a running TUI/socket source with `source <path>`.

Minimal live-test config shape:

```toml
version = 1

[process]
bind = "127.0.0.1:8080"
socket = "/tmp/telnyx-gateway.sock"
artifact_root = "$HOME/artifacts/hf-cache"

[telnyx]
api_base = "https://api.telnyx.com/v2"
api_key_ref = "env:TELNYX_API_KEY"
selected_connection_id = "<telnyx-connection-id>"
selected_phone_number = "<telnyx-phone-number>"

[gateway]
webhook_url = "https://<public-host>/telnyx/webhooks"
media_url = "wss://<public-host>/telnyx/media"
from_number = "<telnyx-phone-number>"

[conversation]
enabled = true
processor = "identity"
tts_backend = "kokoro-82m"

[startup]
warm_models = true

[quality_logging]
path = "./quality-events.jsonl"

[voice_quality.tts]
generation_mode = "streaming"
chunking_enabled = true
prebuffer_chunks = 1

[voice_quality.logging]
enabled = true
redaction_mode = "metrics-only"
include_transcript_text = false
```

### Backend and Processor Selection

| Command / field | Values | Default / current intent | Apply boundary | Effect |
|---|---|---|---|---|
| `asr use <backend>` | e.g. `kroko-2025` | code/config default | next call / source-local selection | Selects ASR backend for new media sessions. |
| `tts use <backend>` | e.g. `kokoro-82m`, `piper` | code/config default | next speech request | Selects conversation/manual TTS backend. |
| `conversation processor identity [call-id]` | `identity` | `identity` | selected call | Sets per-call static processor kind. |
| `conversation smoke-test on|off` | bool | off unless enabled | runtime | Enables/disables gateway-local identity replies; no separate response branch. |
| `conversation barge-in on|off` | bool | on | next ASR/session config bridge | Operator-facing barge-in toggle; also updates quality barge-in config. |
| `warm all`, `warm asr`, `warm tts` | selected backends | manual | immediate | Preloads/probes selected models; gateway must not download artifacts silently. |

### Speech Gate and ASR

| Command / field | Values | Default | Apply boundary | Effect |
|---|---:|---:|---|---|
| `quality speech rms-threshold <value>` | `0.0..20000.0` | `220.0` | next ASR session | RMS speech gate. |
| `quality speech peak-threshold <value>` | `0..32767` | `1100` | next ASR session | Peak speech gate. |
| `quality speech onset-min-silence-ms <ms>` | `0..2000` | `180` | next ASR session | Minimum quiet gap before speech-onset barge-in can fire. |
| `quality asr finish-pad-ms <ms>` | `0..2000` | `320` | next ASR session | Short ASR flush pad after local endpoint. |
| `quality asr repeated-token-run-threshold <n>` | `2..128` | `16` | next ASR session | Repeated-token hallucination suppression. |
| `quality asr repeated-q-run-threshold <n>` | `2..64` | `8` | next ASR session | Repeated `Q` hallucination suppression. |

### Endpointing and Committed-Turn Structure

| Command / field | Values | Default | Apply boundary | Effect |
|---|---:|---:|---|---|
| `quality endpoint trailing-silence-ms <ms>` | `100..5000` | `900` | next ASR session | Acoustic endpoint tail wait. |
| `quality endpoint merge-window-ms <ms>` | `0..5000` | `350` | new conversation turn | Processor-local committed-turn debounce/coalescing window; does not merge M4 caller.turn wire events. |
| `quality endpoint final-settle-ms <ms>` | `0..5000` | `800` | next ASR session | Holds structurally incomplete ASR finals before live dispatch. |
| `quality endpoint conversation-incomplete-tail-hold-ms <ms>` | `0..10000` | `2500` | new conversation turn | Handler-local hold for incomplete committed tails. |
| `quality endpoint conversation-low-confidence-threshold-percent <percent>` | `0..100` | `45` | new conversation turn | Low-confidence non-terminal final hold threshold. |
| `quality endpoint conversation-playback-hold-poll-ms <ms>` | `10..1000` | `100` | new conversation turn | Poll cadence while waiting for active playback before coalescing. |
| `quality endpoint min-turn-words <n>` | `0..50` | `2` | report only | Labels tiny turns. |
| `quality endpoint min-turn-chars <n>` | `0..200` | `6` | report only | Labels tiny turns. |
| `quality endpoint max-turn-words <n>` | `1..500` | `80` | report only | Labels overmerged turns. |
| `quality endpoint max-turn-duration-ms <ms>` | `1000..120000` | `12000` | report only | Labels long turns. |

Config-only endpoint policy lists: `endpoint.final_settle_trailing_punctuation`, `endpoint.final_settle_lead_words`, `endpoint.final_settle_tail_words`, `endpoint.final_settle_dangling_suffixes`, and `endpoint.conversation_tail_words`.

### Early Response Aggregator

| Command / field | Values | Default | Apply boundary | Effect |
|---|---|---:|---|---|
| `quality early-response on|off` / `early_response.enabled` | bool | `false` | new call | Inserts/removes the provisional ASR aggregation stage. |
| `quality early-response boundary <none|clause|sentence>` / `early_response.boundary` | enum | `clause` | new call | Minimum text boundary before provisional processor work can start. |
| `quality early-response start-timing <endpoint-candidate-only|while-speaking>` / `early_response.start_timing` | enum | `while_speaking` | new call | Chooses whether provisional work can start during active speech or only after endpoint-candidate state. |
| `early_response.audio_mode` | `speak_provisionally`, `prepare_only` | `speak_provisionally` | new call | `speak_provisionally` allows provisional gateway TTS; `prepare_only` forwards provisional events but suppresses provisional gateway TTS until the committed-turn path. |
| `early_response.min_text_chars` | count | `12` | new call | Minimum partial length. |
| `early_response.min_text_tokens` | count | `3` | new call | Minimum partial token count. |
| `early_response.min_confidence` | optional score | `0.70` | new call | Minimum model-native ASR confidence when present. |
| `early_response.min_stability` | optional score | `0.80` | new call | Minimum gateway-estimated partial stability. |
| `early_response.missing_signal_policy` | `conservative` | `conservative` | new call | Missing required scores fail closed. |
| `early_response.debounce_ms` | ms | `120` | new call | Minimum time between provisional updates. |
| `early_response.max_updates_per_utterance` | count | `3` | new call | Churn guard for provisional updates. |
| `early_response.append_mode` | `replace_only`, `prefix_monotonic_backend` | `replace_only` | new call | Whether updates replace or append suffix text. |
| `early_response.provisional_max_prebuffer_frames` | count | `1` hard invariant | new call | JIT provisional playback cap to limit stale-audio leakage; validation currently requires exactly `1`. |

Only `enabled`, `boundary`, and `start_timing` are currently exposed as live commands. The remaining early-response fields are gateway TOML config knobs and appear in `quality early-response status` or config snapshots.

### TTS and Outbound Speech

| Command / field | Values | Default | Apply boundary | Effect |
|---|---|---:|---|---|
| `quality tts generation-mode <buffered|streaming>` | enum | `buffered` | new playback request | Selects buffered vs incremental TTS execution inside `SpeechQueueRequest`. |
| `quality tts chunking on|off` | bool | `true` | new playback request | Enables sentence/word text chunking before TTS. |
| `quality tts max-text-chunk-chars <n>` | `40..500` | `90` | new playback request | Steady-state packed text chunk limit. |
| `quality tts first-chunk-max-chars <n>` | `0` or `40..500` | `40` | new playback request | First chunk ramp; `0` disables. |
| `quality tts prebuffer-chunks <n>` | `1..64` | `1` | new playback request | Prepared chunks required before committed playback starts. |

### Barge-In and Echo Suppression

| Command / field | Values | Default | Apply boundary | Effect |
|---|---|---:|---|---|
| `quality barge-in on|off` | bool | `true` | next ASR session | Master quality barge-in toggle. |
| `quality barge-in speech-onset on|off` | bool | `true` | next ASR session | Frame-level onset cancellation. |
| `quality barge-in onset-during-playback <defer-to-partial|trust>` | enum | `defer_to_partial` | next ASR session | Echo guard vs immediate interruption during active playback. |
| `quality barge-in partial-asr on|off` | bool | `true` | next ASR session | Partial-ASR cancellation. |
| `quality barge-in final-asr on|off` | bool | `true` | next ASR session | Final-ASR cancellation. |
| `quality barge-in clear-timeout-ms <ms>` | `100..10000` | `1000` | new cancel request | Clear/terminal wait timeout. |
| `quality echo-suppression on|off` | bool | `true` | next ASR session | Text-domain assistant echo suppression. |
| `quality echo-suppression min-text-chars <n>` | `1..500` | `10` | next ASR session | Minimum transcript length for echo matching. |
| `quality echo-suppression tail-window-ms <ms>` | `0..10000` | `2000` | next ASR session | Window after assistant audio for echo matching. |
| `quality echo-suppression short-token-coverage-percent <n>` | `0..100` | `66` | next ASR session | Short-text coverage threshold. |
| `quality echo-suppression short-longest-token-run <n>` | `1..64` | `2` | next ASR session | Short-text contiguous run threshold. |
| `quality echo-suppression long-min-tokens <n>` | `2..64` | `4` | next ASR session | Token count where long-text rules apply. |
| `quality echo-suppression long-token-coverage-percent <n>` | `0..100` | `60` | next ASR session | Long-text coverage threshold. |
| `quality echo-suppression long-longest-token-run <n>` | `1..64` | `3` | next ASR session | Long-text contiguous run threshold. |

### Text-Call, Logging, and Judge Controls

| Command / field | Values | Default | Apply boundary | Effect |
|---|---|---:|---|---|
| `quality text-call max-active-turns <n>` | `1..1024` | `32` | new text-call session | Outstanding caller-turn cap. |
| `quality text-call media-ready-timeout-ms <ms>` | `1000..120000` | `20000` | new playback request | Wait for media readiness. |
| `quality text-call playback-wait-timeout-ms <ms>` | `1000..600000` | `180000` | new playback request | Playback completion timeout. |
| `quality text-call latest-response-wins on|off` | bool | `true` | new turn | Cancel/replace policy for stale app responses. |
| `quality text-call callback-timeout-ms <ms>` | `100..60000` | `5000` | new callback attempt | Subscriber callback timeout. |
| `quality logging on <path>` / `off` | path/bool | configured | immediate | Enables/disables JSONL quality events. |
| `quality logging include-transcript-text on|off` | bool | `false` | immediate | Sensitive transcript opt-in. |
| `quality logging redaction-mode <mode>` | `metrics-only`, `hashed-text`, `redacted-text`, `sensitive-plaintext` | `metrics-only` | immediate | Transcript payload policy. |
| `quality judge on|off` | bool | `false` | offline/reporting | Enables quality judge sampling when implemented. |

For the full normalized event schema and report fields, see `PROFILING.md`.
