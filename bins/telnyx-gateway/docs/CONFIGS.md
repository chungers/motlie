# Telnyx Gateway Configs

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
| 2026-06-25 PDT | @codex-541 | Added the current barge-in coalesce-after-silence Identity profile and live-run findings. |
| 2026-06-25 PDT | @codex-541 | Documented final-ASR active-playback confidence/stability gates for barge-in policy tuning. |
| 2026-06-24 PDT | @codex-541 | Added a user-facing guide for the strict global TOML config surface, live-run config records, and conversation policy tuning knobs. |
| 2026-06-24 PDT | @codex-541 | Updated the no-barge-in Identity recommended profile with the latest live-run tuning values and follow-up priorities. |
| 2026-06-24 PDT | @codex-541 | Clarified that `first_chunk_max_chars` is enforced by shared streaming TTS chunking and that quality turn counts are source-turn aware for coalesced outputs. |

## Purpose

The Telnyx gateway is configured through one strict TOML document passed with
`--config`. The checked-in canonical example is
[`../gateway.toml`](../gateway.toml). Live routing values must stay in local
operator copies under `$HOME/telnyx-test`; never commit phone numbers,
connection IDs, live public hosts, call IDs, or literal API keys.

The config loader accepts either plain TOML or hybrid live-run records with TOML
front matter:

```toml
+++
version = 1

[process]
bind = "127.0.0.1:8080"
+++

## Run Results

- Verdict: pending
```

Hybrid records are parsed with the `gray_matter` TOML front-matter parser. Only
the TOML front matter is sent to the strict config loader, so appended Markdown
or result TOML below the closing delimiter does not affect startup. Unknown
config keys still fail closed.

## Config Families

Use `bins/telnyx-gateway/gateway.toml` as the redacted canonical baseline and
copy one file per live run:

```text
$HOME/telnyx-test/runs/<YYYYMMDD-HHMMSS>-<hypothesis>-vN/<YYYYMMDD-HHMMSS>-<hypothesis>-vN.toml
```

Local run files are both startup config and run record. After hangup, append the
structured WER, latency, transcript, qualitative feedback, bugs, and next-run
tuning notes below the closing front-matter delimiter.

## Secrets And Routing

Use secret references only:

```toml
[telnyx]
api_key_ref = "env:TELNYX_API_KEY"
selected_connection_id = "<telnyx-connection-id>"
selected_phone_number = "<telnyx-phone-number>"

[gateway]
webhook_url = "https://<public-host>/telnyx/webhooks"
media_url = "wss://<public-host>/telnyx/media"
from_number = "<telnyx-phone-number>"
```

Live values belong only in local files such as
`$HOME/telnyx-test/gateway.toml` or per-run configs.

## Process And Logging

`[process]` controls local process IO:

| Key | Purpose |
| --- | --- |
| `bind` | HTTP/WebSocket bind address. |
| `tui` | Starts the interactive operator TUI when true. |
| `socket` | Unix-domain command socket for local operator control. |
| `artifact_root` | ASR/TTS artifact cache root; `$HOME` and `~` are expanded. |
| `log_file` | Structured process log path. |

`[quality_logging] path` selects the JSONL quality-event sink. The writer is
enabled by `[voice_quality.logging] enabled = true`.

## Call Direction

`[inbound] mode = "manual"` is the preferred live-test setting. The operator
starts the gateway, asks the human to dial the configured Telnyx number, checks
`calls`, then answers explicitly. Use `auto-transcribe` only when auto-answer
itself is under test.

Outbound tests use the same global config plus the socket `dial` command. Do
not paste human destination numbers into chat, docs, commits, or issue comments.

## Conversation

`[conversation]` enables the gateway-local conversation handler:

| Key | Typical live identity value | Purpose |
| --- | --- | --- |
| `enabled` | `true` for inbound identity/repeat; `false` for outbound pre-dial script control | Enables auto conversation for new calls. |
| `final_coalescing_enabled` | `true` | Coalesces committed finals before processor dispatch. |
| `barge_in_enabled` | `false` for no-barge-in smoke tests | Startup bridge for `[voice_quality.barge_in]`. |
| `processor` | `"identity"` | Processor kind. Use `"turn_batched_identity"` only for turn-batching tests. |
| `tts_backend` | `"kokoro-82m"` | Conversation/manual TTS backend. |

For the current no-turn-batching Identity smoke test, keep
`processor = "identity"` and do not set `[conversation.identity_turn_batcher]`.

## Voice Quality

`[voice_quality] profile` applies a baseline before explicit table overrides.
Use explicit overrides in live-run configs so each run is self-describing.

### Speech

`[voice_quality.speech]` controls frame-level speech detection:

| Key | Default baseline | Purpose |
| --- | --- | --- |
| `rms_threshold` | `220.0` | RMS energy gate. |
| `peak_threshold` | `1100` | Peak amplitude gate. |
| `onset_min_silence_ms` | `180` | Quiet gap before a new speech onset can fire. |

### Endpoint

`[voice_quality.endpoint]` controls finalization and conversation dispatch:

| Key | Live identity starting point | Purpose |
| --- | --- | --- |
| `trailing_silence_ms` | `850` | Silence before local endpoint for the current no-barge-in Identity baseline. |
| `min_turn_words` | `2` | Minimum words for committed turn dispatch. |
| `min_turn_chars` | `6` | Minimum characters for committed turn dispatch. |
| `merge_window_ms` | `120` | Merge adjacent ASR finals from one thought. |
| `final_settle_ms` | `500` | Hold structurally incomplete final fragments. |
| `final_settle_trailing_punctuation` | `[",", ":", ";"]` | Punctuation that suggests continuation. |
| `final_settle_lead_words` | See canonical config | Lead words that suggest continuation. |
| `final_settle_tail_words` | See canonical config | Tail words that suggest continuation. |
| `final_settle_dangling_suffixes` | `["'", "-"]` | Dangling suffixes that suggest continuation. |
| `conversation_tail_words` | See canonical config | Conversation-level incomplete-tail words. |
| `conversation_incomplete_tail_hold_ms` | `250` | Hold for incomplete conversation tails. |
| `conversation_low_confidence_threshold_percent` | `45` | Backend confidence threshold for holds. |
| `conversation_playback_hold_poll_ms` | `10` | Poll cadence while waiting for playback clearance. |
| `conversation_playback_max_hold_ms` | `0` for bounded-pending reliability; positive only to test cap behavior | Active-playback max hold for final debounce/deferred Say. |
| `max_turn_words` | `80` | Report/guard threshold for long turns. |
| `max_turn_duration_ms` | `12000` | Report/guard threshold for long turns. |

### ASR

`[voice_quality.asr]` controls ASR flush and repetition suppression:

| Key | Typical value | Purpose |
| --- | --- | --- |
| `finish_pad_ms` | `320` | Extra ASR flush padding after endpoint. |
| `repeated_token_run_threshold` | `16` | Suppress repeated-token hallucinations. |
| `repeated_q_run_threshold` | `8` | Suppress repeated `Q` hallucinations. |

### TTS

`[voice_quality.tts]` controls generation and chunking:

| Key | Live identity value | Purpose |
| --- | --- | --- |
| `generation_mode` | `"streaming"` | Selects buffered vs incremental TTS. |
| `chunking_enabled` | `true` | Splits long replies before synthesis. |
| `max_text_chunk_chars` | `70` | Later chunk packing budget for the current no-barge-in Identity baseline. |
| `first_chunk_max_chars` | `40` | Hard cap for the first incremental TTS request; if the first sentence or unsentenced segment is longer, the remainder goes through normal `max_text_chunk_chars` packing. |
| `prebuffer_chunks` | `1` | Prepared chunks required before playback starts. |

### Early Response

`[voice_quality.early_response]` controls provisional response:

| Key | Live identity value | Purpose |
| --- | --- | --- |
| `enabled` | `true` | Enables provisional processing. |
| `audio_mode` | `"speak_provisionally"` | Plays provisional audio; use `"prepare_only"` for WER-only runs. |
| `min_text_chars` | `18` | Minimum partial text length. |
| `min_text_tokens` | `4` | Minimum partial token count. |
| `boundary` | `"clause"` | Required boundary: `none`, `clause`, or `sentence`. |
| `min_confidence` | `0.70` | Minimum ASR confidence when available. |
| `min_stability` | `0.80` | Minimum partial stability when available. |
| `missing_signal_policy` | `"conservative"` | Behavior when confidence/stability is missing. |
| `allowed_start_speech_states` | `["speaking", "endpoint_candidate"]` | States that may start provisional work. |
| `allowed_update_speech_states` | `["endpoint_candidate", "finalizing"]` | States that may update provisional work. |
| `debounce_ms` | `180` | Minimum interval between provisional updates. |
| `max_updates_per_utterance` | `1` | Per-utterance provisional update cap. |
| `start_timing` | `"while_speaking"` | Start while speaking or wait for endpoint candidate. |
| `append_mode` | `"replace_only"` | Append policy for provisional audio. |
| `provisional_max_prebuffer_frames` | `1` | Provisional leakage prebuffer cap. |

### Barge-In

`[voice_quality.barge_in]` gates interruption triggers:

| Key | No-barge-in identity value | Barge-in policy-test value | Purpose |
| --- | --- | --- | --- |
| `enabled` | `false` | `true` | Enables interruption/cancellation. |
| `speech_onset_cancel_enabled` | `false` | `true` | Allows frame-level speech-onset cancel. |
| `onset_during_playback` | `"defer_to_partial"` | `"defer_to_partial"` or `"trust"` | Echo-guard behavior during active playback. |
| `partial_asr_cancel_enabled` | `false` | `true` | Allows meaningful partial ASR cancel. |
| `final_asr_cancel_enabled` | `false` | `true` | Allows final ASR cancel. |
| `transcript_min_chars` | `6` | `6` | Minimum non-whitespace transcript characters before partial/final ASR may cancel active playback. |
| `transcript_min_words` | `2` | `2` | Minimum transcript words before partial/final ASR may cancel active playback. |
| `partial_min_confidence` | `0.50` | `0.50` | Minimum partial-ASR confidence during active playback; omit in ad-hoc configs to disable this gate. |
| `partial_min_stability` | `0.50` | `0.50` | Minimum partial-ASR stability during active playback; omit in ad-hoc configs to disable this gate. |
| `final_min_confidence` | `0.70` (ignored when disabled) | `0.70` | Minimum final-ASR confidence during active playback; suppresses low-confidence short finals that can self-cancel playback. |
| `final_min_stability` | unset | unset | Optional final-ASR stability floor; leave unset until the backend reports stability on finals. |
| `clear_timeout_ms` | `1000` | `1000` | Media clear/terminal wait budget. |

Set both `[conversation] barge_in_enabled = false` and
`[voice_quality.barge_in] enabled = false` for no-barge-in Identity runs.

### Conversation Policy

`[voice_quality.conversation_policy]` arbitrates assistant output overlap and
valid barge-in actions. Quality summary turn counts are source-turn aware: when
one policy-managed playback covers multiple coalesced caller turns, all linked
source turns count as attempted/played rather than being reported as excluded.

| Key | Values | Purpose |
| --- | --- | --- |
| `mode` | `current_compat`, `no_barge_in_bounded_pending`, `barge_in_cancel_only`, `barge_in_coalesce_after_silence` | Selects the conversation arbitration policy. |
| `active_playback_hold_ms` | `0..180000` | Diagnostic hold budget for policy-managed pending output behind active playback. |
| `max_pending_outputs` | `1..64` | Maximum retained assistant outputs for bounded policies. |
| `pending_output_order` | `latest_only`, `fifo` | Retained-output ordering. |
| `post_barge_in_silence_ms` | `0..30000` | Coalescing window for `barge_in_coalesce_after_silence`. |

Policy modes:

- `current_compat`: default historical behavior.
- `no_barge_in_bounded_pending`: no-barge-in repeat reliability mode; retains
  bounded pending assistant outputs and drains them after playback clears.
- `barge_in_cancel_only`: valid caller interruption cancels active output and
  preserves caller ASR without automatic stale replay.
- `barge_in_coalesce_after_silence`: cancel plus post-barge-in silence
  coalescing before processor dispatch.

Recommended no-barge-in Identity smoke-test profile:

```toml
[conversation]
enabled = true
final_coalescing_enabled = true
barge_in_enabled = false
processor = "identity"
tts_backend = "kokoro-82m"

[voice_quality.tts]
generation_mode = "streaming"
chunking_enabled = true
max_text_chunk_chars = 70
first_chunk_max_chars = 40
prebuffer_chunks = 1

[voice_quality.early_response]
enabled = true
audio_mode = "speak_provisionally"
boundary = "clause"
start_timing = "while_speaking"
debounce_ms = 180
max_updates_per_utterance = 1

[voice_quality.endpoint]
trailing_silence_ms = 850
merge_window_ms = 120
final_settle_ms = 500
conversation_incomplete_tail_hold_ms = 250
conversation_playback_hold_poll_ms = 10
conversation_playback_max_hold_ms = 0

[voice_quality.barge_in]
enabled = false
speech_onset_cancel_enabled = false
partial_asr_cancel_enabled = false
final_asr_cancel_enabled = false

[voice_quality.conversation_policy]
mode = "no_barge_in_bounded_pending"
active_playback_hold_ms = 1000
max_pending_outputs = 3
pending_output_order = "fifo"
post_barge_in_silence_ms = 1200
```

The 2026-06-24 inbound Identity run with this profile completed 12/12 attempted
playbacks with 0 canceled/failed playbacks and good reported audio quality. Keep
this as the repeat-reliability baseline. Next tune ASR fragment/domain accuracy
and TTS/serial playback latency separately.

Recommended barge-in Identity smoke-test profile:

```toml
[conversation]
enabled = true
final_coalescing_enabled = true
barge_in_enabled = true
processor = "identity"
tts_backend = "kokoro-82m"

[voice_quality.tts]
generation_mode = "streaming"
chunking_enabled = true
max_text_chunk_chars = 70
first_chunk_max_chars = 40
prebuffer_chunks = 1

[voice_quality.early_response]
enabled = true
audio_mode = "speak_provisionally"
boundary = "clause"
start_timing = "while_speaking"
debounce_ms = 180
max_updates_per_utterance = 1

[voice_quality.endpoint]
trailing_silence_ms = 850
merge_window_ms = 120
final_settle_ms = 500
conversation_incomplete_tail_hold_ms = 250
conversation_playback_hold_poll_ms = 10
conversation_playback_max_hold_ms = 0

[voice_quality.barge_in]
enabled = true
speech_onset_cancel_enabled = true
onset_during_playback = "defer_to_partial"
partial_asr_cancel_enabled = true
final_asr_cancel_enabled = true
transcript_min_chars = 6
transcript_min_words = 2
partial_min_confidence = 0.50
partial_min_stability = 0.50
final_min_confidence = 0.70
# final_min_stability intentionally unset until final ASR reports stability.
clear_timeout_ms = 1000

[voice_quality.conversation_policy]
mode = "barge_in_coalesce_after_silence"
active_playback_hold_ms = 1000
max_pending_outputs = 1
pending_output_order = "latest_only"
post_barge_in_silence_ms = 1200
```

The 2026-06-25 barge-in Identity run with this profile canceled active playback
within 12 ms, had 0 outbound underruns, and produced clear reported audio. Keep
this as the current interruption profile. For the next run, avoid the spoken
phrase `barge in` because ASR rendered it poorly; use a clearer trigger such as
`Stop now. Please repeat this replacement sentence.` Collect qualitative
feedback only after hangup or after `conversation smoke-test off`, otherwise
Identity/repeat will capture and repeat the feedback as more caller turns.

### Echo Suppression

`[voice_quality.echo_suppression]` suppresses likely assistant echo in ASR:

| Key | Typical value | Purpose |
| --- | --- | --- |
| `enabled` | `true` | Enables text-domain echo suppression. |
| `min_text_chars` | `8` to `10` | Minimum normalized text length. |
| `tail_window_ms` | `2000` to `2500` | Window after playback tail. |
| `short_token_coverage_percent` | `66` | Short-candidate token coverage threshold. |
| `short_longest_token_run` | `2` | Short-candidate contiguous token-run threshold. |
| `long_min_tokens` | `4` | Token count where long thresholds apply. |
| `long_token_coverage_percent` | `60` | Long-candidate token coverage threshold. |
| `long_longest_token_run` | `3` | Long-candidate contiguous token-run threshold. |

### Quality Logging And Judge

`[voice_quality.logging]` controls the quality-event writer:

| Key | Typical value | Purpose |
| --- | --- | --- |
| `enabled` | `true` | Enables JSONL writer at `[quality_logging].path`. |
| `queue_capacity` | `65536` | Async event queue capacity. |
| `per_frame_sample_rate` | `0.0` | High-volume media frame sampling rate. |
| `include_transcript_text` | `false` | Transcript plaintext opt-in. |
| `redaction_mode` | `"metrics-only"` | Redaction mode. |

`[voice_quality.quality_judge]` is optional/report-oriented:
`enabled`, `mode`, `sample_rate`, `model`, `batch_size`, and `timeout_ms`.

`[voice_quality.targets]` contains report-only target thresholds. These values
must not change live ASR finalization, playback ordering, hangup behavior, or
media delivery.

## Validation

Before a live call, verify the local run config parses:

```sh
cargo test -p motlie-telnyx-gateway docs_live_run_example_configs_parse_strictly
```

Then start the gateway from the exact local run config:

```sh
cargo run -p motlie-telnyx-gateway --features "sherpa piper kokoro" -- \
  --config "$HOME/telnyx-test/runs/<run-id>/<run-id>.toml"
```

For the full live-call procedure, use [`TESTING.md`](TESTING.md).
