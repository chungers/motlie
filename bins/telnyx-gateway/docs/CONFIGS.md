# Telnyx Gateway Configs

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
| 2026-07-02 PDT | @codex-541 | Added the implemented #586 `audio_barge_in` strict config surface: media-owned typed evidence bounds, policy-owned uncertain arbitration, and behavior-neutral `measure_only` default. |
| 2026-07-01 PDT | @codex-541 | Linked the #586 audio-first barge-in boundary design and clarified that `echo_characterization` remains measurement-only until media can produce typed caller-onset evidence. |
| 2026-07-01 PDT | @codex-541 | Updated the current no-barge-in Identity baseline and recorded the #587 live barge-in validation pass: 1100 ms endpoint trailing silence, 600 ms ASR finish pad, 450 ms TTS start buffer, bounded FIFO Identity for no-barge-in, and #586 AEC/VAD as the remaining follow-up. |
| 2026-06-29 PDT | @codex-541 | Added opt-in #586 `echo_characterization` diagnostic knobs for measuring inbound/outbound echo correlation before the AEC/VAD follow-up. |
| 2026-06-28 PDT | @codex-541 | Added #587 post-barge-in dispatch guard knobs for suppressing active/recent-playback echo fragments before they cancel or reach the processor. |
| 2026-06-28 PDT | @codex-541 | Marked `barge_in_coalesce_after_silence` experimental and not live-validated pending the post-playback dispatch guard tracked in #587. |
| 2026-06-28 PDT | @codex-541 | Marked the barge-in Identity profile as not yet validated under the 450 ms TTS pacing baseline after post-playback fragments escaped as new turns. |
| 2026-06-28 PDT | @codex-541 | Promoted `streaming_start_buffer_ms = 450` as the current no-barge-in Identity TTS pacing baseline after the live pacing probe eliminated underruns. |
| 2026-06-28 PDT | @codex-541 | Marked the current inbound Identity no-barge-in live-test baseline as the best repeat-reliability starting point after Layer A score-gated barge-in landed. |
| 2026-06-25 PDT | @codex-541 | Added committed streaming TTS start-buffer and tail-pad tuning knobs for outbound pacing reliability. |
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
| `trailing_silence_ms` | `1100` | Silence before local endpoint for the current no-barge-in Identity baseline; the 2026-07-01 probe produced 5/5 expected sentence turns with no split/merge. |
| `min_turn_words` | `2` | Low-information word threshold used for reports and low-confidence conversation-tail holds. |
| `min_turn_chars` | `6` | Low-information character threshold used for reports and low-confidence conversation-tail holds. |
| `merge_window_ms` | `120` | Merge adjacent ASR finals from one thought. |
| `final_settle_ms` | `500` | Hold structurally incomplete final fragments. |
| `final_settle_trailing_punctuation` | `[",", ":", ";"]` | Punctuation that suggests continuation. |
| `final_settle_lead_words` | See canonical config | Lead words that suggest continuation. |
| `final_settle_tail_words` | See canonical config | Tail words that suggest continuation. |
| `final_settle_dangling_suffixes` | `["'", "-"]` | Dangling suffixes that suggest continuation. |
| `conversation_tail_words` | See canonical config | Conversation-level incomplete-tail words. |
| `conversation_incomplete_tail_hold_ms` | `250` | Processor-local hold for incomplete or low-confidence conversation tails; active playback wait does not consume this budget. |
| `conversation_low_confidence_threshold_percent` | `45` | Backend confidence threshold for provisional low-confidence holds, including short final fragments even when ASR inserts terminal punctuation. |
| `conversation_playback_hold_poll_ms` | `10` | Poll cadence while waiting for playback clearance. |
| `conversation_playback_max_hold_ms` | `0` for bounded-pending reliability; positive only to test cap behavior | Active-playback max hold for final debounce/deferred Say. |
| `max_turn_words` | `80` | Report/guard threshold for long turns. |
| `max_turn_duration_ms` | `12000` | Report/guard threshold for long turns. |

### ASR

`[voice_quality.asr]` controls ASR flush and repetition suppression:

| Key | Typical value | Purpose |
| --- | --- | --- |
| `finish_pad_ms` | `600` | Extra ASR flush padding after endpoint for current live Identity runs; keep this with the 1100 ms trailing-silence baseline unless testing ASR tail latency. |
| `repeated_token_run_threshold` | `16` | Suppress repeated-token hallucinations. |
| `repeated_q_run_threshold` | `8` | Suppress repeated `Q` hallucinations. |

### TTS

`[voice_quality.tts]` controls generation and chunking:

| Key | Live identity value | Purpose |
| --- | --- | --- |
| `generation_mode` | `"streaming"` | Selects buffered vs incremental TTS. |
| `chunking_enabled` | `true` | Splits long replies before synthesis. |
| `max_text_chunk_chars` | `70` | Later chunk packing budget for the current no-barge-in Identity baseline. |
| `first_chunk_max_chars` | `40` | First incremental TTS request target for long responses; short complete responses that fit `max_text_chunk_chars` stay in one chunk so tiny terminal phrases are not split into their own playback tail. |
| `prebuffer_chunks` | `1` | Buffered-mode prepared chunks required before playback starts. |
| `streaming_start_buffer_ms` | `450` | Normal streaming TTS frame prebuffer before first playback frame; current no-barge-in Identity pacing baseline. Lower for latency only if underruns remain controlled. |
| `tail_pad_ms` | `200` | Silence frames appended before the final Telnyx mark for normal committed TTS, protecting final syllables from mark/tail clipping. |

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
| `transcript_min_chars` | `6` | `6` | Telemetry-only ASR drift field; non-compat cancellation does not gate on character count. |
| `transcript_min_words` | `2` | `2` | Telemetry-only ASR drift field; non-compat cancellation does not gate on word count. |
| `missing_signal_policy` | `"conservative"` | `"conservative"` | Missing or unconfigured required ASR score signals fail closed for non-compat cancellation. |
| `partial_min_confidence` | `0.50` | `0.50` | Required partial-ASR confidence during active playback for non-compat cancellation. |
| `partial_min_stability` | `0.50` | `0.50` | Required partial-ASR stability during active playback for non-compat cancellation. |
| `final_min_confidence` | `0.70` (ignored when disabled) | `0.70` | Required final-ASR confidence during active playback for non-compat cancellation. |
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
| `post_barge_in_silence_ms` | `0..30000` | Sliding silence window for `barge_in_coalesce_after_silence` before the replacement turn reaches the processor. |
| `post_barge_in_echo_guard_ms` | `0..30000` | Guard window after a coalesced barge-in dispatch; active/recent playback finals are checked before they can cancel playback or reach the processor. |
| `post_barge_in_fragment_max_chars` | `0..200` | Suppress guarded finals at or below this normalized alphanumeric character count. |
| `post_barge_in_fragment_max_words` | `0..50` | Suppress guarded finals at or below this normalized word count. |

Policy modes:

- `current_compat`: default historical behavior.
- `no_barge_in_bounded_pending`: no-barge-in repeat reliability mode; retains
  bounded pending assistant outputs and drains them after playback clears.
- `barge_in_cancel_only`: valid caller interruption cancels active output and
  preserves caller ASR without automatic stale replay.
- `barge_in_coalesce_after_silence`: cancel plus post-barge-in silence coalescing before processor dispatch. #587 adds a post-playback dispatch guard so short/echo-like active-playback finals are suppressed before they can cancel replacement playback or become new processor turns. The 2026-07-01 live Identity/repeat sample validated the dispatch guard: one replacement turn, no stale echo/fragment processor turn, and stable outbound pacing. #586 still owns AEC/VAD-anchored boundary work; the validated production default remains `current_compat` / no-barge-in.

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
streaming_start_buffer_ms = 450
tail_pad_ms = 200

[voice_quality.early_response]
enabled = true
audio_mode = "speak_provisionally"
boundary = "clause"
start_timing = "while_speaking"
debounce_ms = 180
max_updates_per_utterance = 1

[voice_quality.endpoint]
trailing_silence_ms = 1100
merge_window_ms = 120
final_settle_ms = 500
conversation_incomplete_tail_hold_ms = 250
conversation_playback_hold_poll_ms = 10
conversation_playback_max_hold_ms = 0

[voice_quality.asr]
finish_pad_ms = 600

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
post_barge_in_echo_guard_ms = 2000
post_barge_in_fragment_max_chars = 12
post_barge_in_fragment_max_words = 2
```

The current 2026-07-01 inbound Identity no-barge-in baseline keeps these
values as the best repeat-reliability starting point, with
`streaming_start_buffer_ms = 450`, `trailing_silence_ms = 1100`, and
`finish_pad_ms = 600`. The latest local TTS chunking probe preserved 5 expected
sentence turns as 5 processor-visible turns, kept the short `miss it` sentence
in one TTS chunk, and produced strict script WER 3.17% with no deletions. The
remaining `miss it` complaint is therefore not an endpoint, ASR dispatch,
Identity, chunk-loss, or Telnyx mark-handling issue; post-merge follow-up should
inspect synthesized audio/prosody for short terminal phrases and compare a
small TTS text-shaping A/B before changing endpoint knobs again. Layer A keeps
barge-in score knobs visible in the config for telemetry and fail-closed policy
validation, but this profile still disables cancellation.

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
streaming_start_buffer_ms = 450
tail_pad_ms = 200

[voice_quality.early_response]
enabled = true
audio_mode = "speak_provisionally"
boundary = "clause"
start_timing = "while_speaking"
debounce_ms = 180
max_updates_per_utterance = 1

[voice_quality.endpoint]
trailing_silence_ms = 1100
merge_window_ms = 120
final_settle_ms = 500
conversation_incomplete_tail_hold_ms = 250
conversation_playback_hold_poll_ms = 10
conversation_playback_max_hold_ms = 0

[voice_quality.asr]
finish_pad_ms = 600

[voice_quality.barge_in]
enabled = true
speech_onset_cancel_enabled = true
onset_during_playback = "defer_to_partial"
partial_asr_cancel_enabled = true
final_asr_cancel_enabled = true
transcript_min_chars = 6
transcript_min_words = 2
missing_signal_policy = "conservative"
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
post_barge_in_echo_guard_ms = 2000
post_barge_in_fragment_max_chars = 12
post_barge_in_fragment_max_words = 2
```

The 2026-06-28 barge-in Identity run with this profile and
`streaming_start_buffer_ms = 450` failed the human quality bar before #587: short
and post-playback echo-like ASR finals escaped as new turns, Identity repeated
them out of sequence, and outbound pacing recorded 77 underruns. The 2026-07-01
#587 live validation passed with this profile plus `trailing_silence_ms = 1100`,
`finish_pad_ms = 600`, and `echo_characterization.enabled = true`: the assistant
echo final was suppressed before processor dispatch, the replacement caller turn
completed, and outbound underruns were 0. #586 still owns AEC/VAD-anchored
boundary validation after analyzing echo-characterization spans. Collect
qualitative feedback only after hangup or after `conversation smoke-test off`,
otherwise Identity/repeat will capture and repeat the feedback as more caller
turns.

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

### Echo Characterization

`[voice_quality.echo_characterization]` is an opt-in diagnostic #586
instrumentation surface. It does not change ASR, barge-in, cancellation, or
conversation policy decisions. When enabled, the media layer keeps a short
rolling outbound TTS reference and emits `media.echo_characterization` quality
spans while that playback is active or recently completed. The implemented
`audio_barge_in` surface below emits typed evidence and can optionally consume
that evidence for barge-in behavior when explicitly configured.

| Key | Typical value | Purpose |
| --- | --- | --- |
| `enabled` | `false` normally; `true` for #586 live probes | Enables the diagnostic span. |
| `window_ms` | `240` | Outbound reference window compared with inbound media. |
| `max_delay_ms` | `160` | Maximum delayed-playback alignment searched for echo correlation. |
| `emit_interval_ms` | `500` | Minimum interval between diagnostic spans on a media stream. |

Use this in per-run configs when characterizing barge-in false onsets or
post-playback leakage. The #586 design in
[`DESIGN-586-aec-vad-boundary.md`](DESIGN-586-aec-vad-boundary.md) keeps this
diagnostic separate from behavior changes: policy consumes only typed
`CallerOnsetEvidence` from `[voice_quality.audio_barge_in]`, never raw
correlation or RMS fields.

### Audio Barge-In

`[voice_quality.audio_barge_in.media]` owns DSP/evidence bounds. It produces
fresh, playback-ID/epoch-scoped `CallerOnsetEvidence` from normalized inbound
PCM, the rolling outbound TTS reference, per-call ERL/delay calibration, and
transport invalidation. `[voice_quality.audio_barge_in.policy]` owns only how
conversation policy treats ambiguous/unavailable audio evidence when Layer A
ASR transcript gates pass.

Keep `mode = "measure_only"` for default and live measurement runs. Behavior
modes are explicit opt-in and strict validation rejects silent inertness:
`echo_aware_onset` requires `[voice_quality.barge_in] enabled = true` and a
cancel-capable `[voice_quality.conversation_policy] mode` such as
`barge_in_cancel_only` or `barge_in_coalesce_after_silence`; behavior mode is
rejected with `current_compat` and `no_barge_in_bounded_pending`. `aec` is
reserved and currently rejected until a production AEC backend is selected.

| Media key | Default | Purpose |
| --- | --- | --- |
| `mode` | `"measure_only"` | `measure_only` or `echo_aware_onset`. `aec` is reserved and currently rejected until an AEC backend is selected. |
| `max_evidence_age_ms` | `120` | Freshness window for policy consumption and echo-veto expiry. |
| `trusted_onset_min_windows` | `2` | Consecutive valid speech-like media windows required for `TrustedCallerOnset`. |
| `calibration_min_playback_only_ms` | `400` | Per-call playback-only ERL/delay calibration required before trusted audio cancel. |
| `delay_search_min_ms` / `delay_search_max_ms` | `0` / `240` | Plausible echo-delay search bounds. |
| `erl_min_db` / `erl_max_db` | `-60.0` / `0.0` | Plausible per-call echo return loss bounds for calibration sanity checks. |
| `min_echo_margin_db_floor` | `3.0` | Minimum calibrated positive margin above predicted echo for trusted onset. |
| `min_echo_margin_db_ceiling` | `18.0` | Upper sanity bound for confidence scaling; not a global acoustic-path threshold. |
| `max_invalid_frame_ratio` | `0.05` | Loss/reorder/stale-frame ratio above which a window fails safe. |

| Policy key | Default | Purpose |
| --- | --- | --- |
| `uncertain_policy` | `"defer_to_layer_a"` | Allows conservative Layer A ASR transcript gates when audio evidence is ambiguous/unavailable. Use `"continue_playback"` for stricter long-response validation. |

Implementation notes:

- `current_compat` does not consume Layer B audio evidence.
- `LikelyAssistantEcho` vetoes a Layer A cancel only for the fresh evidence
  window and matching playback epoch.
- Stale playback IDs/epochs are ignored; they cannot cancel replacement audio.
- Short outbound references, invalid transport, missing calibration, and
  out-of-range delay produce `Unavailable`/`Ambiguous` and fall back according
  to `uncertain_policy`.
- `mode = "aec"` is fail-closed until a production AEC backend is selected; it must not silently run the echo-aware heuristic path under an AEC label.

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
