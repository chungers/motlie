# Telnyx Gateway Live Testing

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
| 2026-06-28 PDT | @codex-541 | Added #587 post-barge-in dispatch guard knobs to the live-test protocol; the next barge-in run should verify one clean replacement turn before #586 AEC/VAD work. |
| 2026-06-28 PDT | @codex-541 | Recorded the failed barge-in Identity 450 ms run: post-playback ASR fragments escaped as new turns, so the next barge-in step is a code fix rather than knob-only tuning. |
| 2026-06-28 PDT | @codex-541 | Promoted `streaming_start_buffer_ms = 450` for the no-barge-in Identity TTS pacing baseline and recorded that barge-in plus turn-batching N=2 still require separate validation. |
| 2026-06-28 PDT | @codex-541 | Reaffirmed the current best inbound Identity no-barge-in live-test knobs after Layer A score-gated barge-in landed: bounded FIFO pending output, streaming Kokoro first-chunk/tail padding, clause early response, and conservative score telemetry. |
| 2026-06-26 20:51 PDT | @codex-541 | Added processor-visible transcript event and first-audio latency span names to the required live-run result protocol. |
| 2026-06-26 PDT | @codex-541 | Added the PR breadcrumb requirement: each live run must update PLAN.md and commit a redacted `docs/tests/*.toml` run record linked from the roadmap. |
| 2026-06-25 PDT | @codex-541 | Added committed streaming TTS start-buffer and tail-pad tuning knobs for outbound pacing reliability. |
| 2026-06-25 PDT | @codex-541 | Recorded the current barge-in coalesce-after-silence tuning profile and latest live-run result; tightened the next-run protocol to keep qualitative feedback out of Identity repeat capture. |
| 2026-06-25 PDT | @codex-541 | Added final-ASR active-playback confidence gating after a failed barge-in run showed short finals could still cancel replacement playback. |
| 2026-06-21 | @codex-535 | Added run-by-run live tuning ladder for identity/repeat endpoint and playback-hold knobs after PR #558 live test. |
| 2026-06-24 PDT | @codex-541 | Linked the live-test playbook to the comprehensive global TOML config guide. |
| 2026-06-24 PDT | @codex-541 | Recorded latest inbound no-barge-in Identity tuning profile and live-run findings: bounded FIFO policy improved repeat reliability, with remaining ASR fragment and latency work. |
| 2026-06-24 PDT | @codex-541 | Recorded the follow-up decision to defer domain biasing, verify the shared TTS first-chunk latency fix, and treat coalesced source-turn accounting as a gateway metric correctness fix. |
| 2026-06-20 | @ops48-orchestrator | Protocol B: lead with the global run-config TOML as the single source of all batcher knobs (test protocol fully specified in one place); reframe the `conversation smoke-test --…` form as an OPTIONAL runtime operator TUI/agent socket-command override, explicitly NOT process CLI/argv flags. Per David. |
| 2026-06-20 | @codex-541-refactor | Reconciled the gateway-alone turn-batching protocols with approved DESIGN-557 §4.3 event names, deadline/status fields, and batcher-owned config knobs. Timestamp: 2026-06-21 00:46:41 UTC. |
| 2026-06-20 | @codex-541-refactor | Added gateway-alone partials-batching and turns-batching smoke-test protocols plus turn-batch lifecycle observability checks. Timestamp: 2026-06-20 23:27:26 UTC. |
| 2026-06-17 | @codex-535 | Added structured WER/latency run-record blocks and clarified outbound post-dial activation. |
| 2026-06-17 | @codex-535 | Clarified that smoke-test preserves barge-in and live tests must set barge-in explicitly. |
| 2026-06-17 | @codex-535 | Added explicit inbound/outbound identity smoke-test procedures and redacted sample run records. |
| 2026-06-17 | @codex-535 | Added the redacted live-run example config and updated the per-run config-as-record workflow. |
| 2026-06-17 | @codex-535 | Added per-run hybrid config files: strict TOML front matter plus appended run results. |
| 2026-06-17 | @codex-535 | Added the WER passage and noted that identity-repeat TTS can confound ASR-only WER. |
| 2026-06-17 | @codex-535 | Added live-run findings: enable identity/repeat after dial and treat voicemail as an invalid human-quality sample. |
| 2026-06-17 | @codex-535 | Documented the privacy-preserving live-call protocol for config-driven streaming ASR/TTS tests. |

## Purpose

This playbook is for short human-in-the-loop Telnyx gateway checks against the
real phone network. Use it to verify streaming ASR, streaming TTS, early
response, endpointing, echo suppression, and audio quality from a bounded live
run.

Never commit human phone numbers, live Telnyx connection IDs, application IDs,
tailnet hosts, public live hosts, or live routing URLs. Keep live routing values
only in local operator config files under `$HOME/telnyx-test`.

## Preconditions

- Work from `main` unless the test is explicitly for a feature branch.
- Build with all live speech features:

```sh
cargo build -p motlie-telnyx-gateway --features "sherpa piper kokoro"
```

- Source the local Telnyx secret environment. Do not put secrets in TOML:

```sh
set -a
. "$HOME/telnyx-test/telnyx.env"
set +a
```

## Per-Run Config

Every live run gets a fresh, traceable config file. That same file is both the
startup config and the run record after results are appended.

Use this naming pattern:

```text
$HOME/telnyx-test/runs/<YYYYMMDD-HHMMSS>-<hypothesis>-vN/<YYYYMMDD-HHMMSS>-<hypothesis>-vN.toml
```

Create the run config from a committed redacted example. Use the generic template for new hypotheses, or start from the direction-specific identity smoke-test samples when they match the run:

- `bins/telnyx-gateway/docs/LIVE_RUN_CONFIG.example.toml`
- `bins/telnyx-gateway/docs/LIVE_RUN_INBOUND_IDENTITY.example.toml`
- `bins/telnyx-gateway/docs/LIVE_RUN_OUTBOUND_IDENTITY.example.toml`

The global TOML surface and tuning knobs are documented in
[`CONFIGS.md`](CONFIGS.md). Use that guide as the canonical user-facing
reference for conversation, endpoint, TTS, early response, barge-in,
conversation policy, and quality logging knobs.

```sh
RUN_ID="$(date +%Y%m%d-%H%M%S)-clause-coalesce-v1"
RUN_DIR="$HOME/telnyx-test/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
cp bins/telnyx-gateway/docs/LIVE_RUN_CONFIG.example.toml "$RUN_DIR/$RUN_ID.toml"
```

Edit only the local copy. Replace `<run-id>` and the live-routing placeholders:

- `<telnyx-connection-id>`
- `<telnyx-application-name>` if the field is enabled
- `<telnyx-phone-number>`
- `<public-host>`

Keep `api_key_ref = "env:TELNYX_API_KEY"`. Never write a literal API key into
the config.

The committed example uses TOML front matter:

```toml
+++
[conversation]
enabled = true
barge_in_enabled = false
processor = "identity"
tts_backend = "kokoro-82m"

[startup]
warm_models = true

[voice_quality.tts]
generation_mode = "streaming"
chunking_enabled = true
prebuffer_chunks = 1
streaming_start_buffer_ms = 450
tail_pad_ms = 200

[voice_quality.early_response]
enabled = true
boundary = "clause"
start_timing = "while_speaking"
debounce_ms = 180
max_updates_per_utterance = 1

[voice_quality.barge_in]
enabled = false
+++

## Run Results

- Verdict: pending
```

The gateway uses the `gray_matter` TOML front-matter parser for hybrid run
files, then feeds only that TOML block into the strict gateway config loader.
The TOML section remains strict: unknown config or `voice_quality` keys fail the
startup parse. Plain `.toml` files without front matter still parse as regular
standalone gateway configs, but appended run reports require the explicit
`+++` front-matter wrapper.

Before starting the gateway, say exactly which file will be used for startup and
for result recording.

## Startup

Start the gateway with the per-run config file, not a checked-in `.repl` script
or a shared global config:

```sh
cargo run -p motlie-telnyx-gateway --features "sherpa piper kokoro" -- \
  --config "$HOME/telnyx-test/runs/<run-id>/<run-id>.toml"
```

The current default live identity/repeat tuning profile is the 2026-06-28
Layer A no-barge-in bounded-pending profile. It preserves the 2026-06-24
repeat-reliability knobs and records the conservative Layer A score-gating
knobs for traceability:

- `conversation.final_coalescing_enabled = true`
- `conversation.processor = "identity"`
- `conversation.barge_in_enabled = false`
- `conversation.tts_backend = "kokoro-82m"`
- `startup.warm_models = true`
- `voice_quality.tts.generation_mode = "streaming"`
- `voice_quality.tts.chunking_enabled = true`
- `voice_quality.tts.max_text_chunk_chars = 70`
- `voice_quality.tts.first_chunk_max_chars = 40`
- `voice_quality.tts.prebuffer_chunks = 1`
- `voice_quality.tts.streaming_start_buffer_ms = 450`
- `voice_quality.tts.tail_pad_ms = 200`
- `voice_quality.endpoint.trailing_silence_ms = 850`
- `voice_quality.endpoint.merge_window_ms = 120`
- `voice_quality.endpoint.final_settle_ms = 500`
- `voice_quality.endpoint.conversation_incomplete_tail_hold_ms = 250`
- `voice_quality.endpoint.conversation_playback_hold_poll_ms = 10`
- `voice_quality.endpoint.conversation_playback_max_hold_ms = 0`
- `voice_quality.conversation_policy.mode = "no_barge_in_bounded_pending"`
- `voice_quality.conversation_policy.max_pending_outputs = 3`
- `voice_quality.conversation_policy.pending_output_order = "fifo"`
- `voice_quality.early_response.enabled = true`
- `voice_quality.early_response.boundary = "clause"`
- `voice_quality.early_response.start_timing = "while_speaking"`
- `voice_quality.early_response.debounce_ms = 180`
- `voice_quality.early_response.max_updates_per_utterance = 1`
- `voice_quality.barge_in.enabled = false`
- `voice_quality.barge_in.missing_signal_policy = "conservative"`
- `voice_quality.barge_in.partial_min_confidence = 0.50`
- `voice_quality.barge_in.partial_min_stability = 0.50`
- `voice_quality.barge_in.final_min_confidence = 0.70`
- `voice_quality.barge_in.final_min_stability` unset until final ASR reports stability
- `voice_quality.echo_suppression.enabled = true`
- `voice_quality.logging.enabled = true`

Use `conversation_policy.mode = "current_compat"` only for baseline/regression
runs that intentionally reproduce the legacy latest-only drop behavior.

Recent live-run finding: on 2026-06-24, this profile produced a solid audible
improvement on inbound Identity/repeat with good audio quality. The completed
run recorded 12 attempted playbacks, 12 completed playbacks, 0 canceled/failed
playbacks, 13 raw ASR finals, 1 pre-fix excluded turn without playback, and 0
dropped quality events. Follow-up code classified that excluded turn as a
coalesced source-turn accounting artifact rather than a dropped repeat: quality
summary attempted/played/canceled turn counts now include every linked source
turn, not only the selected response turn. Remaining issues were ASR fragments
such as `Motlie -> Martley`, `barge in -> bar J`, and a perceived `hang`/`hang
up` tail miss, plus long first-audio/tail latency during serial playback. Do not
assume early-response is active just because the config enables it; verify
accepted/rejected provisional-response quality events.

## Run-By-Run Tuning Ladder

Change one hypothesis per live run and record the exact knob values in that
run's local TOML record. Keep the identity/repeat smoke test conservative:
streaming TTS, early response on, no turn batching, and barge-in disabled unless
the run is explicitly a barge-in stress test.

### Current Playback-Hold Profile

Use this profile when validating PR #558 no-barge-in playback backlog behavior:

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
trailing_silence_ms = 750
merge_window_ms = 120
final_settle_ms = 500
conversation_incomplete_tail_hold_ms = 250
conversation_playback_hold_poll_ms = 10
conversation_playback_max_hold_ms = 500

[voice_quality.conversation_policy]
mode = "current_compat"
active_playback_hold_ms = 1000
max_pending_outputs = 1
pending_output_order = "latest_only"
post_barge_in_silence_ms = 1200
post_barge_in_echo_guard_ms = 2000
post_barge_in_fragment_max_chars = 12
post_barge_in_fragment_max_words = 2
```

Observed on 2026-06-21 by @codex-535: this profile capped the old serial
identity-repeat backlog. `conversation.final_debounce` recorded playback hold
limit hits and `conversation.say.deferred_playback_hold` recorded max-hold drops
instead of unbounded queuing. Transport was clean, but caller feedback and ASR
finals still showed tail truncation and missing words. Treat this as a known
intermediate point, not the final quality target.

### Bounded Pending Repeat-Reliability Profile

Use this profile when validating issue #545 / PR #558 no-barge-in repeat
reliability. It keeps barge-in disabled, but retains a small FIFO queue of
completed processor outputs while active playback is draining. This tests
whether missed repeats were caused by the legacy max-hold drop policy rather
than ASR loss.

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
post_barge_in_echo_guard_ms = 2000
post_barge_in_fragment_max_chars = 12
post_barge_in_fragment_max_words = 2
```

Success criteria:

- ASR finals appear in the transcript and each expected identity repeat is
  played once after prior playback clears.
- `conversation.say.deferred_playback_hold` may report
  `max_hold_reached_retained`, but should not report legacy
  `max_hold_reached` drops for the policy-managed path.
- Queue growth remains bounded by `max_pending_outputs`; overflow should be
  recorded as dropped pending output rather than unbounded serial playback.

Latest 2026-06-28 TTS pacing result:

- Run record: `docs/tests/20260628-141804-b5ecbbed-tts-startbuf450-nobarge-v1.example.toml`.
- One config knob changed from the prior no-barge-in baseline:
  `streaming_start_buffer_ms = 450` instead of `300`.
- Caller feedback: better.
- Outbound pacing improved materially: 0 underruns and 73 ms max inter-frame
  gap, compared with the prior run's 58 underruns and 1180 ms max gap.
- Playback delivery: 2 attempted playbacks, 2 completed, 0 canceled, 0 failed,
  0 dropped quality events, and 0 excluded turns without playback.
- First-audio cost was modest for this sample: `tts.request_to_first_audio`
  p50/max was 1690 ms, versus 1612 ms in the prior 300 ms run.
- Strict full-script WER was 13.64% over 66 reference words. The core measured
  sentence was recognized exactly, including `hang up`.
- Endpoint segmentation still split one intended passage into two finals, and
  spoken coordination (`hanging up`) entered the transcript after the end marker.

Previous 2026-06-28 no-barge-in result:

- Run record: `docs/tests/20260628-135818-336e9087-layera-nobarge-identity-v1.example.toml`.
- Core measured sentence was recognized exactly, including `hang up`.
- Strict full-script WER was 20.59% over 68 reference words. Errors were mostly
  domain/marker words (`Motlie`, `barge-in`, `smoke`, marker ending) plus one
  inserted phrase (`I will now not`).
- The longer script split into two ASR finals: the first ended at `hang up`, and
  the second restarted with `near the end`. This produced two processor-visible
  Identity turns for one intended passage.
- Final summary: 2 caller turns, 2 raw ASR finals, 3 attempted playbacks, 2
  completed playbacks, 1 caller-hangup-canceled playback, 0 failed playbacks, 0
  dropped quality events, and 0 excluded turns without playback.
- Transport was clean inbound: 4868 packets, 0 lost, jitter p50 2 ms and p95 13
  ms. Outbound pacing still needed work: 58 underruns and one 1180 ms max
  inter-frame gap.

Previous 2026-06-24 result:

- Human feedback: solid improvement, good audio quality, with some remaining
  fragment misses such as `hang` vs `hang up`.
- Structured result: 12/12 attempted playbacks completed, 0 canceled/failed
  playbacks, 13 raw ASR finals, 1 pre-fix excluded turn without playback, and 0
  dropped quality events. The excluded turn was later traced to coalesced
  source-turn accounting; it should report as covered after the shared metrics
  fix.
- Scoped scripted-span WER was 13.33%, but coordination speech happened after
  answer, so treat the WER as diagnostic rather than a clean benchmark.
- Latency tail remains a bottleneck: `turn.finalize_to_first_audio` p95 was
  10.748 s and max was 19.431 s during serial Identity playback.

### Barge-In Policy Profiles

Use `barge_in_cancel_only` when validating today's interrupt semantics through
the unified policy boundary. Caller speech cancels active assistant playback,
ASR is preserved, and the next stable final dispatches normally through the
configured processor.

```toml
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
mode = "barge_in_cancel_only"
active_playback_hold_ms = 1000
max_pending_outputs = 1
pending_output_order = "latest_only"
post_barge_in_silence_ms = 1200
post_barge_in_echo_guard_ms = 2000
post_barge_in_fragment_max_chars = 12
post_barge_in_fragment_max_words = 2
```

Use `barge_in_coalesce_after_silence` when validating interruption followed by
one consolidated caller turn. This mode cancels active assistant playback,
preserves caller ASR, and holds post-barge-in finals until the configured
silence window closes before processor dispatch.

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

Success criteria:

- Barge-in cancel spans include `policy_mode`, `playback_action`,
  `generation_action`, and `caller_turn_action`.
- `barge_in_cancel_only` does not replay stale assistant output automatically.
- `barge_in_coalesce_after_silence` emits one processor turn after the silence
  window when multiple finals arrive during the post-barge-in window.
- Echo-guard still defers likely assistant echo to partial/final ASR before
  cancellation.

Latest 2026-06-28 result:

- Run record: `docs/tests/20260628-155011-09337980-bargein-startbuf450-v1.example.toml`.
- The run failed the human quality bar: caller reported words out of sequence and
  missing phrases.
- Config was the documented `barge_in_coalesce_after_silence` Identity profile
  with `streaming_start_buffer_ms = 450`, no turn batching, and conservative
  missing-signal barge-in policy.
- Prompt-handler-visible sequence had six turns instead of one clean
  replacement: opening sentence, stop/replacement sentence, `but now`, the
  marker sentence missing the end, `up now`, and a garbled assistant-echo
  fragment.
- Final summary: 6 caller turns/raw ASR finals, 6 attempted playbacks, 4
  canceled, 2 completed, 0 failed, 0 dropped quality events, and 0 excluded
  turns without playback.
- Strict full-script WER was 30.30% over 66 reference words, driven mostly by
  insertions from echo/fragment replay.
- Transport inbound was clean enough for this conclusion: 3948 packets, 0 lost,
  0 reordered. Outbound pacing regressed under cancel/replacement churn: 77
  underruns and 1119 ms max inter-frame gap.
- Conclusion: do not continue barge-in as knob-only tuning. The next step is a
  code fix to suppress or hold short/garbled post-playback finals during active
  playback and the short echo window, without relying on Identity semantics,
  script keywords, or domain biasing.

Previous 2026-06-25 result:

- `barge_in_coalesce_after_silence` canceled active playback within 12 ms, had 0
  outbound underruns, and produced clear reported audio in that sample.
- The 2026-06-28 repeat showed the profile is not robust enough under
  replacement-sentence echo/fragment conditions, so treat 2026-06-25 as an
  encouraging sample, not a validated baseline.

Next barge-in Identity run:

- Do not run another config-only barge-in probe first.
- Implement a general post-barge-in dispatch guard: during active assistant
  playback and a short post-playback echo window, short or low-content finals
  should be suppressed or held unless they carry strong non-echo evidence.
- Keep `voice_quality.barge_in.missing_signal_policy = "conservative"`,
  `partial_min_confidence = 0.50`, `partial_min_stability = 0.50`, and
  `final_min_confidence = 0.70`; leave `final_min_stability` unset until final
  ASR emits stability.
- After the code fix, rerun this same profile with the same script shape and
  verify one clean replacement turn, 0 stale echo replay, and stable outbound
  pacing.

### Next No-Barge-In Follow-Up

For the next no-barge-in Identity run, keep the bounded FIFO policy fixed and do
not re-enable turn batching. The 2026-06-28 450 ms pacing run keeps this as the
best repeat-reliability and pacing baseline, while the remaining no-barge-in
work is endpoint segmentation.

Hold this baseline unless testing one explicit hypothesis. Change one
endpointing knob per run:

```toml
[voice_quality.endpoint]
trailing_silence_ms = 850
merge_window_ms = 120
final_settle_ms = 500
conversation_incomplete_tail_hold_ms = 250
conversation_playback_hold_poll_ms = 10
conversation_playback_max_hold_ms = 0

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

Next hypotheses, one per run:

- Endpoint segmentation: test exactly one of `final_settle_ms = 650` or
  `merge_window_ms = 180`; success is one processor-visible turn for the script
  without stale tail latency or overmerging unrelated speech.
- TTS pacing: keep `streaming_start_buffer_ms = 450` as the current baseline.
  Do not test `prebuffer_chunks = 2` unless underruns return under the 450 ms
  setting.
- Barge-in validation: after the endpoint segmentation probe, re-run the current
  `barge_in_coalesce_after_silence` Identity profile with the shared 450 ms TTS
  start buffer to confirm interruption/replacement playback still has 0
  underruns and acceptable first-audio latency.
- Turn-batching validation: after barge-in is revalidated, run
  `processor = "turn_batched_identity"` with fixed N=2 and confirm the prompt
  handler sees one joined prompt with two source turns, not two separate
  single-turn prompts. Keep this separate from barge-in unless explicitly
  testing reset-on-barge-in behavior.
- Protocol cleanliness: provide/read the script before answer or immediately
  from the run config so coordination speech does not contaminate WER, and hang
  up only after confirming the final repeat has completed.

Domain/hotword biasing is intentionally deferred; do not tune domain vocabulary
until endpointing, repeat reliability, and latency are stable.

## Readiness Check

Use targeted operator commands to confirm settings before a call. Avoid broad
status output when it includes live routing values; redact before sharing.

```sh
python3 - <<'PY'
import json, socket

commands = [
    "asr status",
    "tts status",
    "quality tts status",
    "quality early-response status",
    "quality endpoint status",
    "warm all",
]

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/motlie-telnyx-gateway.sock")
reader = sock.makefile("rb")
for command in commands:
    sock.sendall((command + "\n").encode())
    print(command)
    print(json.dumps(json.loads(reader.readline()), indent=2))
PY
```

Expected checks:

- ASR next/default is `kroko-2025`.
- TTS backend is `kokoro-82m`.
- TTS `generation_mode=streaming`.
- Early response is enabled with `boundary=clause`.
- Early response `start_timing=while_speaking`.
- Early response `debounce_ms=180`.
- Early response `max_updates_per_utterance=1`.
- Conversation barge-in is off.
- Warm reports ASR and TTS ready.

## Caller Script

Give the caller the relevant script before dialing or before answering an
inbound manual call. For ASR WER runs, use the WER passage. For identity/repeat
quality runs, use this script:

```text
This is David testing Motlie live streaming voice.

I am going to speak in one continuous turn so we can observe whether the gateway starts responding early, whether it overlaps me, and whether the transcript keeps enough context.

The quick brown fox walked past the live gateway while I was still speaking, and I want the system to repeat this as soon as it can without waiting for a long silence.

Now I am going to stop for a natural endpoint.

Qualitative feedback: the response felt [too early / about right / too late]. The audio sounded [clear / clipped / distorted / echoey]. The biggest issue I noticed was [describe it].
```

## WER Passage

For ASR word-error-rate checks, ask the caller to read the passage exactly. If
the purpose is pure ASR WER, prefer a transcription-only run, set
`voice_quality.early_response.audio_mode = "prepare_only"`, or keep identity TTS
detached/muted after the WER passage starts. Identity-repeat TTS can leak back
into ASR and make the transcript harder to score.

```text
Start WER test. The museum opened before sunrise because the city expected heavy rain and slow traffic. Seven engineers carried blue notebooks, fragile microphones, and a small wooden clock into the quiet control room. Please record every word in this sentence, including numbers like forty two and seventeen, without adding extra phrases. The quick brown fox watched a bright red kite drift above the old stone bridge. End WER test.
```

When scoring, compute one strict WER that includes any extra preface words, and
one trimmed WER that starts at `Start WER test` and ends at `End WER test`.
Every scored pass must be recorded in the structured run-record block below.

## Required Structured Results

After hangup, append machine-readable result blocks below the closing `+++`
delimiter in the same run file. Markdown notes are allowed, but these structured
blocks are required whenever their data exists.

For PR tuning work, also create a redacted committed copy of the completed run
record under `bins/telnyx-gateway/docs/tests/<run-id>.example.toml` and update
[`PLAN.md`](PLAN.md)'s Live Test Breadcrumb Roadmap with the result and next
hypothesis. The committed copy must keep placeholder routing values and secret
references only.

For each scored WER pass, append one TOML block with exact text or checksum and
all edit counts:

```toml
[[wer_pass]]
schema_version = 1
reference_id = "telnyx-default-wer-v1"
reference_text = """Start WER test. The museum opened before sunrise because the city expected heavy rain and slow traffic. Seven engineers carried blue notebooks, fragile microphones, and a small wooden clock into the quiet control room. Please record every word in this sentence, including numbers like forty two and seventeen, without adding extra phrases. The quick brown fox watched a bright red kite drift above the old stone bridge. End WER test."""
reference_sha256 = "76ffa574dc018f2f579ee74aef068a869ae31ec39d70dd0a0cb476a015395ede"
raw_asr_text = """Preface audio outside the scored span. Start WER test. The museum open before sunrise because the city expected heavy rain and slow traffic. Seven engineers carry blue notebooks, fragile microphones, and a small wooden clock into the quiet control room. Please record every word in this sentence, including numbers like forty to and seventeen, without adding extra phrases. The quick brown box watched a bright red kite drift above the old stone bridge. End WER test. Feedback after the scored span."""
scored_hypothesis_text = """Start WER test. The museum open before sunrise because the city expected heavy rain and slow traffic. Seven engineers carry blue notebooks, fragile microphones, and a small wooden clock into the quiet control room. Please record every word in this sentence, including numbers like forty to and seventeen, without adding extra phrases. The quick brown box watched a bright red kite drift above the old stone bridge. End WER test."""
trim_policy = "strict_full | start_end_markers | manual_span"
trim_start_marker = "Start WER test"
trim_end_marker = "End WER test"
normalization_policy = "lowercase; strip punctuation except apostrophes; collapse whitespace; score word tokens"
wer_tool = "motlie_telnyx_gateway::replay::wer"
wer_tool_commit = "<git-sha>"
reference_word_count = 70
hypothesis_word_count = 70
wer_percent = 5.71
substitutions = 4
deletions = 0
insertions = 0
alignment = [
  { op = "sub", reference = "opened", hypothesis = "open" },
  { op = "sub", reference = "carried", hypothesis = "carry" },
  { op = "sub", reference = "two", hypothesis = "to" },
  { op = "sub", reference = "fox", hypothesis = "box" },
]
```

For each live run, append a stable latency/quality block. Use `null` only when a
metric is genuinely unavailable from the run artifacts. The processor-visible
transcript control stream is the `conversation.processor_visible_turn` quality
event, not raw ASR finals, because raw ASR may contain suppressed assistant echo:

```toml
[result_latency]
schema_version = 1
source = "quality_jsonl"
quality_events = 0
caller_turns = 0
raw_asr_final_events = 0
attempted_playbacks = 0
completed_playbacks = 0
canceled_playbacks = 0
dropped_quality_events = 0

[[result_latency.span]]
name = "asr.endpoint_wait"
n = 0
min_ms = 0
p50_ms = 0
p95_ms = 0
max_ms = 0

[[result_latency.span]]
name = "tts.request_to_first_audio"
n = 0
min_ms = 0
p50_ms = 0
p95_ms = 0
max_ms = 0

[[result_latency.span]]
name = "conversation.visible_turn_to_first_audio"
n = 0
min_ms = 0
p50_ms = 0
p95_ms = 0
max_ms = 0

[[result_latency.span]]
name = "barge_in.cancel_terminal_to_replacement_first_audio"
n = 0
min_ms = 0
p50_ms = 0
p95_ms = 0
max_ms = 0

[[result_latency.span]]
name = "conversation.final_debounce"
n = 0
min_ms = 0
p50_ms = 0
p95_ms = 0
max_ms = 0

[[result_latency.span]]
name = "conversation.say.deferred_playback_hold"
n = 0
min_ms = 0
p50_ms = 0
p95_ms = 0
max_ms = 0

[result_transport.inbound]
codec = "PCMU"
sample_rate_hz = 8000
packets_total = 0
lost_packets = 0
reordered_frames = 0
jitter_ms_p50 = 0
jitter_ms_p95 = 0
jitter_ms_max = 0
```

## Privacy-Preserving Dial

Do not paste a human destination number into chat, issue comments, commit
messages, docs, or commands captured by an agent transcript. If the agent cannot
place the call without storing the number, the human operator should run the dial
locally:

```sh
python3 - <<'PY'
import json, socket, getpass

number = getpass.getpass("Destination E.164 (not echoed): ").strip()
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/motlie-telnyx-gateway.sock")
reader = sock.makefile("rb")
sock.sendall((f"dial {number}\n").encode())
print(json.dumps(json.loads(reader.readline()), indent=2))
PY
```

## Outbound Identity Smoke-Test Runs

Use outbound identity smoke tests when the operator needs the gateway to place
the call. `docs/LIVE_RUN_OUTBOUND_IDENTITY.example.toml` is the committed
redacted sample for this path. Start from a new per-run config and complete the
Startup and Readiness Check sections before dialing. The outbound sample selects
the identity processor but intentionally starts with `conversation.enabled=false`;
post-dial `conversation smoke-test on` is the activation boundary. This protects
the script phase from callee greeting, voicemail, or early coordination audio.
Streaming TTS, early response, model warming, quality logging, and disabled
barge-in remain the deterministic baseline knobs.

After the privacy-preserving outbound `dial`, enable the identity/repeat smoke
test processor for the selected call before the caller reads the script, then
explicitly set the intended barge-in mode. `conversation smoke-test on` preserves
the current barge-in setting; it does not reset it.

```sh
python3 - <<'PY'
import json, socket

commands = [
    "conversation smoke-test on",
    "conversation barge-in off",  # use "on" for an interruption stress test
    "conversation status",
]

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/motlie-telnyx-gateway.sock")
reader = sock.makefile("rb")
for command in commands:
    sock.sendall((command + "\n").encode())
    print(command)
    print(json.dumps(json.loads(reader.readline()), indent=2))
PY
```

Expected checks:

- `processor: identity`
- `mode: auto`
- `barge_in: off`
- `attached: true`

Use the identity/repeat caller script unless the run is explicitly WER-only.
After hangup, append the same timing, quality, transcript, WER if applicable,
qualitative feedback, and next-run tuning analysis described below to the run
config.

Inbound tests are the preferred path when the caller can dial the configured
Telnyx number. Do not write that number into committed artifacts.

## Inbound Dial-In Runs

Use inbound dial-in runs for privacy-preserving human tests. The default inbound
live test is audible identity/repeat: the caller should hear the gateway repeat
what it transcribed. WER-only transcription is a separate explicit variant.

Create a new per-run config from the latest stable local run record,
`docs/LIVE_RUN_INBOUND_IDENTITY.example.toml`, or
`docs/LIVE_RUN_CONFIG.example.toml`, then make these run-specific edits in the
local copy only:

```toml
[inbound]
mode = "manual"

[conversation]
enabled = true
final_coalescing_enabled = true
barge_in_enabled = false
processor = "identity"
tts_backend = "kokoro-82m"

[voice_quality.early_response]
enabled = true
audio_mode = "speak_provisionally"
boundary = "clause"
start_timing = "while_speaking"
debounce_ms = 180
max_updates_per_utterance = 1

[voice_quality.barge_in]
enabled = false
```

The stable inbound path is manual-answer: start the gateway, ask the caller to
dial the configured Telnyx number, watch `calls`, then run `answer` when the
waiting inbound call appears. Do not rely on `auto-transcribe` for a live sample
unless that path is the explicit subject of the test and the result will record
whether it answered automatically.

For WER-only transcription, set `conversation.enabled = false` and
`voice_quality.early_response.audio_mode = "prepare_only"`, and label the run
as WER-only before startup. Do not use WER-only settings when the caller expects
audible repeat-back. For inbound identity runs, choose barge-in in the run config
before startup; the smoke-test command will preserve that setting.

Tell the caller exactly which script to read before they dial. Put that script
in the local run config below the closing `+++` delimiter so the transcript can
be scored for WER later. If the run is testing repeat-loop risk, the caller must
read the passage once, stay silent until playback finishes, and give feedback
only after the operator says the repeat phase is complete.

During the call, monitor the run-scoped log and quality JSONL. After hangup,
append to the same run config:

- call setup and answer timing
- ASR first-partial, final, endpoint, and WER metrics
- identity/repeat playback count and whether playback was still active at hangup
- media quality and transport counters
- qualitative caller feedback, if any
- bugs or protocol gaps observed
- proposed next-run config knobs and code fixes

## Gateway-Alone Smoke-Test Protocols

These protocols run entirely inside the gateway: no external agent, no LLM, and
no appserver websocket. They are live-call checks for the staged turn pipeline.
Use the same privacy, startup, readiness, and log-monitoring rules above.

### Protocol A: Partials-Batching Smoke Test

Purpose: validate layers 1-2, from ASR partials through turn formation, early
response, committed turn dispatch, and gateway-local identity echo. This mostly
reframes the existing identity/repeat run as the partials and turn-formation
validation.

Per-run config keeps the default gateway-only processor:

```toml
[conversation]
enabled = true
final_coalescing_enabled = true
barge_in_enabled = false
processor = "identity"
tts_backend = "kokoro-82m"
```

For outbound calls that start detached, enable the selected call after dialing:

```text
conversation smoke-test on identity
conversation barge-in off
conversation status
```

Readiness check: `conversation status` must show `attached: true`,
`processor: identity`, and `processor_state: identity`. The caller should read
the identity/repeat script in one natural turn, then pause for the echo.
Expected behavior is one gateway-local echo of the accepted caller text, with
early-response quality events showing accepted or rejected provisional decisions
and committed playback completing normally.

### Protocol B: Turns-Batching Smoke Test

Purpose: validate layer 3, where the gateway forms committed caller turns, hosts
`IdentityTurnBatcher`, accumulates N turns into one prompt, and echoes that
joined prompt without an external agent or LLM. PerTurn/Identity remains the
default; opt into this protocol only for turn-batching validation.

**Canonical config for the test protocol — the global TOML, fully specified in one place.** None of the batcher knobs are process CLI/argv flags; the run is driven entirely from this run-config file:

```toml
[conversation]
enabled = true
final_coalescing_enabled = true
barge_in_enabled = false
processor = "turn_batched_identity"
tts_backend = "kokoro-82m"

[conversation.identity_turn_batcher]
fixed_batch_size = 3
max_batch_turns = 3
max_batch_wait_ms = 2500
max_idle_wait_ms = 1200
join_separator = " | "
```

This is the single source for the run; the appended run record stays self-describing. The `[conversation.identity_turn_batcher]` table is the strict, opaque `IdentityTurnBatcherConfig`.

Optional runtime override (not required by this protocol): the same knobs can be set live through the **operator TUI / agent socket command** — these are arguments of the runtime `conversation` socket command, parsed from the socket line, **not** process CLI/argv flags:

```text
conversation smoke-test on turn-batched-identity --fixed-batch-size 3 --max-batch-turns 3 --max-batch-wait-ms 2500 --max-idle-wait-ms 1200 --join-separator " | "
conversation barge-in off
conversation status
```

Use the socket form only for dynamic per-call experimentation; for a recorded protocol run, specify everything in the run-config TOML above so the run file is the one self-describing source.

Ownership rule: N and all batching policy values are owned by
`IdentityTurnBatcherConfig`. The gateway is only the host: it constructs the
batcher with this opaque config, feeds ordered committed turns and acoustic
resets, maps `PromptComplete` to speech, and does not decide when a batch is
complete.

Readiness check: `conversation status` must show `attached: true`,
`processor: turn_batched_identity`, and `processor_state: turn_batched_identity`.
While the batch is holding the floor it must also show
`turn_batch.accumulating: true`, the active `turn_batch.batch_id`,
`turn_batch.epoch`, pending/target counts, first-turn age, idle age, and the
effective deadline source/remaining time.

Monitor the run log and quality JSONL for the lifecycle events
`turn_batch_accumulated`, `turn_batch_prompt_complete`, `turn_batch_reset`, and
`turn_batch_output_rejected`. The trace log messages use
`conversation.processor.turn_batch.accumulated`,
`conversation.processor.turn_batch.prompt_complete`,
`conversation.processor.turn_batch.reset`, and
`conversation.processor.turn_batch.output_rejected`. The accumulated/prompt/reset
events carry `batch_id`, `epoch`, pending/source-turn counts, first-turn age,
idle age, effective deadline source, and deadline remaining data so the
`max_batch_wait_ms` vs `max_idle_wait_ms` behavior can be validated from logs
alone.

Caller script for fixed-N validation, using N short distinct turns with clear
endpoint pauses:

```text
Turn one: alpha window.
[pause until endpoint]
Turn two: blue invoice.
[pause until endpoint]
Turn three: cedar marker.
[pause until endpoint]
```

Expected behavior: turns 1 through N-1 are silent while the batch accumulates.
The log and JSONL show `turn_batch_accumulated` with k of N, `batch_id`,
`epoch`, first-turn age, idle age, and the effective deadline source. After turn
N, the gateway emits one `turn_batch_prompt_complete` event with source-turn
count, `completion_reason`, and `accumulation_ms`, then speaks one joined echo.
The existing `quality.turn.playback_linked` event includes the turn-batch id,
epoch, response turn id, and completion reason so delivery/cancel outcome can be
joined back to the prompt. `conversation.processor_visible_turn` records the
exact text the processor saw after suppression/coalescing, and
`conversation.visible_turn_to_first_audio` records the processor-visible to
first-audio latency for the resulting speech. When barge-in cancels active
playback and a replacement is queued,
`barge_in.cancel_terminal_to_replacement_first_audio` records the terminal
cancel-to-replacement-audio latency.

Run three variants when validating a change:

- Fixed-N batch: speak exactly N turns and expect one joined echo after turn N.
- Fixed first-turn ceiling: speak fewer than N turns, wait longer than
  `max_batch_wait_ms`, and expect one partial-batch joined echo with
  `completion_reason=max_batch_wait_timeout`. Later turns must not reset this
  ceiling.
- Idle fallback: speak fewer than N turns, pause longer than `max_idle_wait_ms`
  but shorter than `max_batch_wait_ms`, and expect one partial-batch joined echo
  with `completion_reason=max_idle_timeout`. A new turn resets only this idle
  timer.
- Barge-in reset: while an existing playback is interruptible and a new batch is
  accumulating, barge in and expect `turn_batch_reset` with reason `barge_in`,
  `discarded_turn_count`, and accumulation age; the accumulated pending turns
  must not be echoed after the reset.

Record these layer-3 observations in the run results: accumulation count,
`batch_id`/`epoch` continuity, joined source-turn count, completion-reason
histogram, turns-per-prompt rollup, accumulation latency rollup,
resets-by-reason, stale/output-rejection count, whether timeout fallback fired,
and whether reset-on-barge-in dropped the pending batch.

## Log Monitoring

Monitor the run-scoped gateway log and quality stream from the same per-run
config:

```sh
tail -n 20 -F \
  "$HOME/telnyx-test/runs/<run-id>/<run-id>.log" \
  "$HOME/telnyx-test/runs/<run-id>/<run-id>-quality.jsonl"
```

After the call, retrieve the selected call transcript through the socket:

```sh
python3 - <<'PY'
import json, socket

commands = ["calls", "call show"]

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/motlie-telnyx-gateway.sock")
reader = sock.makefile("rb")
for command in commands:
    sock.sendall((command + "\n").encode())
    print(command)
    print(json.dumps(json.loads(reader.readline()), indent=2))
PY
```

Redact phone numbers, live public hosts, connection IDs, call IDs, and unrelated
personal data before copying logs into issues or PR comments.

Append the structured WER and latency/quality blocks, quantitative metrics,
qualitative caller feedback, bugs/gaps, and proposed tuning changes to the same
per-run TOML file below the closing `+++` delimiter. That file is the run record;
keep it local unless every live routing value and personal datum has been
redacted.

If the call reaches voicemail, classify the run as a media-pipeline smoke test,
not a valid human qualitative sample. The run can still measure dial setup,
media start, ASR, early response, TTS, and transport behavior, but it cannot
answer audio-quality or endpointing questions from the human caller. Record the
voicemail transcript as the reason and rerun against a reachable callee.

## Analysis Checklist

Quantitative:

- Call setup: outbound dial time, media start time, and hangup reason.
- ASR: first partial latency, final latency, final text quality, and endpoint
  timing.
- Early response: accepted and rejected provisional decisions, first
  provisional trigger, and whether generation started before final ASR.
- TTS: first audio latency, chunk/prebuffer behavior, playback completion, and
  dropped or canceled audio.
- Echo suppression: whether playback leakage created false ASR partials or
  finals.
- Quality events: malformed events, missing timestamps, or gaps that prevent
  measurement.
- Layer 3 turn batching: accumulation count matches k of N, `batch_id` and
  `epoch` stay stable until completion/reset, prompt assembly joins the expected
  source turns, `max_batch_wait_ms` is measured from the first turn,
  `max_idle_wait_ms` resets on each new turn, timeout completion reasons match
  the observed deadline source, `quality.turn.playback_linked` can be joined to
  `turn_batch_prompt_complete`, and barge-in reset records discarded turns
  without a later echo.

Qualitative:

- Caller-reported response timing.
- Caller-reported audio clarity, clipping, distortion, or echo.
- Whether the identity/repeat processor preserved enough of the caller's words.
- Whether endpointing felt natural or cut off trailing speech.
- Whether the system talked over the caller despite barge-in being disabled.

Report concrete tuning opportunities with the exact knob, current value,
proposed value, and expected effect.
