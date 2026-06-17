# Telnyx Gateway Live Testing

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
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

The current default live identity/repeat tuning profile is:

- `conversation.final_coalescing_enabled = true`
- `conversation.processor = "identity"`
- `conversation.barge_in_enabled = false`
- `conversation.tts_backend = "kokoro-82m"`
- `startup.warm_models = true`
- `voice_quality.tts.generation_mode = "streaming"`
- `voice_quality.tts.chunking_enabled = true`
- `voice_quality.tts.first_chunk_max_chars = 40`
- `voice_quality.tts.prebuffer_chunks = 1`
- `voice_quality.early_response.enabled = true`
- `voice_quality.early_response.boundary = "clause"`
- `voice_quality.early_response.start_timing = "while_speaking"`
- `voice_quality.early_response.debounce_ms = 180`
- `voice_quality.early_response.max_updates_per_utterance = 1`
- `voice_quality.barge_in.enabled = false`
- `voice_quality.echo_suppression.enabled = true`
- `voice_quality.logging.enabled = true`

Recent live-run finding: the clause/coalescing profile reduced repeated
identity fragments. Do not assume early-response is active just because the
config enables it; verify accepted/rejected provisional-response quality events.

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

Give the caller the relevant script before dialing. For ASR WER runs, use the
WER passage. For identity/repeat quality runs, use this script:

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
Startup and Readiness Check sections before dialing. The run config must enable the identity processor,
streaming TTS, early response, model warming, quality logging, and disabled
barge-in as listed in the default tuning profile above.

After the privacy-preserving outbound `dial`, enable the identity/repeat smoke
test processor for the selected call and re-confirm barge-in is off:

```sh
python3 - <<'PY'
import json, socket

commands = [
    "conversation smoke-test on",
    "conversation barge-in off",
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
audible repeat-back.

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

Append the quantitative metrics, qualitative caller feedback, WER score when
used, bugs/gaps, and proposed tuning changes to the same per-run TOML file below
the closing `+++` delimiter. That file is the run record; keep it local unless
every live routing value and personal datum has been redacted.

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

Qualitative:

- Caller-reported response timing.
- Caller-reported audio clarity, clipping, distortion, or echo.
- Whether the identity/repeat processor preserved enough of the caller's words.
- Whether endpointing felt natural or cut off trailing speech.
- Whether the system talked over the caller despite barge-in being disabled.

Report concrete tuning opportunities with the exact knob, current value,
proposed value, and expected effect.
