# Telnyx Gateway Live Testing

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
| 2026-06-17 | @codex-535 | Added the WER passage and noted that identity-repeat TTS can confound ASR-only WER. |
| 2026-06-17 | @codex-535 | Added live-run findings: enable identity/repeat after dial and treat voicemail as an invalid human-quality sample. |
| 2026-06-17 | @codex-535 | Documented the privacy-preserving live-call protocol for config-driven streaming ASR/TTS tests. |

## Purpose

This playbook is for short human-in-the-loop Telnyx gateway checks against the real
phone network. Use it to verify streaming ASR, streaming TTS, early response,
endpointing, echo suppression, and audio quality from a bounded live run.

Never commit human phone numbers, live Telnyx connection IDs, live public hosts,
or live routing URLs. Keep those only in the local operator config, normally:

```sh
/home/dchung/telnyx-test/gateway.toml
```

## Preconditions

- Work from `main` unless the test is explicitly for a feature branch.
- Build with all live speech features:

```sh
cargo build -p motlie-telnyx-gateway --features "sherpa piper kokoro"
```

- Start the gateway from the local TOML config, not from a checked-in `.repl`
  script:

```sh
set -a
. /home/dchung/telnyx-test/telnyx.env
set +a
cargo run -p motlie-telnyx-gateway --features "sherpa piper kokoro" -- \
  --config /home/dchung/telnyx-test/gateway.toml
```

The local config should set:

- `conversation.processor = "identity"`
- `conversation.barge_in_enabled = false`
- `conversation.tts_backend = "kokoro-82m"`
- `startup.warm_models = true`
- `voice_quality.tts.generation_mode = "streaming"`
- `voice_quality.tts.chunking_enabled = true`
- `voice_quality.tts.prebuffer_chunks = 1`
- `voice_quality.early_response.enabled = true`
- `voice_quality.early_response.boundary = "none"`
- `voice_quality.early_response.start_timing = "while_speaking"`
- `voice_quality.early_response.debounce_ms = 0`
- `voice_quality.barge_in.enabled = false`
- `voice_quality.logging.enabled = true`

## Readiness Check

Use the operator socket to confirm the active settings before a call:

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

Expected mode checks:

- ASR next/default is `kroko-2025`.
- TTS backend is `kokoro-82m`.
- TTS `generation_mode=streaming`.
- Early response is enabled with `boundary=none`.
- Early response `start_timing=while_speaking`.
- Conversation barge-in is off.
- Warm reports ASR and TTS ready.

## Privacy-Preserving Dial

Do not paste a human destination number into chat, issue comments, commit
messages, docs, or shell commands captured by an agent transcript. If the agent
cannot place the call without storing the number, the human operator should run
the dial locally:

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

Inbound tests are also acceptable when the caller already knows the configured
Telnyx number. Do not write that number into committed artifacts.

After an outbound `dial`, enable the identity/repeat test processor for the
selected call and re-confirm barge-in is off:

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

## Caller Script

Give the caller this script before dialing:

```text
This is David testing Motlie live streaming voice.

I am going to speak in one continuous turn so we can observe whether the gateway starts responding early, whether it overlaps me, and whether the transcript keeps enough context.

The quick brown fox walked past the live gateway while I was still speaking, and I want the system to repeat this as soon as it can without waiting for a long silence.

Now I am going to stop for a natural endpoint.

Qualitative feedback: the response felt [too early / about right / too late]. The audio sounded [clear / clipped / distorted / echoey]. The biggest issue I noticed was [describe it].
```

## WER Passage

For ASR word-error-rate checks, ask the caller to read the passage exactly. If
the purpose is pure ASR WER, prefer a transcription-only run or keep TTS silent;
identity-repeat TTS can leak back into ASR and make the transcript harder to
score.

```text
Start WER test. The museum opened before sunrise because the city expected heavy rain and slow traffic. Seven engineers carried blue notebooks, fragile microphones, and a small wooden clock into the quiet control room. Please record every word in this sentence, including numbers like forty two and seventeen, without adding extra phrases. The quick brown fox watched a bright red kite drift above the old stone bridge. End WER test.
```

When scoring, compute one strict WER that includes any extra preface words, and
one trimmed WER that starts at `Start WER test` and ends at `End WER test`.

## Log Monitoring

Monitor both the gateway log and quality event stream during the run:

```sh
tail -n 20 -F \
  /home/dchung/telnyx-gateway-live.log \
  /home/dchung/telnyx-test/quality-events.jsonl
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

Redact phone numbers, live public hosts, connection IDs, and any unrelated
personal data before copying logs into issues or PR comments.

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
- Early response: first provisional trigger, accepted/rejected provisional
  updates, and whether generation started before final ASR.
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
