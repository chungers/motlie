# Telnyx Gateway

Milestone 1 is inbound call transcription in the operator TUI.

## Sherpa Runtime

The Sherpa-backed gateway uses the upstream `sherpa-onnx` Rust crate. That
crate statically links its downloaded prebuilt `sherpa-onnx` native archive,
including the ONNX Runtime library used internally by Sherpa.

This follows the Sherpa exception in the general Motlie ORT/ONNX backend policy
in [`../../libs/model/docs/ORT_ONNX_POLICY.md`](../../libs/model/docs/ORT_ONNX_POLICY.md).

The gateway runbook does not use ORT-specific environment variables. Do not set
`ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`, or `LD_LIBRARY_PATH`, and do not build
ONNX Runtime from source for the gateway.

## Run

### Agent-Assisted Headless Test

For live tests with an agent operator, run the gateway without the TUI and expose
a local Unix-domain command socket:

```sh
cd ~/sessions/issue-358-telnyx-voice/codex-358-research/motlie
rm -f /tmp/motlie-telnyx-gateway.sock
: > /home/dchung/telnyx-gateway-live.log

env -u ORT_LIB_PATH -u ORT_LIB_LOCATION -u ORT_PREFER_DYNAMIC_LINK \
  TELNYX_API_KEY="$TELNYX_API_KEY" \
  cargo run -p motlie-telnyx-gateway --features sherpa -- \
    --bind 127.0.0.1:8080 \
    --load /home/dchung/telnyx-test/config.repl \
    --socket /tmp/motlie-telnyx-gateway.sock \
    --log-file /home/dchung/telnyx-gateway-live.log \
    --capture-dir /home/dchung/telnyx-test/captures
```

The socket accepts one gateway REPL command per line and returns one JSON object
per line:

```sh
python3 - <<'PY'
import json, socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/motlie-telnyx-gateway.sock")
for command in ["status", "inbound enable --manual", "calls", "call show"]:
    sock.sendall((command + "\n").encode())
    print(json.loads(sock.recv(65536)))
PY
```

During a shared live test the agent can start the gateway, enable manual
inbound handling, ask the human to dial the Telnyx number, run `calls`, run
`answer`, and then poll `call show` plus the structured log file for transcript
quality. Stop the gateway with:

```sh
python3 - <<'PY'
import json, socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/motlie-telnyx-gateway.sock")
sock.sendall(b"shutdown\n")
print(json.loads(sock.recv(65536)))
PY
```

The milestone 1 default media request is the known-good live path:

```text
config set media-codec PCMU
config set media-sample-rate 8000
```

To compare ASR quality against Telnyx linear PCM, enable capture and request
`L16` before answering the next call:

```text
config set capture-dir /home/dchung/telnyx-test/captures
config set media-codec L16
config set media-sample-rate 16000
```

The gateway decodes Telnyx `L16` WebSocket payloads as little-endian PCM. The
first captured `L16` live call showed clipped, unusable audio when interpreted as
big-endian and normal speech levels when interpreted as little-endian.

Each accepted media stream creates:

- `telnyx-media.jsonl`: raw Telnyx WebSocket events after `start`
- `decoded-inbound.wav`: decoded inbound media at the observed Telnyx format
- `asr-input-16khz.wav`: samples fed into Sherpa after gating and resampling
- `transcripts.jsonl`: partial/final transcript events, including suppressed events
- `manifest.json`: call, stream, codec, sample-rate, and file metadata

The WAV files are finalized with finite RIFF/data sizes so standard tools can
read their duration and sample count. Older captures written before that fix can
still be replayed through Motlie's permissive decoder.

For WER checks, read this exact reference text at a steady pace after the call is
answered:

```text
The quick brown fox jumps over the lazy dog. Motlie is testing inbound speech through Telnyx. Please record every word in this sentence clearly. The final answer should contain numbers one two three and the phrase blue copper river.
```

Replay a capture and compute WER without another phone call:

```sh
cargo run -p motlie-telnyx-gateway --features sherpa -- \
  --no-asr-download \
  replay-capture /home/dchung/telnyx-test/captures/<gateway-call-id>/<stream-id> \
  --backend sherpa \
  --reference-file /home/dchung/telnyx-test/reference.txt
```

The replay command reads `asr-input-16khz.wav`, feeds it through the selected ASR
backend in fixed chunks, and prints the assembled transcript, raw-ASR WER,
substitution/deletion/insertion counts, token-level errors, and replay latency.
Use `--backend echo` for protocol-only harness checks; `--backend auto` preserves
the live gateway default selection. For Sherpa artifact A/B, use
`--backend sherpa-zipformer-2023` and
`--backend sherpa-zipformer-kroko-2025`.

Replay the golden corpus across comparable backends:

```sh
cargo run -p motlie-telnyx-gateway --features sherpa -- \
  --no-asr-download \
  replay-corpus bins/telnyx-gateway/corpus/asr-golden.json \
  --backend sherpa-zipformer-2023 \
  --backend sherpa-zipformer-kroko-2025 \
  --chunk-ms 20
```

The checked-in corpus manifest records the M1.5 L16 `16 kHz` outbound baseline
(`29.2%`, `19 / 65`) and a PCMU `8 kHz` inbound slot with the read-aloud
reference text. The actual call audio and exact outbound 65-word reference are
private artifacts; place them at the manifest paths or use a local manifest copy
before scoring. The corpus harness reports raw ASR output only; post-ASR or
LLM-based normalization is intentionally not part of these WER numbers. Current
piece-3 artifact A/B status is tracked in
[`docs/ASR_ARTIFACT_AB.md`](docs/ASR_ARTIFACT_AB.md).

### Live Validation Notes

On 2026-06-01, after fixing the Telnyx `L16` byte order, the gateway completed
two manual inbound calls with `L16 16000Hz` media and capture enabled.

- Call `gwc_7fec3c0fa50d49ceafd08638c4c51edb`, stream
  `e0084c78-905f-4939-999c-023a305957e6`, captured under
  `/home/dchung/telnyx-test/captures/gwc_7fec3c0fa50d49ceafd08638c4c51edb/e0084c78-905f-4939-999c-023a305957e6`.
  The reference-section WER was `17.9%` (`7 / 39` words). Main errors:
  `Motlie -> MOTLEY`, `inbound -> BOWED` with inserted `IN GROUNDS IN`,
  `Telnyx -> TONICS`, and `this -> THE`.
- Call `gwc_cbbe4d99f1394bd0a96c5103a33592ea`, stream
  `42271dff-afc5-4ddf-b67b-08ffd517d0b3`, captured under
  `/home/dchung/telnyx-test/captures/gwc_cbbe4d99f1394bd0a96c5103a33592ea/42271dff-afc5-4ddf-b67b-08ffd517d0b3`.
  The intended-snippet WER was `16.2%` (`6 / 37` words). Main errors:
  `Sherpa -> SHIRBA`, inserted `TO`, `Today -> DAY`, `are -> RE`,
  `whether -> WEATHER`, and `improve -> IMPROVED`.

The second run also exposed an operator-control issue: when an ended call stayed
selected, bare `answer` targeted that old call instead of the single new waiting
call. The command now prefers the single `waiting` inbound call when no explicit
call id is provided.

### ASR-Only Outbound Test

For human-assisted ASR testing without waiting for an inbound call, the gateway
can place a test outbound call and transcribe the callee. This is not milestone 2
TTS: the gateway sends only silence keepalive frames on the outbound RTP path.

```text
config set media-codec L16
config set media-sample-rate 16000
config set capture-dir /home/dchung/telnyx-test/captures
test dial-transcribe +14155097294 --from +14159148777
```

The command uses the selected Telnyx application, public media URL, and same
Sherpa ASR/media capture pipeline as inbound calls. If `--from` is omitted, the
gateway uses `config set from-number <e164>`, then the selected Telnyx number.
After the callee answers, `call show` displays the transcript and the capture
directory contains the raw media JSONL, decoded WAV, ASR input WAV, and
transcript JSONL.

The latest prepared outbound `L16 16000Hz` ASR-only call on 2026-06-01 measured
`29.2%` WER on a `65`-word reference using `replay-capture`. The main errors
were phonetic/domain terms such as `outbound -> ALBAN/ALBOW`, `Telnyx -> TAL
NICHS`, `Sherpa -> SHARPA`, and `voice -> BOYS`; issue #371 tracks Sherpa-only
quality tuning with hotwords, model/decoder A/B, and separately scored
normalization.

Telnyx outbound calls require the Call Control application to be assigned to an
Outbound Voice Profile. If Telnyx returns `403 D38` with `Connection has no
Outbound Profile assigned`, create or select an Outbound Voice Profile in the
Telnyx portal and add the Call Control application to that profile before
running `test dial-transcribe` again.

### M2 Outbound TTS Live Test

Milestone 2 adds operator-driven outbound calls plus Piper TTS over the same
bidirectional Telnyx media WebSocket. The inbound ASR read loop remains live
while TTS is playing so the same call can be inspected with `call show` or
`status <call-id>`.

Prerequisites:

- `TELNYX_API_KEY` is exported.
- Tailscale Funnel or equivalent proxies `/` to `http://127.0.0.1:8080`.
- The selected Telnyx Call Control application has an Outbound Voice Profile.
- The `from-number` is outbound-enabled for that profile.
- Piper artifacts can be downloaded through the curated catalog, or are already
  present under the gateway artifact root.

Start a TUI session with an agent socket:

```sh
cd ~/sessions/issue-358-telnyx-voice/codex-358-research/motlie
rm -f /tmp/motlie-telnyx-gateway.sock
: > /home/dchung/telnyx-gateway-live.log

env -u ORT_LIB_PATH -u ORT_LIB_LOCATION -u ORT_PREFER_DYNAMIC_LINK \
  TELNYX_API_KEY="$TELNYX_API_KEY" \
  cargo run -p motlie-telnyx-gateway --features "sherpa piper" -- \
    --tui \
    --bind 127.0.0.1:8080 \
    --load /home/dchung/telnyx-test/config.repl \
    --socket /tmp/motlie-telnyx-gateway.sock \
    --log-file /home/dchung/telnyx-gateway-live.log \
    --capture-dir /home/dchung/telnyx-test/captures
```

If the replay file is not loaded or needs changes, run these in the TUI shell:

```text
config set webhook-url https://<host>/telnyx/webhooks
config set media-url wss://<host>/telnyx/media
config set media-codec PCMU
config set from-number +15551234567
telnyx app use <connection-id>
telnyx app webhook set https://<host>/telnyx/webhooks
tts status
```

Operator TUI flow:

```text
dial +14155550123
calls
call use <gateway-call-id>
status <gateway-call-id>
speak Hello, this is a Motlie outbound voice test.
speak cancel
speak The second sentence should start after the clear command.
call show
hangup
```

Agent socket flow uses the same commands and returns one JSON response per line.
For `status`, `calls`, `call show`, and `tts status`, the response includes a
machine-readable `data` object in addition to human-readable `lines`:

```sh
python3 - <<'PY'
import json, socket

commands = [
    "status",
    "tts status",
    "dial +14155550123",
    "calls",
]

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/motlie-telnyx-gateway.sock")
for command in commands:
    sock.sendall((command + "\n").encode())
    print(command)
    print(json.dumps(json.loads(sock.recv(65536)), indent=2))
PY
```

During validation, record for each call:

- outbound `to` and `from-number`
- selected ASR backend from `status`
- media codec/sample rate from `call show`
- whether `speak` reaches `mark-sent` and then `completed`
- whether `speak cancel` sends clear and stops queued speech
- whether the human's speech still appears in the assembled transcript while TTS
  is active

Expected structured log events include `call.outbound.dial`, `media.started`,
`tts.speak.queued`, `tts.clear.sent`, `tts.mark.sent`, `tts.mark.received`,
`transcript.partial`, and `transcript.final`. A Telnyx `403 D38` response means
the Outbound Voice Profile or outbound-enabled caller ID prerequisite is not
satisfied.

1. Expose the local listener with Tailscale Funnel:

   ```sh
   tailscale funnel --bg 8080
   ```

   Use the Funnel hostname to form:

   - webhook URL: `https://<host>/telnyx/webhooks`
   - media URL: `wss://<host>/telnyx/media`

2. Start the gateway:

   ```sh
   export TELNYX_API_KEY=...
   cargo run -p motlie-telnyx-gateway --features sherpa -- --tui --bind 127.0.0.1:8080
   ```

   The gateway starts idle. Inbound handling is disabled until the operator enables it.
   The M1 command surface intentionally keeps public webhook/media URLs in
   REPL state via `config set`; startup flags for `--webhook-url`,
   `--webhook-path`, and `--media-path` remain deferred until the external
   integration milestone.

3. In the left TUI pane, configure Telnyx:

   ```text
   config set webhook-url https://<host>/telnyx/webhooks
   config set media-url wss://<host>/telnyx/media
   telnyx app list
   telnyx app use <connection-id>
   telnyx app webhook set https://<host>/telnyx/webhooks
   telnyx number list
   telnyx number bind +15551234567 <connection-id>
   inbound enable --manual
   ```

4. Place an inbound call to the bound number.

   The pending call appears in the top-right roster. Select it if needed:

   ```text
   calls
   call use <gateway-call-id>
   answer
   ```

5. Watch the bottom-right selected-call detail pane.

   It shows call state, Telnyx IDs, media metadata, stream ID, partial transcript events, final transcript events, terminal state, and errors.

## Logs

Structured logs include:

- gateway call id
- `call_control_id`
- `call_session_id`
- `call_leg_id`
- `stream_id`
- observed codec
- observed sample rate
- transcript partial/final events

Use `RUST_LOG=debug` for more detail.

Media frame reordering starts from the first observed Telnyx `media.chunk` for
each stream instead of assuming a fixed initial chunk number. If a live call
shows stale chunk warnings, preserve the structured logs so the stream-order
assumption can be revisited with observed Telnyx payloads.

## ASR Artifacts

Sherpa ONNX is the live M1 ASR backend family. The live gateway default is
`kroko-2025` because it is the balanced profile across call-center and
PM/technical golden corpora. Use `sherpa-2023` for call-center-only deployments.
Operators and agents can switch the source-local next-call backend with
`asr use kroko-2025` or `asr use sherpa-2023`. Build the gateway with
`--features sherpa` with no ORT-specific environment variables before running
the live test.

Artifacts are loaded from:

1. `--asr-artifact-root <path>`
2. `MOTLIE_VOICE_ARTIFACT_ROOT`
3. `.agents/skills/voice/artifacts/hf-cache`

By default, missing Sherpa artifacts are downloaded through the curated `motlie-models` catalog. Pass `--no-asr-download` to fail closed instead.

For local protocol testing without model startup:

```sh
MOTLIE_TELNYX_ECHO_ASR=1 cargo run -p motlie-telnyx-gateway -- --tui --dry-run-telnyx
```
