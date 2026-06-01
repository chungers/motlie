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

Each accepted media stream creates:

- `telnyx-media.jsonl`: raw Telnyx WebSocket events after `start`
- `decoded-inbound.wav`: decoded inbound media at the observed Telnyx format
- `asr-input-16khz.wav`: samples fed into Sherpa after gating and resampling
- `transcripts.jsonl`: partial/final transcript events, including suppressed events
- `manifest.json`: call, stream, codec, sample-rate, and file metadata

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

Sherpa ONNX is the live M1 ASR backend. Build the gateway with `--features sherpa`
with no ORT-specific environment variables before running the live test.

Artifacts are loaded from:

1. `--asr-artifact-root <path>`
2. `MOTLIE_VOICE_ARTIFACT_ROOT`
3. `.agents/skills/voice/artifacts/hf-cache`

By default, missing Sherpa artifacts are downloaded through the curated `motlie-models` catalog. Pass `--no-asr-download` to fail closed instead.

For local protocol testing without model startup:

```sh
MOTLIE_TELNYX_ECHO_ASR=1 cargo run -p motlie-telnyx-gateway -- --tui --dry-run-telnyx
```
