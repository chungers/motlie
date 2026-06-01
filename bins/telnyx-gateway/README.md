# Telnyx Gateway

Milestone 1 is inbound call transcription in the operator TUI.

## ONNX Runtime

The Sherpa-backed gateway must link ONNX Runtime statically for live tests and
deployments. The workspace `ort 2.0.0-rc.12` dependency enables
`download-binaries`, so Cargo downloads the matching Pyke ONNX Runtime 1.24.x
static archive and links `libonnxruntime.a`.

This follows the general Motlie ORT/ONNX backend policy in
[`../../libs/model/docs/ORT_ONNX_POLICY.md`](../../libs/model/docs/ORT_ONNX_POLICY.md).

The gateway runbook does not use ORT-specific environment variables. Do not set
`ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`, or `LD_LIBRARY_PATH`, and do not build
ONNX Runtime from source for the gateway.

## Run

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
