# Telnyx Interaction Quality Design

## Changelog

| Date (PDT) | Who | Summary |
|------------|-----|---------|
| 2026-06-11 | @codex-366-impl | Added opt-in, model-agnostic advisory ASR partials for app/agent streams while preserving final `caller.turn` as the committed turn boundary. |

## Problem

Live voice agents feel slow when they learn the caller's intent only after ASR endpointing and finalization. Sending raw backend-specific ASR internals to agents would couple applications to Sherpa, Moonshine, or future decoders, and would make simple turn-based agents fragile.

The gateway needs a middle path: expose enough early text to let capable agents prepare, defer, or start low-risk work, without changing the stable request/response protocol that existing agents use.

## Design

Add an optional text-call extension:

```text
motlie.telnyx.text.partials.v1
```

When the extension is negotiated, the gateway may send advisory ASR partial hypotheses before the final caller turn:

```json
{"type":"caller.partial","utterance_id":"utt_123","sequence":12,"text":"can you check whether","stability":72,"speech_state":"speaking","reply_allowed":false}
```

The committed turn remains the existing final frame:

```json
{"type":"caller.turn","turn_id":"turn_456","utterance_id":"utt_123","sequence":18,"text":"can you check whether the gateway is using kokoro"}
```

The extension is model-agnostic:

- `utterance_id` is a gateway-generated join key for current speech.
- `sequence` preserves stream ordering.
- `text` is the current best hypothesis.
- `stability` is an optional normalized 0..100 score; omitted when the backend does not provide a stable signal.
- `speech_state` is normalized, currently `speaking`, `endpoint_candidate`, or `finalizing`.
- `reply_allowed` is a policy flag. The first implementation emits `false`; agents should treat partials as planning context, not permission to speak over the caller.

## Negotiation

The gateway advertises supported text-stream extensions in callback payloads. An app opts in by returning the extension in `AcceptCallResponse.extensions`:

```json
{
  "protocol": "motlie.telnyx.text.v1",
  "call_url": "wss://agent.example.com/motlie/text-calls/call_123",
  "accept": true,
  "extensions": ["motlie.telnyx.text.partials.v1"]
}
```

The local operator debug stream uses the same capability:

```text
stream attach --partials gwc_...
```

or:

```json
{"type":"debug.attach","protocol":"motlie.telnyx.text.v1","extension":"motlie.telnyx.text.debug.v1","extensions":["motlie.telnyx.text.partials.v1"],"call_id":"gwc_..."}
```

## Conversation Quality Impact

Advisory partials let capable agents:

- prefetch context or start cheap planning before endpointing finishes;
- notice caller corrections before committing a response;
- avoid replying to unstable fragments because final `caller.turn` remains authoritative;
- batch or suppress partial hypotheses locally without the gateway embedding agent-specific NLP policy;
- improve perceived latency for future streamed responders while preserving neutral behavior for current turn-based agents.

## Non-Goals

- Do not emit ASR partials by default.
- Do not expose backend-specific ASR confidence schemas.
- Do not let partials replace final `caller.turn`.
- Do not allow early spoken replies from partials until a later policy sets `reply_allowed=true` and defines interruption semantics.
