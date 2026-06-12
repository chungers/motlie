# Telnyx Interaction Quality Design

## Changelog

| Date (PDT) | Who | Summary |
|------------|-----|---------|
| 2026-06-12 | @codex-366-impl | Addressed #481 and David's stability ruling: opt-in `caller.partial` frames carry optional backend `confidence` plus gateway-estimated `stability`, with strict guidance that stability is only a stream-convergence/churn signal for preparation/routing/debounce decisions. |
| 2026-06-11 | @codex-m6-ds-rv | Reconciled #464 speech-state docs with gateway emission: `speaking`, `endpoint_candidate`, and `finalizing` are emitted from active speech, endpoint wait, and ASR finalization paths. |
| 2026-06-11 | @codex-366-impl | Corrected #464 speech-state documentation before endpoint/finalizing emission was wired. |
| 2026-06-11 | @codex-366-impl | Clarified #464 stopping point: advisory partial text was implemented first; later #481 adds backend-native confidence plus gateway stream-convergence stability as optional advisory scores. |
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
{"type":"caller.partial","utterance_id":"utt_123","sequence":12,"text":"can you check whether","confidence":0.74,"stability":0.63,"speech_state":"speaking","reply_allowed":false}
```

The committed turn remains the existing final frame:

```json
{"type":"caller.turn","turn_id":"turn_456","utterance_id":"utt_123","sequence":18,"text":"can you check whether the gateway is using kokoro"}
```

The extension is model-agnostic:

- `utterance_id` is a gateway-generated join key for current speech.
- `sequence` preserves stream ordering.
- `text` is the current best hypothesis.
- `confidence` is optional backend/model confidence from `motlie_model::TranscriptSegment.confidence`; it is normalized to `0.0..=1.0`, uncalibrated across backends, and omitted rather than synthesized when the backend does not provide it.
- `stability` is optional gateway-estimated stream-convergence/churn signal from current/prior partial continuity for the same utterance. It is for preparation, routing, or debounce decisions only. It must never be used as truth, final response generation input, model/ASR confidence, calibrated probability, or a value to average/combine with `confidence`. It is omitted until there is survival evidence.
- `speech_state` is normalized: `speaking` while speech frames are feeding ASR, `endpoint_candidate` while the low-energy endpoint window is active, and `finalizing` for partials produced by ASR finish-pad/finalization.
- `reply_allowed` is a policy flag. The first implementation emits `false`; agents should treat partials as planning context, not permission to speak over the caller.

Issue #480 supplied the model-layer confidence carrier. Issue #481 layers the scoring fields onto the shared protocol without changing negotiation: the extension remains `motlie.telnyx.text.partials.v1`, and scoring fields are additive and optional. `confidence` and `stability` answer different questions and must not be averaged or collapsed into one score.

## Negotiation

The gateway advertises supported text-stream extensions in callback payloads. An app opts in by returning the extension in `AcceptCallResponse.extensions`; the bundled `bins/telnyx-agent` does this by default so local live calls exercise advisory partial delivery while still waiting for final `caller.turn` before speaking:

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

`bins/telnyx-agent` consumes partial scoring by aggregating latest/max confidence and stability per utterance for logs and future planning, but it still sends no speech until the final `caller.turn` arrives while `reply_allowed=false`. Agents must use stability only as a churn/convergence guard for preparation, routing, or debounce; it is not response truth and must not be combined with confidence.

## Non-Goals

- Do not emit ASR partials by default.
- Do not expose backend-specific ASR confidence schemas or treat gateway-estimated stability as confidence, truth, calibrated probability, final response input, or a value to combine with confidence.
- Do not let partials replace final `caller.turn`.
- Do not allow early spoken replies from partials until a later policy sets `reply_allowed=true` and defines interruption semantics.
