# Telnyx Interaction Quality Design

## Changelog

| Date (PDT) | Who | Summary |
|------------|-----|---------|
| 2026-06-12 | @codex-366-impl | Folded in PR #488 docs nits: Level 3 now places handler-local coalescing and points to the PROFILING knob reference, and Level 4 future function-word ideas are distinguished from structural tail-word policies. |
| 2026-06-12 | @codex-366-impl | Documented the four endpointing levels and clarified that PR #488 covers audio/VAD, ASR finalization, and structural stability only; semantic intent endpointing remains future work. |
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

## Endpointing Levels

Live voice endpointing has four distinct levels. PR #488 intentionally scopes only levels 1-3, and only from an audio processing / ASR stability perspective. It does not decide that the caller's semantic intent is complete.

1. **Acoustic endpointing**: VAD/energy detects speech onset, low-energy tail, and likely stop-of-speech using knobs such as `speech.rms_threshold`, `speech.peak_threshold`, `speech.onset_min_silence_ms`, and `endpoint.trailing_silence_ms`. This is necessary to know when to finish or reset ASR, but it is not an intent boundary.
2. **ASR finalization endpointing**: after acoustic endpointing, the gateway feeds `asr.finish_pad_ms` silence and finishes the ASR session so the backend can emit committed final text. `caller.partial.speech_state=finalizing` describes this processing state; it still does not create a committed agent request.
3. **Transcript structural/stability endpointing**: the gateway repairs obvious ASR finalization artifacts by holding structurally incomplete finals for `endpoint.final_settle_ms`, merging continuation finals when they arrive, and exposing advisory partial `confidence` and `stability`. For coalescing conversation handlers, a second structural stage applies after settle: `endpoint.merge_window_ms` debounces committed turns and the `endpoint.conversation_*` policies extend bounded holds. `confidence` is backend-native ASR confidence; `stability` is hypothesis-survival evidence from partial continuity. Neither is semantic grouping. Full knob reference: `PROFILING.md`.
4. **Semantic intent endpointing**: deciding that the caller has completed an actionable thought, not merely stopped speaking or produced stable text. PR #488 does not implement this layer. Final `caller.turn` remains the current commit boundary for turn-based agents, and `reply_allowed=false` prevents partial-driven speech.

Future level-4 options:

- Add a lightweight structural/semantic readiness classifier that uses cheap lexical features, punctuation/grammar cues, trailing function words, partial confidence/stability, and endpoint timing to set a separate readiness signal. This would be separate from the existing Level 3 structural tail-word lists (`endpoint.final_settle_tail_words` and `endpoint.conversation_tail_words`), which are literal hold policies rather than semantic intent classifiers. It should remain bounded and explainable, not an LLM-in-the-hot-path dependency.
- For richer agents, add a streaming/revisable protocol mode where partials update an utterance-scoped draft and the agent can prepare or revise internally before final `caller.turn`; early speech would require an explicit future `reply_allowed=true` policy plus cancellation/reconciliation rules.

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
