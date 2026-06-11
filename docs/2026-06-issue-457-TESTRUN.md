# Issue 457 Telnyx Gateway Live Test Run

## Changelog

| Date | Who | Summary |
|---|---|---|
| 2026-06-10 | @codex-m6-ds-rv | Added live-call evidence, qualitative transcript notes, latency budget analysis, and scoped fix plan for issue 457. |

## Context

This note captures the 2026-06-10 PDT manual Telnyx gateway run after the first usability fixes on branch `codex-m6-ds-rv/telnyx-usability-fixes`.

Artifacts:

- Log: `/home/dchung/telnyx-test/m6-live-fix-20260610-181250.log`
- Quality JSONL: `/home/dchung/telnyx-test/m6-quality-events-fix-20260610-181250.jsonl`
- Captures: `/home/dchung/telnyx-test/captures/fix-20260610-181250`
- Call: `gwc_19bd67820b3846869089164edcbb3975`

Run setup:

- `tts use piper`
- `conversation barge-in off`
- `quality endpoint merge-window-ms 0`
- `quality logging redaction-mode redacted-text`
- `quality logging include-transcript-text on`
- `warm all`

Warmup timings:

- ASR ready: 2080 ms
- Piper ready: 1051 ms

## Quantitative Observations

The conversation reply backend fix worked: all observed TTS spans used `piper`.

Observed ASR endpoint wait values were tightly clustered around the configured 650 ms tail:

- Values: `607, 663, 606, 616, 620, 665, 614, 616, 616, 641, 623`
- Average: about 626 ms
- P90 in this small sample: about 663-665 ms

Observed `tts.request_to_first_audio` / `turn.finalize_to_first_audio` values:

- Values: `106, 102, 276, 226, 120, 301, 121, 629, 161`
- Average: about 227 ms
- Median: about 161 ms
- P90 in this small sample: 301-629 ms depending interpolation/outlier treatment

Long smoke-echo playback still dominated several turns:

- Playback terminal durations included `13503` ms and `14211` ms.
- This is not first-audio latency; it is long echo content being spoken back.

## Qualitative Transcript Assessment

Positive signal from the caller transcript:

- "I like the snappiness, the responses a lot better, it's a lot smoother."

Remaining failure signal from the caller transcript:

- "end pointing seems to be a real trouble"
- "you missed the last part"
- "endpointing, detecting and flushing the last frame"

Representative clipped final transcripts:

- `Hello, what's`
- `Yeah, I`
- `... endpointing seems to be a real cha`
- `All right, hanging out pa`

Assessment:

- TTS first-audio responsiveness is materially better after switching conversation replies to Piper.
- Endpointing/final ASR text is now the primary UX blocker.
- The issue is intermittent: some later turns were sufficiently complete, but short turns and tail words are still clipped.
- The current logs do not yet distinguish whether tail loss came from media/audio capture, local endpoint timing, ASR finalization, or final-turn selection.

## Agent Latency Budget

For a natural dispatcher/operator flow, the user should hear the first meaningful agent audio in roughly 1.0-1.5 seconds after they stop speaking. With current measurements:

- Endpoint wait floor: about 626 ms average, about 665 ms p90.
- Piper request-to-first-audio: about 227 ms average, 301-629 ms p90 in this small sample.
- Remaining p50 budget for the agent to produce a first speakable clause under a 1.2 second target: roughly 300-400 ms.
- Remaining p90 budget under a 2.0 second target: roughly 700-1000 ms.

For an LLM-backed operator agent, this implies:

- Do not wait for a full response before TTS.
- Stream the response and emit the first semantically complete clause as soon as it is available.
- Prefer short first clauses, usually 8-12 tokens.
- Target TTFT under 200-300 ms and effective first-clause generation around 30-50 tokens/sec where practical.

If the model needs 700-1000 ms before the first usable clause, the total perceived response starts trending toward 1.6-2.3 seconds because endpointing and TTS already consume most of the first second.

## Agent Naturalness Strategy

Do not implement fillers in the gateway media path. Fillers are an agent policy concern and should be evaluated later behind an explicit typed response contract.

Potential naturalness strategies to evaluate:

- Fast acknowledgement when useful: "okay", "got it", "one moment".
- Grounded progress cues for dispatcher tasks: "I am checking that now", "I found the ticket", "I am routing you".
- Confirmation-before-action for safety-critical operator commands: "I heard pull request three seven seven; hold for David, correct?"
- Short first clause plus streamed continuation: this hides the longer reasoning path without pretending work is done.

Risks:

- Fillers can mask real latency but make the system feel evasive if overused.
- Fillers must not delay barge-in, correction, or dispatch actions.
- Fillers are unsafe if the system has not understood the user intent or if ASR tail clipping is unresolved.

Local-model tradeoff:

- A local model can remove network round-trip variance and improve tail latency, which matters when the gateway has only about 300-1000 ms of agent budget.
- A hosted model may still work if TTFT is low and streaming is robust, but multi-hop cloud ASR/LLM/TTS paths will have little slack.
- For the initial "call operator" / dispatcher use case, a small local or LAN-adjacent model is attractive if it reliably handles the narrow task grammar and can stream a first clause quickly.
- The local corpora that match this workload are `bins/telnyx-gateway/corpus/qwen3-call-center-golden.json` and `bins/telnyx-gateway/corpus/qwen3-pm-orchestration-golden.json`.

## Fix Plan

Immediate code fixes:

1. Add ASR final-tail diagnostics to each final turn:
   - previous partial chars/words
   - final chars/words
   - common-prefix size
   - whether final text appears to be a strict prefix of the latest partial
   - whether the emitted caller turn used the ASR final or a conservative partial extension

2. Conservatively reconcile final text with the latest partial:
   - only use the partial when it is a strict normalized extension of the final in the same call state
   - otherwise keep the backend final unchanged and log the divergence

3. Increase ASR final-flush evidence:
   - ensure `asr.finish_pad` and `asr.local_finish` payloads are enough to join finalization events back to `asr_session_id`, `utterance_id`, and `turn_id`
   - keep `asr.finish_pad_ms` operator-tunable

4. Re-test live before changing endpoint defaults:
   - preserve the current 650 ms endpoint wait unless data shows a safe improvement
   - avoid changing smoke echo semantics while measuring ASR quality

Implemented in this pass:

- Added `asr.final_text_reconciliation` trace/quality-span diagnostics with `asr_session_id`, `utterance_id`, selected source, char counts, word counts, common-prefix count, strict-extension flag, and tail-word length.
- Reconciled final ASR text only when the latest partial is a strict normalized extension of the final; divergent partials leave the ASR final unchanged.
- Added regression tests for strict-extension replacement and divergent-partial preservation.

Regression watch:

- Do not reintroduce Kokoro as the conversation reply backend when Piper is selected.
- Do not add global state that leaks across concurrent calls.
- Do not make smoke-test final coalescing affect generic future conversation handlers.
- Do not make transcript text default on outside explicit test runs.
- Do not implement filler speech in this PR.
