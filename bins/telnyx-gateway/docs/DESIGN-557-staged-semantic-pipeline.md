# DESIGN — #557: Explicit pipeline stages + semantic-batching layer (builds on #553/#541)

## Changelog
- 2026-06-20 — @ops48-orchestrator — Initial proposal. Refines the #553 modular pipeline into explicit, contract-enforced stages and adds an optional semantic-batching layer (N turns → 1 coherent prompt). For validation by a codex speech-pipeline architect, a codex streaming-LLM/model expert (turn/streaming LLM integration feasibility), and a claude rust + realtime architect.

## Status
PROPOSAL for review (design-only). Builds on PR #553 (`codex-541-refactor/modular-pipeline @ 5db459f9`), which delivered the modular per-turn processor pipeline. Implementation lands later as a new PR based on #553 (the staged PR chain: `main ← #548-branch ← #553 ← this`). Ping David at design sign-off; then task a codex to implement.

## 1. Problem
The `GatewayOwned`/`AgentOwned` axis conflated two distinct layers. The real decomposition is **three**:

1. **Acoustic substrate (gateway, ALWAYS-on):** ASR partials → turn boundaries (acoustic endpointing/VAD), TTS (buffered/streaming), media playback, **barge-in / safety** (supersede / stale / cancel).
2. **Turn-formation:** partials → one *turn* (one utterance).
3. **Semantic batching (NEW):** N turns → **1 semantically-complete prompt** → response. *Semantic* endpointing ("is the caller's thought complete?"), distinct from acoustic endpointing.

#553 cleanly implements layers 1–2 (per-turn processor boundary + shared `ProvisionalPlaybackRegistry`/speech stages). It has **no layer-3 stage** — the processor responds *per turn*. "B" (`AgentOwned`) tried to express higher-order behavior by reaching *down* into layer-2 turn-formation (re-deriving acoustic boundaries from raw partials), which forces it to re-own safety — the root of the #556 symmetry gap.

## 2. Target — A as substrate + an optional semantic-batching stage (not A *or* B)
- **Gateway always** emits a clean **turn stream** + owns ASR/TTS/barge-in/safety (layer 1+2).
- The **processor** either responds **per-turn** (today) or **accumulates N turns into a prompt** and responds (semantic — layer 3).
- Reframe the policy from "who does aggregation" → **"is a semantic stage layered above the gateway's turns"** (`turn-only` vs `semantic-batched`). The gateway interface is **identical** either way.

This makes the boundary you want enforceable: **gateway = acoustic endpoints/turns + TTS + barge-in; agent/processor = semantic batching of N turns into 1 coherent prompt.**

## 3. Stage contracts to add/enforce
### 3.1 Turn-stream input contract (gateway → processor)
The processor's unit becomes a **turn**, not a raw partial: `{ call_id, turn_id, utterance_id, text, finality: Provisional|Final, supersede/cancel signal, timing }`. Raw partials remain available **advisory** (for richer context) but are not the unit of work. (Today the processor receives `EarlyResponse(event)` + `CommittedTurn` + `AgentTextStream`; this formalizes "turn" as the consumed unit and keeps partials advisory.)

### 3.2 Processor output states (the key addition)
Today `ConversationProcessorOutput` only models per-turn `Speak/Cancel/Commit/CommittedSpeech`. Add the layer-3 states the semantic stage needs:
- **`Accumulating`** — observed a turn; prompt not yet complete; **hold the floor, do not respond yet** (optionally signal "still listening").
- **`PromptComplete(prompt)`** — the batched, semantically-complete prompt; respond now.
- **`Reset`** — discard the accumulated prompt (e.g. on barge-in / supersede).

A per-turn processor (Identity, today) simply never emits `Accumulating` — it's a degenerate case of the same contract.

### 3.3 Two endpointings, explicitly separated
- **Acoustic endpointing (gateway):** silence/VAD → turn boundary. Deterministic, latency-critical.
- **Semantic endpointing (agent):** decides **when to respond**, independent of turn ends — "the acoustic turn ended, but my prompt isn't complete; keep listening."

### 3.4 Safety composes with batching
Barge-in / supersede stay **gateway-owned** (acoustic), but must route **into** the processor's batch state so it `Reset`s the accumulated prompt. The gateway does the acoustic playback cancel; the agent learns of it and drops/rebuilds its prompt. (This is the clean home for what #556 strained to do under B: safety stays on the acoustic layer, the semantic layer above just gets a `Reset`.)

### 3.5 Policy reframe
Replace `TextCallAggregationPolicy::{GatewayOwned, AgentOwned}` with a layer-3 toggle (e.g. `ResponseMode::{PerTurn, SemanticBatched}`), keeping the gateway↔processor turn-stream contract identical for both. `AgentOwned`'s old meaning (agent re-derives acoustic turns) is **dropped** — the gateway always owns acoustic turns; this resolves the #556 symmetry by construction.

## 4. How this builds on #553
#553 already gives the plug points: the `process_stream` boundary, `ConversationProcessorOutput` intents, the shared registry/speech-output stage, and the call-scoped `SpeechOutputConfig`. This design **adds layer 3 on top**: the new output states, the turn-stream framing, and a stateful turn-accumulator processor. The shared downstream stage (registry → speech → TTS, buffered or streaming) is **unchanged** — semantic batching only changes *when* and *with what prompt* the processor responds, not how speech is produced. No re-architecture; an additive layer.

## 5. Streaming-LLM / turn-LLM integration (feasibility — codex streaming-LLM reviewer)
The semantic-batched prompt must integrate with whatever produces the response. Open feasibility questions:
- **Turn-based LLM:** `PromptComplete(prompt)` → one LLM call → response → speech intents (provisional/committed). Straightforward; how does early-response/streaming-TTS compose (LLM streams tokens → processor emits provisional `Speak` chunks → shared streaming-TTS)?
- **Streaming LLM:** can the LLM consume the *accumulating* turn stream and itself decide response-start (i.e. the LLM *is* the semantic endpointer)? If so, does `Accumulating` vs `PromptComplete` get driven by the model rather than a heuristic? What's the contract for "model decided to respond mid-accumulation"?
- **Backpressure/latency:** semantic batching adds a wait (hold for N turns). How is that bounded so it doesn't regress the early-response latency win (#535/#539)? Is there a max-batch / timeout that falls back to per-turn?
- **Cancellation:** mid-LLM-generation barge-in → cancel the in-flight LLM call + the streaming-TTS + `Reset` the batch. Feasible cleanly?

## 6. Reviewer foci
- **Codex speech-pipeline architect:** is the stage boundary (acoustic turns vs semantic batching) clean + enforceable? Does the turn-stream contract + the `Accumulating/PromptComplete/Reset` states correctly model real turn-taking (multi-turn prompts, mid-turn corrections, supersede)? Does barge-in compose?
- **Codex streaming-LLM / model expert:** §5 feasibility — turn-based vs streaming-LLM integration, the model-as-semantic-endpointer option, latency bounds, mid-generation cancel.
- **Claude rust + realtime architect:** can this be expressed as clean Rust contracts on top of #553 (stateful processor, the new output enum, no leaky abstractions, no new races); realtime correctness (the added accumulation wait vs latency; cancel/reset across awaits); does it keep #553's single-authority/no-detach-across-await property?

## 7. Open questions
1. Is `Accumulating` an explicit output, or implicit (processor simply doesn't emit a response)? Tradeoff: explicit lets the gateway/UX know "still listening"; implicit is simpler.
2. Where does the **max-batch timeout / fallback to per-turn** live — gateway or processor?
3. Does semantic batching need access to **partials** (sub-turn) or only **turns**? (Affects the turn-stream contract.)
4. How do provisional **responses** interact with semantic batching — can the agent speak a provisional reply while still accumulating (predictive), or only after `PromptComplete`?
5. Migration: can layer-3 land purely additively on #553 behind the `ResponseMode` toggle (per-turn = today's behavior, byte-for-byte), with semantic-batched as opt-in?

— Implementation (after sign-off): a new PR based on the #553 branch, additive layer-3 stage + the output-state contract; per-turn remains the default/unchanged path.
