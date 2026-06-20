# DESIGN — #557: Explicit pipeline stages + turn-batching layer (builds on #553/#541)

## Changelog
- 2026-06-20 — @ops48-orchestrator — §4.2 ownership rule (David): the gateway owns NO turn-batching logic. `N` and all batching-policy params live in `IdentityTurnBatcherConfig` (`libs/agent`); the smoke-test/gateway config only selects the `ConversationProcessorKind` and passes an opaque batcher config through. Gateway is a dumb host.
- 2026-06-20 — @ops48-orchestrator — §4.2: rename `IdentityPromptHandler` → `IdentityTurnBatcher` (consistent with `TurnBatcher`); add the **gateway-only smoke-test** requirement — the `conversation smoke-test` surface must be configurable to run the gateway in isolation in either mode (a) `Identity` (per-turn echo, validates layers 1–2) or (b) gateway turns + in-gateway `IdentityTurnBatcher` with `fixed_batch_size=N` (validates batched-turns-as-prompts), both model-free / no external agent. Per David.
- 2026-06-20 — @ops48-orchestrator — **Rename: `Semantic*` → `Turn*`** (`SemanticBatcher`→`TurnBatcher`, `ResponseMode::SemanticBatched`→`TurnBatched`, `SemanticReset`→`TurnBatchReset`, "semantic-batching layer"→"turn-batching layer"). Precision per David: **turns** are what get batched; *semantic endpointing* is only **one completion strategy** (alongside fixed-N heuristic and model-as-endpointer), not the name of the mechanism. "Semantic endpointing" / "semantically-complete" retained where they denote that strategy. (File name kept to preserve #558 links.)
- 2026-06-20 — @ops48-orchestrator — Add §4.2: `IdentityTurnBatcher`, the layer-3 identity/debug module — the `TurnBatcher` counterpart of `IdentityRepeatConversationProcessor`. Deterministic, model-free; echoes the assembled prompt; default batch-of-1 + opt-in fixed-N accumulation. The validation harness for the new `Accumulating/PromptComplete/Reset` states, the `TurnBatchReset`/epoch contract, and placement neutrality (same handler in-gateway and in-daemon → identical behavior). Per David.
- 2026-06-20 — @ops48-orchestrator — Add §4.1: the turn-batching logic is a placement-neutral `TurnBatcher` module in `libs/agent`, droppable on either side of the gateway↔daemon process boundary (host adapts transport only). Reframes "where layer-3 runs" as a deployment choice, not a contract change. Added matching reviewer focus (claude rust architect) + open question §7.6 (sync-vs-async batcher for model-as-endpointer). Per David.
- 2026-06-20 — @ops48-orchestrator — Initial proposal. Refines the #553 modular pipeline into explicit, contract-enforced stages and adds an optional turn-batching layer (N turns → 1 coherent prompt). For validation by a codex speech-pipeline architect, a codex streaming-LLM/model expert (turn/streaming LLM integration feasibility), and a claude rust + realtime architect.

## Status
PROPOSAL for review (design-only). Builds on PR #553 (`codex-541-refactor/modular-pipeline @ 5db459f9`), which delivered the modular per-turn processor pipeline. Implementation lands later as a new PR based on #553 (the staged PR chain: `main ← #548-branch ← #553 ← this`). Ping David at design sign-off; then task a codex to implement.

## 1. Problem
The `GatewayOwned`/`AgentOwned` axis conflated two distinct layers. The real decomposition is **three**:

1. **Acoustic substrate (gateway, ALWAYS-on):** ASR partials → turn boundaries (acoustic endpointing/VAD), TTS (buffered/streaming), media playback, **barge-in / safety** (supersede / stale / cancel).
2. **Turn-formation:** partials → one *turn* (one utterance).
3. **Turn batching (NEW):** N turns → **1 semantically-complete prompt** → response. *Semantic* endpointing ("is the caller's thought complete?"), distinct from acoustic endpointing.

#553 cleanly implements layers 1–2 (per-turn processor boundary + shared `ProvisionalPlaybackRegistry`/speech stages). It has **no layer-3 stage** — the processor responds *per turn*. "B" (`AgentOwned`) tried to express higher-order behavior by reaching *down* into layer-2 turn-formation (re-deriving acoustic boundaries from raw partials), which forces it to re-own safety — the root of the #556 symmetry gap.

## 2. Target — A as substrate + an optional turn-batching stage (not A *or* B)
- **Gateway always** emits a clean **turn stream** + owns ASR/TTS/barge-in/safety (layer 1+2).
- The **processor** either responds **per-turn** (today) or **accumulates N turns into a prompt** and responds (turn-batched — layer 3).
- Reframe the policy from "who does aggregation" → **"is a turn-batching stage layered above the gateway's turns"** (`turn-only` vs `turn-batched`). The gateway interface is **identical** either way.

This makes the boundary you want enforceable: **gateway = acoustic endpoints/turns + TTS + barge-in; agent/processor = turn batching of N turns into 1 coherent prompt.**

## 3. Stage contracts to add/enforce
### 3.1 Turn-stream input contract (gateway → processor)
The processor's unit becomes a **turn**, not a raw partial: `{ call_id, turn_id, utterance_id, text, finality: Provisional|Final, supersede/cancel signal, timing }`. Raw partials remain available **advisory** (for richer context) but are not the unit of work. (Today the processor receives `EarlyResponse(event)` + `CommittedTurn` + `AgentTextStream`; this formalizes "turn" as the consumed unit and keeps partials advisory.)

### 3.2 Processor output states (the key addition)
Today `ConversationProcessorOutput` only models per-turn `Speak/Cancel/Commit/CommittedSpeech`. Add the layer-3 states the turn-batching stage needs:
- **`Accumulating`** — observed a turn; prompt not yet complete; **hold the floor, do not respond yet** (optionally signal "still listening").
- **`PromptComplete(prompt)`** — the batched, semantically-complete prompt; respond now.
- **`Reset`** — discard the accumulated prompt (e.g. on barge-in / supersede).

A per-turn processor (Identity, today) simply never emits `Accumulating` — it's a degenerate case of the same contract.

### 3.3 Two endpointings, explicitly separated
- **Acoustic endpointing (gateway):** silence/VAD → turn boundary. Deterministic, latency-critical.
- **Semantic endpointing (agent):** decides **when to respond**, independent of turn ends — "the acoustic turn ended, but my prompt isn't complete; keep listening."

### 3.4 Safety composes with batching
Barge-in / supersede stay **gateway-owned** (acoustic), but must route **into** the processor's batch state so it `Reset`s the accumulated prompt. The gateway does the acoustic playback cancel; the agent learns of it and drops/rebuilds its prompt. (This is the clean home for what #556 strained to do under B: safety stays on the acoustic layer, the turn-batching layer above just gets a `Reset`.)

### 3.5 Policy reframe
Replace `TextCallAggregationPolicy::{GatewayOwned, AgentOwned}` with a layer-3 toggle (e.g. `ResponseMode::{PerTurn, TurnBatched}`), keeping the gateway↔processor turn-stream contract identical for both. `AgentOwned`'s old meaning (agent re-derives acoustic turns) is **dropped** — the gateway always owns acoustic turns; this resolves the #556 symmetry by construction.

## 4. How this builds on #553
#553 already gives the plug points: the `process_stream` boundary, `ConversationProcessorOutput` intents, the shared registry/speech-output stage, and the call-scoped `SpeechOutputConfig`. This design **adds layer 3 on top**: the new output states, the turn-stream framing, and a stateful turn-accumulator processor. The shared downstream stage (registry → speech → TTS, buffered or streaming) is **unchanged** — turn batching only changes *when* and *with what prompt* the processor responds, not how speech is produced. No re-architecture; an additive layer.

### 4.1 Placement-neutral turn batcher (one module, droppable on either side of the process boundary)
The layer-3 logic is a **standalone module in `libs/agent` (`motlie-agent`)** — the crate **both** `telnyx-gateway` (`default-features = false`) and `telnyx-agent` already depend on — sitting next to the existing turn/frame contract (`libs/agent/src/voice/telnyx/text.rs`). It is a **pure state machine** over the turn-stream contract, with **no gateway internals** (no `ProvisionalPlaybackRegistry` / `SpeechOutputConfig` / early-response pipeline) and **no daemon internals** (no websocket transport). Shape:

```rust
/// Layer-3 turn batcher: accumulates clean gateway turns into one prompt.
pub trait TurnBatcher {
    /// Observe one gateway turn (+ advisory partials). Decides hold-vs-respond.
    fn observe(&mut self, turn: Turn) -> BatchDecision;
    /// Barge-in / supersede arrived (acoustic, gateway-owned): drop the prompt.
    fn reset(&mut self);
}

pub enum BatchDecision {
    Accumulating,            // hold the floor, keep listening (§3.2)
    PromptComplete(Prompt),  // semantically complete; respond now (§3.2)
}
```

**Same module, hosted on either side — the host only adapts transport, never re-implements the logic:**
- **In-gateway host:** the `TurnBatched` processor (`ConversationProcessorKind`) owns a `TurnBatcher`, feeds it turns from `process_stream`, maps `PromptComplete` → the response trigger / `CommittedSpeech`, and routes gateway barge-in → `reset()`.
- **In-daemon host:** `telnyx-agent` owns the **same** `TurnBatcher`, fed by the gateway's turn stream over the text-call websocket; emits `AgentTurn*` return frames on `PromptComplete`; gateway barge-in arrives as a `GatewayTextFrame` → `reset()`.

Because the batcher consumes **only clean turns** (never raw acoustic partials) and the gateway **always** owns acoustic endpointing + safety, the daemon-side placement does **not** re-own acoustic boundaries — preserving the §3.5 invariant that dissolves #556. `ResponseMode::{PerTurn, TurnBatched}` selects whether a batcher is in the loop at all; **where** it runs (gateway vs daemon) is a host/deployment choice, **not** a contract change. (Tension to resolve: a heuristic batcher is pure/sync as above, but the *model-as-endpointer* option in §5 makes the decision I/O-bound — see §7.6.)

### 4.2 `IdentityTurnBatcher` — the layer-3 identity/debug module + gateway-only smoke test
The per-turn layers ship a reference identity processor (`IdentityRepeatConversationProcessor`) that echoes partials/turns verbatim — a deterministic, model-free smoke-test for layers 1–2. Layer 3 gets the **same kind of vehicle**: `IdentityTurnBatcher`, a trivial `TurnBatcher` impl (in `libs/agent`, next to the trait) whose only job is to make the new batching plumbing observable and testable end-to-end **without an LLM**.

What it does — the layer-3 analog of "repeat":
- **Batching:** deterministic, no semantics. Default behavior is **batch-of-1** — every turn yields `PromptComplete(turn.text)` immediately, never `Accumulating` (byte-for-byte the `PerTurn` path, but routed *through* the layer-3 `TurnBatcher` contract — so it exercises `PromptComplete` + the host's response trigger).
- **Accumulation mode (opt-in for validation):** a configurable trivial rule — fixed `N` turns, or until a sentinel turn — that emits `Accumulating` for the first N−1 and `PromptComplete(joined_text)` on the Nth. This is what actually drives the *new* code paths: multi-turn→1-prompt assembly, the explicit `Accumulating` state, and `Reset` mid-accumulation.
- **Response:** echo. The assembled prompt is spoken back verbatim (the host maps `PromptComplete` → `CommittedSpeech`/`AgentTurn`), exactly as `IdentityRepeat` echoes a turn. No model call.
- **Reset:** honors `reset()` — drops any in-progress accumulation; with epoch/`batch_id` gating, a late echo after reset is dropped. This makes the `TurnBatchReset` frame + stale-rejection contract directly testable.

Why it's worth specifying now: it is the **deterministic validation harness for layer 3** — it lets us live-validate (a) the new `Accumulating/PromptComplete/Reset` output states, (b) the `TurnBatchReset` boundary frame and epoch gating, and (c) **placement neutrality** (run the identical `IdentityTurnBatcher` in-gateway *and* in-daemon and assert identical observable behavior) before any real LLM/semantic-endpointing handler exists. It is debug/test scaffolding — gated behind `ResponseMode::TurnBatched` + a dev/identity selector, never a production default.

**Gateway-only smoke test (no external agent, no LLM).** The simplest live-call validation of layer 3 is *batch N turns into one prompt and echo it*. The gateway smoke-test surface (the `conversation smoke-test` operator command / live-run config) MUST be configurable to run the gateway **in isolation** in either of two modes — so each layer is independently exercisable on a real call with no daemon and no model:
- **(a) `Identity` processor** — gateway-owned partial→turn formation, echo per turn. Validates layers 1–2 (ASR partials → turn → early-response/commit → streaming TTS → media). This is today's identity/repeat smoke test.
- **(b) gateway-owned partial→turn formation + in-gateway `IdentityTurnBatcher`** (`ConversationProcessorKind::TurnBatchedIdentity`) — accumulates N turns into one prompt and echoes it. Validates layer 3 (**batched turns as prompts**) end-to-end, still entirely inside the gateway.

**Ownership rule: the gateway owns NO turn-batching logic.** The batch size `N` and every batching-policy parameter (`fixed_batch_size`, `max_batch_turns`, `max_batch_wait_ms`) belong to `IdentityTurnBatcher` — they are fields of its own `IdentityTurnBatcherConfig` in `libs/agent`. The smoke-test/gateway config does **not** decide N or hold any batch logic; it only (1) selects the `ConversationProcessorKind` and (2) for mode (b) supplies an opaque `IdentityTurnBatcherConfig` value handed straight to the batcher at construction. The gateway is a dumb host — it constructs the batcher with that config, feeds it turns, and maps `PromptComplete` to speech; all "how to batch" decisions live in the batcher. (Only gateway-side safety stays acoustic: barge-in → `reset()`.) Both modes are gateway-only and model-free; the only difference is whether the layer-3 `IdentityTurnBatcher` stage is engaged. This makes the new turn-batching layer live-validatable before any agent/LLM exists, the same way mode (a) validates the acoustic substrate today.

## 5. Streaming-LLM / turn-LLM integration (feasibility — codex streaming-LLM reviewer)
The turn-batched prompt must integrate with whatever produces the response. Open feasibility questions:
- **Turn-based LLM:** `PromptComplete(prompt)` → one LLM call → response → speech intents (provisional/committed). Straightforward; how does early-response/streaming-TTS compose (LLM streams tokens → processor emits provisional `Speak` chunks → shared streaming-TTS)?
- **Streaming LLM:** can the LLM consume the *accumulating* turn stream and itself decide response-start (i.e. the LLM *is* the semantic endpointer)? If so, does `Accumulating` vs `PromptComplete` get driven by the model rather than a heuristic? What's the contract for "model decided to respond mid-accumulation"?
- **Backpressure/latency:** turn batching adds a wait (hold for N turns). How is that bounded so it doesn't regress the early-response latency win (#535/#539)? Is there a max-batch / timeout that falls back to per-turn?
- **Cancellation:** mid-LLM-generation barge-in → cancel the in-flight LLM call + the streaming-TTS + `Reset` the batch. Feasible cleanly?

## 6. Reviewer foci
- **Codex speech-pipeline architect:** is the stage boundary (acoustic turns vs turn batching) clean + enforceable? Does the turn-stream contract + the `Accumulating/PromptComplete/Reset` states correctly model real turn-taking (multi-turn prompts, mid-turn corrections, supersede)? Does barge-in compose?
- **Codex streaming-LLM / model expert:** §5 feasibility — turn-based vs streaming-LLM integration, the model-as-semantic-endpointer option, latency bounds, mid-generation cancel.
- **Claude rust + realtime architect:** can this be expressed as clean Rust contracts on top of #553 (stateful processor, the new output enum, no leaky abstractions, no new races); realtime correctness (the added accumulation wait vs latency; cancel/reset across awaits); does it keep #553's single-authority/no-detach-across-await property? **Plus §4.1: is the `TurnBatcher` module genuinely placement-neutral — does it sit in `libs/agent` with zero gateway/daemon coupling, are `Turn`/`Prompt` expressible from the existing contract types, and does the in-gateway host keep no-await-across-lock if the batcher stays sync?**

## 7. Open questions
1. Is `Accumulating` an explicit output, or implicit (processor simply doesn't emit a response)? Tradeoff: explicit lets the gateway/UX know "still listening"; implicit is simpler.
2. Where does the **max-batch timeout / fallback to per-turn** live — gateway or processor?
3. Does turn batching need access to **partials** (sub-turn) or only **turns**? (Affects the turn-stream contract.)
4. How do provisional **responses** interact with turn batching — can the agent speak a provisional reply while still accumulating (predictive), or only after `PromptComplete`?
5. Migration: can layer-3 land purely additively on #553 behind the `ResponseMode` toggle (per-turn = today's behavior, byte-for-byte), with turn-batched as opt-in?
6. §4.1 sync-vs-async: the heuristic batcher is a pure/sync `observe`, but the *model-as-endpointer* (§5) makes the decision I/O-bound. Does the `TurnBatcher` contract need an async variant (e.g. `observe` returns a future, or the model decision is driven out-of-band and only its result is fed in)? Whatever the answer, it must hold on **both** host sides and not break the in-gateway no-await-across-lock property.

— Implementation (after sign-off): a new PR based on the #553 branch, additive layer-3 stage + the output-state contract; per-turn remains the default/unchanged path.
