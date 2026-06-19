# DESIGN — #541: Modular conversation-processor pipeline (telnyx-agent on the canonical path)

## Changelog
- 2026-06-18 — @ops48-orchestrator — Add §9: David-approved DECISION (modular refactor = #541 closeout; default A, B opt-in, safety always-on) + the two reviewer-required amendments (committed-output coverage; input-contract channel) + migration + cite fixes.
- 2026-06-18 — @ops48-orchestrator — Initial proposal. Captures the modular-pipeline design that closes the PR #548 P1 concurrency races (A/B/C) and completes #541 structurally, instead of hardening the bespoke agent path in place. For review/validation by @claude-548-rv (software/perf architect) and @codex-548-rv (realtime streaming-voice architect).

## Status
PROPOSAL for review. Grounded in a read of the PR #548 head `2d54a1e5`. Not yet implemented. The recommendation question for the human (David) is the **A-vs-B placement call** in §6.

## 1. Problem
PR #548 set out to make `telnyx-agent` an implementation of the same conversation-processor contract as the local Identity path (#541). Today it is still a **parallel app path**:

- `ConversationProcessorKind::ExternalTextStream::process_input` returns `None` (`processors/mod.rs:68`) — a no-op marker, not load-bearing.
- The real agent work lives in `processors/external_text.rs`, which keeps its **own** provisional/append state machines (`provisional_turns` / `append_turns` maps) and queues TTS **directly** via `speech::queue_*`.
- That bespoke state machine is mutated as `remove → hold across an .await (TTS queueing) → re-insert`, racing the early-response pipeline's **separately spawned** generation-advance / cancel / commit / terminal-cleanup tasks.

Consequences (both #548 reviewers, confirmed at `18c19d21`): P1-A (stale provisional generation accepted; no authoritative gate), P1-B (provisional cancel/commit lost across the queue await), P1-C (same race on committed append-turns). PR #548 head `2d54a1e5` makes the bespoke path race-safe with tombstones + a generation gate — i.e. it **reimplements** the canonical guarantees a second time. The duplication (and divergence risk) remains; #541 RC #1/#3/#5/#7 remain unmet.

## 2. The canonical spine already exists and is generic
`spawn_early_response_pipeline` (`early_response.rs:599`) runs:

```
processor.process_stream(inputs) -> Stream<ConversationProcessorOutput>
  └─ ConversationProcessorOutput::EarlyResponse(intent)
       -> ProvisionalPlaybackRegistry          (early_response.rs:9; authoritative latest/canceled,
                                                 generation gate is_current(), cancel/commit;
                                                 synchronous critical section, NO await inside)
       -> speech-output adapter (call-scoped SpeechOutputConfig)
       -> streaming/buffered TTS -> Telnyx media
```

`ConversationProcessorOutput` (`processors/mod.rs:28`) is `EarlyResponse(intent) | Command | Error`. `process_stream` (`processors/mod.rs:72`) yields `impl Stream<Item = ConversationProcessorOutput>`. This `process_stream → Output::EarlyResponse → registry → speech` chain is **already generic over the processor**; `Identity::process_stream` feeds it today.

## 3. "Early response" is two stages, and the input one is optional
The thing called "early response" conflates:

- **Input stage — partial→turn aggregation** (`aggregate_early_resp_partials`, `early_response.rs:341`). This is the **optional** part (policy-gated via `EarlyResponsePolicy`).
- **Output stage — `ProvisionalPlaybackRegistry` + speech adapter.** This is the **always-on shared authority** for generation gating + cancel/commit + playback.

## 4. Data flow across the bridge (grounded at `2d54a1e5`)
Gateway → agent (`GatewayTextFrame`):
- `CallerPartial` — raw ASR partials, gated by `emit_partials` (`text_calls/websocket.rs:376`). The agent treats these as **advisory only** (`PartialAdvisoryContext::observe`, `text_ws.rs:203-226`).
- `CallerTurn` (final) + `CallerTurnProvisional`/`…Update`/`…Cancel`/`…Commit` — derived from the gateway early-response stage; the agent **answers off these** (`text_ws.rs:154, 229`).

Agent → gateway (`AgentTextFrame`): `AgentTurn`, `AgentTurnProvisional*` — today consumed by the bespoke `external_text` path.

Because the raw partials are **already bridged** (just advisory), the agent could own aggregation if the gateway-side input stage were omitted.

## 5. Proposed design — make the pipeline modular
Target shape, composed per processor path:

```
ASR partials/final
  → [optional INPUT stage: early-response partial-aggregation]   (compose: ON for local/Identity; OMIT for agent)
  → ConversationProcessor::process_stream   (Identity | ExternalTextStream, as PEER processors)
        emits ConversationProcessorOutput::EarlyResponse(intent)
  → [SHARED OUTPUT stage: ProvisionalPlaybackRegistry → speech-output adapter → TTS → media]
```

### Changes required
1. **Make `ExternalTextStream::process_stream` a real processor.** Consume the agent's return frames (`AgentTurnProvisional*` / `AgentTurn`) off the bridge and **emit `ConversationProcessorOutput::EarlyResponse(intent)`** (provisional + committed). The logic currently in `external_text` moves *into* the processor as intents.
2. **Extract the OUTPUT stage into a shared "provisional playback" stage.** Lift `process_stream → ProvisionalPlaybackRegistry → speech-adapter` out of `spawn_early_response_pipeline` so it runs for **any** processor. Identity and ExternalTextStream then share **one** `ProvisionalPlaybackRegistry` = one generation/cancel/commit authority → the bespoke `external_text` maps + direct `speech::queue_*` + the detach-across-await races are **deleted**, not hardened.
3. **Make partial-aggregation an explicit optional INPUT stage**, composed per path (ON for Identity/local; omittable for the agent, which then forms its own turns — option B).
4. **Reduce `external_text` to pure transport** — bridge I/O only; it no longer owns provisional state or queues TTS.

### Why this closes the races structurally
All three P1s are artifacts of a second provisional state machine mutated across awaits. With one shared `ProvisionalPlaybackRegistry` (synchronous critical section, authoritative `latest`/`canceled`, no await while holding state), there is no second machine to race and no detached-entry window — the race is unrepresentable, not merely tombstoned.

### #541 completeness
- RC #3 (agent → canonical `ConversationProcessorOutput` → shared adapter): met by change 1+2.
- RC #2 (separable adapter boundary): met by change 4 (transport-only `external_text`).
- RC #5 (provisional update handling): provisional lifecycle owned by the shared registry; the agent processor maps update/cancel/commit into intents.
- RC #1/#7 (single pipeline, no parallel direct-TTS): met by deleting the direct `speech::queue_*` path.

## 6. A vs B is a composition, not a fork — the open call
Because the INPUT aggregation stage is optional:
- **(A) Gateway owns aggregation** — INPUT stage ON for the agent path; the agent receives gateway-aggregated provisional turns and is a pure responder.
- **(B) Agent owns aggregation** — INPUT stage OMITTED for the agent path; the gateway forwards raw partials and the agent forms its own turns; the gateway does not pre-aggregate for the agent.

Both feed the **same** shared OUTPUT stage. A and B differ only by including/omitting stage 1 for the agent path — they are not separate architectures. **Recommendation request:** which default for the agent path (A or B), and should it be configurable per call? (The brain-side context argument favors B; latency/consistency with the local path favors A.)

## 7. Alternatives considered
- **Harden the bespoke path in place (PR #548 `2d54a1e5`, the current head).** Closes A/B/C via tombstones + a generation gate. Pro: smaller diff, behavior-preserving now. Con: keeps two provisional state machines (divergence risk), `ExternalTextStream` stays a no-op marker, #541 RC #1/#2/#3/#7 unmet, and future processors repeat the duplication. Not recommended as the #541 closeout.
- **The modular refactor (§5, recommended).** Larger change, but removes the duplication, makes races unrepresentable, completes #541, and makes future external processors (e.g. additional agent kinds) first-class with zero new bespoke wiring.

## 8. Open questions for reviewers (validate)
1. Feasibility/shape of `ExternalTextStream::process_stream` ingesting the **async bridge** frame stream and emitting `EarlyResponse` intents (back-pressure, ordering, lifecycle vs. the WS task).
2. Does `ProvisionalPlaybackRegistry` lift cleanly out of `spawn_early_response_pipeline` into a shared stage, or is it entangled with local-path specifics (playback-id clear, JIT provisional prebuffer)?
3. The INPUT-stage composition mechanism — per-`ConversationProcessorKind`? per-call policy? How does `emit_partials` relate?
4. Any correctness gap in B (agent-owned aggregation) vs the gateway's stale/superseded handling that the gateway must still enforce regardless of placement (e.g. barge-in / final-turn supersede)?
5. Migration: can this land incrementally on top of `2d54a1e5` (keep its race-safety as a transitional net) or is it a clean replacement of the `external_text` path?

## 9. DECISION + locked amendments (David approved 2026-06-18)
David approved the **modular refactor as the #541 closeout** (supersedes hardening `2d54a1e` in place). Both reviewers (@claude-548-rv, @codex-548-rv) endorsed the direction. Locked decisions + required amendments to fold into implementation:

### Decision (§6)
- **Default = A (gateway-owned aggregation).** One generation/cancel/supersede authority; matches the local Identity path; makes P1-A unrepresentable.
- **B (agent-owned aggregation) = constrained per-call opt-in, experimental** until its frame contract specifies raw-partial-correction / final-supersede / barge-in delivery + gateway enforcement. Expose a single explicit per-call policy field, e.g. `aggregation: GatewayOwned | AgentOwned` — NOT overloaded onto `emit_partials`/`emit_early_turns` (those are bridge-forwarding flags, orthogonal).
- **HARD INVARIANT:** the gateway-owned **safety-cancel** stage — **barge-in, final-turn supersede, stale-generation rejection** — is **always-on regardless of A/B**. The input stage must be SPLIT into (i) response-trigger aggregation (omittable for B) and (ii) safety cancels/supersede (never omittable), or the safety cancels relocated onto the shared output stage. Omitting the whole input stage for B would silently drop barge-in/supersede.

### Required amendments before/with implementation
1. **Committed-output coverage (REQUIRED — @codex-548-rv).** The shared output stage must own the **committed** agent streaming path (`AgentTurn`/`AgentTurnPartial`), not only provisional. `EarlyResponseIntent` (early_response.rs ~300-318) currently models provisional speak/cancel/commit only, and the committed transcript path ignores early-response outputs (conversation.rs ~579-583). Add a committed speech-output intent/stage under the same authoritative registry — **otherwise P1-C (committed append-turn race) remains a bespoke path and #541 is only partially closed.**
2. **Input contract + channel (REQUIRED — @claude-548-rv V-2).** Add a new `ConversationProcessorInput` variant carrying agent **return** frames (`AgentTurnProvisional*`/`AgentTurn`) + a channel from the WS read task into the pipeline input. The WS task decodes/writes frames and feeds bounded channels; it must NOT call `external_text::handle_agent_message` or `speech::queue_*` directly. Preserve in-order delivery + bounded back-pressure (reuse the early-response mpsc model). No session/text-call state removed-and-held across awaits.

### Scope note (V-1 — smaller than §5 implies)
The shared registry/output spine **already runs per-call for agent calls** (`spawn_early_response_pipeline` gated on `early_response.enabled`, not on kind) — it is live and authoritative but idle while the bespoke `external_text` path does the work. So change #2 (lift the output stage) is largely already present; the real work is change #1 (emit provisional+committed intents from agent frames + the input-contract channel) and change #4 (**delete** the bespoke `provisional_turns`/`append_turns` maps + direct `speech::queue_*`).

### Migration (§8 Q5)
Flag-gated **clean replacement**, not additive coexistence: introduce the shared-registry agent path behind a per-call flag, **never run both paths for one call** (double-drives TTS), use `2d54a1e`'s race tests as the behavioral oracle for stale/cancel/commit semantics, cut over per-call, then **delete** the bespoke maps + direct queueing. Do not keep the tombstoned bespoke machine as a permanent transitional net.

### Cite corrections (per reviewers; non-substantive)
At head `2d54a1e`: `ExternalTextStream → None` is `processors/mod.rs:61` (not :68); `ProvisionalPlaybackRegistry` is defined ~`early_response.rs:1031` (the `:9` cite was a `use` line); `process_stream` `processors/mod.rs:65-73`. The pre-`2d54a1e` `remove→await→re-insert` prose in §1 describes the state before the tombstone fix; §7 credits that fix — the duplication critique holds regardless.

— Implementation directive: build per this section; PR into the #548 branch; both reviewers re-validate; shepherd to merge into PR #548.
