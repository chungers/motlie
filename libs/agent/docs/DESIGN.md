# Design: Agent Channel Managed Message Delivery

## Changelog

| Date (PDT) | Who | Summary |
|------------|-----|---------|
| 2026-06-08 22:22 PDT | @codex-421-design | Reworked the central abstraction from receiver-like inbox to per-process `AgentChannel`, explicitly scoped guarantees to one process targeting one tmux session, and made synchronous send vs asynchronous broadcast/timer semantics first-class in the API sketch. |
| 2026-06-08 22:14 PDT | @codex-421-design | Initial DESIGN for issue #421: new `libs/agent` crate with a per-agent-session managed delivery primitive that centralizes no-barge-in, dedup, attributed coalescing, composer preservation, and verified prompt submit for mstream send/broadcast, mstream timers, and the future M4 Telnyx transcript sink. |

## Status

Draft for [issue #421](https://github.com/chungers/motlie/issues/421), tracked in
[Discussion #422](https://github.com/chungers/motlie/discussions/422).

This is DESIGN-only. No `libs/agent` crate scaffolding or feature code should
land until reviewers approve the design. Although issue #421 uses "inbox" in
the title, this design recommends the crate's central type be named
`AgentChannel` for the scope reasons below.

Product mode is mixed:

- `libs/agent` is a new product surface.
- Its first consumers are brownfield mstream call paths and a planned M4
  Telnyx sink, so this design includes migration paths for those integrations.

## Problem

Motlie currently has multiple ways to inject instructions into agent tmux
sessions, and each path owns a different slice of delivery policy:

| Use case | Current path | Guard | Dedup | Coalescing | Submit |
|----------|--------------|-------|-------|------------|--------|
| `mstream send` | `bins/mstream/src/state.rs:1745` -> `send_text_to_resolved` at `state.rs:3490` | none | none | none | text and Enter in one `KeySequence` |
| `mstream broadcast` | `bins/mstream/src/state.rs:1877` -> `send_text_to_resolved` at `state.rs:3490` | none | none | none | text and Enter in one `KeySequence` |
| `mstream timer` | `timer_fire_once_shared` at `state.rs:2452` | `evaluate_input_guard` at `state.rs:3527` using `HostHandle::session_client_activity` | implicit single timer slot | implicit single timer slot | payload send, then retry Enter helper |
| M4 Telnyx transcript sink | future `TmuxTranscriptSink` described in `bins/telnyx-gateway/docs/DESIGN.md:432` | TBD | TBD | TBD | TBD |

The shared low-level signal already exists in `motlie-tmux`:
`HostHandle::session_client_activity()` summarizes attached client activity
(`libs/tmux/src/host.rs:626`) and returns `SessionClientActivity` with
`latest_client_activity` (`libs/tmux/src/types.rs:1074`). What is missing is one
agent-facing policy layer that all sources use before sending keys.

The broken submit behavior from #420 is part of the same problem. `mstream`
currently builds `KeySequence::literal(payload).then_enter()` in
`send_text_to_resolved`, so a large bracketed paste can absorb Enter into the
composer instead of submitting it. mmux's `$$` flow avoids this by sending text,
waiting, then sending a separate Enter (`bins/mmux/controller.rs:29-30` and
`1797`).

## Goals

- Provide one managed message-delivery primitive: a per-process channel to one
  uniquely identifiable agent tmux session.
- Reuse it for mstream `send`/`broadcast`, mstream timer self-triggers, and the
  future M4 `TmuxTranscriptSink`.
- Apply no-barge-in uniformly by using the existing `motlie-tmux` quiet guard
  signal.
- Deduplicate identical pending messages on that channel without losing source
  attribution.
- Coalesce pending multi-source directives into one naturally readable prompt
  body with attribution.
- Preserve already-typed, unsubmitted composer text by appending below it with
  a separator instead of jamming onto the same line.
- Make prompt submit reliable by decoupling payload from Enter, adding a settle
  delay, sending Enter separately, and retrying with verification where an
  agent UI profile can observe composer state.
- Keep `motlie-tmux` focused on tmux mechanisms and keep agent interaction
  policy in `libs/agent`.

## Non-Goals

- No feature implementation in this DESIGN step.
- No cross-process global mailbox or lock. If two different processes create
  channels to the same tmux session, this crate cannot dedup, coalesce, order,
  or mutually exclude their writes; they can interleave at the tmux layer.
- No durable, cross-daemon-restart queue in the first design. The channel may be
  in-memory, matching current mstream timer state.
- No universal private-state introspection for arbitrary agent TUIs. Composer
  and submit verification should be adapter/profile based, with conservative
  fallbacks when only tmux capture is available.
- No machine envelope prompt format. Coalesced directives must remain natural
  text intended for an agent to read.
- No migration of all historical mstream event schema in the first slice.
  Existing events can be preserved while delivery metadata is added surgically.
- No new third-party dependency is required by this design. Prefer workspace
  `tokio`, `serde`, and `thiserror` before adding crates.

## Recommended Design

Create a new crate, `libs/agent` (`motlie-agent`), depending on `motlie-tmux`.
Its central inbound type should be `AgentChannel`, not `AgentInbox`. The crate
owns reusable agent interaction policy:

- Inbound: managed prompt delivery through per-process channels to agent
  sessions.
- Outbound: future response extraction helpers for marker/history scraping and
  agent reply parsing.

The first implemented surface should focus on inbound delivery while leaving the
crate/module layout ready for outbound response extraction. This keeps the
crate coherent without requiring response extraction implementation in the
first slice.

### Named Decision: Channel, Not Inbox

Use `AgentChannel` as the central abstraction.

David's 2026-06-08 issue comment clarifies the scope: the object is per running
process connecting to one uniquely identifiable tmux session. Multiple threads
or sources inside that same process can queue, dedup, and coalesce through one
destination conduit. The crate cannot enforce a single receiver-owned mailbox
across all processes that might talk to the same tmux session.

`Inbox` implies a globally singular mailbox owned by the receiving agent.
Motlie cannot guarantee that today because tmux accepts keystrokes from any
process with access to the session. `Channel` is more accurate: it is a
sender-side managed conduit from one process to one agent session.

Scope:

- Key: `(process, host/session identity)`.
- Handled: intra-process multi-source queuing, dedup, attributed coalescing,
  no-barge-in checks, composer preservation, and submit verification.
- Not handled: cross-process global ordering, cross-process dedup/coalescing,
  exactly-once delivery across process restarts, or mutual exclusion against
  unrelated tmux writers.

Guarantee within one process:

- one channel serializes its own flush attempts for one target
- all sources using that channel share pending-message state
- synchronous callers can wait for submit confirmation
- asynchronous callers can enqueue without blocking their workflow
- channel delivery still respects observed tmux activity before it writes

If multiple processes/channels target the same tmux session, each channel keeps
its own guarantees, but final keystroke streams can interleave at tmux. That is
out of scope for `libs/agent` unless Motlie later introduces a shared daemon or
tmux-side lock protocol.

### Layering

`motlie-tmux` stays a pure mechanism library:

- discover hosts/sessions/panes
- send literal text and special keys
- capture pane text
- summarize attached-client activity

`libs/agent` owns policy above those mechanisms:

- decide whether it is safe to deliver
- queue, dedup, and coalesce messages
- format attributed prompt bodies
- preserve composer text
- submit and verify agent prompts
- later, extract agent responses from history/markers

mstream and the future Telnyx agent should call `libs/agent` instead of owning
their own tmux delivery policy.

## High-Level System Design

```text
sources
  mstream send
  mstream broadcast
  mstream timer
  M4 TmuxTranscriptSink
        |
        v
AgentChannelManager
  keyed by process-local AgentChannelKey
        |
        v
AgentChannel
  pending segments for one tmux session
  delivery waiters
  coalescing window
  quiet/defer state
        |
        v
QuietGuard
  HostHandle::session_client_activity(session)
        |
        v
PromptAssembler
  dedup identical bodies
  merge source attribution
  coalesce segments with newline separators
  prepend composer separator when needed
        |
        v
PromptSubmitter
  send literal/bracketed payload only
  settle delay
  separate Enter
  retry + verify through AgentUiProfile where available
        |
        v
motlie_tmux::Target::send_keys()
```

### Core Types

The exact names can change in PLAN/implementation, but the library should make
these concepts explicit.

```rust
use std::time::Duration;
use motlie_agent::{
    AgentChannelManager, AgentSessionKey, AsyncDelivery, ChannelConfig, ManagedMessage,
    MessageSource, SubmitPolicy,
};

let manager = AgentChannelManager::new(ChannelConfig {
    input_quiet_for: Duration::from_secs(10),
    coalesce_window: Duration::from_millis(500),
    submit: SubmitPolicy {
        settle: Duration::from_millis(500),
        retries: 1,
        retry_delay: Duration::from_millis(750),
        require_verification: true,
    },
    ..ChannelConfig::default()
});
```

```rust
let session = AgentSessionKey::new("local", "codex-421-design");
let channel = manager.bind(session, host_handle, target);

let outcome = channel
    .send(
        ManagedMessage::new(
            MessageSource::human("@ops48-orchestrator"),
            "Please review the latest design and call out layering concerns.",
        ),
        SubmitPolicy::verified_default(),
        Duration::from_secs(120),
    )
    .await?;

assert!(outcome.submitted());
```

```rust
let queued = channel
    .enqueue(
        ManagedMessage::new(
            MessageSource::timer("issue-421-poll"),
            "Wake up: check issue #421 and summarize only material changes.",
        )
        .dedup_body(),
        AsyncDelivery::FireAndForget,
    )
    .await?;
```

The important ergonomic point is that callers choose the operation that matches
their product semantics:

- `AgentChannel::send(...) -> SubmissionOutcome` for synchronous mstream
  `send`.
- `AgentChannel::enqueue(...) -> QueuedDelivery` for asynchronous broadcast,
  timers, and transcript sink events.

`send` waits until the prompt is submitted into the agent prompt window or until
its timeout/error path resolves. `enqueue` returns after the channel accepts the
message; delivery happens later when quiet/coalescing rules allow it.

The API should not make synchronous send look like a flag on an otherwise
fire-and-forget post. The method names and return types should make the
blocking behavior obvious.

### Synchronous And Asynchronous Semantics

The design uses different operations for the two product semantics:

```rust
pub enum SubmissionOutcome {
    SubmittedVerified,
    SubmittedUnverified,
}

pub struct QueuedDelivery {
    pub message_id: MessageId,
    pub target: AgentSessionKey,
    pub accepted_at: SystemTime,
}

impl AgentChannel {
    pub async fn send(
        &self,
        message: ManagedMessage,
        submit: SubmitPolicy,
        timeout: Duration,
    ) -> Result<SubmissionOutcome, DeliveryError>;

    pub async fn enqueue(
        &self,
        message: ManagedMessage,
        delivery: AsyncDelivery,
    ) -> Result<QueuedDelivery, DeliveryError>;
}
```

`send` is for direct mstream `send`: the caller is blocked until the channel has
submitted the prompt into the agent TUI, or until it can report a timeout/error.
The agent TUI may still queue or process the submitted prompt internally; the
guarantee is confirmation of prompt-window submission, not completion of the
agent's work.

`enqueue` is for mstream broadcast, self timers, and transcript sinks: the
caller gets acceptance into this process's channel and does not wait for quiet
guard, coalescing, or submit completion.

### Pending Message Model

Each pending channel segment stores:

- source kind and display label
- body
- first/last posted time
- dedup identity
- delivery waiter, if any
- submit policy override, if any

Default dedup should operate on normalized message body for one channel target.
If the same pending body arrives from multiple sources in the same process,
keep one body and merge the source list so attribution is not lost:

```text
[from: mstream.timer:issue-421-poll, @ops48-orchestrator]
Wake up: check issue #421 and summarize only material changes.
```

If the same source repeats the same pending body, the channel should refresh
metadata but not duplicate the prompt text.

### Attributed Coalescing

When multiple distinct pending segments flush together, the default body should
be readable as one prompt:

```text
[from: mstream.timer:issue-421-poll]
Wake up: check issue #421 and summarize only material changes.

[from: @ops48-orchestrator]
Also include whether the DESIGN.md commit is ready for the reviewers.
```

Recommended default:

- Use `[from: source]` headers for every segment when a flush contains more
  than one distinct segment or more than one source.
- For a single human `send`, omit the header unless the caller opts in.
- Separate segments with one blank line.
- Do not wrap the whole body in JSON/YAML/XML or any other machine envelope.

### No-Barge-In

Before flushing, the channel asks `HostHandle::session_client_activity(session)`.

- If no attached client has recent activity, flush.
- If the latest writable client activity is younger than `input_quiet_for`,
  keep pending messages queued and schedule the next attempt for the remaining
  quiet interval.
- Read-only attached clients must not block delivery.
- Default `input_quiet_for` should reuse mstream timer's current default of
  10 seconds.

The default design should not force a flush through the quiet guard. For a
never-quiet session:

- asynchronous callers remain queued and observable as pending/deferred
- synchronous callers wait until their configured timeout, then receive a
  `DeliveryError::TimedOut { still_pending: true }`

An explicit future `BargeInPolicy::ForceAfter` can be added, but it should not
be the default because it contradicts the issue's no-barge-in requirement.

### Composer Preservation

The channel must not overwrite or concatenate onto already-typed, unsubmitted
text. The crate should model composer state explicitly:

```rust
pub enum ComposerState {
    Empty,
    NonEmpty,
    Unknown,
}

pub trait AgentUiProfile {
    fn name(&self) -> &'static str;
    async fn composer_state(&self, target: &motlie_tmux::Target) -> Result<ComposerState>;
    async fn verify_submitted(&self, target: &motlie_tmux::Target) -> Result<SubmitVerification>;
}
```

First-slice behavior:

- If the profile reports `NonEmpty`, prefix the managed body with a separator so
  it lands below existing text.
- If the profile reports `Unknown` and the flush was delayed by recent input,
  use the same conservative separator.
- If the profile reports `Empty`, send the body without a leading separator.

Default separator:

```text

---

```

That produces a natural composer body:

```text
human already typed this

---

[from: mstream.timer:issue-421-poll]
Wake up and check status.
```

This is intentionally an agent-readable separator, not metadata. PLAN should
decide which profiles are implemented first. A generic profile can use
`Target::capture_with_options()` and return `Unknown` when it cannot safely
distinguish composer text from transcript text.

### Prompt Submit

Prompt submit must be a library-owned operation, not a caller-owned
`then_enter()`.

Algorithm:

1. Assemble the final body.
2. Apply bracketed paste framing when configured and useful for multiline text.
3. Send only the payload with `Target::send_keys(&KeySequence::literal(...))`.
4. Sleep `settle`.
5. Send a separate `{Enter}` with `KeySequence::parse("{Enter}")`.
6. Ask the active `AgentUiProfile` whether the prompt is submitted.
7. If verification says the body is still in the composer, sleep `retry_delay`
   and send another separate Enter, up to `retries`.
8. Return `SubmittedVerified`, `SubmittedUnverified`, or `StillPending`.

`SubmittedUnverified` is acceptable only when no profile can observe composer
state. The retry loop should still run. Known agent profiles should strive to
return `SubmittedVerified` or `StillPending`.

The default settle delay should start at 500 ms because that matches mmux's
known-good `$$` behavior. mstream may expose `--settle-ms` so smoke tests can
tune that down later.

## Data Flow By Use Case

### 1. mstream `send` / `broadcast`

Current direct paths:

- `send_shared` resolves a target, optionally interrupts, then calls
  `send_text_to_resolved` (`bins/mstream/src/state.rs:1745-1783`).
- `broadcast_shared` loops targets and calls `send_text_to_resolved`
  (`bins/mstream/src/state.rs:1877-1891`).
- `send_text_to_resolved` appends Enter in the same key sequence
  (`bins/mstream/src/state.rs:3490-3507`).

Migration:

1. Add `motlie-agent` as a dependency of `bins/mstream`.
2. Create or retrieve an `AgentChannel` after `resolve_target`.
3. Replace `send_text_to_resolved` in `send_shared` with
   `channel.send(...)`, returning a submission outcome to the caller.
4. Replace `send_text_to_resolved` in `broadcast_shared` with
   `channel.enqueue(...)`, returning after channel acceptance.
5. Keep mstream's workstream membership checks, state transitions, and audit
   events in mstream.
6. Add delivery metadata to event JSON without removing the existing event kinds
   in the first migration slice.
7. Rename prompt-submit flags per #420:
   - `--no-prompt-submit` is the real opt-out.
   - `--no-enter` remains a hidden deprecated alias for one release.
   - `--settle-ms`, `--submit-retries`, and `--submit-retry-delay-ms` apply to
     send and broadcast, not only timers.

Semantics:

- `send` is synchronous. It returns after prompt submission is confirmed or a
  timeout/error occurs. This follows the issue #421 comment from 2026-06-08 PDT:
  send should confirm submission so the caller does not need extra polling.
- `broadcast` is asynchronous fire-and-forget. It returns once messages are
  accepted into each target channel.
- `--interrupt-first` should become an explicit bypass/override policy. It
  remains caller-requested behavior, not the default channel path.

### 2. mstream timer self-trigger

Current path:

- `timer_start_shared` stores prompt, submit retry knobs, and
  `input_quiet_for_secs` (`bins/mstream/src/state.rs:2311-2412`).
- `timer_fire_once_shared` evaluates the quiet guard
  (`bins/mstream/src/state.rs:2510-2555`).
- If quiet, it sends text and then calls `send_submit_retries_to_resolved`
  (`bins/mstream/src/state.rs:2558-2571`).

Migration:

1. Keep timer scheduling and timer record ownership in mstream.
2. Remove timer-owned quiet-guard and submit mechanics after the channel path is
   in place.
3. On fire, enqueue a message with source `mstream.timer:<name>`,
   `AsyncDelivery::FireAndForget`, and a dedup key based on target plus
   normalized prompt body.
4. Map channel deferrals to existing timer observability fields:
   `defer_count`, `last_deferred_at`, `last_defer_reason`, and
   `last_input_activity`.
5. Keep existing `timer_deferred`/`timer_fired` JSON compatibility where
   possible, but make their meaning explicit:
   - `timer_fired`: timer accepted a prompt into the channel.
   - `timer_delivered` or delivery metadata: channel submitted it.

Semantics:

- Timers are asynchronous fire-and-forget.
- The current implicit "one pending timer fire" behavior becomes explicit body
  dedup in the target channel.
- Timer `--no-input-guard` should map to a channel override only for that
  message, not remove the default from the crate.

### 3. M4 Telnyx `TmuxTranscriptSink`

Current design reference:

- `bins/telnyx-gateway/docs/DESIGN.md:432` says a later
  `TmuxTranscriptSink` can map final transcript text to
  `motlie_tmux::KeySequence` and call `Target::send_keys()`.

Migration before implementation:

1. Change the planned sink design from direct `Target::send_keys()` to
   `AgentChannel::enqueue()`.
2. Only final transcript events should enqueue by default; partial transcript
   events remain UI/log feedback unless a future product decision says
   otherwise.
3. Use a source label that identifies the call/session without embedding
   sensitive caller data in prompts or logs, for example
   `telnyx.transcript:<provider_session_id>`.
4. Use `AsyncDelivery::FireAndForget`, so phone transcription never blocks the
   media path on an agent TUI.
5. Reuse the same coalescing behavior if a human or timer message is already
   pending for that agent.

This prevents M4 from becoming a third divergent copy of barge-in, dedup,
coalescing, composer, and submit policy.

## API Ergonomics

### Synchronous mstream send

```rust
let message = ManagedMessage::new(
    MessageSource::human("@ops48-orchestrator"),
    request.text,
)
.paste_mode(PasteMode::Bracketed)
.submit(SubmitPolicy::verified_default());

let outcome = channel
    .send(message, SubmitPolicy::from_cli(&request.submit), request.timeout)
    .await?;

state.record_delivery(&request.workstream, &stable_target, outcome)?;
```

This operation blocks until prompt submission is confirmed or fails. The return
value is about prompt-window submission only; it does not imply the agent has
finished acting on the submitted prompt.

### Asynchronous broadcast

```rust
for target in targets {
    let channel = channels.bind(target.key(), target.host, target.target);
    channel
        .enqueue(
            ManagedMessage::new(MessageSource::human("@ops48-orchestrator"), &request.text)
                .paste_mode(request.paste_mode)
                .submit(SubmitPolicy::from_cli(&request.submit)),
            AsyncDelivery::FireAndForget,
        )
        .await?;
}
```

Broadcast returns after each target channel accepts the message. Delivery may
happen later after quiet-guard and coalescing decisions.

### Timer fire

```rust
channel
    .enqueue(
        ManagedMessage::new(
            MessageSource::timer(&snapshot.name),
            snapshot.prompt,
        )
        .dedup_body()
        .input_quiet_for(snapshot.input_quiet_for),
        AsyncDelivery::FireAndForget,
    )
    .await?;
```

Timer fire remains fire-and-forget. A repeated timer prompt collapses through
channel-local dedup while it is pending.

### Telnyx transcript sink

```rust
async fn on_transcript(&self, event: TranscriptEvent, context: &mut CallContext) -> Result<()> {
    let TranscriptEvent::Final { text, .. } = event else {
        return Ok(());
    };
    let source_id = context
        .ids
        .provider_session_id
        .as_deref()
        .unwrap_or(context.ids.provider_call_id.as_str());

    self.channel
        .enqueue(
            ManagedMessage::new(
                MessageSource::external(format!("telnyx.transcript:{source_id}")),
                text,
            )
            .dedup_body(),
            AsyncDelivery::FireAndForget,
        )
        .await?;

    Ok(())
}
```

## Alternatives Considered

### Alternative A: Keep status quo in mstream and duplicate for M4

Pros:

- Smallest immediate code movement for mstream.
- Timer behavior can remain untouched.
- M4 could ship a minimal direct `Target::send_keys()` sink quickly.

Cons:

- Preserves current `send`/`broadcast` barge-in behavior.
- Duplicates quiet guard, dedup, coalescing, and submit policy in every caller.
- Makes M4 the third divergent behavior path.
- Leaves #420's submit fix as a mstream-specific patch instead of a shared
  delivery guarantee.
- Poor operability: callers must learn which source has which safety behavior.

Verdict: not recommended.

### Alternative B: Fold channel policy into `motlie-tmux`

Pros:

- No new crate.
- The code sits near `Target::send_keys()`, `Target::capture()`, and
  `HostHandle::session_client_activity()`.
- Fewer workspace dependency edges.

Cons:

- Blurs mechanism and policy. `motlie-tmux` would need concepts such as agent
  source attribution, prompt submit verification, composer preservation, and
  coalescing.
- Makes future outbound response extraction even less appropriate, because
  marker/history parsing of agent replies is agent-domain behavior, not tmux
  plumbing.
- Forces all tmux users to carry agent-specific API surface.
- Makes testing harder because low-level tmux behavior and high-level prompt
  policy would share one crate boundary.

Verdict: not recommended. Keep `motlie-tmux` a pure tmux mechanism library.

### Alternative C: New `libs/agent` crate depending on `motlie-tmux`

Pros:

- Clean layering: tmux mechanisms stay below agent interaction policy.
- One reusable inbound channel for all three concrete consumers.
- Natural home for future outbound response extraction, giving the crate a
  coherent inbound/outbound agent-interaction purpose.
- Lets mstream keep workstream/event ownership while delegating prompt delivery.
- Lets M4 start on the shared policy instead of copying mstream internals.
- Easier to test policy with fake tmux/session adapters before full integration.

Cons:

- Adds a workspace crate.
- Requires mstream migration work instead of a local helper patch.
- Needs careful API restraint so the first slice does not become a generic agent
  framework.

Verdict: recommended.

## Components And Subsystems To Test

Detailed test tasks belong in PLAN, but the design requires coverage in these
areas.

### `libs/agent`

- Dedup:
  - identical pending body from same source does not duplicate text
  - identical pending body from multiple sources merges source attribution
  - distinct bodies remain distinct segments
- Coalescing:
  - single segment formatting
  - multi-source `[from: source]` formatting
  - blank-line separation without machine envelopes
- Quiet guard:
  - quiet session flushes
  - recent writable client activity defers
  - read-only clients do not defer
  - never-quiet session leaves async messages pending and times out sync sends
- Composer preservation:
  - `ComposerState::NonEmpty` prefixes separator
  - `ComposerState::Empty` does not
  - `ComposerState::Unknown` follows conservative policy after recent input
- Submit:
  - payload and Enter are separate sends
  - settle delay occurs before Enter
  - retry delay occurs before retry Enter
  - verifier success returns `SubmittedVerified`
  - unsupported verifier returns `SubmittedUnverified` only after retry policy
- Delivery semantics:
  - `send` resolves only on submitted/error/timeout
  - `enqueue` returns after channel acceptance
  - cancellation of a waiter does not drop the pending message unless requested

### `bins/mstream`

- CLI/protocol:
  - `--no-prompt-submit` is accepted
  - hidden `--no-enter` still maps to prompt-submit disabled for one release
  - send/broadcast expose submit knobs
  - timer flags map to channel config
- State integration:
  - `send_shared` waits for submission outcome
  - `broadcast_shared` enqueues all resolved targets
  - timer fire enqueues to channel and preserves existing timer observability
  - stale target freshness checks still run before channel binding
  - workstream event records retain existing useful context
- Manual smoke:
  - typed composer text remains above a separator
  - large multiline prompt submits without an extra manual Enter
  - timer and human messages pending together submit as one attributed prompt

### `motlie-tmux`

No major `motlie-tmux` changes are expected. Existing tests around
`session_client_activity`, `Target::send_keys`, `KeySequence`, and capture APIs
should remain the mechanism-level safety net.

### M4 Telnyx Sink

- Final transcript enqueues one channel message.
- Partial transcript does not enqueue by default.
- Transcript sink uses non-sensitive source labels.
- Media/transcription path is not blocked by agent prompt delivery.

## Open Questions

1. Attribution default:
   Recommended default is `[from: source]` for multi-segment or multi-source
   flushes, omitted for a single human segment. Reviewers should decide whether
   every segment should always carry a header for maximum diarization.

2. Coalescing window and max wait:
   Recommended starting point is `coalesce_window = 500 ms` and
   `input_quiet_for = 10 s`. Default should not force through the quiet guard.
   Reviewers should decide whether any explicit forced-flush override belongs
   in the first implementation.

3. Synchronous send block-vs-queue:
   The issue comment resolves the product semantics: `send` should wait for
   prompt submission, while `broadcast` and timers are async. Remaining design
   detail: the default `send` timeout and JSONL response shape when the message
   is still pending after timeout.

4. Agent UI profiles:
   Which profiles should the first slice include: generic tmux-only,
   Codex-specific, Claude-specific, or a trait plus one generic fallback?

5. Outbound response extraction:
   The crate should reserve module/API space for outbound extraction, but the
   first implementation should probably not build it unless M4 needs it in the
   same phase.
