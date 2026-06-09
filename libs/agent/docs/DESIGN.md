# Design: agent::Channel Managed Message Delivery

## Changelog

| Date (PDT) | Who | Summary |
|------------|-----|---------|
| 2026-06-08 23:27 PDT | @codex-421-design | Addressed PR #423 round-1 review: renamed the API to `Channel`/`ChannelManager`/`SessionKey`/`UiProfile`, made stable resolved tmux identity required, specified the needed `motlie-tmux` writable-client activity signal, removed duplicate submit-policy sources, defined outcome/error/verification contracts, defaulted dedup with zero-or-many waiters, added channel delivery events for mstream observability, clarified M4 as crate-level reuse only, and expanded mixed sync/async test coverage. |
| 2026-06-08 22:22 PDT | @codex-421-design | Reworked the central abstraction from receiver-like inbox to per-process channel, explicitly scoped guarantees to one process targeting one tmux session, and made synchronous send vs asynchronous broadcast/timer semantics first-class in the API sketch. |
| 2026-06-08 22:14 PDT | @codex-421-design | Initial DESIGN for issue #421: new `libs/agent` crate with a per-agent-session managed delivery primitive that centralizes no-barge-in, dedup, attributed coalescing, composer preservation, and verified prompt submit for mstream send/broadcast, mstream timers, and the future M4 Telnyx transcript sink. |

## Status

Draft for [issue #421](https://github.com/chungers/motlie/issues/421), tracked in
[Discussion #422](https://github.com/chungers/motlie/discussions/422). This is a
DESIGN-only update for PR #423; implementation starts only after reviewer
re-accept.

Although issue #421 uses "inbox" in the title, this design recommends the
central inbound type be `agent::Channel`. In Rust API terms, `agent::Channel`
reads better than `agent::AgentChannel` and follows David's review direction to
avoid C-STUTTER.

Product mode is mixed:

- `libs/agent` is a new product surface.
- Its first implementation consumers are brownfield mstream call paths.
- The future M4 Telnyx transcript sink reuses the crate API, but cross-process
  sharing with mstream is explicitly out of scope for #421.

## Problem

Motlie currently has multiple ways to inject instructions into agent tmux
sessions, and each path owns a different slice of delivery policy:

| Use case | Current path | Guard | Dedup | Coalescing | Submit |
|----------|--------------|-------|-------|------------|--------|
| `mstream send` | `bins/mstream/src/state.rs:1745` -> `send_text_to_resolved` at `state.rs:3490` | none | none | none | text and Enter in one `KeySequence` |
| `mstream broadcast` | `bins/mstream/src/state.rs:1877` -> `send_text_to_resolved` at `state.rs:3490` | none | none | none | text and Enter in one `KeySequence` |
| `mstream timer` | `timer_fire_once_shared` at `state.rs:2452` | partial current guard in `state.rs:3527` | implicit single timer slot | implicit single timer slot | payload send, then retry Enter helper |
| future M4 Telnyx transcript sink | `TmuxTranscriptSink` planned in `bins/telnyx-gateway/docs/DESIGN.md:432` | TBD | TBD | TBD | TBD |

`motlie-tmux` already exposes part of the needed signal:
`HostHandle::session_client_activity()` summarizes attached client activity
(`libs/tmux/src/host.rs:626`) and returns `SessionClientActivity` with
`latest_client_activity` (`libs/tmux/src/types.rs:1074`). That current field is
not sufficient for the desired no-barge-in policy because it is computed across
all attached clients, including read-only clients. To keep layering correct,
`motlie-tmux` must expose mechanism facts, and `libs/agent` must own the policy.
This design therefore requires a small `motlie-tmux` extension:
`SessionClientActivity.latest_writable_client_activity: Option<u64>` or an
equivalent per-client activity/read-only fact usable by `libs/agent`.

The broken submit behavior from #420 is part of the same delivery problem.
`mstream` currently builds `KeySequence::literal(payload).then_enter()` in
`send_text_to_resolved`, so a large bracketed paste can absorb Enter into the
composer instead of submitting it. mmux's `$$` flow avoids this by sending text,
waiting, then sending a separate Enter (`bins/mmux/controller.rs:29-30` and
`1797`).

## Goals

- Provide one managed message-delivery primitive: a process-local `Channel` to
  one stable, resolved agent tmux session identity.
- Reuse that primitive for mstream `send`, mstream `broadcast`, mstream timer
  self-triggers, and future Telnyx transcript delivery at the crate level.
- Apply no-barge-in uniformly using writable-client activity reported by
  `motlie-tmux`.
- Deduplicate identical pending messages on a channel by default without losing
  source attribution or waiter notification.
- Coalesce pending multi-source directives into one naturally readable prompt
  body with attribution.
- Preserve already-typed, unsubmitted composer text by appending below it with a
  separator instead of jamming onto the same line.
- Make prompt submit reliable by decoupling payload from Enter, adding a settle
  delay, sending Enter separately, and retrying with verification where a UI
  profile can observe composer state.
- Keep `motlie-tmux` focused on tmux mechanisms and keep agent interaction
  policy in `libs/agent`.

## Non-Goals

- No feature implementation in this DESIGN revision.
- No cross-process global mailbox, lock, ordering, dedup, or coalescing. If two
  different processes create channels to the same tmux session, their final
  keystroke streams can interleave at the tmux layer.
- No durable, cross-daemon-restart queue in the first implementation. The
  channel state may be in-memory, matching current mstream timer state.
- No universal private-state introspection for arbitrary agent TUIs. Composer
  and submit verification are profile based, with conservative generic fallback
  behavior.
- No machine envelope prompt format. Coalesced directives remain natural text
  intended for an agent to read.
- No implementation work for the Telnyx sink in #421. The M4 path is documented
  as future crate-level reuse.

## Recommended Design

Create `libs/agent` (`motlie-agent`) depending on `motlie-tmux`. The crate owns
agent interaction policy above tmux mechanisms:

- Inbound: managed prompt delivery through process-local `Channel` handles.
- Outbound: future response extraction helpers for marker/history scraping and
  agent reply parsing.

The first implemented surface should focus on inbound delivery while reserving
module/API room for outbound response extraction.

### Named Decision: Channel, Not Inbox

Use `Channel` as the central abstraction.

David's 2026-06-08 issue comment clarifies the scope: the object is per running
process connecting to one uniquely identifiable tmux session. Multiple threads
or sources inside that same process can queue, dedup, and coalesce through one
destination conduit. The crate cannot enforce a single receiver-owned mailbox
across all processes that might talk to the same tmux session.

`Inbox` implies a globally singular mailbox owned by the receiving agent.
Motlie cannot guarantee that today because tmux accepts keystrokes from any
process with access to the session. `Channel` is more accurate: it is a
sender-side managed conduit from one process to one resolved agent session.

Scope:

- Key: `SessionKey`, derived from process-local manager identity plus stable
  resolved tmux identity.
- Handled: intra-process multi-source queuing, default dedup, attributed
  coalescing, no-barge-in checks, composer preservation, submit verification,
  and delivery events.
- Not handled: cross-process global ordering, cross-process dedup/coalescing,
  exactly-once delivery across process restarts, or mutual exclusion against
  unrelated tmux writers.

Guarantee within one process:

- one channel serializes its own flush attempts for one target
- all sources using that channel share pending-message state
- synchronous callers can wait for submit confirmation
- asynchronous callers can enqueue without blocking their workflow
- channel delivery observes writable-client activity before it writes
- every accepted message has observable delivery lifecycle events

If multiple processes/channels target the same tmux session, each channel keeps
its own local guarantees, but final keystroke streams can interleave at tmux.
That remains out of scope unless Motlie later introduces a shared daemon or
lock protocol.

### Stable Identity

`SessionKey` must be derived from a stable resolved tmux identity, not from a
human-display session name alone.

Recommended shape:

```rust
pub struct SessionKey {
    pub host_alias: String,
    pub host_connection_id: String,
    pub tmux_session_id: Option<String>,
    pub tmux_session_name: String,
    pub tmux_session_created: Option<u64>,
}
```

Rules:

- Prefer tmux `session_id` plus `session_created` when available.
- Include host identity so two hosts with the same tmux session id do not
  collide.
- Keep `tmux_session_name` for diagnostics and fallback only.
- If only name fallback is available, `session_created` must participate in the
  key and mstream freshness checks must invalidate the channel on reuse.
- On disconnect, reclaim, quarantine, or stale-session detection, mstream must
  remove the channel for that `SessionKey`.

This matches current mstream safety patterns: resolve target, check freshness,
then act on a stable target. A reused tmux session name must never inherit an old
pending queue.

### Layering

`motlie-tmux` stays a mechanism library:

- discover hosts/sessions/panes
- send literal text and special keys
- capture pane text
- summarize attached-client activity facts, including latest writable-client
  activity after the required extension

`libs/agent` owns policy above those mechanisms:

- decide whether it is safe to deliver
- queue, dedup, and coalesce messages
- format attributed prompt bodies
- preserve composer text
- submit and verify agent prompts
- emit delivery lifecycle events
- later, extract agent responses from history/markers

mstream and future Telnyx code call `libs/agent` instead of owning their own
prompt-delivery policy.

## High-Level System Design

```text
sources inside one process
  mstream daemon process: send, broadcast, timer
  future Telnyx process: transcript sink with its own manager
        |
        v
ChannelManager
  daemon/process lifetime owner
  get-or-create by SessionKey
        |
        v
Channel handle
  cheap Clone/Arc-backed shared state
  pending segments for one stable tmux session
  zero-or-many waiters per segment
  coalescing window
  quiet/defer state
        |
        v
QuietGuard
  motlie-tmux latest_writable_client_activity
        |
        v
PromptAssembler
  default dedup of identical pending bodies
  merge source attribution
  coalesce segments with newline separators
  prepend composer separator when needed
        |
        v
PromptSubmitter
  send literal/bracketed payload only
  settle delay
  separate Enter
  retry + verify through UiProfile
        |
        v
motlie_tmux::Target::send_keys()
        |
        v
DeliveryEvent stream/status
  consumed by mstream for audit/timer metadata
```

### Core Types

The exact field names can change in PLAN/implementation, but the public
contract should keep these concepts.

```rust
use std::time::{Duration, Instant};
use motlie_agent::{
    ChannelConfig, ChannelManager, EnqueueOptions, ManagedMessage, MessageSource,
    SendOptions, SessionKey, SubmitPolicy, UiProfile,
};

let manager = ChannelManager::new(ChannelConfig {
    input_quiet_for: Duration::from_secs(10),
    coalesce_window: Duration::from_millis(500),
    default_submit: SubmitPolicy {
        settle: Duration::from_millis(500),
        retries: 1,
        retry_delay: Duration::from_millis(750),
        require_verification: true,
    },
    ui_profile: UiProfile::Generic,
});
```

`ChannelManager::get_or_bind` derives and validates identity from a resolved
session descriptor. It returns a cheap-clone shared handle; it does not create a
fresh queue per request.

```rust
pub struct ResolvedSession {
    pub key: SessionKey,
    pub host: motlie_tmux::HostHandle,
    pub target: motlie_tmux::Target,
}

impl ChannelManager {
    pub fn get_or_bind(&self, resolved: ResolvedSession) -> Result<Channel, DeliveryError>;
    pub fn remove(&self, key: &SessionKey);
    pub fn subscribe(&self) -> DeliveryEvents;
}
```

If `get_or_bind` sees an existing `SessionKey` with a different non-equivalent
resolved target, it returns `DeliveryError::TargetIdentityMismatch` rather than
silently reusing state. mstream must run its existing freshness checks before
calling `get_or_bind`.

### Synchronous And Asynchronous Semantics

The API uses different operations for the two product semantics.

```rust
pub struct SendOptions {
    pub submit: SubmitPolicy,
    pub timeout: Duration,
}

pub struct EnqueueOptions {
    pub submit: SubmitPolicy,
    pub quiet_guard: QuietGuardPolicy,
}

pub enum QuietGuardPolicy {
    Default,
    Disabled,
}

pub enum SubmissionOutcome {
    SubmittedVerified { message_id: MessageId, submitted_at: Instant },
    SubmittedUnverified { message_id: MessageId, submitted_at: Instant },
}

pub struct QueuedDelivery {
    pub message_id: MessageId,
    pub target: SessionKey,
    pub accepted_at: Instant,
}

impl Channel {
    pub async fn send(
        &self,
        message: ManagedMessage,
        options: SendOptions,
    ) -> Result<SubmissionOutcome, DeliveryError>;

    pub async fn enqueue(
        &self,
        message: ManagedMessage,
        options: EnqueueOptions,
    ) -> Result<QueuedDelivery, DeliveryError>;

    pub fn status(&self) -> ChannelStatus;
}
```

`send` is for direct mstream `send`: the caller is blocked until the channel has
submitted the prompt into the agent TUI, or until it reports a timeout/error.
The agent TUI may still queue or process the submitted prompt internally; the
guarantee is confirmation of prompt-window submission, not completion of the
agent's work.

`enqueue` is for mstream broadcast, self timers, and transcript sinks: the
caller gets acceptance into this process's channel and does not wait for quiet
guard, coalescing, or submit completion. The absence of an `AsyncDelivery` enum
is intentional: `enqueue` already means fire-and-forget for this first slice.

Submit policy has one home per operation: `SendOptions` or `EnqueueOptions`.
`ManagedMessage` must not also carry submit policy. `ChannelConfig.default_submit`
is only the value used by `SendOptions::default()` / `EnqueueOptions::default()`;
callers that expose knobs construct explicit options before calling the channel.
CLI conversion stays in the caller; mstream maps CLI flags into `SubmitPolicy`
before calling `libs/agent`.

### Message And Dedup Model

`ManagedMessage` contains source, body, paste mode, and optional display
metadata. `MessageSource` constructors should accept `impl Into<String>` so the
callers can pass either borrowed or owned labels without API churn.

Dedup is default-on for pending messages in the first implementation. There is
no `.dedup_body()` builder in the initial API. The dedup identity is the
normalized body plus the channel `SessionKey`; source attribution is merged
rather than used to keep duplicate prompt text.

Each pending segment stores:

- message id for the segment
- source list and display labels
- body
- first/last accepted time
- dedup identity
- zero-or-many waiters, each with its own timeout/cancellation state
- merged submit policy for the eventual flush

If a second synchronous `send` posts a body already pending from an async timer,
the channel attaches a new waiter to the existing segment rather than adding
more prompt text. Both callers are notified when the coalesced prompt is
submitted, and each waiter's timeout/cancellation applies only to that waiter.
If all sync waiters time out, the pending segment remains if it still has async
sources or other waiters.

When pending segments with different submit policies are coalesced, the channel
uses a deterministic merge: maximum settle/retry delays and retry count, and
`require_verification = true` if any waiter requires verification. This keeps a
single prompt submit conservative without making policy precedence implicit.

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

- Use `[from: source]` headers for every segment when a flush contains more than
  one distinct segment or more than one source.
- For a single human `send`, omit the header unless the caller opts in.
- Separate segments with one blank line.
- Do not wrap the whole body in JSON/YAML/XML or any other machine envelope.

### No-Barge-In

Before flushing, the channel asks `motlie-tmux` for writable-client activity.
The current `SessionClientActivity.latest_client_activity` is not enough because
it includes read-only clients. Implementation must first extend `motlie-tmux` to
expose one of these mechanism facts:

```rust
pub struct SessionClientActivity {
    pub session: String,
    pub attached_clients: usize,
    pub writable_clients: usize,
    pub latest_client_activity: Option<u64>,
    pub latest_writable_client_activity: Option<u64>,
}
```

or equivalent per-client activity/read-only data. `libs/agent` then applies
policy:

- If there is no recent writable-client activity, flush.
- If `latest_writable_client_activity` is younger than `input_quiet_for`, keep
  pending messages queued and schedule the next attempt for the remaining quiet
  interval.
- Read-only attached clients must not block delivery.
- Default `input_quiet_for` should reuse mstream timer's current default of 10
  seconds.

The default design does not force a flush through the quiet guard. For a
never-quiet session:

- async callers remain queued and observable as pending/deferred
- sync callers wait until their configured timeout, then receive
  `DeliveryError::TimedOut { message_id, still_pending: true }`

An explicit future force-through policy can be added later, but it is not part
of the first slice.

### Composer Preservation

The channel must not overwrite or concatenate onto already-typed, unsubmitted
text. The crate should model composer state explicitly and avoid trait-object
async complexity in the first slice by using an enum profile:

```rust
pub enum UiProfile {
    Generic,
    Codex,
    Claude,
}

pub enum ComposerState {
    Empty,
    NonEmpty,
    Unknown,
}

pub enum SubmitVerification {
    Submitted,
    StillComposed,
    Unknown,
}

impl UiProfile {
    pub async fn composer_state(
        &self,
        target: &motlie_tmux::Target,
    ) -> Result<ComposerState, DeliveryError>;

    pub async fn verify_submitted(
        &self,
        target: &motlie_tmux::Target,
    ) -> Result<SubmitVerification, DeliveryError>;
}
```

This avoids `Box<dyn ...>` and native `async fn` in trait object problems while
leaving room to add profiles. If a future implementation needs external profile
plugins, the DESIGN/PLAN should justify dynamic dispatch then.

First-slice behavior:

- If the profile reports `NonEmpty`, prefix the managed body with a separator so
  it lands below existing text.
- If the profile reports `Unknown` and the flush was delayed by recent writable
  input, use the same conservative separator.
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

### Prompt Submit

Prompt submit must be a library-owned operation, not a caller-owned
`then_enter()`.

Algorithm:

1. Assemble the final body.
2. Apply bracketed paste framing when configured and useful for multiline text.
3. Send only the payload with `Target::send_keys(&KeySequence::literal(...))`.
4. Sleep `settle`.
5. Send a separate `{Enter}` with `KeySequence::parse("{Enter}")`.
6. Ask `UiProfile::verify_submitted` whether the prompt is submitted.
7. If verification returns `StillComposed`, sleep `retry_delay` and send another
   separate Enter, up to `retries`.
8. If verification returns `Submitted`, return
   `SubmissionOutcome::SubmittedVerified`.
9. If verification returns `Unknown` and `require_verification` is false, return
   `SubmissionOutcome::SubmittedUnverified` after the retry policy completes.
10. If verification returns `Unknown` and `require_verification` is true, return
    `DeliveryError::VerificationUnavailable { message_id }`.
11. If the prompt is still composed after all retries, return
    `DeliveryError::SubmitNotConfirmed { message_id, attempts }`.

There is no successful `StillPending` state. Pending/not-submitted outcomes are
errors for synchronous `send`; queued async work remains observable through the
channel event/status APIs.

### Errors

`libs/agent` should use `thiserror` and expose an error enum shaped like this:

```rust
#[derive(Debug, thiserror::Error)]
pub enum DeliveryError {
    #[error("target identity mismatch for {key:?}")]
    TargetIdentityMismatch { key: SessionKey },

    #[error("target could not be resolved: {key:?}")]
    TargetUnresolved { key: SessionKey },

    #[error("target is stale or was replaced: {key:?}")]
    TargetStale { key: SessionKey },

    #[error("tmux operation failed: {operation}")]
    Tmux { operation: &'static str, source: motlie_tmux::Error },

    #[error("submit verification unavailable for {message_id:?}")]
    VerificationUnavailable { message_id: MessageId },

    #[error("submit was not confirmed for {message_id:?} after {attempts} attempts")]
    SubmitNotConfirmed { message_id: MessageId, attempts: u8 },

    #[error("delivery timed out for {message_id:?}")]
    TimedOut { message_id: MessageId, still_pending: bool },

    #[error("channel is closed for {key:?}")]
    ChannelClosed { key: SessionKey },
}
```

The exact variants can be refined during implementation, but DESIGN must keep
these categories: identity/freshness, unresolved targets, tmux I/O,
verification unavailable, submit not confirmed, timeout, and closed channel.

### Delivery Observability

Because async `enqueue` returns before quiet-guard deferral and prompt submit,
mstream needs an upward observation surface. `ChannelManager::subscribe()`
should provide a generic event stream, and `Channel::status()` should provide a
snapshot for polling/debugging.

```rust
pub enum DeliveryEvent {
    Accepted { message_id: MessageId, target: SessionKey, source: MessageSource },
    Deferred {
        message_id: MessageId,
        target: SessionKey,
        reason: DeferReason,
        latest_writable_client_activity: Option<u64>,
        retry_after: Duration,
    },
    Coalesced { target: SessionKey, message_ids: Vec<MessageId> },
    Submitted { message_ids: Vec<MessageId>, outcome: SubmissionOutcome },
    Failed { message_id: MessageId, error: DeliveryError },
}
```

mstream consumes these events to update timer `defer_count`, `last_deferred_at`,
`last_defer_reason`, `last_input_activity`, and durable workstream audit events.
The event type stays generic: it contains message id, source, target, reason,
timestamps/durations, and outcome, not mstream-specific record fields.

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
2. Add one `ChannelManager` field to `DaemonState`, owned for the daemon
   lifetime. Do not create a manager per request.
3. Keep existing mstream target resolution and freshness checks before channel
   lookup.
4. Convert the resolved target/session info into `ResolvedSession` and call
   `state.channel_manager.get_or_bind(resolved)`.
5. `send_shared` calls `channel.send(message, SendOptions { submit, timeout })`
   and returns the submission outcome to the client.
6. `broadcast_shared` calls `channel.enqueue(message, EnqueueOptions { submit,
   quiet_guard: Default })` for each resolved target and returns after channel
   acceptance.
7. mstream consumes `DeliveryEvent`s from `ChannelManager::subscribe()` and maps
   them into audit/timer metadata.
8. On disconnect, reclaim, retire/quarantine due stale-session detection, or
   host scan invalidation, mstream removes affected channels by `SessionKey`.
9. Rename prompt-submit flags per #420:
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
- `timer_fire_once_shared` evaluates the current quiet guard
  (`bins/mstream/src/state.rs:2510-2555`).
- If quiet, it sends text and then calls `send_submit_retries_to_resolved`
  (`bins/mstream/src/state.rs:2558-2571`).

Migration:

1. Keep timer scheduling and timer record ownership in mstream.
2. Remove timer-owned quiet-guard and submit mechanics after the channel path is
   in place.
3. On fire, resolve/freshness-check the target, retrieve the same channel from
   the daemon-owned `ChannelManager`, and call `enqueue` with source
   `mstream.timer:<name>`.
4. `--no-input-guard` maps to `EnqueueOptions { quiet_guard: Disabled, .. }` for
   that message only.
5. Timer deferral/delivery metadata comes from `DeliveryEvent`s, not from timer
   code reimplementing channel internals.
6. Keep existing `timer_deferred`/`timer_fired` JSON compatibility where
   possible, but make their meaning explicit:
   - `timer_fired`: timer accepted a prompt into the channel.
   - delivery event metadata: channel deferred/submitted/failed it.

Semantics:

- Timers are asynchronous fire-and-forget.
- The current implicit "one pending timer fire" behavior becomes explicit
  default body dedup in the target channel.

### 3. Future M4 Telnyx `TmuxTranscriptSink`

Current design reference:

- `bins/telnyx-gateway/docs/DESIGN.md:432` says a later
  `TmuxTranscriptSink` can map final transcript text to
  `motlie_tmux::KeySequence` and call `Target::send_keys()`.

Future migration path:

1. Change the planned sink design from direct `Target::send_keys()` to
   `agent::Channel::enqueue()`.
2. Only final transcript events should enqueue by default; partial transcript
   events remain UI/log feedback unless a future product decision says
   otherwise.
3. Use a source label that identifies the call/session without embedding
   sensitive caller data in prompts or logs, for example
   `telnyx.transcript:<provider_session_id>`.
4. Use `enqueue`, so phone transcription never blocks the media path on an
   agent TUI.
5. Because Telnyx gateway/agent code is a separate process from mstream unless a
   future topology routes it through the mstream daemon, it only gets crate-level
   reuse in this design. It shares no pending state with mstream human sends or
   mstream timers. It coalesces only with other sources inside the same Telnyx
   process/channel manager.

This prevents M4 from copying prompt-delivery policy while staying honest about
the accepted process boundary. Telnyx implementation is out of #421; this PR
only documents the future path.

## API Ergonomics

### Synchronous mstream send

```rust
let message = ManagedMessage::new(
    MessageSource::human("@ops48-orchestrator"),
    request.text,
).paste_mode(PasteMode::Bracketed);

let submit = SubmitPolicy {
    settle: Duration::from_millis(request.settle_ms),
    retries: request.submit_retries,
    retry_delay: Duration::from_millis(request.submit_retry_delay_ms),
    require_verification: true,
};

let outcome = channel
    .send(message, SendOptions { submit, timeout: request.timeout })
    .await?;

state.record_delivery(&request.workstream, &stable_target, outcome)?;
```

This operation blocks until prompt submission is confirmed or fails. The return
value is about prompt-window submission only; it does not imply the agent has
finished acting on the submitted prompt.

### Asynchronous broadcast

```rust
for target in targets {
    let channel = state.channel_manager.get_or_bind(target.resolved_session())?;
    channel
        .enqueue(
            ManagedMessage::new(MessageSource::human("@ops48-orchestrator"), &request.text)
                .paste_mode(request.paste_mode),
            EnqueueOptions { submit, quiet_guard: QuietGuardPolicy::Default },
        )
        .await?;
}
```

Broadcast returns after each target channel accepts the message. Delivery may
happen later after quiet-guard and coalescing decisions, with outcomes reported
through `DeliveryEvent`s.

### Timer fire

```rust
let channel = state.channel_manager.get_or_bind(snapshot.resolved_session())?;
channel
    .enqueue(
        ManagedMessage::new(MessageSource::timer(&snapshot.name), snapshot.prompt),
        EnqueueOptions {
            submit: snapshot.submit_policy,
            quiet_guard: snapshot.quiet_guard,
        },
    )
    .await?;
```

Timer fire remains fire-and-forget. A repeated timer prompt collapses through
channel-local default dedup while it is pending.

### Future Telnyx transcript sink

```rust
async fn on_transcript(
    &self,
    event: TranscriptEvent,
    context: &mut CallContext,
) -> Result<(), VoiceAppError> {
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
            ),
            EnqueueOptions::default(),
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

Verdict: not recommended. Keep `motlie-tmux` a tmux mechanism library, with only
mechanism-fact extensions such as latest writable-client activity.

### Alternative C: New `libs/agent` crate depending on `motlie-tmux`

Pros:

- Clean layering: tmux mechanisms stay below agent interaction policy.
- One reusable inbound channel API for all three concrete use cases.
- Natural home for future outbound response extraction, giving the crate a
  coherent inbound/outbound agent-interaction purpose.
- Lets mstream keep workstream/event ownership while delegating prompt delivery.
- Lets future Telnyx code reuse the same crate without copying mstream internals.
- Easier to test policy with fake tmux/session adapters before full integration.

Cons:

- Adds a workspace crate.
- Requires mstream migration work instead of a local helper patch.
- Requires one small `motlie-tmux` mechanism extension for writable-client
  activity.
- Needs careful API restraint so the first slice does not become a generic agent
  framework.

Verdict: recommended.

## Components And Subsystems To Test

Detailed test tasks belong in PLAN, but the design requires coverage in these
areas.

### `libs/agent`

- Session identity:
  - `SessionKey` includes stable host/session id and created generation
  - reused tmux session names do not inherit old pending queues
  - `get_or_bind` returns the same shared channel for the same stable target
  - `get_or_bind` rejects same key with a non-equivalent target
- Dedup:
  - identical pending body from same source does not duplicate text
  - identical pending body from multiple sources merges source attribution
  - distinct bodies remain distinct segments
  - two sync sends of the same pending body both receive completion/error
  - sync send attaches a waiter to an already-pending async/deduped segment
- Coalescing:
  - single segment formatting
  - multi-source `[from: source]` formatting
  - blank-line separation without machine envelopes
- Quiet guard:
  - quiet session flushes
  - recent writable-client activity defers
  - read-only client activity does not defer
  - never-quiet session leaves async messages pending and times out sync sends
- Composer preservation:
  - `ComposerState::NonEmpty` prefixes separator
  - `ComposerState::Empty` does not
  - `ComposerState::Unknown` follows conservative policy after recent writable
    input
- Submit:
  - payload and Enter are separate sends
  - settle delay occurs before Enter
  - retry delay occurs before retry Enter
  - `SubmitVerification::Submitted` returns `SubmittedVerified`
  - `SubmitVerification::Unknown` with verification not required returns
    `SubmittedUnverified`
  - verification required but unavailable returns `DeliveryError`
  - still composed after retries returns `DeliveryError`
- Mixed sync/async delivery:
  - async timer/broadcast/future Telnyx enqueue arrives while a sync send waits
    behind the quiet guard, and one coalesced prompt eventually notifies all
    sync waiters
  - sync send timeout leaves the pending segment and other async messages intact
  - cancellation of one waiter does not drop the pending message or other waiters
  - separate manager instances targeting the same tmux session do not promise
    cross-process coalescing
- Events/status:
  - accepted/deferred/coalesced/submitted/failed events are emitted with message
    id, source, target, reason, and outcome

### `bins/mstream`

- CLI/protocol:
  - `--no-prompt-submit` is accepted
  - hidden `--no-enter` still maps to prompt-submit disabled for one release
  - send/broadcast expose submit knobs
  - timer flags map to channel options
- State integration:
  - `DaemonState` owns one `ChannelManager` for the daemon lifetime
  - `send_shared`, `broadcast_shared`, and `timer_fire_once_shared` retrieve
    channels from that shared manager after freshness checks
  - `send_shared` waits for `SubmissionOutcome`
  - `broadcast_shared` enqueues all resolved targets
  - timer fire enqueues to channel and observes delivery through events
  - stale target/disconnect/reclaim paths remove affected channels
  - workstream event records retain useful delivery context
- Manual smoke:
  - typed composer text remains above a separator
  - large multiline prompt submits without an extra manual Enter
  - timer and human messages pending together submit as one attributed prompt

### `motlie-tmux`

A small mechanism extension is required:

- `SessionClientActivity` exposes `latest_writable_client_activity` or equivalent
  per-client activity/read-only facts
- unit tests prove read-only clients do not update writable activity
- existing `Target::send_keys`, `KeySequence`, and capture tests remain the
  mechanism-level safety net

### Future M4 Telnyx Sink

- Final transcript enqueues one channel message in its own process-local
  `ChannelManager`
- Partial transcript does not enqueue by default
- Transcript sink uses non-sensitive source labels
- Media/transcription path is not blocked by agent prompt delivery
- Contract test/documentation states that cross-process coalescing with mstream
  is not promised unless a future shared daemon topology is introduced

## Open Questions

1. Attribution default:
   Recommended default is `[from: source]` for multi-segment or multi-source
   flushes, omitted for a single human segment. Reviewers should decide whether
   every segment should always carry a header for maximum diarization.

2. Coalescing window and force-through policy:
   Recommended starting point is `coalesce_window = 500 ms` and
   `input_quiet_for = 10 s`. Default should not force through the quiet guard.
   A force-through override is intentionally deferred.

3. Synchronous send timeout and JSON response shape:
   `send` waits for prompt-window submission confirmation. PLAN should choose
   the default timeout and exact mstream JSONL fields for timeout and delivery
   outcomes.

4. First UI profiles:
   The design now recommends a `UiProfile` enum. PLAN should decide whether the
   first implementation includes only `Generic` plus one known Codex/Claude
   profile or stubs known profiles behind conservative behavior.

5. Outbound response extraction:
   The crate should reserve module/API space for outbound extraction, but #421
   should not implement it unless reviewers decide it is needed to keep the API
   coherent.
