# PLAN: agent::Channel Implementation

## Changelog

| Date (PDT) | Who | Summary |
|------------|-----|---------|
| 2026-06-09 16:22 PDT | @codex-421-design | Added api-rv hardening: capped total coalesce wait and final pre-send quiet-guard re-check with integration coverage for sustained input and typing during debounce. |
| 2026-06-09 15:55 PDT | @codex-421-design | Added missed #421/#420 CLI rename gate: public `--no-prompt-submit` with hidden `--no-enter` compatibility alias across send, broadcast, and timer. |
| 2026-06-09 15:22 PDT | @codex-421-design | PR #425 round-2 plan update: add no-enter quiet-guard regression coverage with missing client session id, quiet-guard trace hook, and atomic coalescing wait/drain. |
| 2026-06-09 14:27 PDT | @codex-421-design | Reopened #421 live-validation fix plan: stable session-id client activity matching, quiescent coalescing window, and public `libs/agent` integration coverage for concurrent same-channel senders. |
| 2026-06-09 00:52 PDT | @codex-421-design | Reconciled code review fixes: poison-safe channel locks, pure-sync timeout cancellation, mstream delivery-event observation, channel cleanup on teardown, timer deferral ownership in `motlie-agent`, and send/broadcast submit retry wiring. |
| 2026-06-09 00:07 PDT | @codex-421-design | Marked final implementation gates complete: package-scoped fmt check, build, tests, clippy, and focused tmux writable-client activity tests all pass. Workspace-wide `cargo fmt` remains blocked by unrelated missing `examples/vector2/app/benchmark.rs`. |
| 2026-06-09 00:02 PDT | @codex-421-design | Added implementation PLAN for PR #423 coding phase: `motlie-tmux` writable-client activity, new `motlie-agent` Channel API, mstream send/broadcast/timer migration, docs, and verification gates. |

## Scope

Implement [DESIGN.md](./DESIGN.md) for issue #421 in the same PR as the accepted design. This PLAN covers the code slice only: `motlie-tmux` mechanism extension, new `motlie-agent` crate, and mstream integration. Telnyx transcript delivery remains future crate-level reuse.

## Phases

### 1. `motlie-tmux` Mechanism Extension

- [x] Add `SessionClientActivity.latest_writable_client_activity: Option<u64>` without changing the meaning of `latest_client_activity`.
- [x] Compute writable activity only from clients where `readonly == false`.
- [x] Add/adjust tests proving read-only client activity does not affect writable activity.
- [x] Carry tmux `session_id` through client activity parsing and match activity by stable id when available.

Verify:

```sh
cargo test -p motlie-tmux session_client_activity --lib
```

### 2. `motlie-agent` Crate

- [x] Add workspace crate `libs/agent` / `motlie-agent` depending on `motlie-tmux`.
- [x] Implement public API: `Channel`, `ChannelManager`, `SessionKey`, `ResolvedSession`, `ChannelConfig`, `UiProfile`, `ManagedMessage`, `MessageSource`, `SubmitPolicy`, `SendOptions`, `EnqueueOptions`, `SubmissionOutcome`, `QueuedDelivery`, `DeliveryEvent`, `ChannelStatus`, `DeferReason`, and `DeliveryError`.
- [x] Keep `UiProfile` per session through `ResolvedSession.ui_profile`, with `ChannelConfig.default_ui_profile` fallback.
- [x] Implement synchronous `send` and asynchronous `enqueue`.
- [x] Implement default-on dedup with zero-or-many waiters per pending segment.
- [x] Implement attributed coalescing with natural `[from: source]` headers.
- [x] Wait for a quiet `coalesce_window` before draining pending messages so real concurrent senders coalesce.
- [x] Cap total coalesce debounce with `ChannelConfig.coalesce_max_wait` so sustained sub-window input eventually drains.
- [x] Close the coalescing lock gap by performing the stable-generation check and drain under one queue lock.
- [x] Implement quiet-guard deferral using `latest_writable_client_activity`.
- [x] Re-check quiet guard after coalescing, before payload delivery, and requeue/defer if writable activity appeared during debounce.
- [x] Add public integration coverage proving typing-only/no-enter payloads are also gated by no-barge-in.
- [x] Implement payload/Enter separation with settle delay, retry policy, and profile-based verification fallback.
- [x] Expose delivery observability through a Tokio broadcast receiver and status snapshots.

Verify:

```sh
cargo test -p motlie-agent
cargo clippy -p motlie-agent -- -D warnings
```

### 3. mstream Integration

- [x] Add `DaemonState.channel_manager` so send, broadcast, and timers share channel pending state for the daemon lifetime.
- [x] Migrate `send_shared` to `Channel::send` and return channel message id / verification metadata.
- [x] Migrate `broadcast_shared` to `Channel::enqueue` and return accepted channel message ids.
- [x] Migrate `timer_fire_once_shared` prompt delivery to `Channel::enqueue`; `motlie-agent` now owns no-barge-in deferral and mstream observes deferred/submitted/failed outcomes through `DeliveryEvent`s.
- [x] Update the legacy direct-send helper to use separate payload and Enter for remaining onboarding/handoff paths.
- [x] Remove mstream-owned timer quiet-guard evaluation from the delivery path.
- [x] Remove affected channels during stale-session quarantine, reclaim, and disconnect.

Verify:

```sh
cargo check -p motlie-mstream
cargo test -p motlie-mstream
cargo clippy -p motlie-mstream -- -D warnings
cargo test -p motlie-tmux session_client_activity --lib
```

Note: workspace-wide `cargo fmt` is currently blocked by unrelated missing module `examples/vector2/app/benchmark.rs`; this PLAN uses package-scoped formatting checks for touched Rust packages.

### 4. Final Gate

- [x] Run focused build/test/clippy gates for `motlie-agent` and `motlie-mstream`.
- [x] Run mstream tests.
- [x] Run formatting for touched packages (`cargo fmt --check -p motlie-agent -p motlie-mstream -p motlie-tmux`).
- [ ] Commit with `@codex-421-design` and PDT datetime.
- [ ] Push branch `codex-421-design/agent-inbox` to PR #423.

Commands:

```sh
cargo fmt --check -p motlie-agent -p motlie-mstream -p motlie-tmux
cargo build -p motlie-agent -p motlie-mstream
cargo test -p motlie-agent
cargo test -p motlie-mstream
cargo clippy -p motlie-agent -- -D warnings
cargo clippy -p motlie-mstream -- -D warnings
cargo test -p motlie-tmux session_client_activity --lib
```

Note: workspace-wide `cargo fmt` is currently blocked by unrelated missing module `examples/vector2/app/benchmark.rs`; this PLAN uses package-scoped formatting checks for touched Rust packages.
