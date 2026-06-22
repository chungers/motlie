# Conversation Policy Arbitration

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
| 2026-06-22 PDT | @codex-535 | Started issue #545 design for a unified conversation policy that covers barge-in and no-barge-in playback overlap, with an initial PR #558 implementation slice for bounded pending repeats. |
| 2026-06-22 PDT | @codex-535 | Expanded the design to shipped behavior: enum-backed policy decisions now cover no-barge-in pending output, cancel-only barge-in, and post-barge-in silence coalescing through global TOML config. |

## Problem

Live identity/repeat runs on PR #558 showed that no-barge-in overlap is not only a timing knob. When the caller speaks while assistant audio is still active, the gateway currently defers `ConversationCommand::Say`; positive playback hold caps can then drop complete repeat responses. That protects against unlimited backlog, but it makes identity smoke tests unreliable because valid turns can be transcribed and never spoken back.

The same decision point will grow more complex when barge-in is enabled. A caller may be interrupting the assistant, the assistant may be echoing into ASR, or a new response may be ready while prior playback is still draining. Those cases need one policy surface instead of scattered checks in ASR, conversation, and TTS code.

## Requirements

- Preserve current behavior by default for existing configs.
- Add an opt-in no-barge-in policy that keeps bounded pending assistant outputs instead of dropping them when active playback persists.
- Keep barge-in cancellation behavior unchanged in this PR, but shape the code so barge-in policy decisions can move into the same module later.
- Represent policy with typed config under `voice_quality`, strict TOML parsing, and validation ranges.
- Emit quality spans when pending output waits, is superseded, reaches the active-playback hold budget, drains, or is dropped.
- Avoid coupling the policy to the identity processor. The first proving case is identity/repeat, but the abstraction must apply to any processor that emits `ConversationCommand::Say`.

## Non-Goals

- Do not redesign Telnyx media playback or outbound media pacing in this slice.
- Do not change the existing barge-in clear/cancel path yet.
- Do not change turn batching semantics.
- Do not add model-specific behavior.

## Current Behavior

For auto conversation mode, `ConversationCommand::Say` normally queues speech with `CancelAndReplace`. When `voice_quality.barge_in.enabled = false` and active playback exists, the gateway records the response as a proposal and spawns a deferred task.

The deferred task is latest-only:

- a newer deferred response supersedes the previous one;
- if playback clears before the hold limit, the latest response is queued with `Reject`;
- if `endpoint.conversation_playback_max_hold_ms` is positive and reached, the deferred response is dropped.

This explains the PR #558 live observations: inbound audio and ASR can be clean while repeat reliability still fails because completed `Say` commands are intentionally discarded at the no-barge-in overlap boundary.

## Policy Model

The policy owns decisions at conversation output arbitration boundaries:

- `Say` while playback is active;
- playback clear after one or more pending outputs;
- caller barge-in triggers while playback is active;
- call end while pending output exists.

Initial modes:

- `current_compat`: existing latest-only deferred behavior, including max-hold drop.
- `no_barge_in_bounded_pending`: no-barge-in mode that stores a bounded queue of pending outputs and drains them after active playback clears.
- `barge_in_cancel_only`: reserved for moving today's barge-in cancel behavior into the policy module.
- `barge_in_coalesce_after_silence`: reserved for future caller-interruption handling that can cancel, preserve ASR, and regenerate after a silence window.

Config sketch:

```toml
[voice_quality.conversation_policy]
mode = "current_compat"
active_playback_hold_ms = 1000
max_pending_outputs = 1
pending_output_order = "latest_only"
post_barge_in_silence_ms = 1200
```

For identity/repeat reliability tests:

```toml
[voice_quality.conversation_policy]
mode = "no_barge_in_bounded_pending"
active_playback_hold_ms = 1000
max_pending_outputs = 3
pending_output_order = "fifo"
```

## Implemented Policy Boundary

The implementation is enum-backed rather than trait-backed. `conversation_policy.rs` defines typed policy events/decisions for the stable arbitration boundaries, while the existing pipeline keeps ownership of ASR, early response, processor/turn batching, TTS generation, and media playback.

Implemented decisions:

1. `decide_say_overlap`: chooses `queue_now`, `legacy_defer_latest`, or `retain_bounded_pending` when a processor emits `ConversationCommand::Say` while playback may be active.
2. `decide_barge_in`: chooses playback cancellation, generation cancellation, caller-turn handling, and turn-batch reset behavior for speech-onset, partial-ASR, and final-ASR barge-in triggers.
3. `current_compat`: preserves existing latest-only no-barge-in deferral and existing barge-in cancel semantics.
4. `no_barge_in_bounded_pending`: retains a bounded pending output queue and drains one assistant output at a time after playback clears.
5. `barge_in_cancel_only`: makes today's cancel behavior explicit through the policy boundary: cancel active playback, preserve ASR, reset turn batching, and do not replay stale assistant output automatically.
6. `barge_in_coalesce_after_silence`: cancels active playback, preserves ASR, and coalesces post-barge-in finals until `post_barge_in_silence_ms` before processor dispatch.

Media still owns frame-level speech onset and echo-guard classification. When echo guard classifies onset as likely assistant echo, media defers cancellation to partial/final ASR as before. Once a trigger is valid, the policy decides whether cancellation/coalescing proceeds.

## Global TOML Surface

All policy tuning is exposed through strict global config:

```toml
[voice_quality.conversation_policy]
mode = "current_compat"
active_playback_hold_ms = 1000
max_pending_outputs = 1
pending_output_order = "latest_only"
post_barge_in_silence_ms = 1200
```

Live no-barge-in identity smoke tests should use:

```toml
[voice_quality.conversation_policy]
mode = "no_barge_in_bounded_pending"
active_playback_hold_ms = 1000
max_pending_outputs = 3
pending_output_order = "fifo"
post_barge_in_silence_ms = 1200
```

Barge-in coalescing tests should use:

```toml
[voice_quality.conversation_policy]
mode = "barge_in_coalesce_after_silence"
active_playback_hold_ms = 1000
max_pending_outputs = 1
pending_output_order = "latest_only"
post_barge_in_silence_ms = 1200
```

## Testing

Unit coverage must verify:

- default TOML/config preserves `current_compat`;
- unknown policy keys fail strict config parsing;
- compatibility mode still drops after positive max hold;
- bounded pending FIFO queues multiple no-barge-in outputs behind active playback and drains them in order;
- pending output is bounded and does not grow without limit;
- `barge_in_cancel_only` preserves current cancellation behavior through policy decisions;
- `barge_in_coalesce_after_silence` merges multiple post-interruption finals before dispatch.

Live validation should compare repeat reliability before and after enabling `no_barge_in_bounded_pending`, while keeping barge-in off and using the same identity/repeat script protocol in `TESTING.md`.

## Open Questions

- For non-identity processors, should overflow drop oldest, drop newest, or summarize pending outputs?
- Future pause/duck/resume modes need media support beyond current clear/cancel semantics.
- Operator commands for policy tuning may be useful later; the current PR intentionally exposes policy through global TOML only.
