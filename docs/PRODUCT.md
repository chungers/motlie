# Product Notes

## Competitive Analysis: Motlie vs `tmux-mcp-rs`

Date: 2026-03-21  
Author: @codex

### Scope

This comparison intentionally excludes MCP-specific capability and focuses on the
underlying tmux-over-SSH foundation:

- tmux control coverage
- remote transport and SSH ergonomics
- streaming and monitoring
- robustness and resilience
- operability for long-lived automation or agent workflows

Primary external references:

- `tmux-mcp-rs` docs.rs crate page: <https://docs.rs/tmux-mcp-rs/latest/tmux_mcp_rs/>
- `tmux-mcp` project page / docs: <https://mcpservers.org/servers/bnomei/tmux-mcp>

Primary Motlie references:

- [libs/tmux/docs/API.md](../libs/tmux/docs/API.md)
- [libs/tmux/docs/DESIGN.md](../libs/tmux/docs/DESIGN.md)
- [libs/tmux/docs/PLAN.md](../libs/tmux/docs/PLAN.md)
- [libs/tmux/examples/README.md](../libs/tmux/examples/README.md)

## Summary

As a tmux-over-SSH library foundation, Motlie is stronger on transport design,
streaming architecture, and resilience work. `tmux-mcp-rs` is stronger on breadth
of tmux operations and command-oriented convenience.

Put differently:

- Motlie is closer to a durable foundation for long-lived remote automation and
  external-agent loops.
- `tmux-mcp-rs` is closer to a broad tmux operations facade over the system tmux
  client.

## Where Motlie Exceeds

### 1. Typed SSH foundation

Motlie has an explicit SSH model with:

- structured `SshConfig`
- host-key policy
- timeout and keepalive settings
- tmux socket selection
- explicit key-file authentication
- local / SSH / mock transport parity

This is stronger than a shell-through-to-`ssh` model when the product goal is a
library-grade remote foundation.

### 2. Built-in remote file transfer

Motlie already includes host-level upload/download with local, mock, and SSH/SFTP
 implementations. That materially improves its value as a remote execution
 substrate rather than "just tmux commands."

### 3. Event-driven streaming substrate

Motlie has a deeper streaming model:

- control-mode monitoring
- `OutputBus`
- `Subscription`
- `JoinedStream`
- `HistoryHandle`
- `Fleet`

This is a significant advantage for external-agent or long-running automation
 workflows because it supports incremental consumption rather than repeated
 polling/capture loops.

### 4. Resilience work is materially ahead

Motlie now includes:

- reconnect supervision
- discontinuity modeling separate from subscriber-local gaps
- truthful reconnect snapshot anchoring
- per-session monitor health
- Fleet-level aggregation of monitor health

This is the biggest product difference if the use case is "long-lived tmux over
SSH" rather than one-shot command tooling.

### 5. Multi-host coordination

`Fleet` gives Motlie a better story for coordinating multiple hosts/sessions and
building combined histories across them. That is a meaningful foundation
advantage for real automation systems.

### 6. Testing posture

Motlie's transport and monitoring layers have broader unit/integration coverage,
including mock transport paths and resilience-oriented scenarios. That gives it a
better claim to being a robust library substrate.

## Where Motlie Lags

### 1. tmux operation breadth

`tmux-mcp-rs` appears to cover a wider tmux command surface, including areas like:

- richer pane/window manipulation
- layout management
- synchronize-panes
- join/break/swap pane flows
- client-level operations
- buffer-oriented tools and search helpers

Motlie's current API surface is narrower and more intentionally typed.

### 2. Tracked command execution

`tmux-mcp-rs` exposes a more explicit command lifecycle model via command IDs and
result retrieval. Motlie has `Target::exec()`, but not a comparable command
tracking handle/result model.

### 3. OpenSSH parity / convenience

`tmux-mcp-rs` appears to lean more directly on system `ssh` behavior, which can
make common operator workflows easier:

- `-i`-style identity choice
- likely easier use of existing `~/.ssh/config`
- lower friction for operators already thinking in shell/CLI terms

Motlie's explicit typed SSH model is better for library discipline, but it is not
yet as convenient as broad OpenSSH config parity.

### 4. Socket-isolation ergonomics

Motlie supports tmux socket selection, but the operational ergonomics around
isolated sockets are not yet first-class. `tmux-mcp-rs` highlights socket usage
more directly as an operational isolation tool.

### 5. Policy/scoping controls

`tmux-mcp-rs` appears to have more explicit operational control around what tmux
surfaces are allowed or scoped. Motlie does not currently expose a comparable
policy layer.

## Product Takeaways

If Motlie's product identity is:

> a robust tmux-over-SSH foundation for automation and external-agent workflows

then its current advantage is real and should be protected. The right next moves
are not to mimic MCP, but to close the remaining foundation gaps that matter for
robustness and operability.

The two most relevant gaps surfaced by this comparison are:

1. tracked command execution
2. socket-isolation ergonomics

## Recommendation: Tracked Command Execution

### Should Motlie add it?

Yes, but narrowly and as a library primitive, not as an agent workflow layer.

### Why it is a good fit

Tracked command execution would improve robustness because it separates:

- command launch
- command observation
- command result retrieval

That matters under real-world conditions:

- reconnects
- slow or long-running commands
- polling consumers
- multi-step orchestration

It also reduces pressure on "fire command and block until sentinel" as the only
execution model.

### Why it should stay narrow

Motlie should not become a workflow engine or retained action server. The right
shape is likely:

- start execution
- get a typed command handle / ID
- poll or await result
- optionally inspect status

This would complement `Target::exec()` rather than replace it.

### Product recommendation

Treat tracked command execution as a good medium-priority enhancement for
robustness and operability.

It is especially valuable if Motlie is meant to support:

- remote command orchestration
- long-lived SSH sessions
- external agents that may disconnect/reconnect while a pane command is running

## Recommendation: Socket Isolation

### Should Motlie improve it?

Yes, and more aggressively than tracked command execution.

### Why it matters

Socket isolation directly improves correctness and resilience:

- reduces interference from unrelated human/shared tmux activity
- improves determinism for monitoring and capture
- reduces accidental cross-talk between automation workloads
- makes operational boundaries clearer

For a "robust tmux over SSH" product, isolated sockets are one of the cleanest
ways to improve reliability without adding complex runtime logic.

### What "better support" should mean

Not just "allow socket name/path in the config." Motlie should make isolated
socket workflows easier and more discoverable:

- easier construction and propagation of socket-scoped `HostHandle`s
- example-driven guidance for dedicated automation sockets
- clearer API/documentation guidance that dedicated sockets are the preferred
  robustness path
- possible helpers for creating or validating dedicated automation sockets

### Product recommendation

Treat socket-isolation ergonomics as a high-priority robustness feature.

This aligns directly with Motlie's product identity as a resilient foundation and
likely yields more operational value per unit of implementation than many broader
tmux command-surface additions.

## Recommended Priority Order

From a product perspective, the next useful Motlie gaps to close are:

1. better socket-isolation ergonomics
2. tracked command execution
3. selective expansion of tmux operation breadth
4. OpenSSH parity improvements where they clearly improve operator usability

## Non-Goals For This Comparison

These were intentionally not treated as decision drivers here:

- MCP protocol integration
- prompt/tool UX
- server packaging
- language-model-facing tool schema design

The comparison is specifically about the underlying tmux-over-SSH foundation.
