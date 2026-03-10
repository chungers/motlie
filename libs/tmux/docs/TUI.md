# TUI Reliability and Capture Fidelity

This document defines how to support full-screen TUIs in tmux while reducing
sensitivity to mixed client sizes, reflow, and history overflow.

## Short Answer

- Isolation is **not** the only way to run TUI automation.
- Isolation is the only way to get strong, repeatable determinism.

You can still support mixed-client environments, but the system must treat
captures as best-effort and expose degraded-confidence states.

## Why TUI Is Harder Than Line-Oriented Shell Output

Full-screen TUIs (for example `vim`, `less`, `htop`) are cursor-addressed and
frame-updated. They are not simple append-only line logs.

Key failure modes:

1. Reflow and wrapping changes when different-size clients attach.
2. Cursor/control updates are lossy when reduced to plain normalized text.
3. Scrollback truncation from `history-limit` evicts content permanently.

## Mitigation Strategy

Use a layered strategy instead of a single toggle.

### 1) Split Capture Modes

Use distinct modes:

1. `Line` mode: shell/log-like panes, normalization allowed.
2. `Tui` mode: preserve terminal semantics, avoid destructive normalization.

For `Tui` mode, default to raw/control-preserving capture paths.

### 2) Reconstruct Screen State for TUI

For TUI fidelity, process output as a terminal stream and reconstruct a virtual
screen buffer (cells, cursor, attributes) instead of doing line-only parsing.

This keeps fidelity high even when display updates are cursor-based.

### 3) Stabilize Geometry

Set automation windows to explicit size and manual sizing behavior:

```sh
tmux set-option -w -t <session:window> window-size manual
tmux resize-window -t <session:window> -x 160 -y 48
```

This reduces reflow churn but does not guarantee determinism if mixed clients
keep attaching with different sizes.

### 4) Detect and Gate During Resize Churn

Track pane size (`#{pane_width}`, `#{pane_height}`) during command/capture
windows. If size changes mid-operation:

1. mark capture as degraded, or
2. retry in a quiet period, or
3. fail fast for strict workflows.

### 5) Increase History Capacity

Normalization cannot recover evicted scrollback. Increase history proactively:

```sh
tmux set-option -w -t <session:window> history-limit 200000
# or global for newly created windows:
tmux set-option -g history-limit 200000
```

Note: `history-limit` applies to new windows; existing windows keep their
current limit unless recreated.

### 6) Bind Locking and Execution to the Same Pane Identity

For `exec()`-style operations, resolve the effective pane id first (for
example via `display-message -p '#{pane_id}'`), then:

1. acquire lock keyed by that pane id,
2. execute against that pane id,
3. poll/capture from that same pane id.

This avoids lock-target divergence when active pane focus changes.

## Operational Policy Tiers

### Strict (Deterministic)

1. Dedicated automation session/socket.
2. No mixed interactive clients during exec/capture windows.
3. Fixed geometry + high history limit.

Use for tests, CI, or must-pass agents.

### Guarded (Practical Mixed Use)

1. Mixed clients allowed.
2. Fixed geometry attempted.
3. Resize change detection enabled.
4. Degraded/retry semantics when size churn occurs.

Use for human + automation coexistence where occasional retries are acceptable.

### Best-Effort (Shared Interactive)

1. No isolation guarantee.
2. No strict geometry/control guarantee.
3. Captures are advisory, not deterministic.

Use only when occasional fidelity loss is acceptable.

## Recommended Defaults

1. Default `capture()` behavior should remain explicit about mode.
2. Default to line-oriented normalization only for line-oriented workflows.
3. Default TUI-oriented workflows to `Raw`/terminal-state paths.
4. Always document when output is deterministic vs best-effort.

