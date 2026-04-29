# mmux

Host-wide tmux session selector. Pick a session, preview or monitor it, then attach.

## Overview

`mmux` is a focused TUI binary that lists tmux sessions on a target host, previews or live-monitors the highlighted session, and attaches the user's terminal to the chosen session with a clean PTY handoff. It is designed to be used both interactively from the operator's machine (against a local or remote tmux server) and as the host-wide login entry point via SSH `ForceCommand`.

The selector is built on `motlie-tmux` and never embeds shell or tmux command construction in the binary; all tmux/SSH operations go through typed library APIs.

## Modes

- **Local** — `mmux` (no arguments) targets the local host's tmux server.
- **SSH single-host** — `mmux ssh://user@host` targets one remote host.
- **Multi-host** *(planned, [issue #235](https://github.com/chungers/motlie/issues/235))* — `mmux ssh://a ssh://b ...` aggregates sessions across multiple hosts in a single activity-sorted list. Multi-host mode replaces the per-host MOTD pane and host/IP status with a `mmux - multi-host mode (n)` banner; row format gains a hostname column.
- **Script** — `mmux --script` prints the selected session name to stdout instead of attaching, for shell composition (e.g. `tmux attach -t "$(mmux --script)"`).

## Layouts

Two layouts auto-detected from PTY aspect ratio (override with `-l/--landscape` or `-p/--portrait`):

- **Landscape** — left column has MOTD over the session list; right column shows the detail pane.
- **Portrait** — vertical T/B split (no MOTD pane); session list on top, detail below. Optimized for narrow terminals (mobile SSH, IDE panels, tmux popups).

In multi-host mode the MOTD pane is hidden in landscape too — the selector lists sessions across hosts, not host-specific motd.

## Documentation

| Doc | Purpose |
|-----|---------|
| [`DESIGN.md`](./DESIGN.md) | Functional and non-functional requirements; layout; data flow; library-gap rationale. |
| [`PLAN.md`](./PLAN.md) | Phased implementation tasks with checkboxes, dev harness, test matrix. |
| [`API.md`](./API.md) | Internal API contract (CLI config, detail-source trait, host events). |
| [`CLI.md`](./CLI.md) | User-facing CLI contract (flags, keymap, ForceCommand, exit semantics). |
| [`mmux-mock.svg`](./mmux-mock.svg) | TUI mock states. |

## Library extensions used (`motlie-tmux`)

- `Target::attach_current_pty()` + `AttachExit` — spawn-and-wait PTY handoff with `setpgid` + `tcsetpgrp` signal hygiene.
- `HostHandle::session_by_id()` + `SessionId(String)` newtype — id-based dispatch immune to display-name renames mid-flight.
- `HostHandle::watch_host_events()` + `HostEvent` + `HostEventStream` — host-level event reconciliation.
- `HostHandle::read_text_file(path, max_bytes)` — bounded host-text reads (e.g. `/etc/motd`).
- `HostHandle::list_sessions_now() -> SessionListing` — bundled server-clock for skew-free recency math; `SessionInfo.activity` field.
- `ScrollbackQuery::LinesRange { older_than_lines, count }` — windowed backwards-fetch for the detail pane.

## Issue tracker

- Parent feature: [#226](https://github.com/chungers/motlie/issues/226)
- Multi-host extension: [#235](https://github.com/chungers/motlie/issues/235)
- Phase 9.6 SSH/ForceCommand integration tests: [#232](https://github.com/chungers/motlie/issues/232)
