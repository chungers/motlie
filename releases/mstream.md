# mstream 0.1.0

`mstream` is Motlie's agent-facing tmux workstream orchestrator. It provides a
Unix-domain socket daemon, JSONL client responses, tmux-backed targets, durable
audit events, workstream membership, communication, handoffs, timers, and
bounded observation commands.

## Summary

This is the first `mstream` release artifact set. It packages the implemented
CLI/daemon surface from `main` for direct installation on Linux and macOS.

## Changes

- Ships daemon lifecycle commands: `daemon start`, `daemon status`, and
  `daemon stop`.
- Ships host and workstream management commands: `connect`, `hosts`, `scan`,
  `disconnect`, `open`, `label`, `list`, `show`, and `close`.
- Ships agent/session workflow commands: `join`, `new`, `leave`, `retire`,
  `reclaim`, `send`, `interrupt`, `broadcast`, `rename`, `session list`, and
  `session retag`.
- Ships handoff, timer, status, doctor, events, snapshot, summary-input,
  recruit, and attach surfaces documented in the CLI API.
- Persists durable socket-adjacent JSONL audit events so readable events can
  survive daemon restart.

## Install

```sh
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mstream.sh | sh
/usr/local/bin/mstream --version
```

## Targets

`linux-x64-musl`, `linux-arm64-musl`, and `darwin-arm64`.
Linux targets are static musl builds. The Darwin target is Apple Silicon only, ad-hoc signed, and
verified from the installed path.

## Compatibility

`mstream` is an orchestrator-facing command and is not an SSH `ForceCommand`
entrypoint. It expects reachable tmux targets and uses a local Unix-domain
socket path selected by `--socket`, `MSTREAM_SOCKET`, or the default
`/tmp/mstream-${USER}.sock`.

## Known Issues

- Host metadata is daemon memory only; after daemon restart, reconnect hosts and
  run `scan` to hydrate tagged sessions from tmux.
- Snapshot and summary-input use bounded one-shot tmux capture for joined
  sessions; deeper pane/process-state stuck hints remain follow-up work.
- Archive payloads contain `bin/mstream` only; `README.md` and `LICENSE` files
  are explicitly deferred for this release event.

## References

[`docs/API.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mstream/docs/API.md) |
[`docs/DESIGN.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mstream/docs/DESIGN.md) |
[`docs/PLAN.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mstream/docs/PLAN.md)
