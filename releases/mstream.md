# mstream 0.1.0

`mstream` release notes for `2026-06-bright-beacon`.

## Summary

This is the first `mstream` release artifact set, packaging Motlie's agent-facing tmux workstream orchestrator for direct archive and installer distribution.

## Changes

Ships daemon lifecycle commands: `daemon start`, `daemon status`, and `daemon stop`.

Ships host and workstream management commands: `connect`, `hosts`, `scan`, `disconnect`, `open`, `label`, `list`, `show`, and `close`.

Ships agent and session workflow commands: `join`, `new`, `leave`, `retire`, `reclaim`, `send`, `interrupt`, `broadcast`, `rename`, `session list`, and `session retag`.

Ships handoff, timer, status, doctor, events, snapshot, summary-input, recruit, and attach command surfaces.

Persists durable socket-adjacent JSONL audit events so readable events can survive daemon restart.

## Install

```sh
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mstream.sh | sh
/usr/local/bin/mstream --version
```

The installer defaults to `/usr/local/bin/mstream`. Use `--prefix` to select another prefix:

```sh
curl -fsSLO https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mstream.sh
sh install-mstream.sh --prefix "$HOME/.local"
```

## Targets

| Target | Archive asset |
| --- | --- |
| linux-x64-musl | [motlie-mstream-v0.1.0-linux-x64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mstream-v0.1.0-linux-x64-musl.tar.gz) |
| linux-arm64-musl | [motlie-mstream-v0.1.0-linux-arm64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mstream-v0.1.0-linux-arm64-musl.tar.gz) |
| darwin-arm64 | [motlie-mstream-v0.1.0-darwin-arm64.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mstream-v0.1.0-darwin-arm64.tar.gz) |

Linux targets are static musl builds. The Darwin target is Apple Silicon only, ad-hoc signed, and verified from the installed path by the installer.

## Compatibility

`mstream` is an orchestrator-facing command and is not an SSH `ForceCommand` entrypoint.

`mstream` expects reachable tmux targets and uses a local Unix-domain socket path selected by `--socket`, `MSTREAM_SOCKET`, or the default `/tmp/mstream-${USER}.sock`.

## Known Issues

Host metadata is daemon memory only; after daemon restart, reconnect hosts and run `scan` to hydrate tagged sessions from tmux.

Snapshot and summary-input use bounded one-shot tmux capture for joined sessions; deeper pane/process-state stuck hints remain follow-up work.

Archive payloads contain only `bin/mstream`.

## References

[`bins/mstream/docs/API.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mstream/docs/API.md) |
[`bins/mstream/docs/DESIGN.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mstream/docs/DESIGN.md) |
[`bins/mstream/docs/PLAN.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mstream/docs/PLAN.md)
