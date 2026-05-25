# API: mstream CLI

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-24 | @codex | Addressed PR #330 re-review: request execution now snapshots state under short locks, performs SSH/tmux awaits outside the state mutex, then re-locks briefly to reconcile events and metadata. |
| 2026-05-24 | @codex | Addressed PR #330 feedback: bounded event cursors advance only to the last returned event, handoffs trigger from all explicit state-change paths, recruited sessions persist workstream tags, daemon connections are spawned per client, scan hydrates `cwd`, and broadcast touches `updated-at`. |
| 2026-05-23 | @codex | Documented the first implemented `mstream` CLI/daemon surface, JSONL protocol, and current observation limits. |

## Status

`mstream` is implemented as a `motlie-mstream` package with binary name
`mstream`. It provides a Unix-domain socket daemon, JSONL client responses,
strict `<host>::<session>` target parsing, in-memory workstreams, tmux session
tags, state marking, send/interrupt/broadcast, handoffs, and bounded
observation commands.

The first implementation keeps command/event history in an in-memory
per-workstream ring buffer. `snapshot` and `summary-input` use bounded one-shot
tmux capture for joined sessions. Continuous OutputBus transcript ingestion and
pane/process-state stuck hints remain implementation follow-ups.

## Daemon

```sh
cargo run -p motlie-mstream -- daemon start --socket /tmp/mstream.sock
cargo run -p motlie-mstream -- --socket /tmp/mstream.sock daemon status
cargo run -p motlie-mstream -- --socket /tmp/mstream.sock daemon stop
```

`daemon start` daemonizes by default. Use `--foreground` for tests and manual
debugging:

```sh
cargo run -p motlie-mstream -- --socket /tmp/mstream.sock daemon start --foreground
```

The foreground daemon accepts each socket connection in its own task. Commands
snapshot daemon state under short locks, perform SSH/tmux awaits outside the
state mutex, then re-lock briefly to reconcile metadata and emit events.

Client commands resolve the socket from `--socket`, then `MSTREAM_SOCKET`, then
`/tmp/mstream-${USER}.sock`.

## Host And Workstream Commands

```sh
mstream connect local ssh://localhost --label pool=local --work-root /tmp
mstream hosts
mstream scan local
mstream disconnect local

mstream open pr-324 --title "mstream implementation" --goal "Implement PR 324"
mstream list
mstream show pr-324
mstream close pr-324 --summary "done" --domain tmux --specialty mstream
```

Host metadata is daemon memory only. After daemon restart, reconnect hosts and
run `scan` to hydrate tagged sessions from tmux.

## Session Assignment

```sh
mstream join pr-324 local::codex-reviewer --role reviewer --task "Review this branch."

mstream new pr-324 local::codex-worker \
  --role implementer \
  --cwd /tmp/mstream-worker \
  --agent codex \
  --task "Implement the next phase."

mstream leave pr-324 local::codex-worker --available
mstream kill local::codex-worker
```

`new` validates absolute `--cwd`, creates the directory on the target host, and
starts the agent through a narrow shell bootstrap. Joined/new sessions receive
`@mstream/*` tags and a managed-agent reporting prompt when a task is sent.
Recruited sessions also receive workstream-membership tags before any task is
sent, so restart plus `scan` can hydrate their assignment.

## Communication And Handoff

```sh
mstream send pr-324 local::codex-worker --text "Re-run clippy." --enter
mstream send pr-324 local::codex-worker --interrupt-first --settle-ms 500 \
  --text "Stop and address feedback." --enter --set-state busy

mstream interrupt local::codex-worker
mstream interrupt local::codex-worker --key ctrl-c

mstream broadcast pr-324 --state busy --text "Wrap up and mark done or blocked." --enter

mstream session mark local::codex-worker --state done --summary "Implemented requested fixes."
mstream session mark self --state blocked --summary "Need host credentials."
```

When the target is joined to an open workstream, `interrupt` appends an
`interrupted` event to that workstream and returns the resulting cursor. If the
target is a connected tmux session that is not joined to a known workstream, the
command still sends the key and returns only the command result.

`session mark self` resolves `MSTREAM_TARGET` in the client environment.

Handoffs are daemon-memory edges:

```sh
mstream handoff arm pr-324 \
  --from local::codex-worker \
  --to local::codex-reviewer \
  --on done \
  --task "Worker marked done. Review now."

mstream handoff list pr-324
mstream handoff cancel pr-324 h1
```

If the source is already in the requested state, the handoff fires immediately
unless `--only-on-transition` is supplied. Firing marks the destination `busy`,
updates tags, sends the task, and emits `handoff_fired`.

## Observation

```sh
mstream status pr-324
mstream events pr-324 --limit 50
mstream snapshot pr-324 --max-chars 12000
mstream summary-input pr-324 --max-chars 12000
```

Event cursors are opaque base64 JSON owned by `mstream`; they embed the
workstream timeline generation. A cursor from an older generation returns a
structured `cursor_stale` JSONL error. Bounded `events --limit N` responses
return a cursor that advances only to the last returned event, not to the
workstream watermark.

All machine-facing output is JSONL on stdout. Errors are also JSONL records,
for example:

```jsonl
{"type":"error","kind":"daemon_unreachable","message":"daemon unreachable at /tmp/mstream.sock; start the daemon or provide --socket"}
```
