# Design: mstream Agent Workstreams

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-21 | @codex | Initial design for issue #323: stateless `bins/mstream` daemon/client, tmux-tag hydration, host reconnect flow, workstream CLI, recruiting, and JSONL observation. |

## Status

This document begins the design for [issue #323](https://github.com/chungers/motlie/issues/323).
`mstream` is a greenfield product surface: it is a new binary and command
model over existing `motlie-tmux` capabilities, so there is no migration or
backwards-compatibility contract for existing users.

The related `libs/tmux` timeline dependency is
[issue #322](https://github.com/chungers/motlie/issues/322). Until #322 lands,
`mstream` can expose useful arrival-order merged output, but true timestamp
merge-sorted multi-agent timelines remain a dependency.

## Business Problem

Motlie can already create, inspect, and monitor tmux sessions through
`libs/tmux` and the lower-level `bins/tmux/driver`, but an orchestrating agent
does not have a compact interface for managing several other agents as one
workstream.

The orchestrator currently has to remember operational state in its own
conversation:

- which hosts are reachable and which SSH URI connects to each one
- which tmux sessions are running agents
- which sessions are assigned to a task, PR, issue, or review role
- which sessions are idle or available for new work
- how to watch multiple agents without flooding the orchestrator's context
- how to recover after a watcher process or daemon restart

This is fragile. It pushes every orchestrating agent toward custom local
state, custom buffering, and ad hoc tmux commands. `mstream` should make the
common operation explicit: connect known hosts, model workstreams, join or
create agent sessions, watch bounded output, and periodically summarize or
unstick collaborators.

## Non-Goals

- No TUI. The orchestrating agent needs JSONL stdout, not a rendered interface.
- No durable local database, state directory, or tiny host config file.
- No hidden host discovery. Humans provide host aliases and SSH URIs.
- No replacement for `bins/tmux/driver`; that remains the low-level tmux
  operator/debug tool.
- No separate `libs/workstream` wrapper in the first slice.
- No automatic takeover of arbitrary tmux sessions without explicit join or
  managed-session tags.
- No exact timestamp merge-sort before `libs/tmux` provides the timeline
  support tracked by issue #322.

## Requirements

### Functional Requirements

- FR1: Provide a new `bins/mstream` binary with daemon and client commands.
- FR2: Hide raw tmux, Fleet, and OutputBus mechanics behind a `workstream`
  abstraction.
- FR3: Start the daemon with zero hosts; hosts are connected by explicit
  client commands after startup.
- FR4: Support a workstream with zero or more associated tmux agent sessions.
- FR5: Join an existing tmux session to a workstream and optionally send a
  pivot/task prompt.
- FR6: Create a new tmux session on a connected host, start an agent binary,
  tag the session, and join it to a workstream.
- FR7: Leave a session from a workstream without killing it.
- FR8: Kill a session only through an explicit destructive command.
- FR9: Recruit agents from connected/scanned hosts, preferring tagged available
  sessions before creating new sessions.
- FR10: Scan connected hosts and hydrate workstream/session state from tmux
  session tags.
- FR11: Observe workstreams through bounded polling commands suitable for an
  orchestrating agent.
- FR12: Emit machine-facing output as JSONL on stdout. Human diagnostics and
  debug logs go to stderr.

### State And Recovery Requirements

- SR1: The daemon is an in-memory control-plane process. It may cache host
  connections, sessions, cursors, and timeline buffers while running.
- SR2: The daemon must not persist host aliases, SSH URIs, workstreams, or
  session ledgers to a local file or database.
- SR3: After daemon restart, an orchestrating agent asks the human for the host
  aliases and SSH URIs, reconnects those hosts, and rescans.
- SR4: Durable session/workstream metadata lives in tmux session user options
  under the `mstream` namespace.
- SR5: Tmux history is the durable output snapshot source. OutputBus buffers
  are live/volatile and can be rebuilt only from still-available tmux history.
- SR6: Empty workstreams are ephemeral because there is no durable local store
  and no session tag to attach their metadata to.

### Non-Functional Requirements

- NFR1: Prefer existing `libs/tmux` APIs for tmux operations and command
  construction.
- NFR2: Keep client output bounded by default so an agent can safely process it.
- NFR3: Make restart behavior explicit and boring: reconnect hosts, rescan
  tags, resume monitoring.
- NFR4: Do not invent host aliases, SSH URIs, credentials, host capacity, or
  availability state.
- NFR5: Make all state-changing commands return enough JSONL for an agent to
  understand what changed and what cursor to use next.
- NFR6: Keep the first implementation useful without issue #322 while avoiding
  a CLI contract that blocks timestamp-sorted timelines later.

## Selected Design

### Public Abstraction

The public abstraction is a workstream. A workstream is a named unit of
collaborative agent work, such as `pr-322`, that can have zero or more tmux
sessions attached while the daemon is running.

Users and orchestrating agents should think in these terms:

- connect a host
- scan known sessions
- create a workstream
- join or create agents for the workstream
- recruit available agents
- poll status, events, snapshots, or summary input

The public CLI should not expose Fleet or OutputBus directly. Internally,
`mstream` uses Fleet for connected host/session monitoring and OutputBus for
live output fan-out.

### Binary And Layering

`bins/mstream` owns the workstream business logic. It may use:

- `libs/tmux` for SSH/local host connections, tmux session creation, session
  tags, Fleet, OutputBus, history snapshots, and send-keys/send-text
- `libs/driver` REPL/client infrastructure if it fits the daemon/client
  implementation, but not as the user-facing abstraction

There is intentionally no `libs/workstream` in the initial design. If the
binary logic later becomes reusable and stable, it can be extracted with
evidence from the implementation rather than guessed upfront.

### Daemon Lifecycle

The daemon starts empty and daemonizes by default so an orchestrating agent does
not need to hold a long-running command open:

```sh
mstream daemon start --socket /tmp/mstream.sock
mstream --socket /tmp/mstream.sock daemon status
mstream --socket /tmp/mstream.sock daemon stop
```

`mstream daemon start --foreground --socket /tmp/mstream.sock` may be used for
development and tests. The client connects over a local Unix-domain socket. The
socket path is either provided by `--socket`, by `MSTREAM_SOCKET`, or by a
documented default such as `/tmp/mstream-${USER}.sock`. This default is not
durable state; it is only a connection convention.

If the client cannot connect, it should fail with JSONL that clearly says the
daemon is unreachable. The orchestrating agent should then ask the human to
restart the daemon or provide the correct socket. After restart, the
orchestrating agent asks the human for the host aliases and SSH URIs and runs
`mstream connect` for each host.

### Host Ledger

The host ledger is daemon memory only. It is built by explicit connect
commands:

```sh
mstream connect amd1 ssh://amd1
mstream connect amd2 'ssh://user@amd2?identity-file=/abs/key'
mstream hosts
mstream scan amd1
mstream disconnect amd1
```

Optional runtime-only connect metadata may be accepted when useful:

```sh
mstream connect amd1 ssh://amd1 --label pool=amd --capacity agents=6 --work-root /abs/work
```

Those labels and capacities are not persisted. After daemon restart, the human
must provide them again if they matter. Recruiting must behave conservatively
when capacity or placement metadata is absent.

### Session Target Syntax

Targets use a stable human-readable form:

```text
<host-alias>::<tmux-session-name>
```

Examples:

```text
amd1::gpt55-mmux-reviewer
amd2::ops47-mmux-submitter
```

The host alias must already be connected. The tmux session name is the agent's
operational identity in that session. When `mstream new` starts an agent, the
initial prompt should also tell the agent its identity explicitly.

### Tmux Tag Schema

Durable session/workstream metadata is stored in tmux user-defined session
options under the `mstream` prefix. `libs/tmux` already exposes session tags
through `Target::tags(prefix)` and batch reads through
`HostHandle::list_tags_for_session_infos(prefix, sessions)`.

Initial tags:

```text
@mstream/version=1
@mstream/managed=true
@mstream/workstream=pr-322
@mstream/workstream-title=OutputBus timelines
@mstream/role=reviewer
@mstream/agent=codex
@mstream/identity=ops47-mmux-reviewer
@mstream/state=available|busy|idle|reserved
@mstream/cwd=/abs/path
@mstream/updated-at=2026-05-21T12:34:56Z
```

The tag values must stay small. Larger state belongs in tmux history, in the
repo, or in the human/orchestrator conversation. Tags are for hydration and
selection, not transcripts.

### Hydration Flow

After connecting hosts, the daemon hydrates by scanning tmux:

1. List sessions on each connected host.
2. Batch-read `mstream` tags for those sessions.
3. Build an in-memory session ledger keyed by `(host_alias, session_id)`.
4. Derive workstream membership from `@mstream/workstream`.
5. Start or refresh OutputBus monitoring for joined sessions.
6. Capture bounded tmux history to seed snapshot output.

If a workstream has no tagged sessions, it cannot be reconstructed after
restart. This is acceptable for the first design because the no-local-state
requirement is stronger than durable empty-workstream metadata.

### Workstream Commands

Create an in-memory workstream:

```sh
mstream create pr-322 --title "OutputBus timelines"
mstream list
mstream show pr-322
mstream close pr-322
```

Join an existing tmux session:

```sh
mstream join pr-322 amd1::gpt55-mmux-reviewer \
  --role reviewer \
  --task "Review PR 322 and report blocking findings first."
```

`join` writes the `mstream` tags, starts monitoring the session, and sends the
task prompt if `--task` is provided. It must not create a session.

Create a new session and join it:

```sh
mstream new pr-322 amd2::ops47-mmux-reviewer \
  --role reviewer \
  --cwd /abs/path/pr-322-reviewer \
  --agent codex \
  --task "Review PR 322 and report blocking findings first."
```

`new` validates that `--cwd` is absolute, creates the directory on the target
host if needed, starts the named agent binary in that directory, tags the
session, starts monitoring, and sends the task prompt.

Implementation note: current `CreateSessionOptions` does not expose a start
directory. The first implementation can use a generated shell bootstrap command
that does only `mkdir -p`, `cd`, and `exec <agent>` with validated/escaped
arguments. A later `motlie-tmux` improvement may add `new-session -c` support
to `CreateSessionOptions`.

Leave or kill:

```sh
mstream leave pr-322 amd1::gpt55-mmux-reviewer
mstream kill amd1::gpt55-mmux-reviewer
```

`leave` unsets workstream-specific tags but leaves the session running.
`kill` is explicit and destructive.

### Recruiting

Recruiting uses only connected/scanned hosts and tagged session metadata:

```sh
mstream recruit pr-322 \
  --role reviewer \
  --agent codex \
  --count 2 \
  --selector pool=amd \
  --task "Review PR 322 and summarize blockers."
```

Selection order:

1. Prefer sessions tagged `@mstream/state=available` and matching requested
   role/agent/selector constraints.
2. If creation is allowed and placement metadata is sufficient, create new
   sessions on connected hosts using the same behavior as `mstream new`.
3. Prefer lower observed busy-session count per host when choosing among
   otherwise equivalent hosts.
4. Refuse with an actionable JSONL error if hosts, labels, capacity, work root,
   or credentials are unknown.

`mstream` must not infer that an arbitrary untagged tmux session is available.
Availability is explicit.

### Observation Commands

Polling is the primary agent interface:

```sh
mstream status pr-322
mstream events pr-322 --after <cursor> --limit 200
mstream snapshot pr-322 --after <cursor> --max-chars 12000
mstream summary-input pr-322 --since 15m --max-chars 12000
```

`status` reports current daemon knowledge: connected hosts, participating
sessions, last output time, basic state, and stuck hints.

`events` returns bounded structured event records after a cursor.

`snapshot` returns bounded transcript text suitable for direct summarization.

`summary-input` applies server-side filtering and compaction: strip repeated
terminal chrome, collapse low-value progress noise, preserve human prompts,
tool errors, final answers, and obvious stuck states.

All machine-facing stdout is JSONL. Do not add a `--format` flag until there is
a concrete non-JSONL consumer.

Example output:

```jsonl
{"type":"ok","op":"join","workstream":"pr-322","target":"amd1::gpt55-mmux-reviewer","cursor":"ws/pr-322/000012"}
{"type":"status","workstream":"pr-322","agents":[{"target":"amd1::gpt55-mmux-reviewer","role":"reviewer","state":"busy","last_output_secs":18}]}
{"type":"event","cursor":"ws/pr-322/000013","workstream":"pr-322","target":"amd1::gpt55-mmux-reviewer","text":"Running cargo test -p motlie-tmux"}
```

### Timeline Model

The daemon maintains in-memory ring buffers per workstream while running. A
single connected host/session output bus can feed multiple workstream timelines
because session tags determine membership and subscriptions can filter by
host/session.

Before issue #322 lands, timeline ordering is best-effort arrival order with
source labels. After #322, `mstream` should delegate timestamp-aware merge-sort
and bounded buffering to the `libs/tmux` OutputBus-backed timeline API rather
than implementing its own reorder buffer.

The CLI cursor contract should remain independent from the internal timeline
implementation. Cursors are opaque strings.

## Alternatives Considered

### Alternative A: Local SQLite Or JSON State

This would make empty workstreams and host reconnect automatic, but it creates
another durable state surface for agents to corrupt, migrate, or forget. It
also conflicts with the desired operational model where tmux sessions remain
the source of truth. Rejected.

### Alternative B: Extend `bins/tmux/driver`

The existing driver is useful for raw tmux operations and debugging. Adding
workstream concepts there would mix low-level tmux verbs with higher-level
agent orchestration. Rejected in favor of a new `bins/mstream` surface.

### Alternative C: Build A TUI

A TUI may help humans later, but the first consumer is an orchestrating agent
that needs bounded stdout it can summarize. Rejected for the initial design.

### Alternative D: Add `libs/workstream`

A library may eventually be useful, but the domain model is not proven yet.
Keeping business logic in `bins/mstream` lets the CLI settle before extracting
an API. Deferred.

## Testing Strategy

Detailed test commands belong in [`PLAN.md`](./PLAN.md). The design requires
coverage for:

- CLI parsing and target parsing
- socket client/daemon request handling
- no-local-state restart behavior
- host connect and scan behavior using local tmux where possible
- tmux tag schema read/write/unset
- hydration from tagged sessions
- workstream join/new/leave semantics
- JSONL output shape
- bounded observation and cursor behavior
- arrival-order timeline behavior before issue #322
