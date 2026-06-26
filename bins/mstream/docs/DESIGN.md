# Design: mstream Agent Workstreams

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-26 | @codex-570-impl | Moved attach command construction behind libtmux `AttachMode`; mstream keeps only resolve RPC and visit-window lifecycle policy. |
| 2026-06-26 | @codex-570-impl | Switched local caller-tmux attach visits to direct `env -u TMUX` local attach plus failed-pane diagnostics before reap. |
| 2026-06-25 | @codex-570-impl | Added auto-sweep-on-attach to bound abandoned `--here` visits to at most one stale window. |
| 2026-06-25 | @codex-570-impl | Updated attach design after live dogfood: switch all caller-session clients explicitly and reap on visit-pane exit rather than session active-window polling. |
| 2026-06-25 | @codex-570-impl | Added issue #570 attach/forward-terminal design: daemon attach-command resolution plus client-owned PTY handoff or tagged caller-tmux window injection and reap. |
| 2026-06-07 | @codex-401-impl | Defined issue #409 audit durability contract: non-`agent_output` events are lossless, `agent_output` is best-effort with degraded counters, shutdown drains the writer, and phone scrubbing covers multiline/Unicode digit runs. |
| 2026-06-07 | @codex-401-impl | Reconciled issue #409 durable audit log semantics with the state/recovery requirements: redacted socket-adjacent event JSONL is replayed on startup, but host/session/timer/workflow state remains non-durable. |
| 2026-06-06 | @codex-401-impl | Added issue #401 lifecycle semantics: `quarantined` state, workstream-retained `retire`, gated `reclaim`, and scan-time registry reconciliation. |
| 2026-05-30 | @codex-360-og | Added issue #360 live session rename/retag design using stable tmux session ids, existing freshness checks, and mmux label lifecycle refresh. |
| 2026-05-28 | @gpt55-324-330-og | Added issue #349 mmux-visible workstream labels owned by mstream tags, with selected-key preservation and leave/close cleanup. |
| 2026-05-28 | @codex | Added issue #344 input-quiet timer delivery guard: timed prompts defer when attached-client input was recent, while read-only polling remains unaffected. |
| 2026-05-28 | @codex | Adopted issue #337 tmux Fleet abstractions for host registration, target specs/resolved targets, and batch session tags while keeping workstream business logic in mstream. |
| 2026-05-28 | @codex | Removed `session mark self`; coordinator state marks now require explicit targets. |
| 2026-05-28 | @codex | Added daemon-owned self-wakeup timers that send prompts to the orchestrator's tmux session. |
| 2026-05-23 | @codex | Addressed PR #324 handoff-loop feedback: handoff firing marks the destination busy, already-met handoffs fire immediately by default, and public cursors carry timeline generation for stale-cursor detection. |
| 2026-05-22 | @codex | Aligned timeline dependency language with PR #326's concrete OutputBus timeline APIs: create-or-get, mutable filters, scoped markers, history ingest, stale handles, cleanup, and latest-cursor semantics. |
| 2026-05-22 | @codex | Addressed PR #324 review: added Communication & Handoff, explicit completion/state ownership, OutputBus timeline dependency gates, cursor/time ownership, and timeline cleanup rules. |
| 2026-05-22 | @codex | Addressed issue #323 feedback: renamed workstream creation to `open`, defined `close` as conclusion that frees agents, and added domain/context tags plus `recruit --goal` matching. |
| 2026-05-21 | @codex | Initial design for issue #323: stateless `bins/mstream` daemon/client, tmux-tag hydration, host reconnect flow, workstream CLI, recruiting, and JSONL observation. |

## Status

This document begins the design for [issue #323](https://github.com/chungers/motlie/issues/323).
`mstream` is a greenfield product surface: it is a new binary and command
model over existing `motlie-tmux` capabilities, so there is no migration or
backwards-compatibility contract for existing users.

The related `libs/tmux` timeline dependency is
[issue #322](https://github.com/chungers/motlie/issues/322). Until #322 lands
and the follow-on API gaps are closed, `mstream` can expose useful arrival-order
merged output, but it must not claim exact agent-emission-time merge sorting.
The current `TargetOutput.timestamp` substrate is a daemon receipt timestamp,
not an agent-authored event time, so timestamp-aware timelines are useful for
stable multi-source ordering but not a proof of wall-clock agent emission order.

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
- how to send follow-up instructions, interrupts, and handoffs after agents are
  already running
- how to recover after a watcher process or daemon restart

This is fragile. It pushes every orchestrating agent toward custom local
state, custom buffering, and ad hoc tmux commands. `mstream` should make the
common operation explicit: connect known hosts, model workstreams, join or
create agent sessions, watch bounded output, and periodically summarize or
unstick collaborators.

## Non-Goals

- No TUI. The orchestrating agent needs JSONL stdout, not a rendered interface.
- No durable local database, state directory, or tiny host config file for host,
  session, timer, handoff, or workflow control-plane state. The socket-adjacent
  redacted audit JSONL is the only local durability exception and exists solely
  to replay call-log events after daemon restart.
- No hidden host discovery. Humans provide host aliases and SSH URIs.
- No replacement for `bins/tmux/driver`; that remains the low-level tmux
  operator/debug tool.
- No separate `libs/workstream` wrapper in the first slice.
- No automatic takeover of arbitrary tmux sessions without explicit join or
  managed-session tags.
- No exact agent-emission-time merge-sort in the first slice.
- No durable workflow engine in the first slice. Handoff edges may survive the
  orchestrator's context compaction while the daemon is running, but they do
  not survive daemon restart; replayed audit events are transcript evidence,
  not executable workflow state.
- No durable scheduler in the first slice. Timer state lives only in the
  running daemon and must be recreated after daemon restart.

## Requirements

### Functional Requirements

- FR1: Provide a new `bins/mstream` binary with daemon and client commands.
- FR2: Hide raw tmux, Fleet, and OutputBus mechanics behind a `workstream`
  abstraction.
- FR3: Start the daemon with zero hosts; hosts are connected by explicit
  client commands after startup.
- FR4: Open a workstream with zero or more associated tmux agent sessions.
- FR5: Join an existing tmux session to a workstream and optionally send a
  pivot/task prompt.
- FR6: Create a new tmux session on a connected host, start an agent binary,
  tag the session, and join it to a workstream.
- FR7: Leave a session from a workstream without killing it.
- FR8: Reclaim a session only through an explicit destructive command.
- FR9: Recruit agents from connected/scanned hosts, preferring tagged available
  sessions and goal/context matches before creating new sessions.
- FR10: Scan connected hosts and hydrate workstream/session state from tmux
  session tags.
- FR11: Observe workstreams through bounded polling commands suitable for an
  orchestrating agent.
- FR12: Emit machine-facing output as JSONL on stdout. Human diagnostics and
  debug logs go to stderr.
- FR13: Close a workstream as a conclusion step that marks participating agents
  available for other work and preserves useful domain/context metadata in tmux
  session tags.
- FR14: Accept a high-level `--goal` for workstream opening and recruiting so
  an orchestrating agent can match new work to agents with relevant prior
  context or specialty.
- FR15: Send ad-hoc follow-up messages to running agents after join/new.
- FR16: Interrupt a running agent non-destructively without killing the tmux
  session.
- FR17: Provide coordinator-owned state transitions based on observed agent
  output, pushed commits, PRs, review comments, tests, blockers, and questions.
- FR18: Support handoff and broadcast workflows so one agent's completion can
  trigger another agent's next prompt while the daemon remains alive.
- FR19: Support daemon-owned timers that periodically send a configured prompt
  to a tmux target, primarily the orchestrator's own agent session, so the
  orchestrator can wake itself to poll and unblock workstreams.
- FR20: Guard unattended timer delivery against recent attached-client input so
  a self-wakeup prompt does not corrupt text a human is typing in the target
  session.
- FR21: Support short mmux-visible workstream labels so mmux can group or
  display active sessions by the workstream assigned through mstream.
- FR22: Rename and retag live sessions without recreating them, while keeping
  daemon tracking keyed by stable tmux session id.
- FR23: Support `mstream attach <target> [--here] [--sweep] [--print]`: the
  daemon resolves the connected tmux target to attach argv, while the client
  owns PTY handoff or caller-tmux window injection and reaping.

### State And Recovery Requirements

- SR1: The daemon is an in-memory control-plane process. It may cache host
  connections, sessions, cursors, and timeline buffers while running.
- SR2: The daemon must not persist host aliases, SSH URIs, reusable session
  ledgers, timers, handoffs, or workflow control-plane state to a local file or
  database. The sole local durability exception is a bounded, redacted
  socket-adjacent event audit JSONL (`<socket>.events.jsonl`) for call-log
  replay.
- SR3: After daemon restart, an orchestrating agent asks the human for the host
  aliases and SSH URIs, reconnects those hosts, and rescans.
- SR4: Durable session/workstream metadata lives in tmux session user options
  under the `mstream` namespace.
- SR5: Tmux history remains the durable output snapshot source. OutputBus
  buffers are live/volatile. `mstream` records redacted audit events to the
  socket-adjacent JSONL so readable events can replay the call log after daemon
  restart: all non-`agent_output` control/report/handoff/timer events are
  lossless-enqueued, while high-volume `agent_output` is best-effort and
  exposes dropped/degraded counters.
- SR6: Empty operational workstreams are ephemeral when they have no tagged
  session metadata and no retained audit events. Audit replay may recreate a
  timeline-only workstream for transcript reads, but it must not restore
  recruitable sessions, timers, handoffs, or other executable state.
- SR7: Session state tags are hints owned by `mstream` commands and managed
  agents. Silence or lack of output must never be persisted as completion.
- SR8: Timer state is daemon-memory only. After daemon restart, the
  orchestrator must recreate any needed timers.

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
- NFR7: Coordinator-to-agent communication must use typed `libs/tmux`
  `send_text`/`send_keys`/`KeySequence` APIs rather than shell-built tmux
  commands.
- NFR8: Status output may include stuck hints, prompt heuristics, or process
  state, but those hints must be labeled separately from explicit agent states
  such as `done`, `blocked`, or `needs-input`.
- NFR9: Timer delivery must use typed `libs/tmux` targeting and send-key
  primitives. The target must be explicit; `mstream` must not infer or mutate
  the orchestrator session identity.
- NFR10: Input guards apply only to key/text delivery. Read-only polling
  commands such as `status`, `events`, `snapshot`, `summary-input`,
  `timer list`, `hosts`, `scan`, `list`, and `show` must continue to run
  without waiting for a quiet input window.

## Selected Design

### Public Abstraction

The public abstraction is a workstream. A workstream is a named unit of
collaborative agent work, such as `pr-322`, that can have zero or more tmux
sessions attached while the daemon is running.

Users and orchestrating agents should think in these terms:

- connect a host
- scan known sessions
- open a workstream
- join or create agents for the workstream
- recruit available agents
- close the workstream when the work is concluded
- poll status, events, snapshots, or summary input

The public CLI should not expose Fleet or OutputBus directly. Internally,
`mstream` uses Fleet as the connected-host registry and uses
`FleetTargetSpec` / `ResolvedFleetTarget` for cross-host tmux target addresses.
Full delegation to `libs/tmux` OutputBus timeline APIs remains a follow-up;
the current implementation records sanitized command events and subscribed
OutputBus agent output into bounded per-workstream rings, then mirrors those
events to the socket-adjacent audit JSONL through a bounded writer outside the
daemon state lock.

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
@mstream/workstream-state=open
@mstream/workstream-goal=Add OutputBus-backed timelines
@mstream/workstream-domain=tmux
@mstream/role=reviewer
@mstream/agent=codex
@mstream/identity=ops47-mmux-reviewer
@mstream/state=available|reserved|busy|idle|done|blocked|needs-input|quarantined
@mstream/cwd=/abs/path
@mstream/context-domains=tmux,vmm
@mstream/context-specialties=output-bus,timeline-review
@mstream/context-summary=Reviewed OutputBus timeline design and tmux tag hydration.
@mstream/last-report-kind=done
@mstream/last-report-summary=PR feedback addressed; waiting for reviewer.
@mstream/last-workstream=pr-322
@mstream/last-workstream-title=OutputBus timelines
@mstream/mmux-label=PR 322
@mstream/mmux-selected-key=mstream
@mstream/mmux-previous-selected-key=owner
@mstream/updated-at=2026-05-21T12:34:56Z
```

The tag values must stay small. Larger state belongs in tmux history, in the
repo, or in the human/orchestrator conversation. Tags are for hydration and
selection, not transcripts.

There are two classes of tags:

- active assignment tags such as `workstream`, `workstream-title`,
  `workstream-state`, `workstream-goal`, `workstream-domain`, and `role`
- reusable agent context tags such as `context-domains`,
  `context-specialties`, `context-summary`, and `last-workstream`

The active assignment tags describe the work currently occupying the agent.
The reusable context tags describe the agent's accumulated local context and
specialty so a future `recruit --goal` can find agents whose tmux session,
working tree, scrollback, and model context are likely useful for related work.

State ownership:

- `available`: set by `close`, `leave --available`, or explicit coordinator
  marking when the agent can accept new work.
- `reserved`: set by `recruit` before a task is sent to prevent double
  assignment.
- `busy`: set by `join`, `new`, `recruit`, `send --set-state busy`, or an
  explicit coordinator/agent mark while work is active.
- `idle`: reported as a status hint or set by explicit mark; it means
  quiet/ready, not complete.
- `done`, `blocked`, `needs-input`: set by an explicit coordinator mark after
  the orchestrator observes durable evidence such as pushed commits, PRs,
  review comments, test output, blockers, or direct questions. These are never
  inferred from output silence alone.
- `quarantined`: set by `retire` when an agent must not be recruited for new
  work but should stay alive in its current workstream for audit-logged cleanup.
  `reclaim` is gated on this state and on `managed=true`.

`last-report-*` tags are intentionally small. Detailed reports belong in the
agent pane transcript or PR/issue comments.

When a workstream has an mmux label, `mstream` also writes `@mmux/mstream=<label>`
and sets `@mmux/__selected-key=mstream` on joined, new, or recruited sessions.
Before taking over `@mmux/__selected-key`, it stores the previous selected key
in `@mstream/mmux-previous-selected-key` unless the selected key is already
`mstream`. On `leave` or `close`, `mstream` clears `@mmux/mstream`; if the
selected key is still `mstream`, it restores the previous key or unsets the
selected key when no previous key existed. If another tool changed the selected
key after mstream applied the label, cleanup leaves that selection unchanged.
Selected-key takeover and restore are best-effort against out-of-band tmux
edits because the daemon does not hold its internal lock across tmux or SSH I/O.

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

### Attach / Forward Terminal

`mstream attach <target> [--here] [--sweep] [--print]` is a thin CLI and
lifecycle layer over existing `motlie-tmux` attach transport. The daemon owns
only target resolution: it resolves the connected `FleetTargetSpec` to a fresh
session `Target`, verifies the session has not been reused, and returns the
`AttachCommand` argv. The daemon never attempts to attach because it has no
foreground PTY.

The client chooses the realization:

- `--print` writes the shell-safe attach command and performs no attach side
  effect; cleanup only runs when `--sweep` is also present.
- Outside tmux, the default is PTY handoff: the client requests
  `motlie_tmux::AttachMode::PtyHandoff`, runs the returned opaque attach argv
  with the caller terminal inherited, and returns the attach child shell status.
  Local targets keep the bare local tmux attach command in this path.
- Inside tmux (`$TMUX` set, or explicit `--here`), the client asks the daemon for
  a `motlie_tmux::AttachMode::WindowInjection` command. Libtmux owns the command
  matrix: local targets become direct local attach with `TMUX` removed
  (`env -u TMUX tmux attach ...`) while remote targets keep SSH attach. Both
  forms preserve the same nested-client return behavior, including `Ctrl-b 0`
  back to the caller session. The client auto-sweeps inactive
  `@mstream/attach` windows in the caller tmux server, creates a detached
  caller-tmux window running the resolved attach command, tags it with
  `@mstream/attach` plus target/spawn metadata, switches every attached client
  of the caller session to that window with `switch-client -c <tty>`, then waits
  for the injected visit pane to exit or disappear before cleanup.

Injected windows are ephemeral mstream-owned visit windows. Cleanup is keyed to
two triggers: successful inner detach exits the visit pane and self-kills the
current window; failed attach exits preserve the dead pane long enough for
`mstream` to capture exit status/output, report a target-specific error, and
then reap the window; if a user walks away while the nested attach stays alive,
the next `attach --here` auto-sweep removes that inactive stale window before
creating another one. Because every `--here` attach sweeps before create, at
most one stale attach window can exist at a time. Standalone `--sweep` remains
the explicit cleanup-now command and preserves any active visit window.

### Workstream Commands

Open an in-memory workstream:

```sh
mstream open pr-322 \
  --title "OutputBus timelines" \
  --goal "Add OutputBus-backed timelines for multi-agent monitoring" \
  --domain tmux \
  --mmux-label "PR 322"
mstream label pr-322 --mmux-label "PR 322"
mstream list
mstream show pr-322
mstream close pr-322
```

`open` replaces `create` as the workstream verb. Opening a workstream records
the daemon-memory workstream handle and any provided title/goal/domain metadata.
Because `mstream` has no durable local store, an open workstream becomes
durable only when at least one joined session receives `@mstream/workstream=*`
tags.

`--mmux-label` is optional. It must be one or two whitespace-separated words,
must not contain control or Unicode format characters, and must fit within 24
display columns. Violations are rejected. `label <workstream> --mmux-label` can
update the label for an open workstream and apply it to currently joined
sessions. Scan rehydrates the label from `@mstream/mmux-label`; if participating
sessions disagree, status/show/list expose label conflicts for the orchestrator
to resolve without choosing an order-dependent winner.

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

Close, leave, retire, or reclaim:

```sh
mstream close pr-322 \
  --summary "OutputBus timeline design is ready for implementation." \
  --domain tmux \
  --specialty output-bus \
  --specialty timeline-review
mstream leave pr-322 amd1::gpt55-mmux-reviewer
mstream retire pr-322 amd1::gpt55-mmux-reviewer
mstream reclaim amd1::gpt55-mmux-reviewer
```

`close` concludes a workstream. For each participating session, it marks the
agent available for new work, clears active workstream membership, records the
closed workstream as `last-workstream`, and merges any provided domain,
specialty, and summary text into reusable context tags. Closing does not kill
sessions.

`retire` is different from `leave`: it preserves current workstream membership,
sets the agent state to `quarantined`, and excludes it from recruitment while
allowing cleanup messages in the audit log. `reclaim` is the terminal teardown
step: it kills and deregisters only a live tmux target whose mstream tags prove
`managed=true` and `state=quarantined`.

`mstream` treats the workstream name as an opaque handle. The handle may include
human naming conventions such as `issue-337` or `pr-330`, but the daemon does
not model GitHub issues, PRs, merge state, or closeout-comment publishing.
Closeout publication is orchestrator policy. `mstream` should provide only
neutral primitives: close/free sessions, stop scoped timers, send standby
messages, and expose bounded timeline/transcript output through `events`,
`snapshot`, and `summary-input`.

`leave` unsets workstream-specific tags but leaves the session running.
`reclaim` is explicit and destructive.

### Communication And Handoff

`mstream` is a coordination tool, not only an observation tool. It must provide
coordinator-to-agent communication after the initial `join`/`new` prompt, plus
coordinator-owned state annotations after observing agent output.

#### Send

Ad-hoc messages use:

```sh
mstream send pr-322 amd1::gpt55-mmux-reviewer \
  --text "Please re-run the focused timeline tests." \
  --enter

mstream send pr-322 amd1::gpt55-mmux-reviewer \
  --text "$(cat /tmp/followup.txt)" \
  --paste-mode bracketed \
  --no-enter
```

`send` targets one joined session. It writes a structured `message_sent` event
and uses `libs/tmux` typed send APIs. Multi-line text defaults to bracketed
paste mode so terminal programs that support bracketed paste receive one paste
operation instead of many typed lines. Single-line text may use literal
`send_text`. `--paste-mode literal` is allowed for agents or shells where
bracketed paste is known to be wrong.

`--enter` appends the submit key after the text. `--no-enter` only places the
text in the pane. The command must make this choice explicit; the CLI help may
default to `--enter` for `--text` and `--no-enter` for `--stdin`, but JSONL
must always report the effective value.

If the target is currently `busy`, `send` still sends the text unless
`--require-state <state>` is provided. The JSONL response must include
`target_state` and a `mid_generation_risk` boolean so the orchestrator can
decide whether the message was merely typed into an active TUI or should be
resent with interruption.

`send` does not change `@mstream/state` by default. `--set-state busy` may be
used when the message assigns new work, including handoff-generated messages.

#### Interrupt

Non-destructive interruption is separate from `reclaim`:

```sh
mstream interrupt amd1::gpt55-mmux-reviewer
mstream interrupt amd1::gpt55-mmux-reviewer --key ctrl-c
mstream send pr-322 amd1::gpt55-mmux-reviewer \
  --interrupt-first \
  --settle-ms 500 \
  --text "Stop the current run and inspect PR feedback." \
  --enter
```

The default interrupt key is `Esc` because most agent TUIs use it as a soft
stop. `--key ctrl-c` is available for shell commands or agents that require a
process interrupt. `interrupt` never kills the tmux session.

`send --interrupt-first` is an atomic daemon-side sequence: send the interrupt
key, wait the settle delay, send text, optionally send Enter, then emit one
JSONL result. The timing belongs in `mstream` so the orchestrating agent does
not race a TUI by issuing separate one-shot commands.

#### Broadcast

Round-level coordination uses:

```sh
mstream broadcast pr-322 \
  --text "Wrap up your current step and summarize status." \
  --enter
```

`broadcast` is `send` applied to every joined session that matches optional
filters such as `--role reviewer` or `--state busy`. The response emits one
record per target and a final summary record.

#### Completion And Reports

Project orchestration treats mstream as an orchestrator-only control plane.
Managed agents do not have mstream access and must not be asked to call
`mstream` themselves. They should report progress, blockers, questions, PR
links, pushed commits, and review comments plainly in their normal output.
`mstream new` still passes non-socket environment context so the session can
identify its workstream and role:

```text
MSTREAM_WORKSTREAM=pr-322
MSTREAM_ROLE=reviewer
```

After observing durable evidence, the orchestrator can annotate the explicit
target:

```sh
mstream session mark amd1::gpt55-mmux-reviewer --state done --summary "Posted review comments on PR #340."
mstream session mark amd1::gpt55-mmux-reviewer --state blocked --summary "Cannot post review because GitHub auth failed."
mstream session mark amd1::gpt55-mmux-reviewer --state needs-input --summary "Needs API naming decision from the user."
```

`session mark` requires an explicit `<host>::<session>` target. A successful
mark updates `@mstream/state`, `@mstream/last-report-kind`,
`@mstream/last-report-summary`, and `@mstream/updated-at`, and emits a
structured event:

```jsonl
{"type":"event","kind":"completed","workstream":"pr-322","target":"amd1::gpt55-mmux-reviewer","state":"done","summary":"Implemented requested fixes."}
```

#### Rename And Retag

Session rename and retag commands are coordinator-owned live metadata updates:

```sh
mstream rename amd1::gpt55-mmux-reviewer gpt55-360-reviewer
mstream rename amd1::gpt55-mmux-reviewer gpt55-360-reviewer --role reviewer --workstream issue-360 --mmux-label "360 review"
mstream session retag amd1::$7 --role implementer --workstream issue-360 --mmux-label "360 impl"
```

The daemon resolves friendly names or `host::$id` inputs with the existing
target resolver, freshness-checks the resolved live target, and calls
`Target::rename(new_name)` so tmux is renamed by stable `session_id`. After
tmux succeeds, the daemon updates the existing `SessionRecord` display metadata
and durable `@mstream/identity` tag in place. It must not remove and reinsert a
session record just because the display name changed.

Retagging updates durable role/workstream/mmux metadata without recreating the
session. Moving to another workstream requires an open destination workstream
and an effective role. The stable target is removed from the old workstream set
and inserted into the destination set; timers and handoff endpoints remain
id-keyed. Handoffs owned by the old workstream that reference the moved target
are canceled so they cannot later fire against a session that left that
workstream. Existing mmux label ownership helpers clear the old label and apply
the destination/current label.

All project workflow marks are made by the orchestrator after observing
evidence. A future JSONL extension may add `source:"coordinator"` for
auditability.

Output silence, prompt-looking text, or unchanged tmux history can create
`idle` or `stuck_hint` status fields, but cannot transition an agent to
`done`, `blocked`, or `needs-input`.

#### Handoff

The coordinator owns the dependency graph. The minimum reliable pattern is:

```sh
mstream status pr-322
mstream events pr-322 --after <cursor> --limit 100
mstream send pr-322 amd2::ops47-reviewer \
  --text "A marked done. Re-review PR 322 now." \
  --enter
```

For context compaction resilience while the daemon remains alive, `mstream`
also provides a thin armed handoff primitive:

```sh
mstream handoff arm pr-322 \
  --from amd1::gpt55-submitter \
  --to amd2::ops47-reviewer \
  --on done \
  --task "The submitter marked done. Re-review PR 322 and post verdict."

mstream handoff arm pr-322 \
  --from amd1::gpt55-submitter \
  --to amd2::ops47-reviewer \
  --on done \
  --only-on-transition \
  --task "Re-review only after the submitter next reports done."

mstream handoff list pr-322
mstream handoff cancel pr-322 <handoff-id>
```

An armed handoff is daemon-memory state. When the `from` target reports the
matching terminal state, the daemon atomically marks the `to` target `busy`,
updates `@mstream/updated-at`, sends the task to the `to` target, emits a
`handoff_fired` event, and marks the handoff fired. The destination state update
is part of firing so a target that was previously `done`, `blocked`, or
`needs-input` is not misclassified while it works on the new task.

If the `from` target is already in the requested state when `handoff arm` runs,
the handoff fires immediately by default. This makes the common coordinator
sequence race-free: poll status, observe `A=done`, then arm `A -> B`. Callers
that need edge-triggered behavior can pass `--only-on-transition`; in that mode
the handoff waits for a future state transition even if the current state already
matches. Armed handoffs do not survive daemon restart, but they let the
orchestrator set a dependency edge and safely compact or poll less often.

### Self-Wakeup Timers

The agent harness does not provide a first-class cron primitive, so `mstream`
provides a small daemon-owned timer surface. The timer does not decide project
state. It only sends a prompt to an explicit tmux target on an interval, using
the same typed `libs/tmux` targeting and send-key path as other coordinator
messages.

The main use case is an orchestrator waking its own agent session:

```sh
mstream timer start issue-337-poll \
  --every 5m \
  --target local::codex-orchestrator \
  --prompt "[mstream:issue-337-poll] Wakeup: check issue-337-tmux-fleet-api with mstream status and summary-input. Unblock agents, summarize material changes, and decide whether to keep, change, or stop this timer." \
  --paste-mode bracketed \
  --settle-ms 500 \
  --verify-delivery \
  --submit-retries 1 \
  --submit-retry-delay-ms 750

mstream timer list
mstream timer fire issue-337-poll
mstream timer stop issue-337-poll
```

Timer names are daemon-unique and should describe their purpose. `--every`
accepts seconds or minutes. `--target` uses `<host-alias>::<session>` and must
resolve when the timer starts. `--prompt` is not persisted outside daemon
memory. The default behavior submits the prompt with Enter; `--no-enter` leaves
the text in the pane and disables submit retries. `--paste-mode
bracketed|literal`, `--settle-ms`, and `--verify-delivery` match the
send/broadcast delivery primitive surface.

Timer prompts default to one extra Enter after 750ms because agent TUIs
occasionally miss the first submit key after pasted text. The retry policy is
configurable with `--settle-ms`, `--submit-retries`, and
`--submit-retry-delay-ms`. Retries send only extra Enter keys, never the prompt
text, so the duplicate-submission risk is limited to the submit action.

Timers, send, and broadcast default to an attached-client input guard. Before
sending prompt text or submit retries, `mstream` asks `libs/tmux` for the target
session's most recent attached-client activity. If input was observed within
`--input-quiet-for` (default `10s`), the managed channel does not send keys.
Timers additionally increment `defer_count`, record `last_deferred_at`,
`last_defer_reason=recent_client_input`, record the latest input timestamp, and
reschedule the next attempt after the remaining quiet window. `--no-input-guard`
disables this behavior for cases where collision avoidance is not wanted.

Timers are intentionally best-effort. If the target is gone, the host is
disconnected, or the activity query/send-keys path fails, `mstream` records
`last_error` and tries again on the next tick until the timer is stopped.
`timer fire` is an immediate smoke-test trigger and does not replace the next
scheduled tick. It follows the same input-quiet guard.

Timers do not survive daemon restart. They are not a substitute for status
polling, workstream events, review-loop judgment, or future alerting. They are
the smallest wakeup primitive needed so an orchestrator can periodically ask
itself to poll and unblock active workstreams.

### Recruiting

Recruiting uses only connected/scanned hosts and tagged session metadata:

```sh
mstream recruit pr-322 \
  --role reviewer \
  --agent codex \
  --count 2 \
  --goal "Clean up vmm examples" \
  --selector pool=amd \
  --task "Review PR 322 and summarize blockers."
```

Selection order:

1. Prefer sessions tagged `@mstream/state=available`.
2. Within available sessions, prefer role/agent/selector matches and sessions
   whose `context-domains`, `context-specialties`, `context-summary`, and
   recent `last-workstream` tags match the requested `--goal`.
3. If multiple candidates remain, return or choose candidates with enough
   context metadata for the orchestrating agent to explain the selection.
4. If creation is allowed and placement metadata is sufficient, create new
   sessions on connected hosts using the same behavior as `mstream new`.
5. Prefer lower observed busy-session count per host when choosing among
   otherwise equivalent hosts.
6. Refuse with an actionable JSONL error if hosts, labels, capacity, work root,
   or credentials are unknown.

`mstream` is not required to be an LLM. The first implementation may use
structured/lexical scoring over the session context tags and include candidate
metadata in JSONL so the orchestrating agent can make the semantic judgment.
A future model-backed scorer can improve `--goal` matching without changing
the CLI contract.

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

Read-only observation commands are not gated by timer input quieting. The guard
exists only to prevent unattended key/text injection into a session with recent
attached-client input; it must not slow down workstream polling or stuck-agent
detection.

`status` reports current daemon knowledge plus a live tmux activity refresh:
connected hosts, participating sessions, explicit `@mstream/state`, last output
time, last report summary, and stuck hints. The refresh uses connected
`HostHandle::list_sessions()` calls and the tmux library's session activity
semantics, where `SessionInfo.activity` is the maximum of tmux
`session_activity` and per-window `window_activity`. This lets orchestrators
poll liveness through `mstream` instead of bypassing it with direct SSH/tmux
commands.

Status activity hints are advisory and distinct from explicit agent states:

- `active`: latest tmux activity is within `--active-window-secs`
- `quiet`: activity is older than the active window but newer than `--idle-after-secs`
- `idle`: activity is at least `--idle-after-secs` old
- `missing`: the target session is absent from the connected host snapshot
- `unknown`: the host activity refresh failed or timestamps cannot be compared

Stuck hints must be based on observable facts such as output silence, pane
process state, prompt heuristics, or monitor health, and must not be conflated
with explicit completion states.

`events` returns bounded structured event records after a cursor. The retained
ring is mirrored to a socket-adjacent redacted audit JSONL so readable events
and closeout transcripts survive daemon restart; malformed or partial replay
lines are skipped with diagnostics rather than aborting startup. The `status`
and `events` APIs include an `audit` object so orchestrators can detect degraded
persistence, best-effort `agent_output` drops, lossless enqueue failures, and
writer persistence failures.

`snapshot` returns bounded transcript text suitable for direct summarization.

`summary-input` applies server-side filtering and compaction: strip repeated
terminal chrome, collapse low-value progress noise, preserve human prompts,
tool errors, final answers, and obvious stuck states.

These observation commands are the closeout transcript boundary. They can feed
an orchestrator-authored GitHub issue or PR comment, but `mstream` should not
take issue/PR identifiers or post to external systems itself.

`last_output_secs`, `--since`, and `@mstream/updated-at` are owned by
`mstream`. The daemon should derive wall-clock fields from its own clock when
it ingests output or writes tags. Internal `libs/tmux` timeline cursors may use
`Instant`, but public `mstream` cursors are opaque serialized strings that do
not expose `Instant` or require `serde` support from `libs/tmux`.

Public cursors must embed the workstream timeline generation or epoch alongside
the internal timeline cursor. `close` followed by re-`open` of the same
workstream name creates a fresh timeline generation; using a cursor from the old
generation against the new timeline must return a structured JSONL error such as
`{"type":"error","kind":"cursor_stale",...}`. The orchestrator can then
re-baseline with `latest` instead of receiving plausible but wrong replay/skip
behavior.

All machine-facing stdout is JSONL. Do not add a `--format` flag until there is
a concrete non-JSONL consumer.

Example output:

```jsonl
{"type":"ok","op":"join","workstream":"pr-322","target":"amd1::gpt55-mmux-reviewer","cursor":"ws/pr-322/000012"}
{"type":"status","workstream":"pr-322","agents":[{"target":"amd1::gpt55-mmux-reviewer","role":"reviewer","state":"busy","tmux_present":true,"last_output_secs":18,"activity_hint":"active"}]}
{"type":"event","cursor":"ws/pr-322/000013","workstream":"pr-322","target":"amd1::gpt55-mmux-reviewer","text":"Running cargo test -p motlie-tmux"}
```

### Timeline Model

The daemon maintains in-memory ring buffers per workstream while running and
replays retained audit JSONL records on startup. A single connected
host/session output bus can feed multiple workstream timelines because session
tags determine membership and subscriptions can filter by host/session. Durable
audit writes are queued to a bounded writer so high-frequency agent output does
not perform filesystem I/O while the daemon state mutex is held; compaction is
batched by writer-owned count/size thresholds. The writer has separate
lossless and best-effort queues: control-plane, to-agent, explicit from-agent
reports, handoff, timer, and system events are lossless-enqueued, while raw
`agent_output` may be dropped under backpressure and increments observable
degraded counters. Foreground daemon shutdown aborts output/timer producers and
drains the writer before returning.

Before issue #322 lands, timeline ordering is best-effort arrival order with
source labels. After #322, `mstream` should delegate timestamp-aware merge-sort
and bounded buffering to the `libs/tmux` OutputBus-backed timeline API rather
than implementing its own reorder buffer.

Delegation to `libs/tmux` is gated on the OutputBus timeline APIs needed by
workstreams:

- idempotent timeline hydration through `open_timeline`
- mutable `TimelineHandle::set_filters` / `add_filter` when a workstream gains
  or loses sessions mid-round
- scoped continuity/gap markers such as `publish_discontinuity_for` and
  `publish_gap_for`, so unrelated sessions do not pollute a workstream
  timeline
- `TimelineHandle::ingest_historical` for post-restart backfill from tmux
  history before live output resumes
- stale-handle errors plus explicit `detach`, `remove_timeline`, or idle
  cleanup so closed workstreams do not keep collecting output
- cursor behavior that cannot skip or replay retained entries when timestamp
  ordering, bounded reads, `render_after`, and bounded `latest` are combined
- wall-clock receipt/ingest metadata sufficient for `mstream` to derive JSONL
  fields while keeping public cursors opaque

Until those are present, `mstream` should keep a local daemon-memory timeline
layer over the `OutputBus` subscription stream.

The CLI cursor contract should remain independent from the internal timeline
implementation. Cursors are opaque strings.

Timeline lifecycle:

- `open` performs get-or-create for the daemon-memory workstream timeline.
- `join`/`new` adds the target to the workstream timeline filter set without
  dropping retained output.
- `scan` after restart creates or gets the timeline, backfills from bounded tmux
  history, then resumes live monitoring.
- `leave` removes the target from the filter set and detaches or removes the
  timeline when the last session leaves an otherwise closed/empty workstream.
- `close` removes or detaches the active workstream timeline after final
  status/summary output is emitted.
- re-`open` after close creates a fresh timeline and must not reuse stale
  cursors.

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
- no-local-control-plane-state restart behavior, including the explicit
  durable audit-log exception
- host connect and scan behavior using local tmux where possible
- tmux tag schema read/write/unset
- hydration from tagged sessions
- workstream join/new/leave semantics
- JSONL output shape
- bounded observation and cursor behavior
- arrival-order timeline behavior before issue #322
- durable audit replay, redaction, lossless-vs-best-effort queueing outside
  the state lock, degraded counters, shutdown drain, malformed line tolerance,
  and bounded JSONL compaction
- coordinator-to-agent `send`, `interrupt`, and `broadcast`
- coordinator-owned `session mark <target>` state transitions
- handoff arming, firing, cancellation, and daemon-restart loss behavior
- status hints that distinguish explicit completion from idle/stuck heuristics
