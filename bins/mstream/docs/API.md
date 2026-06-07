# API: mstream CLI

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-07 | @codex-401-impl | Clarified issue #409 audit durability: non-`agent_output` events are lossless-enqueued, high-volume `agent_output` is best-effort with observable degraded counters, shutdown drains the writer, and phone scrubbing covers multiline/Unicode digit runs. |
| 2026-06-06 | @codex-401-impl | Added issue #409 durable audit call-log events: redacted `to_agent`/`from_agent` text, OutputBus agent output capture, socket-adjacent JSONL replay, and readable transcripts that survive daemon restart. |
| 2026-06-06 | @codex-401-impl | Added issue #410 `new`/`recruit --agent-arg` passthrough for agent argv flags such as Claude `--permission-mode auto`. |
| 2026-06-06 | @codex-401-impl | Added issue #401 session lifecycle commands: `retire` to mark live agents `quarantined` and `reclaim` for gated teardown. |
| 2026-05-30 | @codex-360-og | Added issue #360 live session rename/retag commands with id-stable tmux rename, role/workstream retagging, and mmux label refresh. |
| 2026-06-06 | @codex-386-impl | Documented `mstream new` agent executable preflight on the remote non-login PATH and absolute-path guidance. |
| 2026-05-30 | @codex-355-rv | Switched daemon session bookkeeping to stable `(host, session_id)` targets while keeping tmux session names as display metadata. |
| 2026-05-28 | @gpt55-324-330-og | Added issue #349 mmux-visible workstream labels with `open --mmux-label`, `label --mmux-label`, status/show/list fields, and close/leave cleanup. |
| 2026-05-28 | @gpt55-324-330-og | Added issue #347 closeout ergonomics: `timer --self`, workstream-scoped timers, readable events, and close-time timer/standby flags. |
| 2026-05-28 | @codex | Added timer input-quiet guards so scheduled prompt delivery defers instead of colliding with recent attached-client input. |
| 2026-05-28 | @codex | Clarified release-binary installation and removed the obsolete `session mark self` workflow. |
| 2026-05-28 | @codex | Added daemon-owned `timer` commands for orchestrator self-wakeup prompts delivered to a tmux target. |
| 2026-05-27 | @codex | Added live tmux activity refresh to `status` so orchestrators can detect active, quiet, idle, missing, or unknown sessions without direct SSH/tmux probes. |
| 2026-05-26 | @codex | Added per-workstream `--event-limit` setting to replace the fixed event ring size while keeping 1000 as the default. |
| 2026-05-24 | @codex | Addressed PR #330 re-review: request execution now snapshots state under short locks, performs SSH/tmux awaits outside the state mutex, then re-locks briefly to reconcile events and metadata. |
| 2026-05-24 | @codex | Addressed PR #330 feedback: bounded event cursors advance only to the last returned event, handoffs trigger from all explicit state-change paths, recruited sessions persist workstream tags, daemon connections are spawned per client, scan hydrates `cwd`, and broadcast touches `updated-at`. |
| 2026-05-23 | @codex | Documented the first implemented `mstream` CLI/daemon surface, JSONL protocol, and current observation limits. |

## Status

`mstream` is implemented as a `motlie-mstream` package with binary name
`mstream`. It provides a Unix-domain socket daemon, JSONL client responses,
strict `<host>::<session-or-id>` target parsing, in-memory workstreams, durable
audit events, tmux session tags, state marking, live rename/retag,
send/interrupt/broadcast, handoffs, and bounded observation commands. It also
provides daemon-memory self-wakeup timers that send a configured prompt to an
orchestrator tmux session on an interval, with default input-quiet guarding to
avoid colliding with attached human typing.

Workstream event history is served from an in-memory per-workstream ring and is
also appended to a socket-adjacent JSONL audit log named
`<socket>.events.jsonl`. Daemon startup replays retained audit records so
`events --readable` survives daemon restart. Non-`agent_output` audit events are
lossless-enqueued to the writer; high-volume `agent_output` entries are
best-effort and report dropped/degraded counters through `status` and `events`
`audit` fields. `snapshot` and `summary-input` use bounded one-shot tmux capture
for joined sessions; pane/process-state stuck hints remain implementation
follow-ups.

Install the release binary from the Motlie checkout before using this API:

```sh
cargo install --path bins/mstream --locked
```

Make sure Cargo's bin directory, usually `~/.cargo/bin`, is on `PATH` so
`mstream` resolves to the installed binary. The examples below assume the
release binary is available on `PATH`.

## Daemon

```sh
mstream --socket /tmp/mstream.sock daemon start
mstream --socket /tmp/mstream.sock daemon status
mstream --socket /tmp/mstream.sock daemon stop
```

`daemon start` daemonizes by default. Use `--foreground` for tests and manual
debugging:

```sh
mstream --socket /tmp/mstream.sock daemon start --foreground
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

mstream open pr-324 --title "mstream implementation" --goal "Implement PR 324" --mmux-label "PR 324" --event-limit 1000
mstream label pr-324 --mmux-label "PR 324"
mstream list
mstream show pr-324
mstream close pr-324 --summary "done" --domain tmux --specialty mstream
mstream close pr-324 --summary "done" --stop-timers --standby-agents
```

Host metadata is daemon memory only. Workstream `settings.event_limit` controls
the in-memory event ring and `events` API retention, and defaults to 1000 when
omitted. Re-opening an existing workstream can raise or lower this limit;
lowering it trims old events immediately. The daemon also compacts the durable
audit JSONL to retained events, with a global safety cap, so call-log output
cannot grow without bound. On daemon stop, timer/output audit tasks are stopped
and the audit writer is explicitly drained before the foreground daemon returns.
`--mmux-label` stores a short label that `join`, `new`, and `recruit` apply to
participating sessions as `@mmux/mstream`, and sets
`@mmux/__selected-key=mstream` so mmux can group/display the workstream label.
Labels are enforced as one or two whitespace-separated words, with no control
or Unicode format characters, and no more than 24 display columns.
`label <workstream> --mmux-label <label>` changes the label for an open
workstream and applies it to currently joined sessions. `list`, `show`, and
`status` include `mmux_label` and `mmux_label_conflicts`.

After daemon restart, reconnect hosts and run `scan` to hydrate tagged sessions
from tmux. Scan reads `@mstream/mmux-label` from joined sessions to recover the
workstream label. If joined sessions disagree, `mmux_label_conflicts` reports
the observed labels.

Daemon records are keyed by `(host, tmux session_id)`, for example
`local::$7`; tmux session names are display metadata. Commands may still accept
`<host>::<tmux-session-name>` for convenience, but JSONL responses and events
use the resolved stable target when the session exists. `scan` reconciles by
session id, updates display names after rename, and drops a stale in-memory
record if tmux reuses an id with a different session creation timestamp.

## Session Assignment

```sh
mstream join pr-324 local::codex-reviewer --role reviewer --task "Review this branch."

mstream new pr-324 local::codex-worker \
  --role implementer \
  --cwd /tmp/mstream-worker \
  --agent codex \
  --task "Implement the next phase."

mstream new pr-324 local::claude-reviewer \
  --role reviewer \
  --cwd /tmp/mstream-reviewer \
  --agent /opt/homebrew/bin/claude \
  --agent-arg --permission-mode \
  --agent-arg auto \
  --task "Review this branch."

mstream leave pr-324 local::codex-worker --available
mstream retire pr-324 local::codex-worker
mstream reclaim local::codex-worker

mstream rename local::codex-worker codex-reviewer
mstream rename local::codex-worker codex-reviewer --role reviewer --workstream pr-324 --mmux-label "PR 324"
mstream session retag local::$7 --role reviewer --workstream pr-324 --mmux-label "PR 324"
```

`new` validates absolute `--cwd` and validates `--agent` before session
creation with a non-login `command -v` on the target host. Remote agents must
be visible on that non-login PATH; pass an absolute executable path when login
shell setup is required. `--agent-arg <ARG>` is repeatable and forwards each
value as a separate agent argv entry, so flags such as Claude
`--permission-mode auto` do not need wrapper scripts or shell-joined
`--agent` strings. After validation, `new` creates the directory on the target
host and starts the agent through a narrow shell bootstrap whose final command
is an `exec <agent> <arg>...` argv. Joined/new sessions receive
`@mstream/*` tags and a managed-agent reporting prompt when a task is sent. If
the workstream has an mmux label, assignment also writes:

```text
@mstream/mmux-label=<label>
@mstream/mmux-selected-key=mstream
@mstream/mmux-previous-selected-key=<previous @mmux/__selected-key, when any>
@mmux/mstream=<label>
@mmux/__selected-key=mstream
```

Recruited sessions also receive workstream-membership tags before any task is
sent, so restart plus `scan` can hydrate their assignment. `recruit --agent-arg`
uses the same repeated argument form to match available sessions that were
created with that agent argv profile and preserves that metadata on the
recruited assignment.

`rename <target> <new-name>` resolves the target to the stable `(host,
session_id)` record, runs `tmux rename-session -t $id <new-name>`, and updates
the existing daemon record's display name without rekeying timers, handoffs, or
workstream membership. `<new-name>` is trimmed and must not be empty, contain
`:` or `::`, contain control/Unicode format characters, or look like a tmux
session id such as `$7`.

`rename` also accepts `--role`, `--workstream`, and `--mmux-label` for a
single-step rename plus retag. `session retag <target>` exposes the same
metadata path without changing the tmux session name. Retagging to another
workstream requires an existing open destination workstream and an effective
role. The daemon clears the previous workstream's mmux label, applies the
destination/current label, updates durable `@mstream/*` tags, and cancels stale
handoffs owned by the old workstream while preserving the stable session target.

`leave` and `close` clear mstream-owned mmux labels. If mstream still owns
`@mmux/__selected-key`, cleanup restores the saved previous selected key or
unsets the selected key when there was no previous value. If a user or another
tool changed `@mmux/__selected-key` after mstream applied the label, cleanup
leaves that selected key unchanged.

`retire` keeps the target in its workstream, writes `@mstream/state=quarantined`,
and records the transition so cleanup directives remain audit-logged. `reclaim`
then kills and deregisters only a managed target whose live tmux tags are
`quarantined`.

## Communication And Handoff

```sh
mstream send pr-324 local::codex-worker --text "Re-run clippy." --enter
mstream send pr-324 local::codex-worker --interrupt-first --settle-ms 500 \
  --text "Stop and address feedback." --enter --set-state busy

mstream interrupt local::codex-worker
mstream interrupt local::codex-worker --key ctrl-c

mstream broadcast pr-324 --state busy --text "Wrap up your current step and summarize status." --enter

mstream session mark local::codex-worker --state done --summary "Implemented requested fixes."
mstream session mark local::codex-worker --state blocked --summary "Need host credentials."
mstream send pr-324 local::codex-worker --require-state quarantined \
  --text "Rename your worktree with a TBR- prefix, then summarize." --enter
```

When the target is joined to an open workstream, `interrupt` appends an
`interrupted` event to that workstream and returns the resulting cursor. If the
target is a connected tmux session that is not joined to a known workstream, the
command still sends the key and returns only the command result.

`session mark` is coordinator-owned in the project workflow: the orchestrator
marks explicit targets after observing durable evidence such as pushed commits,
PRs, review comments, test output, or blockers. Collaborating agents do not
have mstream access and should not be instructed to call mstream state commands.
There is no `self` alias; use explicit `<host>::<session>` targets so marks are
auditable coordinator actions.

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

## Timers

Timers solve the orchestrator wakeup problem when the agent harness has no
first-class cron or scheduler. A timer lives in the daemon and periodically
sends a configured prompt into a tmux target through `motlie-tmux` send-keys.
Timer state is daemon memory only; daemon restart loses timers.

```sh
mstream timer start issue-337-poll \
  --every 5m \
  --workstream issue-337-tmux-fleet-api \
  --self \
  --prompt "[mstream:issue-337-poll] Wakeup: check issue-337-tmux-fleet-api with mstream status and summary-input. Unblock agents, summarize only material changes, then decide whether to keep, change, or stop this timer." \
  --submit-retries 1 \
  --submit-retry-delay-ms 750

mstream timer start issue-337-poll \
  --every 5m \
  --workstream issue-337-tmux-fleet-api \
  --target local::codex-orchestrator \
  --prompt "[mstream:issue-337-poll] Wakeup: check issue-337-tmux-fleet-api with mstream status and summary-input. Unblock agents, summarize only material changes, then decide whether to keep, change, or stop this timer." \
  --submit-retries 1 \
  --submit-retry-delay-ms 750

mstream timer list
mstream timer list --workstream issue-337-tmux-fleet-api
mstream timer fire issue-337-poll
mstream timer stop issue-337-poll
```

`--every` accepts raw seconds, second suffixes such as `30s`, or minute
suffixes such as `5m`. Use `--self` for orchestrator wakeup timers when the
client is running inside the orchestrator tmux session. It resolves the current
session with `tmux display-message -p '#S'` and targets `local::<session>` by
default. Use `--self-host <alias>` when the orchestrator host alias is not
`local`. `--self` and `--target` are mutually exclusive.

Explicit `--target` accepts `<host-alias>::<tmux-session-name>` or a stable
`<host-alias>::$<tmux-session-id>` target. The host must already be connected
and `timer start` validates that the target exists before scheduling.

`--workstream <name>` associates a timer with a workstream for filtering and
closeout. `timer list --workstream <name>` returns only scoped timers.

Timer prompts default to sending Enter after the text, matching the
orchestrator self-prompt use case. They also default to one extra Enter after
750ms to handle agent TUI cases where the first submit key is missed. Tune with
`--submit-retries` and `--submit-retry-delay-ms`; retries send only extra Enter
keys and never re-send prompt text. Use `--no-enter` when the text should be
placed in the pane without submission; this disables submit retries.

Timer delivery also defaults to `--input-quiet-for 10s`. When attached-client
input in the target session is newer than that quiet window, the timer defers
without sending prompt text or Enter retries, records
`last_defer_reason=recent_client_input`, and schedules the next attempt after
the remaining quiet time. Use `--input-quiet-for <duration>` to tune the guard,
or `--no-input-guard` when unattended delivery should not wait for a quiet
window.

`timer fire` is a manual immediate trigger for smoke testing the target and
prompt. It follows the same input-quiet guard and does not replace the next
scheduled wakeup. `timer list` reports `next_fire_at`, `last_fired_at`,
`fire_count`, `defer_count`, `last_deferred_at`, `last_defer_reason`,
`last_input_activity_at`, `input_quiet_for_secs`, `last_error`,
`submit_retries`, `submit_retry_delay_ms`, optional `workstream`, and prompt
length without echoing the prompt body.

`close --stop-timers` stops timers scoped to the closing workstream.
`close --standby-agents` sends a standby message to joined sessions before
freeing them. Standby send attempts are recorded in the workstream timeline as
`standby_sent` or `standby_failed`; successful standby events include the
redacted standby message text as a `to_agent` audit entry. Failed standby sends
are reported in `standby_failed` and do not abort the rest of closeout.

`mstream close` is intentionally workflow-neutral. It does not know about
GitHub issues, PRs, merges, or external closeout comments, and should not grow
`--issue`, `--pr`, or posting flags. Use the observation primitives below,
especially `events --readable`, `summary-input`, and `snapshot`, to dump the
workstream timeline or transcript. The orchestrator or project skill owns
turning that material into issue comments, PR comments, or user-facing
closeout summaries.

## Observation

```sh
mstream status pr-324
mstream status pr-324 --active-window-secs 30 --idle-after-secs 300
mstream events pr-324 --limit 50
mstream events pr-324 --limit 50 --readable
mstream snapshot pr-324 --max-chars 12000
mstream summary-input pr-324 --max-chars 12000
```

`status` refreshes live tmux session activity through each connected
`HostHandle::list_sessions()` call before returning JSONL. The activity value is
the tmux library's session-level maximum of `session_activity` and
`window_activity`, so it reflects either attached-client input or program
output. Use this command for liveness instead of direct SSH/tmux probing.
Timer input guards do not apply to observation commands; `status`, `events`,
`snapshot`, `summary-input`, `timer list`, `hosts`, `scan`, `list`, and `show`
remain read-only polling operations.

Each status agent includes:

- `tmux_present`: `true`, `false`, or `null` when host activity could not be read
- `tmux_session_id`, `tmux_activity`, and `tmux_activity_at`
- `last_output_secs`: age of the latest tmux activity according to the daemon clock
- `observed_activity_idle_secs`: how long the daemon has observed the same activity value
- `activity_hint`: `active`, `quiet`, `idle`, `missing`, or `unknown`
- `activity_error`: host activity refresh error when known

The default hint thresholds classify sessions with activity in the last 30
seconds as `active` and sessions quiet for at least 300 seconds as `idle`. The
workstream `status` response also includes an `audit` object with
`degraded`, `agent_output_dropped`, `lossless_enqueue_failures`, and
`persist_failures` counters.

Event cursors are opaque base64 JSON owned by `mstream`; they embed the
workstream timeline generation. A cursor from an older generation returns a
structured `cursor_stale` JSONL error. Bounded `events --limit N` responses
return a cursor that advances only to the last returned event, not to the
workstream watermark.

Event records include `direction` (`to_agent`, `from_agent`, or `system`),
`actor` (`orchestrator`, `agent`, or `mstream`), target, optional `source_pane`,
state, summary, text, and `redacted`/`truncated` flags. Outbound messages from
`send`, `broadcast`, handoffs, timer fires, closeout standby sends, and initial
managed prompts are stored as `to_agent` entries. Agent terminal output observed
through tmux monitoring is stored as `from_agent` `agent_output` entries, and
explicit `session mark --summary` reports are also agent-authored events. All
non-`agent_output` events use the lossless audit queue; `agent_output` uses a
best-effort queue so bursty terminal output cannot block control-plane commands
or evict acknowledged control audit messages. Structured `events` and
`events_readable` responses include the same `audit` counters as `status`.

Before entering memory or disk, event text and summaries are scrubbed for
phone-like digit runs, including runs split by tabs/newlines/Unicode whitespace
or using non-ASCII numeric digits, and replaced with `[REDACTED_PHONE]`. Text
fields are capped per event and marked `truncated=true` when capped. Daemon
startup skips malformed or partial audit-log JSONL lines with a stderr warning
rather than failing startup.

`events --readable` prints a plain-text audit transcript for human-facing
summaries. Call-log message bodies are rendered as indented blocks instead of
single-line markers. The daemon wire response still carries an
`events_readable` record with a `text` field, but the CLI unwraps that field
before writing stdout. The default `events` response remains structured JSONL.

Machine-facing output is JSONL on stdout except for the explicit human-readable
`events --readable` mode. Errors are also JSONL records, for example:

```jsonl
{"type":"error","kind":"daemon_unreachable","message":"daemon unreachable at /tmp/mstream.sock; start the daemon or provide --socket"}
```
