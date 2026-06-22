# Design: Issue #561 mstream Session-State Fidelity

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-22 | @codex-561-design | Folded in broadened #561 scope for boot-liveness settle/retry and genuine boot-death diagnostics. |
| 2026-06-22 | @codex-561-design | Initial design for reconciling canonical agent names and liveness against live tmux state, with stable session-id binding, drift signals, doctor output, and #401 lifecycle hygiene. |

## Status

This is brownfield design work for
[issue #561](https://github.com/chungers/motlie/issues/561). `mstream` and
`libs/tmux` already have the important substrate: stable tmux session ids
(`$N`), session metadata tags, `Target::rename()` by stable id, OutputBus
routing by session id, and #401 quarantine/reclaim lifecycle commands. This
design tightens the session-state contract without replacing those layers. The
same fidelity principle also applies during session creation: a newly created
tmux session must be reconciled with settle/retry and useful diagnostics, not a
single immediate probe.

## Problem

Project invariant 12 says an agent's canonical identity is the live tmux
session name. `mstream` currently keeps enough cached session metadata that
operators can see stale identity or stale existence:

- a record can keep an old `identity` even after `tmux rename-session`
- `session list` can report daemon-memory records without proving the tmux
  session still exists
- status-style liveness signals are not a complete session ledger
  reconciliation contract
- host unreachable, tmux server unavailable, and confirmed missing session are
  not surfaced consistently enough for safe cleanup
- `libs/tmux::HostHandle::create_session()` treats one immediate
  `list_sessions()` miss after `new-session` as boot death, which can false-fail
  a live agent that needs a short settle window
- when startup genuinely fails, `mstream new` reports a generic "exited
  immediately" message without pane output or exit diagnostics

The user-visible failure is serious: an orchestrator may call an agent by a
stale handle and tell it to self-identify with the wrong `@handle` in commits
or GitHub comments.

## Non-Goals

- No durable local database for session state. mstream remains daemon-memory
  plus tmux tags and audit JSONL, as described in the main DESIGN.
- No replacement for `libs/tmux::Fleet`, `HostHandle`, `Target`, or OutputBus.
- No raw SSH/tmux escape hatch in mstream UX. Reconciliation uses typed
  `libs/tmux` APIs.
- No automatic destructive cleanup on read-only observation commands.
- No guarantee that historical events are rewritten after a rename. Audit
  records remain immutable; rendered responses may add current live identity as
  a sidecar field.
- No exact agent self-report parser in this issue. Existing
  `@mstream/identity` data is treated as the best available
  `self_reported_handle` until a future explicit self-report channel exists.

## Current Implementation Grounding

### mstream state

`bins/mstream/src/state.rs` stores sessions in:

```rust
sessions: BTreeMap<FleetTargetSpec, SessionRecord>
```

`FleetTargetSpec` is aliased as `SessionTarget`. Created, joined, scanned, and
renamed sessions are generally normalized to `FleetTargetSpec::session_id(host,
"$N")`, so the daemon already has the right stable key. `SessionRecord` stores:

- `identity: String`: currently a cached display identity
- `agent: Option<String>` and `agent_args`: currently the executable metadata
- `tmux_session_created: Option<u64>`: used to detect reused `$N` ids
- activity observation fields used by `status`

This design splits the overloaded identity meaning instead of changing the
stable target key model.

### new, join, scan, and rename

`new_session_shared` creates a named tmux session through `HostHandle`, then
rekeys the daemon record to `host::$N` using the returned `SessionInfo.id`.

`join_shared` resolves a user-provided target through `HostHandle::target()`,
then stores the resolved stable target.

`scan_shared` calls `HostHandle::list_sessions()`, batch reads mstream tags with
`list_tags_for_session_infos()`, and rehydrates records keyed by `session.id`.
It already calls `record.observe_tmux_session(&session)`, so scan is the right
place to make the live tmux name authoritative.

`session_retag_shared` implements `mstream rename`. It resolves the target,
checks freshness, calls `Target::rename(&new_name)`, observes the returned
target, writes tags, and keeps the stable target unchanged. In `libs/tmux`,
`Target::rename()` calls `rename-session -t <stable-session-id> <new-name>` for
session targets.

### create_session boot liveness

`libs/tmux/src/host.rs::HostHandle::create_session()` currently runs
`control::create_session_with_prefix()`, then immediately calls
`self.list_sessions()` once and looks up the requested name. If that one listing
does not contain the new session, it returns a generic created-but-not-found
state error.

`mstream new` catches that exact message through
`is_created_session_not_found()` and turns it into a generic exited-immediately
startup error.

This is a boot-time version of the same fidelity bug: a single immediate tmux
read is treated as truth. It also erases the diagnostic evidence an operator
needs to distinguish "alive but not listed yet" from a real startup failure.

### status, snapshot, and events

`status_shared` already calls `HostHandle::list_sessions()` grouped by host and
builds `LiveActivity::{Present, Missing, Error}`. It detects session-id reuse
with `tmux_session_created`. However, `SessionRecord::status_json()` does not
promote the live session name to the public canonical identity field, and
`session list` does not use this reconciliation path.

`snapshot_shared` and `summary_input_shared` capture tmux panes by resolving
targets and calling `ensure_resolved_target_fresh_shared()`, which can observe
`SessionInfo`. These are natural refresh points.

Output audit records route `TargetOutput` by `session_id()` first in
`target_for_output()`, then fall back to name and cached identity. That is good
for rename safety, but rendered events do not expose current live name or name
drift.

### libs/tmux substrate

`libs/tmux::SessionInfo` contains the live tmux name, stable id, creation time,
and activity:

```rust
pub struct SessionInfo {
    pub name: String,
    pub id: SessionId,
    pub created: u64,
    pub activity: u64,
    // ...
}
```

`HostHandle::list_sessions()` is the core inventory call. Today
`discovery::list_sessions_with_prefix()` collapses "no server running" and "no
sessions" into `Ok(Vec::new())`. Issue #561 needs mstream to distinguish
confirmed dead sessions from unreachable or unavailable host/tmux inventory, so
mstream needs a richer inventory result without breaking existing
`list_sessions()`.

## Root Cause

Two concepts are conflated:

- canonical live tmux name: mutable, authoritative, read from live tmux
- stored identity/self-report: cached metadata, useful for detecting drift but
  not authoritative
- boot liveness: a just-created tmux session needs a short reconciliation
  window before mstream can conclude that the agent died

The daemon also has a session ledger that can outlive tmux reality. A stable
target `host::$N` is a good key, but serving the ledger without a live inventory
pass makes stale records look live. Reconciliation exists in pieces, especially
`status`, but it is not the contract for every operator-facing session view or
for the boot path.

## Requirements

The design must satisfy the ten #561 points:

1. Derive the live name from tmux using the stable tmux session id as the key.
2. Piggyback name refresh on snapshot/status/timeline-style reads.
3. Make scan rehydration treat tmux name as ground truth.
4. Surface live tmux name as canonical `agent`; label cached/self value as
   `self_reported_handle`; expose `name_drift`.
5. Make `mstream rename` rename tmux and daemon state as one control-plane
   operation.
6. Reconcile liveness as `live`, `dead`, or `unreachable`.
7. Provide an in-band drift/death query.
8. Tie cleanup to the #401 quarantine/reclaim lifecycle.
9. Make `create_session` boot-liveness settle/retry instead of single-probe.
10. On genuine boot-death, capture and report pane output/exit diagnostics.

## Alternatives Considered

### A. Refresh on read, chosen

Every operator-facing read that reports sessions performs or reuses a bounded
tmux inventory for the relevant hosts. `session list`, `status`, `snapshot`,
`summary-input`, and rendered `events` all return reconciled identity/liveness
fields. `scan` remains the explicit rehydration command.

Pros:

- fits existing `HostHandle::list_sessions()` and `SessionInfo` data
- low conceptual change because `status` already does most of this
- exact on the next operator read, which matches the incident acceptance tests
- no background task lifecycle, timers, or extra host load while idle
- works after daemon restart because scan/status rebuild from tmux tags

Cons:

- not instantaneous between reads
- each read can pay one inventory call per involved host
- needs careful error classification so unreachable hosts do not look dead
- rendered historical events need sidecar live identity rather than mutation

### B. Periodic daemon reconcile task

The daemon runs a background reconciliation loop per connected host and updates
the session ledger every N seconds.

Pros:

- `session list` can be cheap and still usually fresh
- can precompute doctor summaries
- detects drift even if the operator does not poll status

Cons:

- still stale between ticks
- adds another long-lived daemon task with retry/backoff behavior
- complicates daemon shutdown and host disconnect semantics
- wastes host/SSH traffic when no orchestrator is observing
- still needs a foreground read path for "tell me now" correctness

### C. Event-driven tmux host watch/hooks

Use `HostHandle::watch_host_events()` or tmux control-mode rename/session close
notifications as the primary source of truth.

Pros:

- best latency for rename/death detection
- `libs/tmux` already has event primitives keyed by stable session id
- can eventually reduce repeated list calls on active hosts

Cons:

- not sufficient for cold start or daemon restart; scan is still required
- connection drops are themselves the hard case for issue #561
- more moving parts per connected host
- tmux hook/control-mode differences make correctness harder than a direct
  inventory pass

Event-driven watch can be a later optimization after the read-time contract is
correct.

## Chosen Design

### 1. Stable-id key, live-name value

The stable mstream target remains `host::$N`, backed by
`FleetTargetSpec::session_id()`. User-facing name targets are accepted only as
input selectors and normalized to the stable target after `HostHandle::target()`
resolves a live `SessionInfo`.

`SessionRecord` should separate:

- `live_name: Option<String>`: last name observed from live tmux
- `self_reported_handle: Option<String>`: cached/self metadata, initially
  migrated from existing `identity` tag/field
- `agent_executable: Option<String>`: current meaning of `agent`

Implementation can keep the stored tag name `@mstream/identity` for migration,
but public JSON must not call it canonical identity. Render it as
`self_reported_handle`.

`observe_tmux_session(&SessionInfo)` updates only live tmux facts:

- `live_name`
- `tmux_session_created`
- activity-related fields when relevant

It must not overwrite `self_reported_handle`; otherwise out-of-band renames
would hide drift.

### 2. Reconciliation helper

Add an internal mstream helper, conceptually:

```rust
struct ReconciledSession {
    target: SessionTarget,
    liveness: SessionLiveness,
    live: Option<SessionInfo>,
    self_reported_handle: Option<String>,
    name_drift: Option<bool>,
    error: Option<String>,
}

enum SessionLiveness {
    Live,
    Dead,
    Unreachable,
}
```

The helper groups requested `SessionTarget`s by host, inventories each host
once, indexes returned `SessionInfo` by stable `session.id`, updates
`SessionRecord.live_name` for live rows, and returns per-session reconciliation
results.

Classification:

- `live`: host inventory succeeded, stable `$N` exists, and
  `tmux_session_created` is absent or matches `SessionInfo.created`
- `dead`: host inventory succeeded, stable `$N` is absent, or `$N` was reused
  with a different `session_created`
- `unreachable`: host/SSH/tmux inventory could not prove absence

For the no-server case, add a small `libs/tmux` API that preserves diagnostic
state instead of only returning `Vec<SessionInfo>`:

```rust
pub enum SessionInventory {
    Available(Vec<SessionInfo>),
    NoTmuxServer { reason: String },
}

impl HostHandle {
    pub async fn session_inventory(&self) -> Result<SessionInventory>;
}
```

Existing `HostHandle::list_sessions()` remains backward compatible and can keep
collapsing no-server/no-sessions for existing callers. `mstream` uses
`session_inventory()` so "tmux server unavailable" maps to `unreachable`, not
mass dead/prune.

### 3. Read-time refresh points

`session list`:

- reconciles all daemon-known sessions for connected hosts
- returns `liveness`, live name fields, and drift flags
- marks disconnected host records as `unreachable`

`status`:

- reuses the current host-grouped inventory path
- updates live names before rendering agents
- keeps activity fields and maps them to the same `liveness`

`snapshot` and `summary-input`:

- already resolve/capture targets; after resolution, observe live `SessionInfo`
- aggregate output headers should include stable target and live agent when
  known, for example `=== amd1::$69 agent=codex-561-design ===`
- missing/unreachable targets produce empty capture as today, but the JSON
  response should include a `sessions` sidecar with liveness details

`events` / timeline:

- rendered JSON responses add current reconciliation sidecars:
  `agents: [...]` and per-event `agent` when the event target has a live match
- persisted audit events are not rewritten
- OutputBus `TargetOutput` with `session_id` should refresh `live_name` from
  `output.session_name()` before recording the audit event

### 4. Public identity fields

For every session row in `session list`, `status`, doctor output, snapshot
sidecars, and event sidecars:

```json
{
  "target": "amd1::$69",
  "agent": "codex-561-design",
  "agent_source": "live_tmux",
  "last_observed_agent": "codex-561-design",
  "self_reported_handle": "codex-535",
  "name_drift": true,
  "agent_executable": "codex",
  "agent_args": [],
  "liveness": "live",
  "tmux_present": true,
  "tmux_session_id": "$69"
}
```

Rules:

- `agent` is the canonical live tmux session name when `liveness=live`
- `agent` is `null` when the session is `dead` or `unreachable`
- `last_observed_agent` may show the last cached live name, but never implies
  current liveness
- `self_reported_handle` is the old `@mstream/identity` value or future
  self-report value
- `name_drift=true` only when the row is live and
  `self_reported_handle != agent`
- `name_drift=null` when mstream cannot compare because the session is not live
- the executable currently exposed as `agent` moves to `agent_executable`

This is a brownfield JSON contract change. To soften migration, one release can
also emit deprecated `identity` and `agent_command` aliases, but docs and new
tests should treat `agent` as the only canonical name.

### 5. Scan rehydration

`scan` remains the explicit "rebuild from tmux tags" operation:

1. inventory host sessions
2. read mstream tags for each live `SessionInfo`
3. key records by `host::session.id`
4. set `live_name` from `SessionInfo.name`
5. set `self_reported_handle` from `@mstream/identity` if present
6. rebuild workstream membership from tags
7. start monitoring by stable session id

If a live tmux name differs from `@mstream/identity`, scan does not overwrite
the self-reported handle. It returns a drift issue so the orchestrator can
message the agent to re-self-identify as `@<live tmux name>`.

### 6. Rename semantics

`mstream rename <target> <new-name>` is the only supported mstream-owned rename
path. It must:

1. resolve the input target to `host::$N`
2. verify `tmux_session_created` when known
3. call `Target::rename(new_name)`, which uses `rename-session -t '$N'`
4. observe the returned `SessionInfo` and set `live_name=new_name`
5. update `@mstream/identity` / `self_reported_handle` to `new_name`
6. leave the stable target key unchanged
7. return the reconciled identity fields

The tmux rename is the commit point. mstream must not mutate daemon identity
before tmux rename succeeds. If tag writing fails after tmux rename succeeds,
the command reports the partial failure; the next reconciliation still derives
the correct canonical name from live tmux and reports drift until tags are
fixed.

Example:

```sh
mstream rename amd1::$69 codex-561-design
```

```json
{
  "type": "ok",
  "op": "rename",
  "target": "amd1::$69",
  "old_agent": "codex-541",
  "agent": "codex-561-design",
  "self_reported_handle": "codex-561-design",
  "name_drift": false,
  "liveness": "live"
}
```

### 7. Liveness reconciliation

Public liveness values:

- `live`: session exists in live tmux under the stable id
- `dead`: connected host inventory succeeded and the stable id is absent or
  reused
- `unreachable`: host/SSH/tmux inventory failed or tmux server state is
  unavailable, so mstream cannot prove the session dead

Compatibility fields map as:

- `live` -> `tmux_present=true`, `activity_hint=active|quiet|idle`
- `dead` -> `tmux_present=false`, `activity_hint=missing`
- `unreachable` -> `tmux_present=null`, `activity_hint=unknown`

Dead example:

```json
{
  "target": "local::$12",
  "agent": null,
  "last_observed_agent": "opus47-543-rv",
  "self_reported_handle": "opus47-543-rv",
  "name_drift": null,
  "liveness": "dead",
  "tmux_present": false,
  "death_reason": "stable tmux session id not present in reachable host inventory"
}
```

Unreachable example:

```json
{
  "target": "mac1::$4",
  "agent": null,
  "last_observed_agent": "codex-535",
  "self_reported_handle": "codex-535",
  "name_drift": null,
  "liveness": "unreachable",
  "tmux_present": null,
  "liveness_error": "ssh connection failed"
}
```

### 8. Boot-liveness settle/retry

`libs/tmux::HostHandle::create_session()` must stop treating one immediate
post-`new-session` listing as authoritative. After `new-session` succeeds, it
should reconcile for a bounded startup window:

```rust
struct CreateSessionProbe {
    settle_ms: u64,
    retry_interval_ms: u64,
    max_attempts: u8,
}
```

Default behavior should be conservative and fast, for example an immediate
probe plus retries over roughly 1-2 seconds. The exact values belong in PLAN,
but the contract is:

1. run `tmux new-session`
2. poll `list_sessions()` for the requested name and stable id
3. if found, return the live `Target`
4. if inventory is transiently unreachable, retry inside the same bounded
   window
5. only after the window expires, classify startup as failed or diagnostically
   uncertain

The retry loop should prefer stable id after it is known, but the first lookup
can still be by requested name because tmux does not return the new session id
from the current `new-session` wrapper. A follow-up implementation may improve
this by using `tmux new-session -P -F` with `session_id` and `session_name`, but
the design does not require changing the public `HostHandle::create_session()`
return type.

`mstream new` should surface this as boot reconciliation, not as an
agent-authored state transition. A live session that appears on a retry is a
successful boot. It must not be marked dead, quarantined, or removed because the
first probe missed it.

### 9. Boot-death diagnostics

When the bounded boot probe expires, mstream must collect diagnostics before
returning a boot-death error. The diagnostic attempt should be best-effort and
bounded:

- try to resolve the session by requested name and, if known, stable id
- capture the primary pane scrollback with a small cap
- query pane/process death fields when available, such as pane dead status,
  current command, or exit status
- include the tmux command error if inventory itself failed
- return diagnostics in structured JSONL and in the human error message

For managed `mstream new` agent sessions, the bootstrap path should preserve
short-lived failure evidence long enough to capture it. A genuine immediate
exit often removes the tmux session before a later capture can run, so the
implementation should use one of these equivalent strategies:

- a tmux remain-on-exit/startup diagnostic option that leaves the pane
  capturable until mstream classifies the boot result, then clears/cleans it up
- a narrow managed-agent bootstrap that records exit status and holds the pane
  briefly only on failure

The normal long-running agent process must still become the pane foreground
workload. The diagnostic mechanism is only for startup failure evidence; it
must not turn a healthy agent into a shell-supervised process with different
runtime semantics unless that tradeoff is explicitly documented in PLAN.

Example genuine failure response:

```json
{
  "type": "error",
  "code": "agent_boot_failed",
  "target": "mac1::codex-562-impl",
  "agent_executable": "/opt/homebrew/bin/codex",
  "boot_probe_attempts": 8,
  "boot_probe_elapsed_ms": 1500,
  "tmux_present": false,
  "exit_status": 127,
  "pane_output": "sh: /opt/homebrew/bin/codex: No such file or directory\n"
}
```

Example delayed-but-live response:

```json
{
  "type": "ok",
  "op": "new",
  "target": "mac1::$42",
  "agent": "codex-562-impl",
  "boot_probe_attempts": 3,
  "boot_probe_elapsed_ms": 320,
  "liveness": "live"
}
```

This keeps boot behavior aligned with the rest of #561: live tmux/pane state is
the source of truth, and mstream reconciles with retry plus real diagnostics
before making an operator-visible claim.

### 10. In-band doctor query

Add a read-only `mstream doctor` command:

```sh
mstream doctor
mstream doctor --workstream issue-561-mstream-fidelity
mstream doctor --host amd1
```

It returns one bounded JSONL record:

```json
{
  "type": "doctor",
  "summary": {
    "live": 6,
    "name_drift": 1,
    "dead": 2,
    "unreachable": 21,
    "unreachable_hosts": 1
  },
  "name_drift": [
    {
      "target": "amd1::$69",
      "agent": "codex-541",
      "self_reported_handle": "codex-535"
    }
  ],
  "dead_sessions": [
    {
      "target": "local::$12",
      "last_observed_agent": "opus47-543-rv",
      "death_reason": "stable tmux session id not present"
    }
  ],
  "unreachable_hosts": [
    {
      "host": "mac1",
      "reason": "ssh connection failed"
    }
  ]
}
```

`session list` and `status` also include per-row flags so an orchestrator does
not have to run doctor for every poll. Doctor is the aggregate audit/query
surface for drift and death.

### 11. Hygiene and #401 lifecycle

Read-only reconciliation commands do not kill tmux sessions and do not silently
delete records.

Cleanup uses explicit lifecycle operations:

```sh
mstream doctor --quarantine-dead
mstream doctor --prune-quarantined
```

`--quarantine-dead` applies only to confirmed `dead` sessions, never
`unreachable` sessions. It removes them from recruitable rosters, cancels
handoffs/timers using the existing `deregister_session_target` behavior, and
marks/reports them as quarantined in daemon memory where a record remains.

`--prune-quarantined` removes daemon records that are already quarantined and
dead. Live managed sessions still use the #401 `retire` then gated `reclaim`
path because reclaim is the destructive tmux kill operation.

`scan` can continue to clean up stale id reuse for safety, but it should report
which records were quarantined/pruned in its JSON output. It must not classify
an unreachable host as a set of dead sessions.

## Data Flow

### Create or join

1. User supplies `host::name` or `host::$N`.
2. mstream resolves it through `HostHandle::target()`.
3. `libs/tmux` returns `SessionInfo { name, id, created, activity }`.
4. mstream stores the record under `host::id`.
5. mstream writes assignment tags, including current
   `self_reported_handle`.
6. Public output renders `agent` from live `SessionInfo.name`.

### Create session boot

1. `mstream new` builds the managed bootstrap command and calls
   `HostHandle::create_session()`.
2. `libs/tmux` runs `tmux new-session`.
3. `libs/tmux` polls live tmux inventory over the startup probe window.
4. If the session appears, `libs/tmux` returns a `Target` with live
   `SessionInfo`; mstream normalizes it to `host::$N`.
5. If the session does not appear, mstream captures startup diagnostics before
   returning `agent_boot_failed`.
6. A single missed listing is never enough to classify boot death.

### Reconcile on read

1. mstream snapshots all relevant `SessionTarget`s.
2. Targets are grouped by host.
3. Each host is inventoried once.
4. Returned sessions are indexed by `SessionInfo.id`.
5. Records are updated with live facts for present sessions.
6. Rows are rendered with canonical `agent`, drift, and liveness fields.
7. Confirmed dead and unreachable are surfaced distinctly.

### Out-of-band tmux rename

1. User or another tool runs `tmux rename-session -t '$69' codex-541`.
2. Next `status`, `session list`, `snapshot`, `events`, or `doctor` inventories
   the host.
3. mstream matches `$69`, updates `live_name=codex-541`, keeps
   `self_reported_handle=codex-535`.
4. Output shows `agent=codex-541` and `name_drift=true`.
5. The orchestrator can message the agent to self-identify as
   `@codex-541`.

## High-Level CLI/API Changes

### `mstream new`

`mstream new` keeps the same user-facing command shape, but its result and
errors include boot reconciliation fields:

```json
{
  "type": "ok",
  "op": "new",
  "workstream": "issue-561-mstream-fidelity",
  "target": "mac1::$42",
  "agent": "codex-562-impl",
  "boot_probe_attempts": 3,
  "boot_probe_elapsed_ms": 320,
  "liveness": "live",
  "cursor": "..."
}
```

On genuine boot failure:

```json
{
  "type": "error",
  "code": "agent_boot_failed",
  "target": "mac1::codex-562-impl",
  "agent_executable": "/opt/homebrew/bin/codex",
  "boot_probe_attempts": 8,
  "boot_probe_elapsed_ms": 1500,
  "tmux_present": false,
  "exit_status": 127,
  "pane_output": "..."
}
```

The exact cap for `pane_output` belongs in PLAN, but it must be bounded and
large enough to show ordinary shell, PATH, and auth errors.

### `mstream session list`

Current behavior is cached ledger output. New behavior reconciles before
rendering:

```sh
mstream session list
```

```json
{
  "type": "session",
  "target": "amd1::$69",
  "agent": "codex-541",
  "agent_source": "live_tmux",
  "last_observed_agent": "codex-541",
  "self_reported_handle": "codex-535",
  "name_drift": true,
  "agent_executable": "codex",
  "agent_args": [],
  "role": "implementer",
  "state": "busy",
  "workstream": "issue-561-mstream-fidelity",
  "liveness": "live",
  "tmux_present": true,
  "tmux_session_id": "$69"
}
```

### `mstream status <workstream>`

Status keeps current activity fields and adds canonical identity/liveness:

```json
{
  "type": "status",
  "workstream": "issue-561-mstream-fidelity",
  "agents": [
    {
      "target": "amd1::$69",
      "agent": "codex-541",
      "self_reported_handle": "codex-535",
      "name_drift": true,
      "agent_executable": "codex",
      "state": "busy",
      "liveness": "live",
      "tmux_present": true,
      "tmux_session_id": "$69",
      "last_output_secs": 18,
      "activity_hint": "active"
    }
  ]
}
```

### `mstream events <workstream>`

Persisted event records keep their original `target`. Rendered JSON adds
current live identity:

```json
{
  "kind": "agent_output",
  "target": "amd1::$69",
  "agent": "codex-541",
  "name_drift": true,
  "text": "..."
}
```

The top-level response also includes an `agents` reconciliation sidecar so a
reader can inspect drift/death even when the event limit excludes recent output.

### `mstream snapshot <workstream>`

Snapshot keeps bounded text and adds sidecar session state:

```json
{
  "type": "snapshot",
  "workstream": "issue-561-mstream-fidelity",
  "sessions": [
    {
      "target": "amd1::$69",
      "agent": "codex-541",
      "self_reported_handle": "codex-535",
      "name_drift": true,
      "liveness": "live"
    }
  ],
  "text": "=== amd1::$69 agent=codex-541 ===\n..."
}
```

### `mstream doctor`

Doctor is the aggregate in-band check for #561 drift/death:

```sh
mstream doctor --workstream issue-561-mstream-fidelity
mstream doctor --host amd1 --quarantine-dead
```

Default doctor is read-only. Cleanup flags are explicit.

## Migration Strategy

This is a brownfield API change because current JSON uses `agent` for the
executable and `identity` for a cached name.

Migration plan:

1. Add internal fields for live tmux name, self-reported handle, and executable.
2. Keep reading existing `@mstream/identity` tags as `self_reported_handle`.
3. Emit new canonical fields in all changed commands.
4. Move executable metadata to `agent_executable`.
5. Keep deprecated aliases for one release only if needed by known callers.
6. Update `bins/mstream/docs/API.md` and `PLAN.md` during implementation.
7. Remove any remaining code path that treats cached identity as canonical.
8. Replace the string-matched `is_created_session_not_found()` boot-death path
   with typed startup probe and diagnostic errors from `libs/tmux` or a narrow
   mstream wrapper.

No tmux tag migration is required for existing sessions. The first scan or
status after the change derives `live_name` from tmux and reuses existing tags
for `self_reported_handle`.

## Components To Test

### mstream unit tests

- `SessionRecord::observe_tmux_session` updates live name without overwriting
  `self_reported_handle`.
- `session list` reconciles and renders `agent`, `self_reported_handle`,
  `name_drift`, and `agent_executable`.
- `status` updates live name from `LiveActivity::Present`.
- dead vs unreachable classification maps to `tmux_present` and
  `activity_hint` correctly.
- `mstream rename` keeps the stable target, calls tmux rename before daemon
  mutation, and returns drift-free fields.
- `mstream new` succeeds when a session appears on a later startup probe rather
  than the first post-create listing.
- `mstream new` reports `agent_boot_failed` with bounded pane output and exit
  diagnostics on genuine startup failure.
- scan rehydrates by `SessionInfo.name` even when `@mstream/identity` is stale.
- OutputBus `TargetOutput` with a stable session id refreshes `live_name` and
  routes output after an out-of-band rename.
- doctor summary counts live, drift, dead, and unreachable rows.
- `doctor --quarantine-dead` never applies to unreachable rows.

### libs/tmux tests

- new inventory API distinguishes successful empty inventory from tmux server
  unavailable when tmux reports "no server running" or equivalent.
- existing `HostHandle::list_sessions()` behavior remains backward compatible.
- `HostHandle::create_session()` retries session discovery over the startup
  probe window and returns the live target when a delayed listing appears.
- startup probe exhaustion returns a typed or inspectable diagnostic error
  instead of only `session created but not found in list`.
- `Target::rename()` continues using stable session id in the tmux command.
- mock transport coverage for host command errors and no-server diagnostics.

### integration/smoke tests

- Create a tmux session, join it, out-of-band `tmux rename-session`, then verify
  `mstream status` and `session list` show the new `agent` and
  `name_drift=true`.
- Run `mstream rename` and verify live tmux `list-sessions` shows the new name,
  while mstream target remains `host::$N`.
- Kill a joined tmux session and verify `liveness=dead` on the next reconcile.
- Simulate SSH/host failure and verify `liveness=unreachable`, not dead.
- Scan after daemon restart and verify live tmux names are ground truth.
- Doctor reports drift/dead/unreachable without direct tmux probing.
- Create a session whose first `list_sessions()` response omits the new session
  but a later response includes it; verify mstream does not boot-death.
- Create a managed agent command that exits immediately; verify mstream reports
  captured output and exit status.

## Risks And Mitigations

- Field-name churn can break JSON consumers. Mitigate with an explicit
  migration note and short-lived aliases if required.
- No-server classification needs careful `libs/tmux` error handling. Mitigate
  by adding a richer API rather than changing `list_sessions()` behavior for all
  callers.
- Event history cannot be rewritten safely. Mitigate with rendered current
  identity sidecars and immutable audit records.
- Cleanup can destroy useful forensic state if automatic. Mitigate by keeping
  reconciliation read-only by default and requiring explicit doctor cleanup
  flags tied to #401 lifecycle semantics.
- Boot diagnostics can perturb healthy agent process semantics if implemented
  as a permanent wrapper. Mitigate by limiting diagnostic hold behavior to the
  startup window and documenting any bootstrap tradeoff in PLAN.
