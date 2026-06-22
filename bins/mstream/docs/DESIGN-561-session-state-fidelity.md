# Design: Issue #561 mstream Session-State Fidelity

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-22 | @codex-561-design | Simplified David's boot design: `new-session -d -P -F` is the correct-by-construction create-session fix, and remain-on-exit is only genuine boot-exit diagnostics. |
| 2026-06-22 | @codex-561-design | Addressed PR #563 review: made boot items 9-10 first-class acceptance criteria, chose remain-on-exit diagnostics, bounded per-host reconcile, documented handle semantics, promoted `new-session -P -F`, named migration consumer, and distinguished reused session ids. |
| 2026-06-22 | @codex-561-design | Folded in broadened #561 scope for create-session fidelity and genuine boot-exit diagnostics. |
| 2026-06-22 | @codex-561-design | Initial design for reconciling canonical agent names and liveness against live tmux state, with stable session-id binding, drift signals, doctor output, and #401 lifecycle hygiene. |

## Status

This is brownfield design work for
[issue #561](https://github.com/chungers/motlie/issues/561). `mstream` and
`libs/tmux` already have the important substrate: stable tmux session ids
(`$N`), session metadata tags, `Target::rename()` by stable id, OutputBus
routing by session id, and #401 quarantine/reclaim lifecycle commands. This
design tightens the session-state contract without replacing those layers. The
same fidelity principle also applies during session creation: mstream must take
the stable id and name from tmux's `new-session -d -P -F` output instead of
creating by name and then running a separate list probe. David explicitly folded
these boot-fidelity items into #561 on 2026-06-22, so they are in scope
for this DESIGN and for the follow-on PLAN.

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
- `libs/tmux::HostHandle::create_session()` runs bare `new-session -d -s` and
  then a separate `list_sessions()` to learn the stable id; if that second read
  is empty or stale, mstream can false-fail a live session
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

### create_session stable-id discovery

`libs/tmux/src/host.rs::HostHandle::create_session()` currently runs
`control::create_session_with_prefix()`, then immediately calls
`self.list_sessions()` once and looks up the requested name. If that separate
listing does not contain the new session, it returns a generic
created-but-not-found state error.

`mstream new` catches that exact message through
`is_created_session_not_found()` and turns it into a generic exited-immediately
startup error.

This is a boot-time version of the same fidelity bug: mstream creates the
session in tmux but then asks a separate cached/lagging listing to prove what
tmux already knows. The fix is to make `create_session_with_prefix()` use
`new-session -d -P -F`, parse the stable id/name from tmux's create output,
and key by that stable id immediately. Genuine process exit remains a separate
diagnostics problem.

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
- create-session identity discovery: tmux can return the stable id/name from
  `new-session -d -P -F`, but the current wrapper creates by name and then
  separately lists sessions to discover the id

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
9. Make `create_session` use `new-session -d -P -F` and parse the stable id/name
   from tmux's create output, with no post-create list probe.
10. On genuine agent boot-exit, capture and report pane output/exit diagnostics.

## Acceptance Criteria

The issue is accepted only when all ten points are covered by PLAN and
implementation:

1. `session list`, `status`, `snapshot`, and `events` derive canonical `agent`
   from live tmux session name by stable session id.
2. Monitoring reads refresh names without a manual `scan`.
3. `scan` treats live tmux name as ground truth after daemon restart.
4. Public rows expose live `agent`, `self_reported_handle`, and `name_drift`
   with documented handle semantics.
5. `mstream rename` performs tmux `rename-session` by stable id before daemon
   identity mutation.
6. Reconciliation distinguishes `live`, `dead`, and `unreachable`.
7. `session list`/`doctor` expose in-band name drift, dead sessions,
   unreachable hosts, and reused-session-id cases.
8. Confirmed-dead cleanup is explicit and tied to the #401 quarantine/reclaim
   lifecycle; unreachable hosts are never pruned as dead.
9. `HostHandle::create_session()` / `mstream new` use tmux
   `new-session -d -P -F` and parse stable id/name from the create command
   output, with no post-create `list_sessions()` probe required to prove
   existence.
10. Genuine agent boot-exit returns bounded pane output and exit/process
    diagnostics collected through remain-on-exit plus post-exit pane capture.

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
- `self_reported_handle: Option<String>`: mstream-authored last-written
  `@mstream/identity` handle. Until a real agent self-report channel
  exists, this is assigned state, not an agent assertion
- `agent_executable: Option<String>`: current meaning of `agent`

Implementation can keep the stored tag name `@mstream/identity` for migration,
but public JSON must not call it canonical identity. Render it as
`self_reported_handle`. The field name is retained because #561 asks for it,
but consumers must read it as the last assigned handle until an explicit
agent self-report channel exists.

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
once under a bounded timeout, indexes returned `SessionInfo` by stable
`session.id`, updates `SessionRecord.live_name` for live rows, and returns
per-session reconciliation results. A host timeout or inventory error is
contained to that host; it must not delay or poison rows for other hosts.

Classification:

- `live`: host inventory succeeded, stable `$N` exists, and
  `tmux_session_created` is absent or matches `SessionInfo.created`
- `dead`: host inventory succeeded, stable `$N` is absent, or `$N` was reused
  with a different `session_created`
- `unreachable`: host/SSH/tmux inventory could not prove absence

Dead rows include `death_kind`. Use `absent_session_id` when stable `$N` is
not present in a reachable inventory. Use `reused_session_id` when stable
`$N` is present but `SessionInfo.created` differs from the record; include the
observed live name so doctor/scan can report "tracked agent gone, id reused by
X" instead of treating it as a simple absence.

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

- reconciles all daemon-known sessions for connected hosts by default
- inventories hosts concurrently under a bounded per-host inventory timeout
- isolates timeout/error to that host: those rows render `liveness=unreachable`,
  `tmux_present=null`, `activity_hint=unknown`, and `liveness_error`
- never waits indefinitely on one bad or hanging host; other hosts render normally
- returns `liveness`, live name fields, and drift flags
- marks disconnected host records as `unreachable`
- supports `--cached` with alias `--no-reconcile` for high-frequency
  orchestrator polls; cached output skips host I/O, returns last observed
  ledger facts, and marks rows with `reconciled=false` and
  `agent_source=last_observed`

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
- `self_reported_handle` is the last mstream-authored `@mstream/identity`
  value until a future explicit agent self-report channel exists; it is not
  an agent assertion today
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
the assigned handle. It returns a drift issue so the orchestrator can remind
the agent to use `@<live tmux name>` in future commits, comments, and
operator-visible self-identification.

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
  "death_kind": "absent_session_id",
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

### 8. Create-session stable-id capture

`libs/tmux::HostHandle::create_session()` must stop creating by name and then
running a separate `list_sessions()` probe to discover the stable id. The
correct-by-construction fix is to make `control::create_session_with_prefix()`
use tmux print-format output:

```sh
tmux new-session -d -P -F "#{q:session_id} #{q:session_name}" -s <name> ...
```

If constructing the existing `Target::Session(SessionInfo)` needs more fields,
the format should include the required tmux values such as
`#{q:session_created}` and `#{q:session_activity}`. The invariant is that all
creation identity fields come from the `new-session` command output itself, not
from a later inventory pass.

Contract:

1. run `tmux new-session -d -P -F` through `create_session_with_prefix`
2. parse the returned stable `session_id` and live `session_name`
3. construct the returned `Target` keyed by stable id immediately
4. do not call `list_sessions()` to prove the just-created session exists
5. if `new-session` fails, return the command error with context
6. if parsing fails, return a typed parse/state error rather than falling back to
   a racing name lookup

This is an internal breaking change and is acceptable for #561. `new-window` and
`split-pane` already use the same `-P -F` pattern in
`libs/tmux/src/control.rs`; `create_session_with_prefix()` is the holdout. Once
create returns the stable id/name atomically, the alive-but-not-listed race is
eliminated. No post-create discovery loop remains in the design because such a
loop only patched the old create-then-separately-list pattern.

`mstream new` should surface this as create-session fidelity, not as an
agent-authored liveness transition. A successful `new-session -P -F` is enough
to key the daemon record as `host::$N`; later normal reconciliation can refresh
activity and liveness, but it must not be required to accept the boot.

### 9. Genuine boot-exit diagnostics

The create-session fix and boot diagnostics are orthogonal mechanisms:

- `new-session -d -P -F` solves the alive-but-not-listed race by returning the
  stable id/name from the tmux create command itself
- `remain-on-exit` plus bounded post-exit capture preserves evidence only when
  the managed agent process genuinely exits during boot

For managed `mstream new` agent sessions, the bootstrap path should preserve
short-lived failure evidence long enough to capture it. Decision: use tmux
`remain-on-exit` plus bounded post-exit pane capture for the startup diagnostic
window. This diagnostic path must not decide whether the session exists; it only
explains real process exit after tmux has created the session.

When a managed agent exits during that diagnostic window, mstream collects
best-effort bounded diagnostics before returning `agent_boot_failed`:

- capture the primary pane scrollback with a small cap
- query pane/process death fields when available, such as pane dead status,
  current command, or exit status
- include the tmux command error if the diagnostic query itself failed
- return diagnostics in structured JSONL and in the human error message

When the managed agent remains alive past the diagnostic window, clear the
`remain-on-exit` diagnostic option so normal later exits are not held. The
healthy agent keeps the direct pane foreground process model; no shell wrapper
is introduced for every managed agent boot.

Decision criteria:

- failed startup panes remain capturable long enough to report useful output
- healthy agents keep the direct pane foreground process model
- diagnostic retention is bounded and explicitly cleaned up
- diagnostics do not mask or replace the `new-session -P -F` creation contract

Example genuine failure response:

```json
{
  "type": "error",
  "code": "agent_boot_failed",
  "target": "mac1::codex-562-impl",
  "agent_executable": "/opt/homebrew/bin/codex",
  "tmux_session_id": "$42",
  "tmux_present": true,
  "exit_status": 127,
  "pane_output": "sh: /opt/homebrew/bin/codex: No such file or directory\n"
}
```

Example successful create response:

```json
{
  "type": "ok",
  "op": "new",
  "target": "mac1::$42",
  "agent": "codex-562-impl",
  "tmux_session_id": "$42",
  "create_source": "tmux_new_session_print",
  "liveness": "live"
}
```

This keeps boot behavior aligned with the rest of #561: tmux stable id/name are
the source of truth for creation, and pane diagnostics explain only genuine
managed-process death.

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
      "death_kind": "absent_session_id",
      "death_reason": "stable tmux session id not present"
    },
    {
      "target": "amd1::$69",
      "last_observed_agent": "codex-535",
      "death_kind": "reused_session_id",
      "death_reason": "tracked agent gone; stable id now belongs to codex-541",
      "reused_by_agent": "codex-541"
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
surface for drift and death. Doctor and scan output must distinguish
`absent_session_id` from `reused_session_id`; reused-id reports include the
current live tmux name that owns the stable id.

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
which records were quarantined/pruned in its JSON output, and whether each
confirmed-dead record was `absent_session_id` or `reused_session_id`. It must
not classify an unreachable host as a set of dead sessions.

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
2. `libs/tmux` runs `tmux new-session -d -P -F` and captures the created
   `session_id` and `session_name` from the command output.
3. `libs/tmux` constructs the returned `Target` from that output and returns it
   keyed by stable id; no post-create `list_sessions()` proof is needed.
4. mstream normalizes the target to `host::$N`, writes assignment tags, and
   starts normal monitoring/reconciliation.
5. If the managed agent process exits during the bounded diagnostic window,
   mstream captures remain-on-exit pane output and process fields before
   returning `agent_boot_failed`.
6. Creation fidelity and process-exit diagnostics stay separate.

### Reconcile on read

1. mstream snapshots all relevant `SessionTarget`s.
2. Targets are grouped by host.
3. Each host is inventoried once under a per-host timeout.
4. A timeout or error marks only that host rows as `unreachable`; other hosts
   continue rendering.
5. Returned sessions are indexed by `SessionInfo.id`.
6. Records are updated with live facts for present sessions.
7. Rows are rendered with canonical `agent`, drift, and liveness fields.
8. Confirmed dead and unreachable are surfaced distinctly.

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
errors include create-session fidelity and genuine boot-exit diagnostic fields:

```json
{
  "type": "ok",
  "op": "new",
  "workstream": "issue-561-mstream-fidelity",
  "target": "mac1::$42",
  "agent": "codex-562-impl",
  "tmux_session_id": "$42",
  "create_source": "tmux_new_session_print",
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
  "tmux_session_id": "$42",
  "tmux_present": true,
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
mstream session list --cached        # alias: --no-reconcile
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

`--cached` / `--no-reconcile` is the high-frequency poll path. It skips host
I/O, never blocks on a slow host, and renders the last observed ledger with
`reconciled=false` and `agent_source=last_observed`. The default path reconciles
with live tmux and marks only the timed-out host rows as `unreachable`.

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
executable and `identity` for a cached name. The known in-repo consumer is the
project/orchestrator skill and its mstream command references, which currently
read `agent` from status/session rows. During the one-release alias window,
update that skill and related CLI docs to consume canonical `agent` and
`agent_executable` explicitly.

Migration plan:

1. Add internal fields for live tmux name, assigned handle, and executable.
2. Keep reading existing `@mstream/identity` tags as `self_reported_handle`.
3. Emit new canonical fields in all changed commands.
4. Move executable metadata to `agent_executable`.
5. Keep deprecated aliases for one release only if needed by known callers.
6. Update `bins/mstream/docs/API.md` and `PLAN.md` during implementation.
7. Remove any remaining code path that treats cached identity as canonical.
8. Replace the string-matched `is_created_session_not_found()` startup-error path
   with parsed `new-session -d -P -F` create output and the selected
   remain-on-exit diagnostics surfaced through `libs/tmux` / `mstream new`.

No tmux tag migration is required for existing sessions. The first scan or
status after the change derives `live_name` from tmux and reuses existing tags
for `self_reported_handle`.

## Components To Test

### mstream unit tests

- `SessionRecord::observe_tmux_session` updates live name without overwriting
  `self_reported_handle`.
- `session list` reconciles and renders `agent`, `self_reported_handle`,
  `name_drift`, and `agent_executable`.
- `session list --cached` / `--no-reconcile` skips host I/O and marks
  `reconciled=false` with last-observed identity fields.
- `status` updates live name from `LiveActivity::Present`.
- live vs dead vs unreachable classification maps to `tmux_present` and
  `activity_hint` correctly, including host-local timeout isolation.
- reused stable session ids render `death_kind=reused_session_id` and the
  observed replacement live name, distinct from absent ids.
- `mstream rename` keeps the stable target, calls tmux rename before daemon
  mutation, and returns drift-free fields.
- `mstream new` keys the daemon record from the stable id returned by
  `new-session -d -P -F`, without requiring a post-create `list_sessions()`
  success.
- `mstream new` reports `agent_boot_failed` with bounded pane output and exit
  diagnostics from the remain-on-exit path on genuine startup failure.
- scan rehydrates by `SessionInfo.name` even when `@mstream/identity` is stale.
- OutputBus `TargetOutput` with a stable session id refreshes `live_name` and
  routes output after an out-of-band rename.
- doctor summary counts live, drift, dead, and unreachable rows.
- `doctor --quarantine-dead` never applies to unreachable rows.

### libs/tmux tests

- new inventory API distinguishes successful empty inventory from tmux server
  unavailable when tmux reports "no server running" or equivalent.
- existing `HostHandle::list_sessions()` behavior remains backward compatible.
- `HostHandle::create_session()` uses `new-session -d -P -F` to capture the
  created session id/name and returns a stable-id target without calling
  `list_sessions()` to prove creation.
- malformed or missing `new-session -P -F` output returns a typed or inspectable
  parse/state error instead of falling back to a racing name lookup.
- remain-on-exit startup diagnostics preserve failed pane output for genuine
  process exit without being part of the creation liveness path or leaving
  healthy sessions held after successful boot.
- `Target::rename()` continues using stable session id in the tmux command.
- mock transport coverage for host command errors and no-server diagnostics.

### integration/smoke tests

- Create a tmux session, join it, out-of-band `tmux rename-session`, then verify
  `mstream status` and `session list` show the new `agent` and
  `name_drift=true`.
- Run `mstream rename` and verify live tmux `list-sessions` shows the new name,
  while mstream target remains `host::$N`.
- Kill a joined tmux session and verify `liveness=dead` on the next reconcile.
- Simulate SSH/host failure and verify `liveness=unreachable`, not dead, while
  other hosts still render normally.
- Run `mstream session list --cached` during a simulated hanging host inventory
  and verify it returns without host I/O.
- Scan after daemon restart and verify live tmux names are ground truth.
- Doctor reports drift/dead/unreachable and distinguishes absent ids from
  reused ids without direct tmux probing.
- Create a session in a test harness where any post-create `list_sessions()`
  response would be empty; verify mstream still records `host::$N` from
  `new-session -d -P -F` output and does not report `agent_boot_failed`.
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
- Boot diagnostics can leave failed panes or `remain-on-exit` options behind if
  cleanup fails. Mitigate with bounded diagnostic retention, explicit cleanup
  after live boot, and tests that prove healthy agents are not shell-wrapped.
