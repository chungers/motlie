# Plan: mstream Agent Workstreams

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-22 | @codex | Aligned timeline implementation tasks with PR #326's concrete OutputBus timeline APIs and latest-cursor/backfill/cleanup contracts. |
| 2026-05-22 | @codex | Addressed PR #324 review: added Communication & Handoff implementation phase, explicit completion state ownership, timeline dependency gates, and status/process-state follow-ups. |
| 2026-05-22 | @codex | Synced PLAN with issue #323 feedback: use `open`/`close`, add close-time agent availability/context tags, and require `recruit --goal` matching. |
| 2026-05-21 | @codex | Initial implementation plan for issue #323, covering scaffolding, stateless daemon/client, host connection, tmux-tag hydration, workstream commands, recruiting, observation, and validation. |

## Scope

This plan implements the design in [`DESIGN.md`](./DESIGN.md) for
[issue #323](https://github.com/chungers/motlie/issues/323). The first
deliverable is a usable agent-facing CLI/daemon. It may expose arrival-order
merged timelines until [issue #322](https://github.com/chungers/motlie/issues/322)
adds OutputBus timelines with the additional mutable-filter, filtered-continuity,
and cursor guarantees required by workstream observation.

## Phase 0. Project Setup And Traceability

Design references:

- [Status](./DESIGN.md#status)
- [Business Problem](./DESIGN.md#business-problem)
- [Requirements](./DESIGN.md#requirements)

Tasks:

- [x] 0.1 Create the product issue documenting the business problem,
  requirements, and selected design: GitHub issue #323.
- [x] 0.2 Create `bins/mstream/docs/DESIGN.md` with the greenfield scope,
  non-goals, requirements, and CLI/system design.
- [x] 0.3 Create this `bins/mstream/docs/PLAN.md` with traceable implementation
  phases and validation gates.
- [ ] 0.4 Add follow-up issues or DESIGN notes for known `libs/tmux` gaps:
  session start-directory support, pane/process-state status for better stuck
  hints, mutable OutputBus timeline filters, filter-respecting timeline
  continuity markers, and bounded timestamp-merge cursor safety.

Validation:

```sh
git diff --check -- bins/mstream/docs/DESIGN.md bins/mstream/docs/PLAN.md
```

## Phase 1. Binary Scaffolding

Design references:

- [Binary And Layering](./DESIGN.md#binary-and-layering)
- [Daemon Lifecycle](./DESIGN.md#daemon-lifecycle)
- [Observation Commands](./DESIGN.md#observation-commands)

Tasks:

- [ ] 1.1 Add `bins/mstream/Cargo.toml` with package name
  `motlie-mstream` and binary name `mstream`.
- [ ] 1.2 Add `bins/mstream/main.rs` with a `clap` command tree and no
  side-effecting command behavior yet.
- [ ] 1.3 Register `bins/mstream` in the workspace root `Cargo.toml`.
- [ ] 1.4 Add internal modules for `cli`, `protocol`, `daemon`, `jsonl`,
  `target`, `tags`, `workstream`, `hosts`, and `timeline`.
- [ ] 1.5 Define a consistent JSONL envelope for success, error, status, and
  event records.
- [ ] 1.6 Make all diagnostics go to stderr and all machine-facing command
  output go to stdout as JSONL.

Validation:

```sh
cargo check -p motlie-mstream
cargo run -p motlie-mstream -- --help
cargo run -p motlie-mstream -- daemon --help
```

## Phase 2. Socket Protocol And Stateless Daemon

Design references:

- [Daemon Lifecycle](./DESIGN.md#daemon-lifecycle)
- [Host Ledger](./DESIGN.md#host-ledger)
- [State And Recovery Requirements](./DESIGN.md#state-and-recovery-requirements)

Tasks:

- [ ] 2.1 Implement `mstream daemon start --socket <path>` so it daemonizes by
  default after the socket is ready; provide `--foreground` for development and
  tests.
- [ ] 2.2 Implement `mstream daemon status` and `mstream daemon stop`.
- [ ] 2.3 Implement a local Unix-domain socket protocol using one JSON request
  per line and one or more JSONL response records per request.
- [ ] 2.4 Add client connection handling for `--socket`, `MSTREAM_SOCKET`, and
  the documented default socket path.
- [ ] 2.5 Return a structured JSONL error when the daemon is unreachable,
  including the socket path attempted.
- [ ] 2.6 Keep daemon state entirely in memory: no state directory, no database,
  no host config file, no workstream ledger file.
- [ ] 2.7 Add shutdown cleanup for the socket path the daemon owns.

Validation:

```sh
cargo test -p motlie-mstream daemon
cargo run -p motlie-mstream -- daemon start --socket /tmp/mstream-test.sock
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock daemon status
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock daemon stop
```

## Phase 3. Target Parsing And JSONL Contracts

Design references:

- [Session Target Syntax](./DESIGN.md#session-target-syntax)
- [Observation Commands](./DESIGN.md#observation-commands)
- [Public Abstraction](./DESIGN.md#public-abstraction)

Tasks:

- [ ] 3.1 Implement strict parsing for `<host-alias>::<tmux-session-name>`.
- [ ] 3.2 Reject empty host aliases, empty session names, and malformed target
  strings with actionable JSONL errors.
- [ ] 3.3 Define opaque cursor types for workstream event streams. The encoding
  is owned by `mstream`; it must not expose `std::time::Instant` or depend on
  `libs/tmux` timeline serde support.
- [ ] 3.4 Add snapshot-safe text fields that preserve content without control
  characters leaking into JSONL.
- [ ] 3.5 Add golden tests for JSONL output shape for representative success
  and failure cases.

Validation:

```sh
cargo test -p motlie-mstream target
cargo test -p motlie-mstream jsonl
```

## Phase 4. Host Connect, Scan, And Volatile Ledger

Design references:

- [Host Ledger](./DESIGN.md#host-ledger)
- [Hydration Flow](./DESIGN.md#hydration-flow)
- [State And Recovery Requirements](./DESIGN.md#state-and-recovery-requirements)

Tasks:

- [ ] 4.1 Implement `mstream connect <alias> <ssh-uri>` using `motlie-tmux`
  host connection APIs.
- [ ] 4.2 Store host alias, URI, connection handle, optional labels, and
  optional capacity only in daemon memory.
- [ ] 4.3 Implement `mstream hosts`, `mstream scan <alias>`, and
  `mstream disconnect <alias>`.
- [ ] 4.4 Implement host scan: list tmux sessions and build an in-memory
  session ledger keyed by host alias and stable tmux session id.
- [ ] 4.5 Expose scan results as JSONL without persisting them.
- [ ] 4.6 Verify restart semantics manually: after daemon stop/start, `hosts`
  should be empty until `connect` is run again.

Validation:

```sh
cargo test -p motlie-mstream hosts
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock connect local ssh://localhost
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock hosts
```

## Phase 5. Tmux Tag Schema And Hydration

Design references:

- [Tmux Tag Schema](./DESIGN.md#tmux-tag-schema)
- [Hydration Flow](./DESIGN.md#hydration-flow)
- [Workstream Commands](./DESIGN.md#workstream-commands)

Tasks:

- [ ] 5.1 Implement typed `MstreamTags` serialization/deserialization for
  `@mstream/*` session tags.
- [ ] 5.2 Use `HostHandle::list_tags_for_session_infos("mstream", sessions)`
  for batch hydration.
- [ ] 5.3 Implement tag writes for `open`, `join`, `new`, `close`, state
  changes, role/agent metadata, small `last-report-*` fields, and reusable
  agent context metadata.
- [ ] 5.4 Implement tag unsets for `leave` and close-time clearing of active
  workstream membership.
- [ ] 5.5 Preserve unknown `@mstream/*` tags unless a command explicitly owns
  that key.
- [ ] 5.6 Add tests for malformed tag values, missing version, unknown version,
  and partial metadata.
- [ ] 5.7 Add tests for domain/specialty/context-summary tags used by
  `recruit --goal`.
- [ ] 5.8 Add tests for state values `available`, `reserved`, `busy`, `idle`,
  `done`, `blocked`, and `needs-input`, including ownership rules that prevent
  output silence from becoming completion.

Validation:

```sh
cargo test -p motlie-mstream tags
cargo test -p motlie-tmux session_tags
```

## Phase 6. Workstream Commands

Design references:

- [Workstream Commands](./DESIGN.md#workstream-commands)
- [Public Abstraction](./DESIGN.md#public-abstraction)
- [Hydration Flow](./DESIGN.md#hydration-flow)

Tasks:

- [ ] 6.1 Implement `mstream open <workstream> --title <title>
  [--goal <goal>] [--domain <domain>]` as an in-memory workstream until
  sessions are joined.
- [ ] 6.2 Implement `mstream list`, `mstream show <workstream>`, and
  `mstream close <workstream>`.
- [ ] 6.3 Implement `mstream join <workstream> <target> --role <role>
  [--task <text>]`.
- [ ] 6.4 Make `join` write session tags before sending a task prompt.
- [ ] 6.5 Include the managed-agent reporting contract in `join` task prompts:
  `mstream session mark self --state done|blocked|needs-input --summary ...`.
- [ ] 6.6 Implement `mstream leave <workstream> <target>` by removing
  workstream membership tags while preserving non-workstream metadata.
- [ ] 6.7 Implement `mstream kill <target>` as a separate explicit destructive
  command.
- [ ] 6.8 Ensure each state-changing command returns a JSONL cursor or enough
  metadata for the orchestrating agent to poll next.
- [ ] 6.9 Make `mstream close <workstream>` mark participating agents available,
  clear active workstream membership, and merge optional `--summary`,
  `--domain`, and repeated `--specialty` values into reusable context tags.

Validation:

```sh
cargo test -p motlie-mstream workstream
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock open pr-323 --title "mstream" --goal "Build mstream orchestration"
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock show pr-323
```

## Phase 7. New Agent Session Bootstrap

Design references:

- [Workstream Commands](./DESIGN.md#workstream-commands)
- [Session Target Syntax](./DESIGN.md#session-target-syntax)
- [Non-Functional Requirements](./DESIGN.md#non-functional-requirements)

Tasks:

- [ ] 7.1 Implement `mstream new <workstream> <target> --role <role>
  --cwd <abs-path> --agent <binary> [--task <text>]`.
- [ ] 7.2 Reject relative `--cwd` values.
- [ ] 7.3 Build a narrow shell bootstrap for `mkdir -p`, `cd`, and
  `exec <agent>` with validated and escaped arguments.
- [ ] 7.4 Start the tmux session with the target session name as the agent's
  operational identity.
- [ ] 7.5 Set initial environment variables such as `MSTREAM_SOCKET`,
  `MSTREAM_WORKSTREAM`, `MSTREAM_TARGET`, and `MSTREAM_ROLE` when useful.
- [ ] 7.6 Send an initial prompt that explicitly states the agent identity,
  role, workstream, cwd, task, and completion/report command contract.
- [ ] 7.7 Add a design follow-up if `CreateSessionOptions::start_directory`
  should be added to `libs/tmux` instead of using the bootstrap command.

Validation:

```sh
cargo test -p motlie-mstream bootstrap
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock new pr-323 local::codex-test --role reviewer --cwd /tmp/mstream-pr-323-reviewer --agent codex --task "Report status."
```

## Phase 8. Communication, Completion, And Handoff

Design references:

- [Communication And Handoff](./DESIGN.md#communication-and-handoff)
- [Workstream Commands](./DESIGN.md#workstream-commands)
- [Tmux Tag Schema](./DESIGN.md#tmux-tag-schema)

Tasks:

- [ ] 8.1 Implement `mstream send <workstream> <target> --text <text>
  (--enter|--no-enter)` using typed `motlie-tmux` send APIs.
- [ ] 8.2 Implement multi-line send behavior with explicit
  `--paste-mode bracketed|literal`, and report the effective paste mode in
  JSONL.
- [ ] 8.3 Implement `send --require-state <state>` and `send --set-state busy`
  so new assignments can update state atomically without guessing from output.
- [ ] 8.4 Implement `mstream interrupt <target> [--key esc|ctrl-c]` as a
  non-destructive command distinct from `kill`.
- [ ] 8.5 Implement `mstream send --interrupt-first [--settle-ms N]` as one
  daemon-side sequence: interrupt, wait, text, optional Enter, JSONL result.
- [ ] 8.6 Implement `mstream broadcast <workstream> --text <text>` with
  optional `--role` and `--state` filters and one result record per target.
- [ ] 8.7 Implement `mstream session mark <target|self> --state
  done|blocked|needs-input|available|reserved|busy|idle --summary <text>`.
- [ ] 8.8 Make `self` resolve from `MSTREAM_TARGET`; update `@mstream/state`,
  `@mstream/last-report-kind`, `@mstream/last-report-summary`, and
  `@mstream/updated-at` on successful marks.
- [ ] 8.9 Emit structured events for `message_sent`, `interrupted`,
  `broadcast_sent`, `completed`, `blocked`, and `needs_input`.
- [ ] 8.10 Implement `mstream handoff arm/list/cancel` as daemon-memory state
  that fires when a source target reaches a requested terminal state.
- [ ] 8.11 Ensure handoff firing sends the configured task to the destination,
  emits `handoff_fired`, and does not claim durability across daemon restart.
- [ ] 8.12 Add tests that silence, prompt heuristics, and missing output do not
  transition a session to `done`, `blocked`, or `needs-input`.

Validation:

```sh
cargo test -p motlie-mstream communication
cargo test -p motlie-mstream handoff
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock send pr-323 local::codex-test --text "Report status." --enter
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock interrupt local::codex-test
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock session mark local::codex-test --state done --summary "manual smoke completed"
```

## Phase 9. Monitoring, Timelines, And Observation

Design references:

- [Observation Commands](./DESIGN.md#observation-commands)
- [Timeline Model](./DESIGN.md#timeline-model)
- [Testing Strategy](./DESIGN.md#testing-strategy)

Tasks:

- [ ] 9.1 Use `motlie-tmux` Fleet/OutputBus to start monitoring joined
  sessions.
- [ ] 9.2 Maintain per-workstream in-memory ring buffers with opaque cursors.
- [ ] 9.3 Implement `mstream status <workstream>`.
- [ ] 9.4 Implement `mstream events <workstream> --after <cursor> --limit N`.
- [ ] 9.5 Implement `mstream snapshot <workstream> --after <cursor>
  --max-chars N`.
- [ ] 9.6 Implement `mstream summary-input <workstream> --since <duration>
  --max-chars N` with server-side filtering/compaction.
- [ ] 9.7 Mark ordering as arrival-order in JSONL metadata until issue #322's
  OutputBus timeline API is available on the target branch with create-or-get,
  mutable filters, scoped markers, history ingest, stale-handle cleanup, and
  bounded `latest` cursor safety.
- [ ] 9.8 Make `status` include explicit session state, last report summary,
  last output age, monitor health, and process/prompt-based stuck hints when
  available.
- [ ] 9.9 Keep stuck hints separate from explicit completion states in JSONL.
- [ ] 9.10 Remove or detach a workstream timeline on `close` and when `leave`
  removes the last session from an otherwise empty/closed workstream.
- [ ] 9.11 After issue #322 lands with the required APIs, replace the local
  ring-buffer/timeline layer with the `libs/tmux` OutputBus-backed timeline API
  where possible. Use `create_or_get_timeline` for hydration, `set_filters` /
  `add_filter` for dynamic membership, scoped gap/discontinuity APIs for
  continuity markers, `ingest_historical` for restart backfill,
  `remove_timeline`/`detach`/idle cleanup for lifecycle, and the timeline cursor
  contract for `entries_after`, `render_after`, and bounded `latest`.

Validation:

```sh
cargo test -p motlie-mstream timeline
cargo test -p motlie-mstream observation
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock status pr-323
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock events pr-323 --limit 20
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock summary-input pr-323 --max-chars 12000
```

## Phase 10. Recruiting And Availability

Design references:

- [Recruiting](./DESIGN.md#recruiting)
- [Host Ledger](./DESIGN.md#host-ledger)
- [Tmux Tag Schema](./DESIGN.md#tmux-tag-schema)

Tasks:

- [ ] 10.1 Implement `mstream session list`. `session mark` is implemented in
  Phase 8 because it is part of the completion/report channel.
- [ ] 10.2 Implement `mstream recruit <workstream> --role <role>
  --agent <agent> --count N [--goal <goal>] [--selector key=value]
  [--task <text>]`.
- [ ] 10.3 Prefer explicitly tagged available sessions.
- [ ] 10.4 Refuse to recruit untagged sessions unless the target is explicitly
  named by the human.
- [ ] 10.5 Score available sessions against `--goal` using structured/lexical
  matching over `context-domains`, `context-specialties`, `context-summary`,
  and `last-workstream` tags; include candidate metadata in JSONL so the
  orchestrating agent can make a semantic selection.
- [ ] 10.6 Add optional creation only when placement, cwd/work-root, and capacity
  information are explicitly available in the daemon's current memory.
- [ ] 10.7 Return actionable JSONL errors when recruiting cannot proceed because
  host metadata is unknown after restart.

Validation:

```sh
cargo test -p motlie-mstream recruit
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock session list
cargo run -p motlie-mstream -- --socket /tmp/mstream-test.sock recruit pr-323 --role reviewer --agent codex --count 1 --goal "clean up vmm examples" --task "Review the current branch."
```

## Phase 11. Documentation And Agent Skill Follow-Up

Design references:

- [Daemon Lifecycle](./DESIGN.md#daemon-lifecycle)
- [Host Ledger](./DESIGN.md#host-ledger)
- [Observation Commands](./DESIGN.md#observation-commands)

Tasks:

- [ ] 11.1 Add `bins/mstream/docs/API.md` or `CLI.md` after the implemented
  command behavior exists.
- [ ] 11.2 Document daemon restart recovery with exact orchestrating-agent
  behavior: ask the human to restart/provide socket, then ask for host aliases
  and SSH URIs, reconnect, rescan.
- [ ] 11.3 Update the project skill only after the implemented CLI has been
  validated, so the skill instructions match reality.
- [ ] 11.4 Add examples for one submitter plus one reviewer workstream,
  including send, mark-done, handoff, and reviewer re-review.
- [ ] 11.5 Add a release-note candidate entry only if `mstream` is intended for
  the next release.

Validation:

```sh
rg -n "mstream connect|daemon restart|summary-input" bins/mstream/docs
```

## End-To-End Validation Plan

The first acceptable implementation should pass:

```sh
cargo fmt --check
cargo test -p motlie-mstream
cargo check -p motlie-mstream
cargo clippy -p motlie-mstream --all-targets -- -D warnings
```

Manual local tmux smoke:

```sh
cargo run -p motlie-mstream -- daemon start --socket /tmp/mstream-smoke.sock
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock connect local ssh://localhost
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock open pr-323 --title "mstream smoke" --goal "validate mstream local orchestration"
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock new pr-323 local::mstream-smoke-reviewer --role reviewer --cwd /tmp/mstream-smoke-reviewer --agent codex --task "Say ready, then wait."
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock send pr-323 local::mstream-smoke-reviewer --text "Mark done after this." --enter --set-state busy
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock session mark local::mstream-smoke-reviewer --state done --summary "local send/mark smoke completed"
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock status pr-323
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock summary-input pr-323 --max-chars 4000
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock close pr-323 --summary "local smoke completed" --domain tmux --specialty mstream-smoke
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock daemon stop
```

Restart recovery smoke:

```sh
cargo run -p motlie-mstream -- daemon start --socket /tmp/mstream-smoke.sock
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock hosts
# expected: no connected hosts
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock connect local ssh://localhost
cargo run -p motlie-mstream -- --socket /tmp/mstream-smoke.sock scan local
# expected: tagged session membership is hydrated from tmux
```
