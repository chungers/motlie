---
name: project
description: Manage Motlie engineering projects and multi-agent workstreams with mstream. Use when planning or orchestrating issues, PRs, workstreams, agent recruitment, reviewer/implementer collaboration, progress monitoring, handoffs, or workstream closeout.
---

# Motlie Project Workstreams

Use this skill when acting as project manager, tech lead, or orchestrator for Motlie work. The primary tool is `mstream`: use it to open workstreams, connect hosts, recruit or create agent sessions, communicate tasks, monitor timelines, summarize progress, unblock collaborators, and close workstreams.

Always follow repo process from `AGENTS.md` or `CLAUDE.md` in the target checkout. If both exist, read both and reconcile; if they conflict, ask the user.

## Operating Charter

Your job is to maximize efficient and timely execution by agents in active workstreams while preserving correctness. Keep agents moving, keep reviewers prepared, unblock issues with the best available context, and surface material risks or delays to humans before they have to ask.

Optimize for:

- velocity: start the right agents, give complete context, avoid idle reviewers, and nudge local work toward pushed PRs
- correctness: enforce project bars, tests, review discipline, and clear ownership
- low human polling: monitor often enough that humans do not need to ask whether work is stalled
- low human intervention: solve local, obvious, reversible blockers yourself; escalate only when credentials, product direction, destructive actions, or external decisions are needed

If an agent appears to be doing dangerous work, such as destructive git operations, editing the wrong worktree, using the wrong remote/base, leaking credentials, broadening scope without approval, or ignoring stated safety/review bars, interrupt the session promptly with `mstream interrupt` or `mstream send --interrupt-first`, then report the risk and mitigation to the human.

## Core Invariants

These rules hold across every section below. They are stated once here; later sections reference them by number instead of restating them.

1. **mstream is orchestrator-only.** Only you run mstream. Never tell a collaborator agent to run mstream, inspect the socket, or manage its own session state. Infer agent state from observable evidence: commits, PRs, posted comments, test output, snapshots, and timeline.
2. **mstream is the coordination boundary.** Do not bypass it with direct `ssh`/`tmux` for liveness, listing, snapshots, monitoring, or messaging. If mstream lacks a needed signal, extend it in the active PR or ask the user before any temporary manual path.
3. **Keep the daemon alive.** Once assigned, keep one daemon on one stable socket across turns. Do not stop or restart it after routine commands or transient SSH errors. Never stop it unless the user explicitly asks. If it seems unreachable, report and ask before replacing it.
4. **Durable context outranks runtime state.** tmux and daemon memory are recoverable runtime, not the source of truth. Persist issue/PR/branch/commit/host/session/role facts to durable places (GitHub, worktrees, notes) so resets are survivable.
5. **Side effects need approval.** Treat `connect`, `open`, `join`/`new`/`recruit`, `send`, `interrupt`, `handoff`, `label`, `leave`, `kill`, `close`, and `daemon start` as side-effecting. If the latest request asks for an outline, staffing, or risk plan, deliver that and wait. Otherwise summarize intended side effects before running them. On failure, stop the sequence, report the exact failure and implication, then decide the next step with the user unless the fix is local and obvious.
6. **Never invent state.** Do not fabricate credentials, host aliases, SSH URIs, issue numbers, or product direction. Ask the user (see Prerequisites To Collect).
7. **Report quietly, by outcome.** Surface material state changes, blockers, risks, and decisions needing a human. Do not narrate routine tool calls or paste raw logs; summarize the relevant result.
8. **TUI submit retry.** Agent TUIs sometimes miss the submit newline. If the timeline shows typed-but-unsubmitted text after a send, wait briefly and send one extra empty `--enter`. This is a targeted retry, not a default double-send.

## Prerequisites To Collect

Before opening a workstream, gather these. Ask the user for anything missing (Invariant 6):

- **Hosts:** alias(es), SSH URI(s), and per-host `--work-root` (`/home/dchung/sessions` on Linux, `/Users/dchung/sessions` on macOS).
- **Work scope:** GitHub issue number(s), workstream goal, and greenfield vs. brownfield in the product sense.
- **Repository:** clone/worktree base. Default Motlie URL is `https://github.com/chungers/motlie.git` unless the user specifies otherwise.
- **Socket:** the `MSTREAM_SOCKET` value for the run (default `/tmp/mstream-${USER}.sock`).
- **Agent runtime:** executable name or path per agent (`claude`, `codex`, or a workstream-local wrapper) and the approved permission mode.
- **GitHub auth:** confirm agents can post to GitHub (`gh` auth). Treat missing auth as a foreseeable blocker, not a surprise mid-review.

## mstream Command Reference

Authoritative surface, pinned to mstream **v0.1.0** (motlie `main` @ `b1799d6`). Every command also accepts `--socket <SOCKET>`; prefer `MSTREAM_SOCKET` or the default socket path over repeating the flag. Consult `--help` only if the installed binary differs from this table (Invariant: do not rediscover basic shapes during normal orchestration).

Shared enums:

- **session state** (`--state`, `--set-state`, `--require-state`, `handoff --on`): `available`, `reserved`, `busy`, `idle`, `done`, `blocked`, `needs-input`.
- **activity_hint** (`status` output, not an input): `active`, `quiet`, `idle`, `missing`, `unknown`.
- **`--agent`** is a free-form executable name/path, not an enum (e.g. `claude`, `codex`, or a wrapper).
- **duration args** (`--every`, `--input-quiet-for`) accept suffixes `s`/`sec`/`secs`/`min`/`m` (e.g. `30s`, `5m`) or bare seconds; `*-ms`/`*-secs` integer flags are raw numbers.

| Group | Command | Purpose | Notable args (defaults) |
|---|---|---|---|
| Daemon | `daemon start` | Start daemon | `--foreground` |
| | `daemon status` | Check daemon liveness | |
| | `daemon stop` | Stop daemon (only on user request) | |
| Hosts | `connect <alias> <ssh-uri>` | Register a host | `--work-root`, `--label k=v` (repeatable), `--capacity` |
| | `hosts` | List connected hosts | |
| | `scan <alias>` | Hydrate tagged tmux sessions | |
| | `disconnect <alias>` | Drop a host | |
| Workstream | `open <ws> --title <t>` | Open workstream | `--goal`, `--domain`, `--mmux-label`, `--event-limit` (1000) |
| | `label <ws> --mmux-label <l>` | Set/replace mmux label | |
| | `list` | List workstreams | |
| | `show <ws>` | Workstream detail | |
| | `close <ws>` | Close workstream | `--summary`, `--domain`, `--specialty` (repeatable), `--stop-timers`, `--standby-agents` |
| Sessions | `new <ws> <target> --role --cwd --agent` | Create a session | `--task` |
| | `join <ws> <target> --role` | Join existing session | `--task` |
| | `recruit <ws> --role` | Recruit tagged agents | `--agent`, `--count` (1), `--goal`, `--selector k=v` (repeatable), `--task` |
| | `leave <ws> <target>` | Remove from workstream, keep tmux | `--available` |
| | `kill <target>` | Destroy tmux session (terminal) | |
| | `session list` | List sessions | |
| | `session mark <target> --state --summary` | Annotate session state | |
| Messaging | `send <ws> <target> --text` | Send to one session | `--enter`/`--no-enter`, `--interrupt-first`, `--settle-ms` (500), `--paste-mode bracketed\|literal`, `--require-state`, `--set-state` |
| | `broadcast <ws> --text` | Send to many | `--enter`/`--no-enter`, `--role`, `--state`, `--paste-mode` |
| | `interrupt <target>` | Non-destructive interrupt | `--key esc\|ctrl-c` (esc) |
| Handoff | `handoff arm <ws> --from --to --on <state> --task` | Conditional sequencing | `--only-on-transition` |
| | `handoff list <ws>` / `handoff cancel ...` | Inspect/cancel armed handoffs | |
| Timers | `timer start <name> --every <dur> --prompt <p>` | Self-wakeup timer | `--workstream`, `--self`/`--self-host` (local) or `--target`, `--enter`/`--no-enter`, `--submit-retries` (1), `--submit-retry-delay-ms` (750), `--input-quiet-for` (10s), `--no-input-guard` |
| | `timer list` | List timers | `--workstream` |
| | `timer stop <name>` / `timer fire <name>` | Stop / test-fire a timer | |
| Observation | `status <ws>` | Liveness snapshot | `--active-window-secs` (30), `--idle-after-secs` (300) |
| | `events <ws>` | Timeline | `--limit` (200), `--readable`, `--after` |
| | `snapshot <ws>` | Pane capture | `--max-chars` (12000), `--after` |
| | `summary-input <ws>` | Input/activity summary | `--max-chars` (12000), `--since` |

`<target>` is `<host-alias>::<tmux-session>` (e.g. `amd1::opus47-337-rv`).

## Identity

Before acting as orchestrator, establish your identity:

1. Use the identity explicitly given by the user, for example `You are "@codex-project-manager"`.
2. If not given, look for a `WHOAMI` file in the current directory or child directories.
3. If still unknown, ask whether to use the current tmux session name or a user-provided identity.

To get the current tmux session name:

```sh
tmux display-message -p '#S'
```

When a tmux session name is the identity, use `@{session_name}` in comments, issues, PRs, changelogs, and handoffs.

## Workstream Intake

Workstreams usually map to one or more GitHub issues. Before opening a workstream:

- identify the issue(s), PR(s), project area, and user goal
- ask for the issue number if it is not known
- ask whether the work is greenfield or brownfield in the product sense
- if no issue exists yet, enter brainstorm/plan mode before DESIGN and PLAN
- create one or more GitHub issues that capture the problem, goal, high-level approach, non-goals, and strategy

Use issue-first language in the workstream name and agent session names. Prefer names like `issue-337-tmux-fleet-api` or `pr-330-mstream-review`.

Decision checkpoint: apply Invariant 5. If the latest request asks for an
outline, staffing proposal, risk assessment, or bars to enforce, deliver that
and wait for approval before opening workstreams, connecting hosts, or creating
sessions.

## mstream Daemon And Hosts

Client commands use JSONL stdout. The socket resolves from `--socket`, then
`MSTREAM_SOCKET`, then `/tmp/mstream-${USER}.sock`. Use `MSTREAM_SOCKET` for
normal orchestration so client commands stay short; do not use `MSTREAM_SOCK`.

Install the release `mstream` binary on `PATH` from the Motlie checkout before orchestration:

```sh
cargo install --path bins/mstream --locked
```

Ensure Cargo's bin directory, usually `~/.cargo/bin`, is on `PATH`. Use the release binary name `mstream` in orchestration commands, not `cargo run` or `./target/debug/mstream`.

Once assigned as orchestrator for an active workstream, keep the mstream daemon running across turns for the duration of that orchestration assignment. Do not stop and restart it after routine `list`, `status`, `summary-input`, progress checks, transfer delivery, or SSH-channel failures. Never stop the daemon unless the user explicitly asks you to stop it. If the daemon appears failed or unreachable, report the symptom and ask before stopping or replacing it.

Use a stable socket for the whole assignment. In a persistent shell, such as a
human shell or a dedicated tmux daemon shell, export it once:

```sh
export MSTREAM_SOCKET=/tmp/mstream-${USER}.sock
mstream daemon status
```

Normal rehydration/playbook after daemon start or restart:

```sh
mstream connect <host-alias> '<ssh-uri>' --work-root <host-work-root>
mstream scan <host-alias>
mstream list
mstream status <workstream> --active-window-secs 30 --idle-after-secs 300
mstream summary-input <workstream> --max-chars 12000
```

Use the concrete host aliases, SSH URIs, work roots, and workstream names supplied by the user or recovered from durable context. Do not rediscover basic CLI shapes with `--help` during normal orchestration; keep this playbook as the default sequence and inspect help only when a command actually fails because the local binary changed.

Start the daemon when the user asks you to operate mstream and no daemon is running. In Codex/harness sandbox environments, use foreground mode in a managed sandbox exec session as the one normal startup method:

```sh
MSTREAM_SOCKET=/tmp/mstream-${USER}.sock mstream daemon start --foreground
```

Keep that foreground exec session running and issue client commands from
separate exec calls. Record the foreground exec session id in your context if
available. In Codex/harness tool calls, `export MSTREAM_SOCKET=...` in one exec
does not persist to future exec calls. Prefer the default socket path so client
commands need neither `--socket` nor an env prefix; if using a non-default
socket, set `MSTREAM_SOCKET=...` inline for each client command or fall back to
`--socket`.

Do not use `mstream daemon start` or `nohup ... &` as the normal Codex/harness
playbook (they can be reaped by the harness even after reporting success);
foreground mode is the orchestrator default. Outside Codex/harness, daemonized
`mstream daemon start` is still a valid human/manual mode.

Fallback only if the managed foreground exec session is unavailable or lost while the daemon must survive for timers or cross-turn orchestration: run the same foreground daemon inside a dedicated tmux session.

```sh
tmux new-session -d \
  -s mstream-daemon-${USER} \
  -c '<absolute-motlie-worktree>' \
  'export MSTREAM_SOCKET=/tmp/mstream-${USER}.sock; mstream daemon start --foreground'
mstream daemon status
```

Before starting any daemon path, use `mstream daemon status` on the chosen
socket. If it is already running, reuse it. Keep the same socket value for the
full orchestration run once chosen, preferably through `MSTREAM_SOCKET` or the
default socket path rather than repeated `--socket` flags. Keep it alive across
requests (Invariant 3).

When assigned as orchestrator and the harness has no first-class cron, start a
daemon-owned self-wakeup timer targeted at your own tmux session. The timer
sends a prompt to you through tmux, which creates a queued self-reminder to
poll and unblock the workstream. Start this timer immediately after agents are
recruited or joined for an active workstream:

```sh
mstream timer start <workstream>-poll \
  --every 5m \
  --workstream <workstream> \
  --self \
  --prompt "[mstream:<workstream>-poll] Wakeup: check <workstream> with mstream status and summary-input. Unblock agents, summarize only material changes, and decide whether to keep, change, or stop this timer." \
  --submit-retries 1 \
  --submit-retry-delay-ms 750
```

If the installed `mstream` does not support `--self`, fall back to resolving
your tmux session name and using an explicit target:

```sh
tmux display-message -p '#S'
mstream timer start <workstream>-poll \
  --every 5m \
  --workstream <workstream> \
  --target <orchestrator-host-alias>::<your-tmux-session> \
  --prompt "[mstream:<workstream>-poll] Wakeup: check <workstream> with mstream status and summary-input. Unblock agents, summarize only material changes, and decide whether to keep, change, or stop this timer."
```

Use `mstream timer list --workstream <workstream>` to verify active scoped
timers, `mstream timer fire <name>` to test prompt delivery, and
`mstream timer stop <name>` when the workstream is closed or no longer needs
periodic attention. Timer state is daemon memory only and must be recreated
after daemon restart. Do not target collaborator sessions with orchestrator
timers unless the user explicitly asks for that behavior.
Timer prompts default to one extra Enter after 750ms because agent TUIs
occasionally miss the first submit key. Retries send only extra Enter keys, not
the prompt text; `--no-enter` disables retries. Timer delivery also defaults to
an input-quiet guard (`--input-quiet-for 10s`): if an attached client typed in
the target session recently, mstream defers the timer and reports the deferral
in `timer list` instead of interleaving prompt text with user input. Use
`--no-input-guard` only when you explicitly want unattended delivery to ignore
attached-client input.

The input-quiet guard applies to key/text delivery, not observation. Keep
polling workstreams with `status`, `events`, `snapshot`, `summary-input`, and
`timer list` even when a timer is deferring because recent user input was
detected.

If the daemon is unreachable, ask the user whether to restart it or provide the correct socket. Do not stop an existing daemon on your own. After daemon restart, ask the user for the host aliases and SSH URIs; mstream does not persist the host ledger.

Keep mstream as the orchestration boundary (Invariant 2) and keep user-facing
updates quiet and outcome-focused (Invariant 7): include command-level detail
only when the user explicitly asks, and summarize relevant results instead of
pasting full logs.

For timer wakeups, keep responses concise. If a timer is stale, first verify
`mstream timer list`; if the timer still exists, stop it. If it is already gone,
treat the wakeup as previously queued input and do not keep re-reporting the
same stale state unless the user asks. For active workstream timers, report only
material changes since the last poll, such as a PR update, review verdict,
blocker, merge, issue closeout, timer change, or agent state change.

Only the orchestrator runs mstream (Invariant 1).

Connect hosts only from human-provided aliases and SSH URIs. Set `--work-root` for the target host filesystem, not the orchestrator's local machine. Common defaults:

- macOS hosts: `/Users/dchung/sessions`
- Linux hosts: `/home/dchung/sessions`

```sh
mstream connect amd1 ssh://amd1 --label pool=amd --work-root /home/dchung/sessions
mstream connect local ssh://localhost --label pool=local --work-root /Users/dchung/sessions
mstream hosts
mstream scan amd1
```

Use `scan` after connecting or daemon restart to hydrate tagged sessions from tmux.

If `scan` reports no tmux server or no sessions, do not silently infer available capacity. Tell the user there are no discoverable existing sessions on that host and ask whether to create fresh sessions, unless the latest user instruction already explicitly approved fresh session creation.

Do not treat `mstream scan` as an exhaustive inventory of every tmux session on
a host. `scan` hydrates sessions visible to the connected tmux socket and may
only import sessions with mstream metadata. When the user asks for "all
sessions" on a host, first perform a coverage audit: identify every relevant
tmux socket/server for that host, connect each socket as a distinct mstream host
alias when needed, and compare the raw session names against `mstream session
list`. Prefer mstream/lib-tmux primitives for this inventory; if mstream lacks a
raw session listing primitive, state the gap and use the smallest diagnostic
needed to discover socket/session names, then perform transfer and monitoring
through mstream. Do not declare a host fully transferred until every discovered
session is either transferred, explicitly excluded, or recorded as needing
targeted inspection.

If fresh session creation is approved but the remote host has no tmux server,
try `mstream new`; tmux should create its server as part of creating the
session. If mstream cannot bootstrap the session, treat that as an mstream
capability gap to fix or escalate to the user instead of running ad hoc
SSH/tmux commands.

## Rehydrating Lost Sessions

Users may stop tmux sessions or the mstream daemon out of band. Treat tmux and daemon state as recoverable runtime state, not the source of truth. Rehydrate from durable context:

- this conversation and your own notes
- GitHub issues, PRs, branches, and comments
- local worktree directories under `~/sessions/{workstream}/`
- git branch/commit state in each agent checkout
- any available memory tools or project notes that preserve workstream goals, host aliases, SSH URIs, roles, branches, and PR URLs

When sessions are gone:

1. Verify durable work first: inspect the agent cwd, branch, commit, `git status`, remotes, and PR state.
2. Restart or reconnect the mstream daemon with the same socket convention when possible.
3. Ask for or recover the host alias and SSH URI, then run `mstream connect`.
4. If no remote tmux server exists, start a bootstrap tmux session and run `mstream scan`.
5. Reopen the workstream with the known issue/PR title and goal.
6. Recreate only the sessions still needed for the current phase, using the existing per-agent cwd.
7. Send rehydrated assignments that include the durable facts: issue, PR, branch, commit, cwd, role, and next action.
8. Verify with `mstream status` and `mstream summary-input`.

As project manager, keep enough durable context to survive these resets. When a workstream reaches a meaningful transition, record the issue, PR, branch, latest commit, host, session names, role assignments, tests, and blockers in the discussion or an available memory/project note. Do not rely only on tmux scrollback or daemon memory.

## Transferring Or Replacing Work

Use when an agent must be replaced, a host/session is going away, or the user
asks to move ownership of in-flight work. Build transfer from existing mstream
primitives (`send`, `new`/`join`/`recruit`, `session mark`, `leave`, `kill`); do
not request a new `mstream transfer` command — if a primitive gap blocks you,
stop and report it.

The model is simple. **The predecessor writes a packet; you (the orchestrator)
judge it and may interrogate until it is high quality; you hand it to the
successor and may interrogate to gauge readiness. Transfer is complete when you
have determined the successor holds quality context and the correct
work/artifact state to take over the predecessor's duties.** Agents never run
mstream (Invariant 1) — you broker the whole exchange.

Core principles (these always hold; how you get there is flexible):

- **One source -> one successor**, no fan-in. Host-wide transfer is one successor
  per source, finished one at a time.
- **No new or synthetic workstream just to transfer.** The successor inherits the
  predecessor's existing workstream, role, and tags; never invent a transfer
  workstream or label. If the predecessor has none, ask the user which existing
  workstream governs.
- **Durable work first.** mstream moves context, not uncommitted files — the
  predecessor must commit/push, update the PR, or explicitly flag local-only risk
  before it is safe to retire.

Keep names, tags, and state summaries terminal-friendly: ASCII, no spaces, short
and descriptive (e.g. session `gpt55-337-og`, summary `succeeded by gpt55-337-og-2`).

### Usual flow

A workable default, not a rigid script — adapt the order, the depth of
interrogation, and the exact commands to the situation, as long as the core
principles hold and you retire the predecessor only after judging the successor
ready.

1. **Ask the predecessor for a packet:** its work state (issue/PR, branch,
   commit, pushed-vs-local, cwd, dirty state, next action) and a context summary
   a replacement of any model could act on. If the predecessor is itself an
   orchestrator, the packet must also carry operating context (hosts, sockets,
   timers, recent decisions). Don't quarantine or retire it yet.

```sh
mstream send <workstream> <source-target> --interrupt-first --set-state busy \
  --text "Stop new work. Make your work durable (commit/push/PR), then write a succession packet: work state (issue/PR/branch/commit/pushed-vs-local/cwd/next action) and a context summary a replacement can act on." --enter
```

2. **Interrogate until the packet is high quality.** Read it via `summary-input`;
   if durability, next action, or where the code lives is thin, ask the
   predecessor for just that and repeat until satisfied. If the source is gone or
   rate-limited, assemble the best packet from durable facts and mark it
   `INCOMPLETE`.

3. **Create the successor and deliver the packet.** It inherits the predecessor's
   workstream/role/tags (suffix the name, e.g. `gpt55-pm` -> `gpt55-pm-2`) and
   prepares its own checkout under the target work-root. TUIs can drop a `--task`
   sent at the welcome screen, so confirm the prompt with `snapshot` and send the
   packet explicitly (one extra empty `--enter` if it didn't submit, Invariant 8).

```sh
mstream new <workstream> <host>::<successor-session> --role <role> --cwd <abs-cwd> --agent <agent>
mstream send <workstream> <successor-target> --set-state busy --enter \
  --text "<packet> — check out the named repo/branch, read AGENTS.md/CLAUDE.md in the repo root (e.g. ./motlie/, not your cwd), then confirm your workspace state and how you'll continue."
```

4. **Optionally interrogate the successor to gauge readiness:** have it restate
   the context in its own words and report its prepared repo/branch/commit/dirty
   state. Declare transfer complete only when you judge it has the context and
   work state to replace the predecessor.

5. **Quarantine, then retire the predecessor** once the successor is ready: mark
   it `reserved` to keep it out of recruitment, then prefer `leave` (keeps the
   tmux session for later inspection) over `kill` (terminal).

```sh
mstream session mark <source-target> --state reserved --summary "succeeded by <successor-target>; retiring"
mstream leave <workstream> <source-target>   # or: mstream kill <source-target>
```

Record the outcome in the closeout log: source, successor, durable checkpoint,
and any follow-up.

## Opening Workstreams

Open and annotate workstreams with issue-aware names and descriptive titles:

```sh
mstream open issue-337-tmux-fleet-api \
  --title "Issue 337: tmux Fleet API improvements" \
  --goal "Design and implement cross-host Fleet target helpers" \
  --mmux-label "337 fleet" \
  --event-limit 2000
```

When supported by the installed mstream, set `--mmux-label` at open time so
mmux can display and group the active sessions by workstream. Keep the label
short: mstream rejects labels that are more than two whitespace-separated words,
more than 24 display columns, or contain control/Unicode format characters. Use
labels such as `337 fleet` or `PR 330`. If the label needs to change after
sessions have joined, use:

```sh
mstream label issue-337-tmux-fleet-api --mmux-label "337 fleet"
```

mstream owns the mmux workstream label lifecycle. It writes the display label
when sessions join the workstream and clears/restores it on `leave` or `close`.
Do not ask collaborator agents to edit `@mmux/*` tags directly for workstream
labels.

Before recruiting, propose a staffing plan to the user:

- candidate session or new session name
- model or agent type, if known
- role: implementer, reviewer, product manager, release manager, etc.
- why the candidate fits the role based on availability, prior context, metadata, or recent work
- diversity rule: use different agent families where practical, for example Codex implementer and Claude-family reviewer
- distinguish session identity from executable name. For example, use `opus47` in the tmux session name to record the model, but start it with the `claude` CLI executable if that is the available command.

Also include a risk and quality-bar briefing for the team before work starts:

- product mode: greenfield or brownfield, and what that changes
- success metric, especially any downstream simplification the work must enable
- layering boundaries and non-goals
- correctness, safety, error-handling, and test bars
- reviewer expectations and must-fix criteria

Prefer available sessions with relevant experience. If no known session fits, propose creating a new session.

## Agent Session Naming

Session names must include enough context to reconstruct the work:

```text
<model>-<issue-or-pr>-<role-abbrev>
```

Examples:

```text
gpt55-337-og
opus47-337-rv
gpt55-330-prod
```

Common role abbreviations:

- `og`: original author / implementer
- `rv`: reviewer
- `prod`: product manager
- `rel`: release manager
- `qa`: validation / testing

If reassigning a free agent, rename the tmux session when appropriate so the identity matches the workstream. Tell agents to self-identify as `@{session_name}`.

Use the concrete model name in the session name when known, not just the CLI family. For example, name an Opus 4.7 reviewer `opus47-337-rv` even if the command used to start it is `claude`.

## Work Directories And Worktrees

Agent working directories always live under `~/sessions` on the target host. Resolve `~` according to the target host platform and user, for example `/home/dchung/sessions` on Linux hosts and `/Users/dchung/sessions` on macOS hosts.

Use the workstream root as a grouping directory, not as the shared cwd for every agent:

```text
~/sessions/{workstream}/
```

Each agent session gets a unique cwd under the workstream root:

```text
~/sessions/{workstream}/{session_name}/
```

Never put two active agents on the same host in the same cwd or shared repo checkout unless the user explicitly asks for that coordination. This is especially important when an author and reviewers run on the same host.

Stage implementation work in a local checkout or git worktree inside the agent cwd. Prefer a git worktree from an existing Motlie clone when available; otherwise clone the repo, fetch the target base, and create a local working branch. Keep implementation commits local until the work is ready for review.

```text
~/sessions/{workstream}/{session_name}/motlie
```

Use local working branches named with the agent identity and a short scope, for example:

```text
@gpt55-337-og/fleet-target-specs
```

If additional worktree directories are needed for that agent, place them under the agent cwd, not inside a repo checkout.

## Creating Or Joining Agents

Recruit available tagged agents first when the workstream can use existing capacity:

```sh
mstream recruit issue-337-tmux-fleet-api \
  --role reviewer \
  --agent claude \
  --count 1 \
  --goal "Review tmux Fleet API changes for issue #337" \
  --selector pool=amd \
  --task "Review the implementation for issue #337 and post must-fix feedback."
```

Join an existing agent session:

```sh
mstream join issue-337-tmux-fleet-api amd1::opus47-337-rv \
  --role reviewer \
  --task "Review issue #337 and prepare to review the implementation branch."
```

Create a new session:

```sh
mstream new issue-337-tmux-fleet-api amd1::gpt55-337-og \
  --role implementer \
  --cwd /home/dchung/sessions/issue-337-tmux-fleet-api/gpt55-337-og \
  --agent codex \
  --task "Confirm identity from tmux, check out Motlie main into ./motlie, read ./motlie/AGENTS.md (repo root, not your cwd), then prepare the implementation plan for issue #337."
```

For an Opus reviewer on the same workstream, use the model in the target/session name but the Claude CLI executable:

```sh
mstream new issue-337-tmux-fleet-api amd1::opus47-337-rv \
  --role reviewer \
  --cwd /home/dchung/sessions/issue-337-tmux-fleet-api/opus47-337-rv \
  --agent claude \
  --task "Confirm identity from tmux, check out Motlie main into ./motlie, read ./motlie/AGENTS.md (repo root, not your cwd), then review issue #337 and wait for the implementation branch."
```

When starting a new agent, include:

- the GitHub issue(s), PR(s), and goal
- the assigned role and expected deliverable
- whether the work is greenfield or brownfield
- instruction to read `AGENTS.md` or `CLAUDE.md`, and **the path where they live**.
  These sit in the repo root, which is the checkout dir under the agent's cwd
  (e.g. `~/sessions/{workstream}/{session}/motlie/AGENTS.md`), not the agent's
  workspace root. Never tell an agent to "read AGENTS.md" without naming the path,
  or it will look in the wrong directory.
- instruction to check its tmux session name with `tmux display-message -p '#S'`
- instruction to use `@{session_name}` in comments and submissions
- working directory and branch naming expectations
- exact repository URL and issue context. Do not make agents guess repository owner/name; include `https://github.com/chungers/motlie.git` for Motlie work unless the user gives a different URL.
- **Permission mode is MANDATORY INITIAL SETUP for every Codex agent — set it BEFORE assigning any work.** `mstream new --agent` takes a bare executable path (`claude` or `codex`); never pass CLI flags or wrappers. As soon as the Codex TUI is ready (confirm with `mstream snapshot`), the orchestrator sets the mode by driving the TUI through mstream, in this exact order:
  1. `mstream send <ws> <target> --enter --text "/permission"` — opens the Codex permission selector.
  2. `mstream snapshot <ws>` — read the selector (options: 1 Default, 2 Auto-review, 3 Full Access).
  3. `mstream send <ws> <target> --enter --text "2"` — pick **option 2, Auto-review** (workspace-write + on-request approvals routed through the auto-reviewer subagent).
  4. `mstream snapshot <ws>` — confirm the pane shows `Permissions updated to Auto-review`.
  NEVER pick option 3 (Full Access / `danger-full-access` = yolo). Do this only with the user's approval for the host/workstream. (For Claude agents, set the equivalent auto permission mode.) A bare empty `--enter` submits stuck/queued input.

Remote non-login shells may not have the same `PATH` as an interactive shell.
Prefer user-provided absolute executable paths when creating sessions (do NOT
create workstream-local wrapper scripts). If you need to discover executable
paths and mstream lacks a safe host probe, ask the user or add the needed
mstream capability instead of running direct SSH probes.

NEVER create launcher wrapper scripts (e.g. a `codex-auto` that bakes in `--ask-for-approval`/`--sandbox danger-full-access` flags), and NEVER create a wrapper that execs uncommitted code. Launch the bare executable (`codex` or `claude`) via `mstream new --agent <absolute-executable-path>`, then run the mandatory permission-mode setup as the FIRST step before the agent does any work — the `/permission` → pick **2 Auto-review** procedure above. Permission posture is set interactively via mstream — never via on-disk wrapper flags.

When a TUI is sensitive to startup timing, create the session first without a long `--task`, confirm the prompt is ready with `mstream snapshot`, then send the assignment with `mstream send --interrupt-first --enter`. Verify the timeline shows the agent started acting on the assignment.

Give agents as much context as possible, including user feedback, issue links, PR links, relevant design decisions, and known constraints.

For implementers, include the delivery path explicitly:

- stage work in the local agent worktree
- commit completed work locally
- unless the user has specified a different destination, push the branch and create a GitHub PR targeting `origin/main`
- if the correct target branch is unknown, ask before pushing or opening the PR
- make the PR address the issue, for example "Closes #337" or "Addresses #337" depending on whether merge should close it
- report the PR URL, branch, commit, and tests run

Nudge implementers about this at kickoff and again when they report local completion. Reviewers should wait for the PR or pushed branch unless explicitly asked to review an unpushed local branch through another access path.

After each `new`, `join`, or `recruit`, verify with `mstream status <workstream>` or `mstream events <workstream> --limit 20` before creating the next agent. If one agent creation fails, do not continue creating the remaining agents until the failure is understood.

## Design & Brainstorm Workstreams

For exploration that produces a design, not merged code. The deliverable is
either a detailed GitHub issue with comments, or a PR carrying `DESIGN.md`
and/or `PLAN.md` — nothing more. Greenfield work weighs 2-3 alternatives with
pros/cons; brownfield adds a migration strategy (CLAUDE.md DESIGN rules).

Core principles:

- **No implementation starts until the human gives a tacit greenlight** — even
  exploratory code. Brainstorms and designs end in docs and comments, not commits
  to feature logic.
- **Agents think through you.** They never talk directly or run mstream
  (Invariant 1); you relay proposals and critiques between them.
- **Capture is durable.** Discussion lives in GitHub issue/PR comments;
  substantial alternatives are short docs each proposer writes in its own cwd
  checkout on a branch like `@gpt55-401-prop/option-a` — never a shared checkout
  (see Work Directories). You post major decisions to the issue and flag them for
  the human.

Roles (`<model>-<issue>-<role>`): **proposer** (independent alternatives,
agent-family diverse), **critic** (adversarial — pitfalls, false assumptions,
conflicting requirements, user confusion, dependency fit), **synthesizer** (a
dedicated agent that reads the proposers' branches/comments and folds the chosen
direction into `DESIGN.md` in its own checkout), optional **researcher**
(crate/dependency fit, maturity, safety, support). You facilitate and broker
decisions; the human reviews the synthesizer's output and sends questions and
feedback back through you.

### Usual flow

A flexible default — adapt depth and rounds to the problem, as long as the
principles hold.

1. **Frame.** Confirm/create the GH issue, then broadcast the goal, constraints,
   and bars.

```sh
mstream broadcast <workstream> --text "Issue #401: <goal>. Constraints: <...>. Each proposer drafts one alternative on its own branch; do not coordinate yet." --enter
```

2. **Diverge.** Each proposer drafts an alternative independently — an issue
   comment for a sketch, a branch doc for a substantial one — with no cross-talk
   yet, to avoid anchoring.
3. **Harvest & relay.** Collect proposals (`summary-input` / posted comments) and
   relay them to the critics.

```sh
mstream send <workstream> <critic-target> --text "Critique these on correctness/perf/resilience/UX/operability/dependency fit; post inline + issue comments: <links>." --enter
```

4. **Critique.** Critics score each alternative and post inline + issue comments.
5. **Converge.** The synthesizer tallies pros/cons and recommends a leading
   alternative, naming the open decision forks.
6. **Decide.** For genuine forks (product direction, conflicting requirements,
   irreversible dependency choices), the synthesizer drafts a decision summary
   with options/tradeoffs; you post it to the issue, flag it for the human, and
   wait — never fabricate direction (Invariant 6).

```sh
mstream session mark <synth-target> --state needs-input --summary "decision fork posted to #401; awaiting human pick"
```

7. **Capture.** On the human's call, the synthesizer writes the deliverable:
   `DESIGN.md` (body = chosen alternative, appendix = the rest, Changelog entry)
   and/or `PLAN.md` as a PR, or a fully-commented issue. Iterate via the PR
   Review Loop.

The workstream closes when the human accepts the issue/PR deliverable; the
merge-centric checks in Closing Workstreams apply only when there is a
DESIGN/PLAN PR. mstream fits persistent, long-lived agents; for broad, ephemeral
idea generation prefer Workflow/subagents and bring the synthesis back here.

## PR Review Loop

PR review is iterative and reviewer-owned. Follow repo guidance from
`CLAUDE.md` or `AGENTS.md` for how reviews must be posted.

Default loop:

1. Nudge reviewers to post feedback directly to the PR after every review round.
2. Require reviewers to post both inline comments and a verdict after each round.
3. Treat all nits as must-fix. No issue is too small, and no code smell should be accepted.
4. When you detect that reviewers have posted all feedback, notify the PR author to pull the PR feedback and address it completely.
5. Monitor the author while they make fixes, commit, push, and update the PR.
6. Once the PR is updated, start another review round by nudging reviewers to re-review the latest commit.
7. Repeat until every issue is addressed, every open thread is resolved and closed, and the reviewer accepts the PR.

If a reviewer cannot post to GitHub because of auth or tooling failure, treat
that as a blocker to fix or escalate. A local/session-only review verdict is not
equivalent to a posted PR review unless the user explicitly accepts that
fallback.

Acceptance and merge rules:

- Reviewers must post comments and a verdict after every review round, including the final accepting round.
- When the reviewer finally accepts, verify that all open PR threads are resolved and closed.
- Only the reviewer may merge the accepted PR.
- After merge, have the reviewer verify the merged commit fully addresses the issue, then close the original issue or create a follow-up issue for remaining gaps.
- Treat "PR merged but issue still open" as a closeout gap and nudge the reviewer to resolve it.
- After issue closeout or follow-up creation, tell all parties to stand by.
- Do not close the workstream automatically after merge; the user decides whether the workstream is good enough to close.

## Communication

Use `mstream` as the communication channel:

```sh
mstream send issue-337-tmux-fleet-api amd1::gpt55-337-og \
  --text "Please address the latest reviewer feedback on FleetTargetSpec parsing." \
  --enter

mstream broadcast issue-337-tmux-fleet-api \
  --text "Status check: summarize your current state, latest durable output, and any blocker or question." \
  --enter
```

Tell collaborators that they can communicate with each other through you as mediator. Relay decisions, blockers, and review findings explicitly between agents.

Agent TUIs sometimes miss a submitted newline after pasted text. If the timeline suggests the prompt was typed but not submitted, wait briefly and send an extra empty submit:

```sh
mstream send issue-337-tmux-fleet-api amd1::gpt55-337-og \
  --text "" \
  --enter
```

Use this as a targeted retry, not as a default duplicate-send, so you do not accidentally submit stale text twice.

For non-destructive interruption:

```sh
mstream interrupt amd1::gpt55-337-og
mstream send issue-337-tmux-fleet-api amd1::gpt55-337-og \
  --interrupt-first \
  --settle-ms 500 \
  --text "Stop current work and switch to the new user priority." \
  --enter \
  --set-state busy
```

## Monitoring And Unblocking

Poll progress periodically:

```sh
mstream status issue-337-tmux-fleet-api --active-window-secs 30 --idle-after-secs 300
mstream events issue-337-tmux-fleet-api --limit 50
mstream snapshot issue-337-tmux-fleet-api --max-chars 12000
mstream summary-input issue-337-tmux-fleet-api --max-chars 12000
```

These read the daemon's **passively-buffered timeline**: mstream records workstream
activity continuously, so you can answer "what is agent X doing?" on demand —
including ad-hoc user questions — with no prior trigger or capture step. Query these
instead of shelling out to raw `ssh <host> tmux capture-pane` (or other direct host
probes): `status` is the cheapest liveness check, `summary-input` the fastest "what
is it doing right now", `events --readable` the human timeline, and `snapshot` the
live pane. Reserve direct tmux/host probes for state mstream genuinely cannot surface,
and per Prerequisites prefer asking the user or adding an mstream capability over
ad-hoc SSH.

Use `mstream status` liveness fields to decide when to look deeper:
`activity_hint`, `tmux_present`, `last_output_secs`, and
`observed_activity_idle_secs`. `active` means recent tmux input or output,
`quiet` means activity is outside the active window but not yet idle, `idle`
means the session has been quiet beyond the idle threshold, `missing` means the
tmux session is gone, and `unknown` means the host activity refresh failed.

Treat `mstream status` as the first polling layer:

- during setup, handoff, or after sending a task, check within 30-60 seconds
- during normal active work, check about every 3-5 minutes
- during known long-running tests or builds, check about every 5-10 minutes unless the user is waiting on an update
- after an agent produces a durable output, review verdict, blocker, or question, act immediately rather than waiting for the next poll

Use a self-wakeup timer to make those checks happen without relying on the
human to poll you. For active review/implementation loops, choose an interval
that matches the risk: about 3-5 minutes for normal active work and 5-10
minutes for long tests or builds. If a timer wakes you and the workstream is
complete or waiting on a human decision, stop or lengthen the timer yourself.

Workstream state is coordinator-owned (Invariant 1). Determine whether an agent
is done, blocked, needs input, should continue, should stand by, or should look
for feedback from observable evidence: pushed commits, PR creation or updates,
posted review comments, test output, explicit terminal messages, snapshots, and
timeline history. Then use mstream state commands yourself if they help
coordinate handoffs or summaries.

Stopping points:

- pushed PR/branch update: verify branch, commit, PR target, and tests; decide whether to recruit reviewers, request fixes, or put the author on standby
- posted review comments or verdict: inspect whether feedback is must-fix, relay it to the author, and decide whether another review pass is needed
- explicit blocker in output: inspect events and `summary-input`, unblock locally if safe, otherwise surface the blocker to the user
- explicit question in output: relay the question to the user or another agent with the right context
- `tmux_present=false` or `activity_hint=missing`: assume the session was stopped out of band; rehydrate from durable context before recreating work
- `activity_hint=unknown`: treat as a daemon or host visibility problem; fix mstream connectivity before interpreting agent progress

Stuck or stopped detection should require evidence, not just a single quiet poll:

- If `activity_hint=active`, avoid interrupting. Use `events` or `summary-input` only if the user asks for a current summary.
- If `activity_hint=quiet`, compare with the previous poll. If `last_output_secs` and `observed_activity_idle_secs` keep growing and there are no new events, inspect `summary-input`.
- If `activity_hint=idle` for two consecutive polls while `state=busy`, inspect `summary-input` and then `snapshot`.
- If the snapshot shows a shell prompt, final answer, PR URL, completed test output, or review verdict, decide the next step yourself: continue, standby, review, address feedback, or report completion.
- If the snapshot shows an approval prompt, credential failure, merge conflict, failing tests, missing branch/PR access, or a direct question, treat it as `blocked` or `needs-input` and intervene.
- If the snapshot shows a known long-running command such as `cargo test`, `cargo clippy`, dependency build, or `gh` operation still producing plausible progress, do not classify it as stuck solely from quiet output.
- If the snapshot shows text typed into an agent TUI but not submitted after a send, wait briefly and send one extra empty `--enter`.

Use the mstream timeline/history to infer what each agent is doing. Summaries to the user should be succinct, actionable, and current:

- who is active, done, blocked, or needs input
- what changed since the last summary
- current risks or blockers
- decisions needed from the user
- next recommended action

When the user asks to see a timeline, prefer the human-readable event view if
the installed binary supports it:

```sh
mstream events <workstream> --limit 50 --readable
```

This prints plain text in current mstream builds. The underlying daemon record
uses an `events_readable.text` field, but the CLI unwraps it before writing
stdout.

Fall back to normal JSONL `events` and summarize it yourself only when
`--readable` is unavailable.

When an agent is stuck, unblock by sending focused instructions, clarifying requirements, relaying feedback, or asking the user for missing external state. Do not invent credentials, host aliases, issue numbers, or product direction.

If a state annotation helps coordinate the workstream, mark the target yourself
with an explicit target and evidence-based summary. This is an orchestrator
action, not a collaborator protocol:

```sh
mstream session mark amd1::gpt55-337-og --state done --summary "Pushed PR #340 commit cc77516b; tests passed."
mstream session mark amd1::gpt55-337-og --state blocked --summary "Cannot push because GitHub auth failed."
mstream session mark amd1::gpt55-337-rv --state needs-input --summary "Reviewer asks whether API should expose aliases publicly."
```

Use handoffs for explicit reviewer/implementer sequencing:

```sh
mstream handoff arm issue-337-tmux-fleet-api \
  --from amd1::gpt55-337-og \
  --to amd1::opus47-337-rv \
  --on done \
  --task "The implementer pushed the issue #337 branch and the coordinator marked it ready. Review the branch and post must-fix feedback."
```

## Closing Workstreams

Close a workstream only after the user accepts the final work or after associated PRs are merged. Check with the user before closing unless they explicitly instructed automatic closeout.

Close with useful context tags:

```sh
mstream close issue-337-tmux-fleet-api \
  --summary "Fleet API issue created and implementation work completed." \
  --domain tmux \
  --specialty fleet \
  --specialty mstream \
  --stop-timers \
  --standby-agents
```

Use `--stop-timers` for workstream-scoped self-wakeup timers and
`--standby-agents` to send a final standby message before freeing agents. If
the installed `mstream` does not support these flags, manually broadcast the
standby instruction, stop workstream timers, then run the older `close`
command.

Use `leave --available` to free individual agents without killing sessions:

```sh
mstream leave issue-337-tmux-fleet-api amd1::opus47-337-rv --available
```

Use `kill` only when explicitly destructive session cleanup is intended:

```sh
mstream kill amd1::gpt55-337-og
```

Before closeout, verify the complete chain: PR merged, original issue closed
or follow-up issue created, reviewers posted final verdict, agents were told to
stand by, and active workstream timers are stopped. At closeout, tell the user
what merged, what remains open, which agents were freed, which timers were
stopped, and whether any local uncommitted changes remain.

Keep mstream boundaries clean during closeout. `mstream` is responsible for
workstream/session/timer/timeline primitives; it should not decide which GitHub
issue or PR to comment on and should not post closeout logs itself. When a
closeout log is useful, build it as the orchestrator from neutral primitives:

```sh
mstream events <workstream> --limit 100 --readable
mstream summary-input <workstream> --max-chars 12000
mstream snapshot <workstream> --max-chars 12000
```

Then use your own GitHub context to decide whether to post the synthesized log
to the PR, the issue, both, or neither. The log should be concise: result,
agents/roles, important timeline points, validation, remaining risks, and
follow-up issues. Every closeout log and posted transcript MUST include both,
led by a compact stats block (table or list) before the narrative:
1. **Timestamps** (with timezone) on key milestones: workstream open, agent
   recruitment, PR open, each review verdict, fix pushes, merge, issue close,
   and workstream close.
2. **Stats:** number of **review rounds** (per project/skill), agent count by
   role/model, elapsed wall-clock from first known milestone to closeout,
   commits/pushes with merge/commit SHAs, timer wakeups or orchestrator turns if
   known, and any follow-up issue count.

If a timestamp or count is genuinely unavailable, mark it `unknown` rather than
inventing precision or dropping the block. (Invariant 1: do not ask collaborator
agents to use mstream for this.)
