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

Decision checkpoint:

- If the latest user request asks for an outline, staffing proposal, risk assessment, or bars to enforce, provide that first and wait for approval before opening workstreams, connecting hosts, or creating sessions.
- If the user has explicitly said to proceed with setup, still summarize the proposed side effects before running them: daemon start, host connect, workstream open, session creation, and prompts sent to agents.
- Treat host connection, workstream open, session join/new/recruit, send, interrupt, handoff, close, and kill as side-effecting operations.
- If a command fails, stop the setup sequence, report the exact failure and implication, then decide the next step with the user unless the fix is purely local and obvious.

## mstream Daemon And Hosts

Client commands use JSONL stdout. The socket resolves from `--socket`, then `MSTREAM_SOCKET`, then `/tmp/mstream-${USER}.sock`.

Once assigned as orchestrator for an active workstream, keep the mstream daemon running across turns for the duration of that orchestration assignment. Do not stop and restart it after routine `list`, `status`, `summary-input`, or progress checks. Stop it only when the user tells you to, when the workstream is closed and no active orchestration remains, or when replacing a failed daemon instance.

Use a stable explicit socket for the whole assignment:

```sh
mstream --socket /tmp/mstream-${USER}.sock daemon status
```

Normal rehydration/playbook after daemon start or restart:

```sh
mstream --socket /tmp/mstream-${USER}.sock connect <host-alias> '<ssh-uri>' --work-root <host-work-root>
mstream --socket /tmp/mstream-${USER}.sock scan <host-alias>
mstream --socket /tmp/mstream-${USER}.sock list
mstream --socket /tmp/mstream-${USER}.sock status <workstream> --active-window-secs 30 --idle-after-secs 300
mstream --socket /tmp/mstream-${USER}.sock summary-input <workstream> --max-chars 12000
```

Use the concrete host aliases, SSH URIs, work roots, and workstream names supplied by the user or recovered from durable context. Do not rediscover basic CLI shapes with `--help` during normal orchestration; keep this playbook as the default sequence and inspect help only when a command actually fails because the local binary changed.

Start the daemon when the user asks you to operate mstream and no daemon is running. In Codex/harness sandbox environments, use foreground mode in a managed sandbox exec session as the one normal startup method:

```sh
mstream --socket /tmp/mstream-${USER}.sock daemon start --foreground
```

Keep that foreground exec session running and issue client commands from separate exec calls. Record the foreground exec session id in your context if available. This was validated locally: daemonized `mstream daemon start` returned success but was immediately unreachable, while foreground mode stayed reachable from separate client commands after a delay.

Do not use `mstream daemon start` or `nohup ... &` as the normal Codex/harness playbook; they may be reaped by the harness even after reporting success. Outside Codex/harness, daemonized `mstream daemon start` is still a valid human/manual mode, but it is not the orchestrator playbook.

Fallback only if the managed foreground exec session is unavailable or lost while the daemon must survive for timers or cross-turn orchestration: run the same foreground daemon inside a dedicated tmux session.

```sh
tmux new-session -d \
  -s mstream-daemon-${USER} \
  -c '<absolute-motlie-worktree>' \
  './target/debug/mstream --socket /tmp/mstream-${USER}.sock daemon start --foreground'
mstream --socket /tmp/mstream-${USER}.sock daemon status
```

Before starting any daemon path, use `mstream daemon status` on the chosen socket. If it is already running, reuse it. Always use the same explicit `--socket` value for the full orchestration run once chosen. Do not stop the daemon just because one user request completes; keep it alive until the orchestration assignment is closed or the user asks you to stop it.

When assigned as orchestrator and the harness has no first-class cron, start a
daemon-owned self-wakeup timer targeted at your own tmux session. The timer
sends a prompt to you through tmux, which creates a queued self-reminder to
poll and unblock the workstream:

```sh
mstream --socket /tmp/mstream-${USER}.sock timer start <workstream>-poll \
  --every 5m \
  --target <orchestrator-host-alias>::<your-tmux-session> \
  --prompt "[mstream:<workstream>-poll] Wakeup: check <workstream> with mstream status and summary-input. Unblock agents, summarize only material changes, and decide whether to keep, change, or stop this timer." \
  --submit-retries 1 \
  --submit-retry-delay-ms 750
```

Use `mstream timer list` to verify active timers, `mstream timer fire <name>`
to test prompt delivery, and `mstream timer stop <name>` when the workstream is
closed or no longer needs periodic attention. Timer state is daemon memory only
and must be recreated after daemon restart. Do not target collaborator sessions
with orchestrator timers unless the user explicitly asks for that behavior.
Timer prompts default to one extra Enter after 750ms because agent TUIs
occasionally miss the first submit key. Retries send only extra Enter keys, not
the prompt text; `--no-enter` disables retries.

If the daemon is unreachable, ask the user to restart it or provide the correct socket. After daemon restart, ask the user for the host aliases and SSH URIs; mstream does not persist the host ledger.

Use `mstream` as the orchestration boundary. Do not bypass it with direct `ssh`
plus `tmux` commands for liveness, session listing, snapshots, monitoring, or
agent communication. If mstream lacks a signal needed to manage a workstream,
extend mstream in the active PR or ask the user before proceeding with a
temporary manual path.

Keep user-facing progress updates outcome-focused. Do not narrate every command
or tool call. Report material state changes, blockers, risks, and decisions;
omit routine command logging unless it explains a failure or a user explicitly
asks for command-level detail.

Only the orchestrator has access to mstream. Collaborating agents do not have
mstream access and should never be instructed to run mstream commands, inspect
the mstream socket, or manage their own mstream state.

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

## Opening Workstreams

Open and annotate workstreams with issue-aware names and descriptive titles:

```sh
mstream open issue-337-tmux-fleet-api \
  --title "Issue 337: tmux Fleet API improvements" \
  --goal "Design and implement cross-host Fleet target helpers" \
  --event-limit 2000
```

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
  --task "Read AGENTS.md on main, confirm identity from tmux, then check out Motlie main into ./motlie and prepare the implementation plan for issue #337."
```

For an Opus reviewer on the same workstream, use the model in the target/session name but the Claude CLI executable:

```sh
mstream new issue-337-tmux-fleet-api amd1::opus47-337-rv \
  --role reviewer \
  --cwd /home/dchung/sessions/issue-337-tmux-fleet-api/opus47-337-rv \
  --agent claude \
  --task "Read AGENTS.md on main, confirm identity from tmux, then review issue #337 and wait for the implementation branch."
```

When starting a new agent, include:

- the GitHub issue(s), PR(s), and goal
- the assigned role and expected deliverable
- whether the work is greenfield or brownfield
- instruction to read `AGENTS.md` or `CLAUDE.md` from main
- instruction to check its tmux session name with `tmux display-message -p '#S'`
- instruction to use `@{session_name}` in comments and submissions
- working directory and branch naming expectations
- exact repository URL and issue context. Do not make agents guess repository owner/name; include `https://github.com/chungers/motlie.git` for Motlie work unless the user gives a different URL.
- permission-mode guidance when useful. Treat `mstream new --agent` as the executable name unless the current CLI supports arguments; otherwise start the executable, such as `claude` or `codex`, and use the initial task to request auto permission mode, for example Claude auto mode or Codex `/permission` auto mode.

Remote non-login shells may not have the same `PATH` as an interactive shell.
Prefer user-provided absolute executable paths or workstream-local wrappers when
creating sessions. If you need to discover executable paths and mstream lacks a
safe host probe, ask the user or add the needed mstream capability instead of
running direct SSH probes.

If an agent needs CLI flags and `mstream new --agent` accepts only an executable path, create a small wrapper under the workstream root, for example `~/sessions/{workstream}/bin/codex-auto`, and pass the wrapper path as `--agent`. Use this only after the user has approved the permission mode for the target host and workstream.

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
- After merge, tell all parties to stand by.
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

Workstream state is coordinator-owned. Do not ask collaborating agents to run
any mstream command. Determine whether an agent is done, blocked, needs input,
should continue, should stand by, or should look for feedback from observable
evidence: pushed commits, PR creation or updates, posted review comments, test
output, explicit terminal messages, snapshots, and timeline history. Then use
mstream state commands yourself if they help coordinate handoffs or summaries.

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
  --specialty mstream
```

Use `leave --available` to free individual agents without killing sessions:

```sh
mstream leave issue-337-tmux-fleet-api amd1::opus47-337-rv --available
```

Use `kill` only when explicitly destructive session cleanup is intended:

```sh
mstream kill amd1::gpt55-337-og
```

At closeout, tell the user what merged, what remains open, which agents were freed, and whether any local uncommitted changes remain.
