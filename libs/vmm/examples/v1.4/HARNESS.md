# v1.4 Harness

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-08 | @codex | Add `apt-get update` to the baseline agent/bootstrap validation and document the default egress allocator compatibility rule: keep slot-based capacity growth, but start the guest-facing vnet range at `10.0.2.0/24` so harness egress stays aligned with the previously validated path |
| 2026-04-08 | @codex | Promote HARNESS from a runbook to the harness contract and evolution log: document the stable JSON scenario format, shell/wrapper layering, allocator UX, and PTY/VTE artifacts for future agent-driven troubleshooting and feature work |
| 2026-04-07 | @codex | Add the first `harness_v1_4` runbook covering smoke, shell, multi-guest wrappers, and live SSH validation |

## Purpose

The `v1.4` harness is the primary automation and troubleshooting surface for
`libs/vmm`.

Success for this harness design is not merely "smoke tests pass". Success is:

- a future coding agent can boot isolated harness instances without human setup
- the agent can reproduce product bugs through `vfs`, `vnet`, SSH proxy, and
  PTY flows
- the agent can inspect host-side state, guest-side state, serial logs, launch
  logs, and terminal sessions from one harness run
- the agent can codify the reproduction as a saved scenario and reuse it for
  validation while iterating on a fix

This is why the harness now has both:

- a stable machine-readable scenario format
- thin convenience wrappers (`*.sh`) for common end-to-end flows

The scenario format is the durable contract. The shell wrappers are convenience
entrypoints and compatibility checks.

## Layering

The harness stack is intentionally split:

1. `harness_v1_4 scenario <file.json>`
   - stable machine-driven contract
   - deterministic action/expectation steps
   - machine-readable per-step results
   - PTY transcript and VTE screen artifacts
2. `harness_v1_4 shell`
   - ad-hoc/manual control surface
   - first customer for allocator UX and runtime discovery
   - useful for debugging, one-off exploration, and live SSH attach
3. `integration/*.sh`
   - thin wrappers over harness behavior
   - useful for regression convenience and external SSH checks
   - should avoid reimplementing orchestration logic that belongs in the
     harness itself

The shell wrappers still matter because real external `ssh -tt` behavior caught
bugs that pure `exec` did not. But the harness owns the reusable scenario
contract.

## Preconditions

Work from:

```bash
cd /tmp/vmm-v1.4
```

Build or rebuild the guest image when guest contents or overlay behavior
changes:

```bash
./libs/vmm/examples/v1.4/build-guest.sh
```

Build the harness:

```bash
cargo build -p motlie-vmm --example harness_v1_4
```

## Core Modes

### `smoke`

Fast single-guest lifecycle smoke over the extracted library API:

```bash
./target/debug/examples/harness_v1_4
./target/debug/examples/harness_v1_4 --result-json /var/tmp/motlie-v14/smoke-result.json
```

This covers:

- boot
- readiness
- `exec`
- `vfs`
- `vnet` route and outbound HTTPS
- shutdown

### `pty`

Single-guest PTY/VTE smoke:

```bash
./target/debug/examples/harness_v1_4 pty
```

This covers:

- PTY open
- login banner / MOTD capture
- prompt interaction
- terminal resize
- raw PTY transcript capture
- rendered terminal state capture
- terminal close/exit evidence

### `scenario`

Stable machine-readable scenario driver:

```bash
./target/debug/examples/harness_v1_4 scenario \
  ./libs/vmm/examples/v1.4/scenarios/pty-login.json
```

Multi-guest example:

```bash
./target/debug/examples/harness_v1_4 scenario \
  ./libs/vmm/examples/v1.4/scenarios/multiguest-validate.json
```

The scenario mode is the preferred non-interactive surface for future agent
work.

### `shell`

Ad-hoc/manual harness shell:

```bash
./target/debug/examples/harness_v1_4 shell
```

Optional root override:

```bash
./target/debug/examples/harness_v1_4 shell --root /var/tmp/motlie-v14
```

Or:

```bash
MOTLIE_VMM_ROOT=/var/tmp/motlie-v14 ./target/debug/examples/harness_v1_4 shell
```

Useful shell commands now include:

```text
capacity
where
where alice
boot alice
exec alice /bin/uname -s
pty-open alice alice-shell
pty-expect alice-shell Start tmux session?
pty-send-line alice-shell n
pty-screen alice-shell
pty-resize alice-shell 120 40
shutdown alice
quit
```

Each harness run allocates its own namespace and roots, for example:

```text
v1.4 harness instance: motlie-vmm-v14-h12345
  demo_root=/var/tmp/motlie-v14/motlie-vmm-v14-h12345-demo
  socket_root=/var/tmp/motlie-v14/motlie-vmm-v14-h12345-sockets
  proxy=ssh://localhost:34345
```

## Scenario Format

The canonical format is JSON. JSON is intentional:

- agents already generate it reliably
- it is unambiguous in PR review
- it maps directly to machine-readable results
- it avoids inventing a bespoke parser before the action model stabilizes

Canonical top-level shape:

```json
{
  "name": "pty-login",
  "description": "Boot alice and validate the PTY login path.",
  "steps": [
    { "action": "boot", "guest": "alice" },
    { "action": "ready", "guest": "alice" },
    {
      "action": "pty_open",
      "guest": "alice",
      "session": "alice-shell",
      "cols": 80,
      "rows": 24
    },
    {
      "action": "pty_expect",
      "session": "alice-shell",
      "contains": "alice@motlie-alice",
      "timeout_ms": 10000
    },
    { "action": "shutdown", "guest": "alice" }
  ]
}
```

Supported actions today:

- `boot`
- `ready`
- `exec`
- `pty_open`
- `pty_send`
- `pty_send_line`
- `pty_read`
- `pty_resize`
- `pty_expect`
- `pty_expect_terminal`
- `pty_snapshot`
- `shutdown`

Current expectation model:

- `exec` supports `expect.exit_code`, `expect.stdout_contains`,
  `expect.stderr_contains`
- PTY expectations are explicit step actions instead of being hidden in
  shell-wrapper `grep` chains

Current limits:

- steps run sequentially
- there is no branching or looping
- `validate` is not yet a first-class scenario action
- shell mode is not yet a thin frontend over the exact same engine

Those are acceptable `v1.4` limits. The important part is that saved
reproductions are now harness-native and reviewable.

## PTY / VTE Artifacts

The harness now keeps both raw and rendered terminal state.

Per PTY session it writes:

- `pty-transcript.ndjson`
- `pty-screen.json`

Why both:

- raw transcript is the source-of-truth event stream
- rendered VTE screen state is what agents and humans usually reason about

Why NDJSON for the raw transcript instead of one large pretty JSON array:

- easier for agents to stream and chunk
- cheaper to append and inspect
- smaller and less awkward than one giant nested JSON blob

Why JSON for the rendered screen:

- agents want direct structured access to rows, cols, cursor position, and
  visible text
- the rendered screen is a snapshot, so one structured JSON object is natural

This means the harness artifact strategy is:

- `scenario-result.json` for high-level structured results
- `pty-transcript.ndjson` for compact raw terminal events
- `pty-screen.json` for rendered terminal state

That split is more useful for agents than either:

- raw bytes only
- giant pretty transcript JSON only

## Machine-Readable Result Shape

`smoke` and `pty` still emit the legacy top-level harness result envelope.

`scenario` emits the scenario-native result envelope, including:

- overall status
- scenario name and description
- artifact root
- proxy URI
- allocator capacity
- per-step result records
- per-session artifact paths
- structured classified error record on failure

Step results include:

- action
- guest/session identity when relevant
- detail string
- `exec` output when relevant
- PTY read output when relevant
- rendered screen snapshot when relevant
- shutdown report when relevant

This is intended to be directly consumable by agents and CI.

## Allocator UX

The harness is the first customer of the new allocation API.

CLI overrides:

```bash
./target/debug/examples/harness_v1_4 shell \
  --max-guests 64 \
  --first-cid 100 \
  --admin-base 172.22.0.0/16 \
  --admin-guest-prefix 30 \
  --egress-base 10.32.0.0/12 \
  --egress-guest-prefix 24
```

Shell inspection:

```text
capacity
where
where alice
```

`capacity` shows:

- configured base pools
- per-guest subnet size
- computed capacity
- next slot and remaining capacity

`where <guest>` now shows:

- slot
- CID
- admin subnet/IP/MAC
- egress subnet/IP/MAC
- runtime paths and logs

Default policy today:

- admin base `172.20.0.0/16`, guest `/30`
- egress base `10.0.0.0/8`, guest `/24`
- default effective capacity `16384`

This replaces the old implicit 7-guest stopgap.

## Scenario Examples

Saved examples live in:

- [`scenarios/agent-bootstrap.json`](./scenarios/agent-bootstrap.json)
- [`scenarios/pty-login.json`](./scenarios/pty-login.json)
- [`scenarios/multiguest-validate.json`](./scenarios/multiguest-validate.json)

The multi-guest example proves the stable format is not single-guest-only:

- boot `alice`
- boot `bob`
- validate guest-specific `vfs` content
- validate guest-specific `vnet` routing
- validate outbound HTTPS
- shut both guests down

The agent bootstrap example is the baseline future-agent scenario:

- boot `alice`
- verify `vfs` is mounted
- verify `sudo -n true`
- verify `git` is preinstalled
- verify `codex --version`
- verify outbound HTTPS
- verify `sudo -n apt-get update`
- shut the guest down

Current allocator compatibility note:

- the harness still uses the slot-derived allocation API and larger capacity
  model
- the default egress pool now reserves child slots `0` and `1`, so slot `0`
  starts at `10.0.2.0/24`
- this keeps the first harness guests aligned with the previously validated
  libslirp/guest path while still allowing capacity growth well beyond `7`

## Shell and Wrapper Flows

Main multi-guest wrapper:

```bash
./libs/vmm/examples/v1.4/integration/harness-shell-smoke.sh
```

Isolation wrapper:

```bash
./libs/vmm/examples/v1.4/integration/harness-isolation-smoke.sh
```

These still matter because they validate:

- shell UX
- multi-guest bring-up from the harness shell
- live external SSH login behavior
- concurrent harness isolation

But the wrappers should stay thin. New orchestration logic belongs in the
harness scenario engine, not in bash.

Saved shell command sequence:

```bash
cat ./libs/vmm/examples/v1.4/setup-multiguest.harness
```

## Live External SSH

When validating real login UX, use the printed proxy port:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 38306 alice@localhost
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 38306 bob@localhost
```

Useful manual probes:

```bash
pwd
sudo -n true
sudo -n apt-get update
sudo -n apt-get install -y git
cat ~/.env
ls -ald /home/$USER /workspace /agent-state
curl -fsSL https://example.com -o ~/example.html && stat ~/example.html
ip route
cat /etc/resolv.conf
```

This path remains important because `exec` success does not prove full external
interactive SSH correctness.

Current expected guest privilege model:

- `alice` and `bob` should have passwordless sudo through
  `/etc/sudoers.d/90-motlie-demo`
- `sudo -n true` should succeed without prompting
- `sudo -n apt-get update` should succeed without DNS or route failures
- `git` should already be present in the base image

## Regression Matrix

When changing `vmm`, `vfs`, `vnet`, SSH proxying, allocator behavior, or guest
image seeding, rerun at least:

1. `cargo test -p motlie-vmm --lib`
2. `cargo build -p motlie-vmm --example harness_v1_4`
3. `./target/debug/examples/harness_v1_4`
4. `./target/debug/examples/harness_v1_4 pty`
5. `./target/debug/examples/harness_v1_4 scenario ./libs/vmm/examples/v1.4/scenarios/multiguest-validate.json`
6. `./libs/vmm/examples/v1.4/integration/harness-shell-smoke.sh`

And when changing login/banner/proxy or PTY handling, also run a live external
SSH check.

## Evolution Notes

The current harness evolution is:

1. library lifecycle extraction
2. basic smoke harness
3. structured machine-readable results
4. transcript/log bundle capture
5. PTY session support
6. VTE screen capture
7. stable scenario/action-expectation format

Still open:

- make shell mode a thin frontend over the same scenario engine
- add typed validation profiles instead of hand-authored `exec` checks
- decide whether to emit optional human-first artifacts such as asciinema or
  ttyrec in addition to the current agent-first NDJSON/JSON split

That is acceptable for `v1.4`. The important change in this branch is that the
harness is now strong enough for future agents to save and rerun reproductions
through the same VM/VFS/VNET/PTY path they are trying to debug.
