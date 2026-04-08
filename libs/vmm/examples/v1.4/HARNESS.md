# v1.4 Harness Runbook

This file is the repeatable validation guide for `examples/v1.4`.

Use it for two different purposes:

- fast non-interactive regression checking through the programmatic harness
- higher-fidelity interactive/manual testing through `harness_v1_4 shell`
- transitional compatibility checks through the migrated `repl_host_v1_4`
- future harness-first work:
  - named scenarios
  - interactive/manual operation
  - PTY transcript-driven debugging

The main lesson from the recent multi-guest bug is important:

- REPL/library `exec` checks are useful, but they are not enough by
  themselves
- the real interactive SSH shell path must also be tested
- the smoke coverage should always include both guests when validating Motlie
  `vfs` + `vnet` + SSH proxy behavior

## Preconditions

Work from:

```bash
cd /tmp/vmm-v1.4
```

Build or rebuild the image when the guest image or launch overlay behavior
changes:

```bash
./libs/vmm/examples/v1.4/build-guest.sh
```

## Path 1: Programmatic Harness

This is the fastest end-to-end rootless smoke for the extracted library API.

Build:

```bash
cargo build -p motlie-vmm --example harness_v1_4
```

Run:

```bash
./target/debug/examples/harness_v1_4
```

Write a machine-readable result artifact:

```bash
./target/debug/examples/harness_v1_4 \
  --result-json /var/tmp/motlie-v14/smoke-result.json
```

Explicit PTY scenario:

```bash
./target/debug/examples/harness_v1_4 pty
```

Current note:

- `smoke` is the reliable machine-readable scenario today
- `pty` now emits a stable machine-readable evidence block plus a persisted
  transcript artifact, but rendered terminal-state / VTE assertions are still
  future work

Interactive/manual harness mode:

```bash
./target/debug/examples/harness_v1_4 shell
```

Run under an explicit host root:

```bash
./target/debug/examples/harness_v1_4 shell --root /var/tmp/motlie-v14
```

or:

```bash
MOTLIE_VMM_ROOT=/var/tmp/motlie-v14 ./target/debug/examples/harness_v1_4 shell
```

Each harness run now allocates its own per-process namespace and demo root
under the selected host root. At startup it prints where to look, for example:

```text
v1.4 harness instance: motlie-vmm-v14-h12345
  demo_root=/var/tmp/motlie-v14/motlie-vmm-v14-h12345-demo
  socket_root=/var/tmp/motlie-v14/motlie-vmm-v14-h12345-sockets
  proxy=ssh://localhost:34345
```

What it proves today:

- guest boot through the extracted lifecycle API
- guestfs-backed home/workspace/agent-state bring-up
- SSH control-plane readiness
- library `VmHandle::exec(...)`
- outbound HTTPS over Motlie `vnet`
- shutdown
- first PTY/session path through:
  - `VmHandle::open_pty(...)`
  - banner/MOTD capture
  - prompt interaction
  - terminal resize
  - transcript capture

What the JSON result includes today:

- status (`passed` / `failed`)
- scenario name
- guest id
- pid and shutdown outcome
- `VmHandle::observability()` snapshot, including:
  - runtime/log/socket paths
  - host mount metadata
  - typed run-bundle metadata
  - standard capture paths for `scenario-result.json` and `pty-transcript.json`
- named scenario checks
- structured classified error record on failure
- PTY transcript summary and raw PTY transcript events for the `pty` scenario

Each booted run also writes internal bundle artifacts under the runtime root:

- `.../runtime/<guest>/bundle/scenario-result.json`
- `.../runtime/<guest>/bundle/pty-transcript.json` for the `pty` scenario

Current limitation:

- `harness_v1_4` is single-guest today
- use the harness shell or REPL path below for multi-guest validation
- the harness does not yet expose the full ad-hoc interactive/manual shell mode
  that should eventually replace the standalone `repl_host_v1_4`

Expected success line:

```text
v1.4 harness smoke passed: guest=alice ...
```

or:

```text
v1.4 harness pty passed: guest=alice ...
```

## Path 2: Automated Harness Shell Smoke

This is the main multi-guest regression check and the preferred replacement for
the old standalone REPL-driven smoke.

Run:

```bash
./libs/vmm/examples/v1.4/integration/harness-shell-smoke.sh
```

What it validates:

- drives `harness_v1_4 shell`
- replays [`setup-multiguest.harness`](./setup-multiguest.harness)
- boots `alice`
- boots `bob`
- runs `validate alice`
- runs `validate bob`
- opens a real interactive SSH shell for `alice`
- opens a real interactive SSH shell for `bob`
- checks in each guest:
  - MOTD is shown
  - `pwd` lands in the correct home
  - `~/.env` is guest-specific
  - `curl https://example.com` succeeds
  - file lands in the writable guest home overlay
  - `/workspace` and `/agent-state` ownership is correct
- shuts both guests down

Expected success line:

```text
v1.4 harness shell smoke passed
```

Two-harness isolation smoke:

```bash
./libs/vmm/examples/v1.4/integration/harness-isolation-smoke.sh
```

This proves two separate harness shells can run concurrently, both boot
`alice`, and not collide on namespace, sockets, demo roots, or proxy ports.

If this smoke fails, inspect:

```bash
tail -n 200 /tmp/motlie-vmm-v14-harness-shell-smoke.log
```

Saved command script:

```bash
cat ./libs/vmm/examples/v1.4/setup-multiguest.harness
```

That saved script is the command sequence:

- `where`
- `boot alice`
- `where alice`
- `boot bob`
- `where bob`
- `status`
- `validate alice`
- `validate bob`

The expectation layer lives in:

- [`integration/harness-shell-smoke.sh`](./integration/harness-shell-smoke.sh)
- [`integration/harness-isolation-smoke.sh`](./integration/harness-isolation-smoke.sh)

## Path 3: Harness Shell

This is the preferred ad-hoc/manual control surface going forward.

Run:

```bash
./target/debug/examples/harness_v1_4 shell
```

Optional host root override:

```bash
./target/debug/examples/harness_v1_4 shell --root /var/tmp/motlie-v14
```

Core commands:

```text
boot alice
boot bob
status
where
where alice
validate alice
validate bob
exec alice /bin/uname -s
exec bob /bin/uname -s
shutdown bob
shutdown alice
quit
```

Notes:

- harness shell uses the same `libs/vmm` lifecycle API as the smoke scenarios
- harness shell allocates its own namespace/demo root/proxy port per run
- use `where` to print the current roots, sockets, and per-guest logs

## Path 4: Live REPL

Build:

```bash
cargo build -p motlie-vmm --example repl_host_v1_4
```

Run:

```bash
./target/debug/examples/repl_host_v1_4
```

Core commands:

```text
boot alice
boot bob
status
validate alice
validate bob
exec alice /bin/uname -s
exec bob /bin/uname -s
shutdown bob
shutdown alice
where
where alice
quit
```

Notes:

- `v1.4` uses `boot`, not `launch`
- the REPL is intentionally thinner than `v1.3` and should stay a client of
  `libs/vmm`, not grow orchestration logic again
- the long-term direction is to fold this ad-hoc/manual surface into the
  harness rather than keep two separate control planes
- `where` prints the current namespace/demo root/socket root/proxy, and
  `where <guest>` prints the per-guest runtime and log paths

## Path 5: Manual Interactive SSH

This is the path that caught the earlier smoke gap. Use it when validating
login UX, MOTD, and the real shell transport.

After booting guests in the harness shell or REPL, use the printed proxy port.
For example, if `where` shows `proxy=ssh://localhost:38306`:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 38306 alice@localhost
```

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 38306 bob@localhost
```

Inside each guest, run:

```bash
pwd
cat ~/.env
ls -ald /home/$USER /workspace /agent-state
curl -fsSL https://example.com -o ~/example.html && stat ~/example.html
```

What to expect:

- MOTD is visible before the shell prompt
- `alice` lands in `/home/alice`
- `bob` lands in `/home/bob`
- `~/.env` contains the guest-specific demo API key
- `curl` succeeds for both guests

Current expected egress layout:

- `alice`: `10.0.2.0/24`
- `bob`: `10.0.3.0/24`

So a route check inside the guest should look like:

```bash
ip route
cat /etc/resolv.conf
```

with guest-specific gateway/DNS values, not a hardcoded `10.0.2.2` assumption
for every guest.

## Suggested Regression Matrix

When changing `vmm`, `vfs`, `vnet`, SSH proxying, or guest image seeding,
rerun at least:

1. `cargo test -p motlie-vmm --lib`
2. `cargo test -p motlie-vnet`
3. `./target/debug/examples/harness_v1_4`
4. `./libs/vmm/examples/v1.4/integration/harness-shell-smoke.sh`

And when changing any login/banner/proxy behavior, also do one live manual
interactive check:

1. start `harness_v1_4 shell`
2. `boot alice`
3. `boot bob`
4. `ssh -p 2224 alice@localhost`
5. `ssh -p 2224 bob@localhost`

## Failure Triage

If `alice` works and `bob` fails:

- check REPL log:
  - `/tmp/motlie-vmm-v14-harness-shell-smoke.log`
- check launch logs:
  - `/tmp/motlie-vmm-v14-launch/alice/launch.log`
  - `/tmp/motlie-vmm-v14-launch/bob/launch.log`
- check whether the failure is:
  - REPL/library `exec` only
  - interactive SSH shell only
  - both

That distinction matters:

- REPL `exec` success does not prove interactive shell success
- interactive shell success does not prove all library `exec` semantics

Both must remain green.
