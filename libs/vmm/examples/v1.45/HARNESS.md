# v1.45 Apple Vz Harness Contract

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-26 | @codex-vz | Replace copied v1.4/CH runbook text with the actual v1.45 Vz backend split, validation commands, and Vz-only image hardening caveat |
| 2026-04-25 | @codex-vz | Require prebuilt Vz runner and egress helper artifacts by default so first-contact startup cannot hide host cargo builds |
| 2026-04-25 | @codex-vz | Document that v1.45 Vz first-contact SSH follows the shared convergence contract: `ready` means interactive SSH plus required mounts, and full VFS/VNET/egress certification remains `validate` or a saved scenario |

## Purpose

The `v1.45` harness is the macOS Apple Vz validation surface for `libs/vmm`.
It exists to prove that the library lifecycle can support a non-CH backend
while keeping platform-specific adaptation visible and reviewable.

The durable contract is the saved scenario format plus the library API. Shell
mode is for exploration and manual certification.

## Actual Runtime Configuration

The current harness creates this runtime:

```text
HypervisorBacking::AppleVirtualizationShell(VzShellBackend)
FilesystemBacking::MotlieVfs(MotlieVfsBacking)
NetworkBacking::HypervisorManaged
ControlPlaneBacking::MotlieSshProxy
```

Interpretation:

- Vz owns the VM process, serial log, guest disk clone, and userspace egress
  helper launch.
- Motlie VFS is still the filesystem backing for guest workspace/home/state
  mounts in this slice.
- There is no macOS `motlie-vnet` runtime handle. The Vz userspace egress
  helper provides outbound connectivity for the Vz VM.
- The SSH proxy remains the common control plane for exec, PTY, and external
  SSH.

Any documentation or scenario text that says this slice validates "Motlie
vnet" is wrong. The correct phrase is "Vz userspace egress".

## Backend Boundary Rules

Backend choice is owned by `HypervisorBacking`.

- CH backends render `launch-ch.sh` and accept CH/common network modes.
- Vz backends render `launch-vz.sh` and require
  `--admin-net=none --egress-net=vz-userspace`.
- `EgressNetMode::VzUserspace` is a Vz backend network mode, not a global
  launch selector.

This rule is required for later common-core extraction. Network vocabulary may
still be imperfect, but backend-specific behavior must not be hidden behind a
network-mode-only switch.

## Readiness Contract

`ready` is intentionally the first-contact interactive gate:

- the VM is running
- the SSH proxy can execute `/bin/true`
- required VFS mounts are connected
- dynamic CA/principal/identity state has been staged

`ready` does not include:

- package installation
- guest Rust builds
- npm repair
- package-manager quiescence
- outbound HTTPS certification
- optional benchmarks

Those checks belong in `validate <guest>` or saved scenarios.

## Image Contract

`build-guest.sh` is the Vz base-image builder for this slice. It currently
bakes Vz-only smoke-image hardening:

- masks `apt-daily`, `apt-daily-upgrade`, and `unattended-upgrades`
- forces apt IPv4 with `/etc/apt/apt.conf.d/99motlie-force-ipv4`

This is not yet a converged CH/Vz image builder contract. The `v1.4` CH builder
does not have matching hooks. Treat these as explicit Vz slice assumptions
until image convergence moves them into shared image content or removes them.

The first-contact path must fail fast when immutable base-image content is
missing. Rebuild the image; do not add hidden runtime repair to `launch-vz.sh`.

## Preconditions

From the repository root:

```bash
cargo build -p motlie-vmm --example harness_v1_45 --example repl_host_v1_45
cargo build -p motlie-vnet --example vz_egress_helper_v1_25
```

Build the guest image when base content changes:

```bash
./libs/vmm/examples/v1.45/build-guest.sh
```

Required launch artifacts:

- `libs/vmm/examples/v1.45/artifacts/build/vz-vsock-runner`
- `target/debug/examples/vz_egress_helper_v1_25`
- `disk.img` and `nvram.bin` under the base VM directory

Developer rebuild knobs:

- `MOTLIE_VZ_SKIP_RUNNER_BUILD=0`
- `MOTLIE_VZ_SKIP_EGRESS_HELPER_BUILD=0`

These knobs are for explicit rebuilds. They are not the default startup path.

## Core Modes

Smoke:

```bash
./target/debug/examples/harness_v1_45
```

PTY:

```bash
./target/debug/examples/harness_v1_45 pty
```

Scenario:

```bash
./target/debug/examples/harness_v1_45 scenario \
  ./libs/vmm/examples/v1.45/scenarios/multiguest-validate.json
```

Shell:

```bash
./target/debug/examples/harness_v1_45 shell
```

Useful flags:

```bash
./target/debug/examples/harness_v1_45 --root /tmp/motlie-v145
./target/debug/examples/harness_v1_45 --result-json /tmp/result.json
./target/debug/examples/harness_v1_45 --terminal-backend shadow
```

`shadow` is the default terminal backend. `vt100` is retained as an explicit
fallback/debugging option.

## Standard Verification Matrix

Use this matrix before posting or merging v1.45 changes:

```bash
cargo check -p motlie-vmm --example harness_v1_45 --example repl_host_v1_45
cargo check -p motlie-vnet --example vz_egress_helper_v1_25
cargo test -p motlie-vfs
cargo clippy -p motlie-vmm --example harness_v1_45 --example repl_host_v1_45 -- -D warnings
./target/debug/examples/harness_v1_45 scenario ./libs/vmm/examples/v1.45/scenarios/multiguest-validate.json
./target/debug/examples/harness_v1_45 scenario ./libs/vmm/examples/v1.45/scenarios/auto-provision-ssh.json
```

After harness shutdown, the runtime root printed by the harness should not
contain leaked Vz disk artifacts unless `MOTLIE_VZ_KEEP_GUEST_DISKS` was set.
`v1.45` launches Apple Virtualization.framework directly through
`vz-vsock-runner`; Tart is not part of the runtime verification path.

## Manual Certification

In shell mode:

```text
boot alice
ready alice
validate alice
where alice
```

Then SSH through the printed proxy:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p <port> alice@localhost
```

Inside the guest:

```bash
test -w ~/.codex && echo CODEX_WRITABLE
mount | grep -E 'agent-state|/home/alice|workspace'
curl -fsS https://example.com >/dev/null && echo EGRESS_OK
```

Expected:

- `CODEX_WRITABLE`
- `/workspace` and `/agent-state` mounted via FUSE/Motlie VFS
- `EGRESS_OK` over Vz userspace egress

## Troubleshooting

Use `where <guest>` first. It prints the runtime directory, launch log, serial
log, VFS socket, and captured artifacts.

Primary files:

- `<runtime_dir>/launch.log`
- `<runtime_dir>/serial.log`
- `<runtime_dir>/vz-launch-result.json`
- `<runtime_dir>/vz-phases.log`
- `<runtime_dir>/control-plane-ready`
- `<runtime_dir>/interactive-ready`
- `<runtime_dir>/validation-complete`

If first-contact SSH is slow, inspect `vz-phases.log` before changing code.
The expected long pole is VM/Linux boot plus transitional dynamic identity,
CA, and VFS staging. Runtime package installs, npm installs, Rust builds, and
host cargo builds are contract violations on the default path.

## Related Docs

- [README.md](./README.md)
- [../../docs/CONVERGENCE.md](../../docs/CONVERGENCE.md)
- [../../docs/DESIGN_VZ.md](../../docs/DESIGN_VZ.md)
