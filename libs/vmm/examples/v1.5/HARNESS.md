# v1.5 Common CH/VZ Harness Contract

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-15 | @vmm-cdx | Record the CH-to-VZ rootfs handoff: `mbuild build --target ch` emits `assembled-rootfs.tar` with digest evidence, VZ consumes that tarball during image build without baking demo guest identities, and the Codex PTY scenario now checks stable welcome text instead of a layout-dependent footer |
| 2026-05-05 | @vmm-cdx | Fix CH guest boot-graph cycles by moving the VFS/agent-state units under `cloud-init.target`, and update the PTY Codex welcome assertion to the current footer text |
| 2026-05-04 | @codex-vz | Remove the standalone `vz_egress_helper_v1_5`; VZ egress is hosted by `harness_v1_5` either as embedded runtime state or the `vz-egress` image-build subcommand |
| 2026-05-04 | @codex-vz | Move default VZ userspace egress into the VMM runtime/harness path and make `launch-vz.sh` consume, not own, the egress socket |
| 2026-05-03 | @codex-vz | Promote guest-level VFS memfs views, apt-backed egress, and Codex/Claude startup into the v1.5 conformance scenario and harness validation surface |
| 2026-05-03 | @codex-vz | Add `build-image.sh` as the common builder entrypoint and `build-ch-artifacts.sh` as the Linux CH emitter for the common v1.5 guest contract |
| 2026-05-03 | @codex-vz | Mark v1.5 as the unified CH/VZ script home, add common-contract.sh as the shell-to-Rust extraction seam, and require CH launch to consume v1.5 guest-contract metadata instead of legacy v1.4 artifacts |
| 2026-04-26 | @codex-vz | Replace copied v1.4/CH runbook text with the actual v1.5 Vz backend split, validation commands, and Vz-only image hardening caveat |
| 2026-04-25 | @codex-vz | Require prebuilt Vz runner artifacts by default so first-contact startup cannot hide host cargo builds |
| 2026-04-25 | @codex-vz | Document that v1.5 Vz first-contact SSH follows the shared convergence contract: `ready` means interactive SSH plus required mounts, and full VFS/VNET/egress certification remains `validate` or a saved scenario |

## Purpose

The `v1.5` harness is the common CH/VZ validation surface for `libs/vmm`.
It exists to prove that both backend paths can share the same guest contract,
seed schema, service graph, scenario format, and operator shell while keeping
platform-specific adaptation visible and reviewable.

The durable contract is the saved scenario format plus the library API. Shell
mode is for exploration and manual certification.

## Actual Runtime Configuration

The current harness creates this runtime:

```text
HypervisorBacking::AppleVirtualizationShell(VzShellBackend)
FilesystemBacking::MotlieVfs(MotlieVfsBacking)
NetworkBacking::VzUserspaceEgress(VzUserspaceEgressBacking)
ControlPlaneBacking::MotlieSshProxy
```

Interpretation:

- Vz owns the VM process, serial log, and guest disk clone through the external
  `vz-vsock-runner` boundary.
- Motlie VFS is still the filesystem backing for guest workspace/home/state
  mounts in this slice.
- The VMM runtime owns the Vz userspace egress backend. `launch-vz.sh` consumes
  the rendered datagram socket and must not spawn a helper in the harness path.
- There is no macOS CH-style `MotlieVnet` vhost-user handle. The Vz egress
  backend uses the same libslirp core through the VZ file-handle NIC transport.
- The SSH proxy remains the common control plane for exec, PTY, and external
  SSH. Vz first-contact SSH uses the host-forward port selected by the embedded
  egress backend and written to `control-port`.

Any documentation or scenario text that says this slice validates "Motlie
vnet" on macOS is wrong. The correct phrase is "Vz userspace egress".

There is no standalone v1.5 VZ egress binary. The normal launch path embeds
egress in the VMM runtime before `launch-vz.sh` starts the Apple VZ runner.
The VZ image-builder path can still need a temporary host egress process while
customizing the base VM, but that process is `harness_v1_5 vz-egress`, not a
separate helper artifact.

## Backend Boundary Rules

Backend choice is owned by `HypervisorBacking`.

- CH backends render `launch-ch.sh` and accept CH/common network modes.
- Vz backends render `launch-vz.sh` and require
  `--admin-net=none --egress-net=vz-userspace`.
- `EgressNetMode::VzUserspace` is a Vz backend network mode, not a global
  launch selector.
- Both scripts live in `examples/v1.5` and source `common-contract.sh`; common
  constants and seed schema must not be duplicated in backend-specific scripts.

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

`build-image.sh` is the common builder entrypoint. It currently dispatches to
backend shell emitters and is the CLI seam for the future Rust image builder.
`build-guest.sh` packages the common v1.5 guest payload through the Vz native
base-image path. `build-ch-artifacts.sh` is the Linux CH emitter under the same
`examples/v1.5` tree; it emits `artifacts/base/rootfs.squashfs`,
`artifacts/base/Image|vmlinux.bin`, and `artifacts/base/guest-contract.json`
from the same guest contract.

The guest contract metadata must record:

- `MOTLIE_VMM_GUEST_MOUNTER_V1_5`
- `--no-default-features --features guest-vfs`
- VMM-owned guest binary paths under `/opt/motlie/v1.5/guest/bin` and
  `/usr/local/bin`

`motlie-vfs-guest.service` and `motlie-agent-state.service` must be installed
under `cloud-init.target`. The guest mounter orders itself after
`cloud-final.service`, and `motlie-agent-state.service` orders itself after the
guest mounter, so enabling either one under `multi-user.target` creates an
ordering cycle that prevents CH harness readiness from completing.
This is an image-profile invariant for OCI-derived `ubuntu-systemd` guests, not
a CH-only runtime workaround: image importers and backend emitters must preserve
the same pre-boot service graph for both CH and VZ artifacts.

The v1.5 builders bake smoke-image hardening during image assembly:

- masks `apt-daily`, `apt-daily-upgrade`, and `unattended-upgrades`
- forces apt IPv4 with `/etc/apt/apt.conf.d/99motlie-force-ipv4`

This is image content, not first-contact repair. If a backend cannot use this
hardening, document the technical reason before diverging.

For the VZ #271 bridge, `mbuild build --target ch` emits
`assembled-rootfs.tar` and `mbuild-common-rootfs.json` from the common
OCI-derived rootfs before CH-specific boot adaptations. `build-guest.sh` accepts
that tarball through `MOTLIE_V15_ASSEMBLED_ROOTFS_TARBALL`. The tarball is
copied through the build seed disk and applied while producing VZ artifacts; the
emitted `build-result.json` and `guest-contract.json` record `rootfs_input`
with the canonical tarball path, byte size, and sha256 digest. The VZ adapter
pre-scans the tarball before copying and verifies the seeded guest copy before
root extraction. After extraction it normalizes OpenSSH StrictModes path
ancestors (`/`, `/etc`, and `/etc/ssh/*`) to `root:root 0755`; launch-time
provisioning must fail fast if this image contract is broken rather than
weakening sshd. `mbuild build --target vz --rootfs-tarball <tar>` is the
preferred macOS entrypoint because the builder validates the tarball and passes
it through the configured VZ adapter env. This is the only allowed transitional
handoff from the common rootfs assembler into the Apple VZ adapter. Harness
launch, `ready`, first SSH, and scenario steps must consume the completed
artifacts only; they must not run rootfs assembly, package installation, npm
repair, or guest binary builds. Reusable VZ image build must not bake demo
guest identities; `alice`, `bob`, and future harness guests are provisioned from
per-guest seed/runtime inputs.

The first-contact path must fail fast when immutable base-image content is
missing. Rebuild the image; do not add hidden runtime repair to `launch-vz.sh`
or `launch-ch.sh`.

## Guest Conformance Exit Criteria

`scenarios/multiguest-validate.json` is the v1.5 guest functionality
conformance suite. Passing `ready` is not enough for exit.

The suite must prove, for multiple guests:

- VFS/FUSE memfs layer views exist for `/home/<user>`, `/workspace`, and
  `/agent-state`, are mounted as FUSE, and are writable from the guest.
- Guest-specific `/workspace` content is isolated between guests.
- Backend internet egress is functional through DNS/HTTPS and through
  `sudo -n apt-get update`.
- Codex and Claude CLIs are present and start with `--version` without OS-level
  execution errors such as permission, missing executable, or operation errors.

The harness shell `validate <guest>` and default smoke path use the same core
commands: `check_vfs_memfs`, `wait_egress_ready`, `apt_update`, and
`check_agent_cli`.

## Preconditions

From the repository root:

```bash
cargo build -p motlie-vmm --example harness_v1_5
```

Build the guest image when base content changes:

```bash
./libs/vmm/examples/v1.5/build-image.sh --backend vz
```

Build CH artifacts on a Linux CH host when common base content changes:

```bash
./libs/vmm/examples/v1.5/build-image.sh --backend ch
```

Required launch artifacts:

- `libs/vmm/examples/v1.5/artifacts/build/vz-vsock-runner`
- `disk.img` and `nvram.bin` under the base VM directory

`launch-vz.sh` and the harness do not compile or sign host helpers. Build the
runner before launch; missing artifacts are contract failures. The default Vz
egress path is embedded in `harness_v1_5`.

## Core Modes

Smoke:

```bash
./target/debug/examples/harness_v1_5
```

PTY:

```bash
./target/debug/examples/harness_v1_5 pty
```

Scenario:

```bash
./target/debug/examples/harness_v1_5 scenario \
  ./libs/vmm/examples/v1.5/scenarios/multiguest-validate.json
```

Shell:

```bash
./target/debug/examples/harness_v1_5 shell
```

The v1.5 REPL entrypoint is intentionally removed. `harness_v1_5 shell` is the
single interactive operator surface for VZ now and the intended CH/VZ shared
surface for the next convergence slice.

Useful flags:

```bash
./target/debug/examples/harness_v1_5 --root /tmp/motlie-v15
./target/debug/examples/harness_v1_5 --result-json /tmp/result.json
./target/debug/examples/harness_v1_5 --terminal-backend shadow
./target/debug/examples/harness_v1_5 --backend vz
./target/debug/examples/harness_v1_5 --backend ch
./target/debug/examples/harness_v1_5 shell --auto-provision on
```

`shadow` is the default terminal backend. `vt100` is retained as an explicit
fallback/debugging option.

`--backend vz` is the current validated local path and remains the default.
`--backend ch` is the convergence hook for running the same harness on a Linux
CH host after `build-image.sh --backend ch` emits
`artifacts/base/rootfs.squashfs`, `Image` or `vmlinux.bin`, and
`guest-contract.json`. On non-Linux hosts the CH path fails early because
Motlie VNET is Linux-only.

Interactive SSH auto-provisioning is explicit harness state. Shell mode starts
with `auto-provision=off` unless `--auto-provision on` is passed, and operators
can toggle it with `auto-provision on`, `auto-provision off`, or
`auto-provision status`. Scenario mode owns its own resolver state and keeps the
auto-provision scenario enabled for deterministic e2e coverage.

## Standard Verification Matrix

Use this matrix before posting or merging v1.5 changes:

```bash
cargo check -p motlie-vmm --example harness_v1_5
cargo test -p motlie-vfs
cargo clippy -p motlie-vmm --example harness_v1_5 -- -D warnings
./target/debug/examples/harness_v1_5 --backend vz scenario ./libs/vmm/examples/v1.5/scenarios/multiguest-validate.json
./target/debug/examples/harness_v1_5 --backend vz scenario ./libs/vmm/examples/v1.5/scenarios/auto-provision-ssh.json
./libs/vmm/examples/v1.5/integration/harness-auto-provision-smoke.sh
```

After harness shutdown, the runtime root printed by the harness should not
contain leaked Vz disk artifacts unless `MOTLIE_VZ_KEEP_GUEST_DISKS` was set.
`v1.5` launches Apple Virtualization.framework directly through
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
sudo -n apt-get update
codex --version
claude --version
```

Expected:

- `CODEX_WRITABLE`
- `/workspace` and `/agent-state` mounted via FUSE/Motlie VFS
- `EGRESS_OK` over Vz userspace egress
- `apt-get update` succeeds over backend internet egress
- `codex --version` and `claude --version` start without OS-level execution errors

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
- [../../docs/DESIGN_GUEST_IMAGE.md](../../docs/DESIGN_GUEST_IMAGE.md)
- [../../docs/CONVERGENCE.md](../../docs/CONVERGENCE.md)
- [../../docs/DESIGN_VZ.md](../../docs/DESIGN_VZ.md)
