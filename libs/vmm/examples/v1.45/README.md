# v1.45 Apple Vz Harness

`v1.45` is the Apple Virtualization.framework slice for the `libs/vmm`
library lifecycle. It is not a `v1.4` Cloud Hypervisor runbook under a new
directory.

The slice proves that the same high-level API used by the CH harness can drive
a Vz-backed guest on macOS:

- `prepare()`
- `boot()`
- `VmHandle::ready(...)`
- `VmHandle::exec(...)`
- `VmHandle::open_pty(...)`
- `VmHandle::shutdown()`
- `VmHandle::observability()`

## Current Backend Split

The live `v1.45` harness configuration is:

- `HypervisorBacking::AppleVirtualizationShell`
  - materializes `launch.sh`
  - runs `launch-vz.sh`
  - supervises the Objective-C `vz-vsock-runner`
- `FilesystemBacking::MotlieVfs`
  - provides the guest-visible `/home/<user>`, `/workspace`, and
    `/agent-state` mounts through the Motlie VFS server and the Vz vsock
    bridge
  - this is intentionally not `FilesystemBacking::HypervisorManaged` in this
    slice
- `NetworkBacking::HypervisorManaged`
  - no `NetworkHandle::MotlieVnet` is provisioned on macOS
  - egress is the Vz userspace helper path selected by the Vz backend
- `ControlPlaneBacking::MotlieSshProxy`
  - owns external SSH, guest exec, PTY, and auto-provisioning

Backend selection is owned by the hypervisor backing. `EgressNetMode` alone
must not route a CH caller into `launch-vz.sh`.

## Preconditions

Run commands from the repository root:

```bash
cargo build -p motlie-vmm --example harness_v1_45 --example repl_host_v1_45
cargo build -p motlie-vnet --example vz_egress_helper_v1_25
```

Build or refresh the Vz base image when immutable guest content changes:

```bash
./libs/vmm/examples/v1.45/build-guest.sh
```

The launch path expects these host-side artifacts to exist before first
contact:

- `libs/vmm/examples/v1.45/artifacts/build/vz-vsock-runner`
- `target/debug/examples/vz_egress_helper_v1_25`
- a Vz base VM directory with `disk.img` and `nvram.bin`

The base VM path defaults through the `v1.35` source-base artifact and can be
overridden with `MOTLIE_VZ_BASE_VM_DIR`.

## Image Contract

`v1.45/build-guest.sh` currently contains Vz-specific smoke-image hardening:

- masks `apt-daily`, `apt-daily-upgrade`, and `unattended-upgrades`
- writes `Acquire::ForceIPv4 "true";`

Those settings are not present in the `v1.4` CH builder today. They are
documented Vz slice behavior for the current userspace egress helper and
first-contact latency work, not a claim that the CH and Vz image builders are
already identical. Future image convergence must either move a shared contract
into both builders or remove/narrow the Vz-only hardening with explicit
validation.

First-contact SSH must not repair stale base-image content. If the launcher
reports a missing package, CLI, service unit, SSH CA config, profile script, or
`MOTLIE_CONVERGENCE_AGENT_STATE_SETUP_V3`, rebuild the base image instead of
adding runtime fallback copies to `launch-vz.sh`.

## Standard Validation

Fast single-guest smoke:

```bash
./target/debug/examples/harness_v1_45
```

Multi-guest parity scenario:

```bash
./target/debug/examples/harness_v1_45 scenario \
  ./libs/vmm/examples/v1.45/scenarios/multiguest-validate.json
```

Unknown-user SSH auto-provisioning scenario:

```bash
./target/debug/examples/harness_v1_45 scenario \
  ./libs/vmm/examples/v1.45/scenarios/auto-provision-ssh.json
```

Manual shell:

```bash
./target/debug/examples/harness_v1_45 shell
```

The harness prints the per-run demo root, socket root, and proxy URL. Use the
printed `ssh://localhost:<port>` endpoint for manual external SSH checks.

## Manual Verification

After booting a guest from `shell`, run:

```bash
validate <guest>
where <guest>
```

Then connect through the printed proxy port and verify:

```bash
test -w ~/.codex && echo CODEX_WRITABLE
mount | grep -E 'agent-state|/home/manual|workspace'
curl -fsS https://example.com >/dev/null && echo EGRESS_OK
```

Expected backend-specific observations:

- `/workspace` and `/agent-state` are Motlie VFS/FUSE mounts
- outbound network uses the Vz userspace egress helper
- `ready` means interactive/control readiness, not full egress certification
- full certification is `validate <guest>` or a saved scenario

## Related Docs

- [HARNESS.md](./HARNESS.md)
- [CONVERGENCE.md](../../docs/CONVERGENCE.md)
- [DESIGN_VZ.md](../../docs/DESIGN_VZ.md)
