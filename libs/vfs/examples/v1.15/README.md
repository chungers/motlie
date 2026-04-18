# `v1.15` Apple Vz Managed Guestfs PoC

`v1.15` is the Apple Vz parallel to
[`libs/vfs/examples/v1.1`](../v1.1/README.md).

The target is the same high-level behavior as the Cloud Hypervisor path:

- one host-side `FsServer` per guest
- multiple tags per guest
- one listener per guest, with `TAG <name>` routing inside that listener
- a generic shared guest base
- per-guest runtime state created at launch time

The difference is the launcher and host bridge:

- CH `v1.1`: guest vsock -> host Unix socket
- Vz `v1.15`: guest virtio-socket -> Apple `VZVirtioSocketListener` ->
  host Unix socket, with Tart used only to provision the runtime disk

This remains a PoC:

- it keeps the current managed `FsServer` semantics
- it intentionally uses a small Objective-C helper instead of the final
  `libs/vmm` `vz-runner`
- it is allowed to be backend-specific while we prove the Vz socket path

## What `v1.15` Implements

- Apple Vz guests can boot from a shared Tart-backed base image
- a guest-specific clone can be reprovisioned, stopped, and handed off to an
  Apple Vz helper using `VZVirtioSocketDeviceConfiguration`
- the host-side `FsServer` and Unix socket layout match the `v1.1` CH shape
- the guest-side binary now uses Linux AF_VSOCK, not guest TCP
- the helper bridges `VZVirtioSocketConnection.fileDescriptor` onto the same
  host Unix socket contract already consumed by `VsockConnectionHandler`
- the runtime remains all-userspace and ephemeral on the host

## What `v1.15` Does Not Yet Prove

- a locally unsigned helper can complete VM startup on this Mac
- full `libs/vmm` integration
- final transport traits or adapter boundaries
- final VFS policy work from `#134`
- signed distribution of the helper on developer machines without a local Apple
  development identity

## Current Validation Status

`v1.15` has been validated through these stages on this M4 Pro host:

- `repl_host_v1_15` successfully provisions and listens on:
  - `/tmp/motlie-vfs-alice.vsock_5000`
  - `/tmp/motlie-vfs-bob.vsock_5000`
- `build-guest.sh` successfully builds `motlie-vfs-guest-v1_15` in-guest with
  `--features vsock,client` and installs the systemd units
- `build-vz-runner.sh` successfully compiles the example-local Apple Vz helper
- `launch-vz.sh --guest alice` successfully:
  - clones and reprovisions the Tart base guest
  - stops Tart after provisioning
  - launches the Apple Vz helper with the expected virtio-socket / Unix-socket
    arguments

The current runtime stop point is Apple entitlement enforcement:

- the helper exits at VM configuration/start with:
  `The process doesn’t have the "com.apple.security.virtualization" entitlement.`

So `v1.15` now proves the transport wiring and the launch handoff, but a fully
successful Apple Vz runtime still requires the signed helper boundary already
called out in the Vz design docs.

## Transport Model

Each guest gets one host Unix socket:

- Alice: `/tmp/motlie-vfs-alice.vsock_5000`
- Bob: `/tmp/motlie-vfs-bob.vsock_5000`

Each mount inside the guest:

1. connects to host CID `2`, guest port `5000`, over Linux AF_VSOCK
2. sends `TAG <name>\n`
3. switches to framed `FsOp` / `FsResult` traffic

On the host:

1. the helper boots the guest from the provisioned Tart disk image
2. the helper configures one `VZVirtioSocketDeviceConfiguration`
3. the helper installs a `VZVirtioSocketListener` on port `5000`
4. on each guest connection, the helper bridges the
   `VZVirtioSocketConnection.fileDescriptor` onto the guest's Unix socket path

This preserves the same guest/tag routing shape as `v1.1`, but with Apple
Virtualization.framework in place of Cloud Hypervisor's vsock socket bridge.

The relevant Apple API surface is:

- `VZVirtioSocketDeviceConfiguration`
- `VZVirtioSocketDevice`
- `VZVirtioSocketListener`
- `VZVirtioSocketConnection`

`v1.15` therefore now follows the intended Apple Vz transport direction rather
than the earlier Tart NAT/TCP feasibility bridge.

## Files

| File | Purpose |
|------|---------|
| `build-guest.sh` | Build one generic Tart-backed base guest with `motlie-vfs-guest-v1_15` installed |
| `build-vz-runner.sh` | Compile the Apple Vz helper that exposes `VZVirtioSocketListener` |
| `launch-vz.sh` | Clone and provision one guest-specific Tart VM, stop it, then boot the disk through the Apple Vz helper |
| `vz-vsock-runner.m` | Minimal Apple Vz helper that bridges `VZVirtioSocketConnection.fileDescriptor` to a host Unix socket |
| `repl_host.rs` | Host-side multi-guest REPL example using the same Unix socket / vsock bridge shape as `v1.1` |
| `mounts.alice.yaml` | Alice guest mount template |
| `mounts.bob.yaml` | Bob guest mount template |
| `setup-multiguest.sh.vfs` | Host REPL provisioning script for both guests |
| `motlie-vfs-guest.service` | Guest-side systemd service for `motlie-vfs-guest-v1_15` |
| `motlie-vfs-validate.service` | Guest-side oneshot validation unit that writes host-visible sentinel files |

## Host Requirements

- macOS on Apple Silicon
- Tart installed and working
- Rust toolchain on the host
- the ability to run a helper with `com.apple.security.virtualization`
- no `sudo` required on the host

The guest provisioning step installs build dependencies and Rust inside the VM.
The Apple Vz helper itself still requires the virtualization entitlement at
runtime.

## Quick Start

### 1. Build the generic Vz base guest

```bash
cd libs/vfs/examples/v1.15
./build-guest.sh
```

This produces a reusable Tart VM named `motlie-v1-15-base` by default.

### 2. Start the host-side managed VFS servers

From the repo root:

```bash
cd /path/to/motlie
cat libs/vfs/examples/v1.15/setup-multiguest.sh.vfs | \
  cargo run --manifest-path libs/vfs/Cargo.toml --example repl_host_v1_15 -- --empty
```

That one host process owns:

- Alice `FsServer` on `/tmp/motlie-vfs-alice.vsock_5000`
- Bob `FsServer` on `/tmp/motlie-vfs-bob.vsock_5000`

### 3. Launch Alice and Bob

In separate terminals:

```bash
cd libs/vfs/examples/v1.15
./launch-vz.sh --guest alice
```

```bash
cd libs/vfs/examples/v1.15
./launch-vz.sh --guest bob
```

Each launch:

- clones the generic base guest
- boots it once with Tart for guest provisioning only
- installs guest-specific `mounts.yaml`
- creates the guest user and mount directories
- installs a oneshot validation unit that writes host-visible sentinel files
- stops Tart
- launches the same disk image through the Apple Vz helper with a
  `VZVirtioSocketListener`
- waits for host-visible validation sentinels under the mounted backing
  directories

## Current Caveats

- Tart is still used for guest provisioning because it is the easiest signed
  launcher available on this host
- the final runtime launch now goes through Apple virtio-socket rather than
  guest TCP/NAT
- the helper still requires the virtualization entitlement, so unsigned local
  runs will fail at VM configuration validation/start
- the guest user IDs differ from `v1.1` because the base Ubuntu Tart image already reserves uid `1000` for `admin`
- the generic base build still patches the throwaway guest copy of the root workspace manifest to include `libs/vfs` before building
- default run VM names use a timestamped suffix so repeated launches do not
  depend on `tart delete` for cleanup

## Expected Next Step

If `v1.15` proves stable enough with a signed helper, the next step is not more
PoC scripting.
The next step is the real cross-backend cleanup in:

- `libs/vfs/docs/DESIGN_XBACKENDS.md`
- `libs/vfs/docs/PLAN_XBACKENDS.md`

## External References

- Apple Virtualization sockets overview:
  https://developer.apple.com/documentation/virtualization/sockets
- `VZVirtioSocketListenerDelegate`:
  https://developer.apple.com/documentation/virtualization/vzvirtiosocketlistenerdelegate
- `VZVirtioSocketDevice.connect(toPort:)`:
  https://developer.apple.com/documentation/virtualization/vzvirtiosocketdevice/connect%28toport%3Acompletionhandler%3A%29
- VM runtime `socketDevices`:
  https://developer.apple.com/documentation/virtualization/vzvirtualmachine/socketdevices
