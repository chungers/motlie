# `v1.15` Apple Vz Managed Guestfs PoC

`v1.15` is the Apple Vz parallel to
[`libs/vfs/examples/v1.1`](../v1.1/README.md).

The target is the same high-level behavior as the Cloud Hypervisor path:

- one host-side `FsServer` per guest
- multiple tags per guest
- one listener per guest, with `TAG <name>` routing inside that listener
- a generic shared guest base
- per-guest runtime state created at launch time

The difference is the transport and launcher:

- CH `v1.1`: guest vsock -> host Unix socket
- Vz `v1.15`: guest TCP -> host TCP listener, booted by Tart

This remains a PoC:

- it keeps the current managed `FsServer` semantics
- it intentionally uses an ugly transport shim under `libs/vfs/src/vz/`
- it is allowed to be backend-specific while we prove feasibility
- it does **not** claim final transport parity with CH yet

## What `v1.15` Proves

- Apple Vz guests can boot from a shared Tart-backed base image
- a guest-specific clone can connect back to a host-managed `FsServer`
- the existing framed `FsOp` / `FsResult` protocol works over plain TCP
- the `TAG <name>` handshake still provides per-guest multi-tag routing
- the runtime remains all-userspace and ephemeral on the host
- the managed guestfs semantics are portable enough to survive the Apple Vz path

## What `v1.15` Does Not Yet Prove

- full `libs/vmm` integration
- final transport traits or adapter boundaries
- final VFS policy work from `#134`
- a native `vz-runner` launcher
- final transport parity with CH's vsock path

## Transport Model

Each guest gets one host TCP listener:

- Alice: `0.0.0.0:5501`
- Bob: `0.0.0.0:5502`

Each mount inside the guest:

1. connects to the guest's listener address via the Tart NAT gateway
2. sends `TAG <name>\n`
3. switches to framed `FsOp` / `FsResult` traffic

This preserves the same guest/tag routing shape as `v1.1`, but it is only a
temporary feasibility bridge.

The intended final Apple Vz transport is **not** Tart NAT. Apple
Virtualization.framework exposes a virtio-socket device through:

- `VZVirtioSocketDeviceConfiguration`
- `VZVirtioSocketDevice`
- `VZVirtioSocketListener`
- `VZVirtioSocketConnection`

That API is the closest equivalent to the CH/KVM virtio-vsock path already used
by `v1` and `v1.1`. The likely final Vz shape is therefore:

- guest `motlie-vfs-guest` speaking the existing framed protocol over a
  `VZVirtioSocketConnection`
- a host-side Vz adapter inside `vz-runner`
- no dependence on general guest networking for filesystem transport

`v1.15` proves filesystem feasibility, not final transport choice.

## Files

| File | Purpose |
|------|---------|
| `build-guest.sh` | Build one generic Tart-backed base guest with `motlie-vfs-guest-v1_15` installed |
| `launch-vz.sh` | Clone and boot one guest-specific Tart VM, install guest-specific `mounts.yaml`, and start the guest mounter |
| `repl_host.rs` | Host-side multi-guest REPL example using TCP listeners instead of Unix sockets |
| `mounts.alice.yaml` | Alice guest mount template with placeholder host gateway / port |
| `mounts.bob.yaml` | Bob guest mount template with placeholder host gateway / port |
| `setup-multiguest.sh.vfs` | Host REPL provisioning script for both guests |
| `motlie-vfs-guest.service` | Guest-side systemd service for `motlie-vfs-guest-v1_15` |

## Host Requirements

- macOS on Apple Silicon
- Tart installed and working
- Rust toolchain on the host
- no `sudo` required on the host

The guest provisioning step installs build dependencies and Rust inside the VM.

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

- Alice `FsServer` on `0.0.0.0:5501`
- Bob `FsServer` on `0.0.0.0:5502`

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
- boots it headlessly
- computes the host Tart NAT gateway from the guest IP
- renders guest-specific `mounts.yaml`
- creates the guest user and mount directories
- restarts `motlie-vfs-guest.service`
- waits for the guest mounts to appear and validates host-backed content under
  `/workspace` and `/home/<guest>`

## Current Caveats

- the Vz PoC currently relies on Tart's guest agent for launch-time guest config
- the host TCP listener uses `0.0.0.0` so the guest can reach it through the Tart NAT gateway
- that TCP/NAT path is a PoC-only bridge and should be replaced by Apple
  virtio-socket in the final Vz transport
- the guest user IDs differ from `v1.1` because the base Ubuntu Tart image already reserves uid `1000` for `admin`
- the generic base build still patches the throwaway guest copy of the root workspace manifest to include `libs/vfs` before building
- default run VM names use a timestamped suffix so repeated launches do not
  depend on `tart delete` for cleanup

## Expected Next Step

If `v1.15` proves stable enough, the next step is not more PoC scripting.
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
