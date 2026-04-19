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
  host Unix socket, with guest-specific runtime provisioning delivered over the
  native runner's NAT/SSH path after boot

This remains a PoC:

- it keeps the current managed `FsServer` semantics
- it intentionally uses a small Objective-C helper instead of the final
  `libs/vmm` `vz-runner`
- it is allowed to be backend-specific while we prove the Vz socket path

## What `v1.15` Implements

- Apple Vz guests can boot from a shared Tart-backed base image
- a guest-specific clone can be booted directly by the Apple Vz helper without
  a Tart runtime handoff
- guest-specific userspace provisioning happens after native boot over the
  runner's NAT/SSH path, not through Tart runtime commands
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
- `build-guest.sh` successfully produces a reusable stock Ubuntu base clone
- `build-vz-runner.sh` successfully compiles the example-local Apple Vz helper
- a signed `vz-vsock-runner` has already been shown to boot the guest directly
  to a serial login prompt on this machine
- `launch-vz.sh --guest alice` now completes a native boot, provisions the live
  guest over runner NAT/SSH, starts `motlie-vfs-guest.service`, and validates
  the mounted guest view against the same `README.md`, `.env`, and
  `.ssh/authorized_keys` expectations used by `v1.1`

The current operator stop point is helper signing from the developer shell:

- the example-local helper must be signed with
  `com.apple.security.virtualization` after each rebuild
- `build-vz-runner.sh` supports this through
  `MOTLIE_VZ_CODESIGN_IDENTITY` and `MOTLIE_VZ_ENTITLEMENTS_FILE`
- from my noninteractive session, `codesign` still fails against the user
  keychain with `errSecInternalComponent`, so the final signing step must be
  done from the developer shell on this host

So `v1.15` now proves the native transport wiring and native launch model, and
the remaining stop point is the signed-helper developer workflow rather than
Tart runtime dependence.

## Transport Model

Each guest gets one host Unix socket:

- Alice: `/tmp/motlie-vfs-alice.vsock_5000`
- Bob: `/tmp/motlie-vfs-bob.vsock_5000`

Each mount inside the guest:

1. connects to host CID `2`, guest port `5000`, over Linux AF_VSOCK
2. sends `TAG <name>\n`
3. switches to framed `FsOp` / `FsResult` traffic

On the host:

1. the helper boots the guest from the cloned Tart disk image
2. the helper configures one `VZVirtioSocketDeviceConfiguration`
3. the helper installs a `VZVirtioSocketListener` on port `5000`
4. after boot, `launch-vz.sh` provisions the guest over NAT/SSH
5. on each guest connection, the helper bridges the
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
| `build-guest.sh` | Build one generic Tart-backed base guest image |
| `build-vz-runner.sh` | Compile the Apple Vz helper that exposes `VZVirtioSocketListener` |
| `launch-vz.sh` | Clone one guest-specific VM disk, boot it through the Apple Vz helper, provision the guest over native NAT/SSH, then validate the mounted guest view |
| `vz-vsock-runner.m` | Minimal Apple Vz helper that bridges `VZVirtioSocketConnection.fileDescriptor` to a host Unix socket |
| `repl_host.rs` | Host-side multi-guest REPL example using the same Unix socket / vsock bridge shape as `v1.1` |
| `mounts.alice.yaml` | Alice guest mount template |
| `mounts.bob.yaml` | Bob guest mount template |
| `setup-multiguest.sh.vfs` | Host REPL provisioning script for both guests |
| `motlie-vfs-guest.service` | Guest-side systemd service for `motlie-vfs-guest-v1_15` |

## Host Requirements

- macOS on Apple Silicon
- Tart installed and working
- Rust toolchain on the host
- the ability to sign and run a helper with `com.apple.security.virtualization`
- no `sudo` required on the host

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

- clones the generic base guest disk
- launches the guest directly through the Apple Vz helper with:
  - the cloned root disk
  - a `VZVirtioSocketListener`
- resolves the guest IP over the helper's NAT network
- provisions the guest directly over SSH:
  - creates the guest user
  - installs build prerequisites
  - builds and installs `motlie-vfs-guest-v1_15`
  - installs `mounts.yaml` and the systemd units
- validates the mounted guest view directly over SSH and stores the validation
  JSON locally under `artifacts/`

## Current Caveats

- Tart is still used to source and clone the reusable base disk image, but not
  to run the guest for the `v1.15` native path
- the final runtime launch now goes through Apple virtio-socket rather than
  guest TCP/NAT
- guest provisioning currently happens after native boot over SSH; it is not
  yet an offline image-build flow
- the helper still requires the virtualization entitlement, so rebuilt helpers
  must be signed before launch
- the guest user IDs differ from `v1.1` because the base Ubuntu Tart image
  already reserves uid `1000` for `admin`
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
