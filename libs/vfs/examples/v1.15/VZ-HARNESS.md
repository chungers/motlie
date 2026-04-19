# v1.15 Apple Vz Harness

`v1.15` is the Apple Vz parallel to
[`libs/vfs/examples/v1.1`](../v1.1/README.md).

The harness keeps the same high-level topology as `v1.1`:

- one `FsServer` per guest VM
- multiple mount tags per guest
- one guest-scoped Unix socket per host-side server
- one guest-side FUSE mount thread per tag
- one shared host process managing all guest-scoped `FsServer` instances

The runtime difference is the VMM boundary:

- `v1.1` uses Cloud Hypervisor's vsock -> Unix-socket bridge
- `v1.15` uses a signed Apple helper that accepts virtio-socket connections
  and bridges them onto the same Unix socket contract

## Topology

```text
Host                                          Guest VMs
----                                          ---------
repl_host_v1_15 (one process)
  guest alice -> FsServer + socket /tmp/motlie-vfs-alice.vsock_5000
    mounts:
      alice-home -> /tmp/.../alice-home       alice VM
      alice-workspace -> /tmp/.../alice-workspace

  guest bob -> FsServer + socket /tmp/motlie-vfs-bob.vsock_5000
    mounts:
      bob-home -> /tmp/.../bob-home           bob VM
      bob-workspace -> /tmp/.../bob-workspace

signed vz-vsock-runner
  guest AF_VSOCK port 5000 -> VZVirtioSocketListener -> host Unix socket
```

Each guest mount connection still sends `TAG <name>\n` immediately after the
transport connection is established. `repl_host_v1_15` keeps the same
guest-socket-first / tag-second routing model as the CH harness.

## Prerequisites

- macOS on Apple Silicon
- Tart installed and working
- Rust toolchain on the host
- Apple Development signing identity available in Keychain
- `vz.entitlements` with `com.apple.security.virtualization`
- a signed `vz-vsock-runner` built through `build-vz-runner.sh`

## Start Host Server

From the repo root:

```bash
cat libs/vfs/examples/v1.15/setup-multiguest.sh.vfs | \
  cargo run --manifest-path libs/vfs/Cargo.toml --example repl_host_v1_15 -- --empty
```

That host process owns:

- Alice `FsServer` on `/tmp/motlie-vfs-alice.vsock_5000`
- Bob `FsServer` on `/tmp/motlie-vfs-bob.vsock_5000`

## Launch Guests

Build the base guest image once:

```bash
cd libs/vfs/examples/v1.15
./build-guest.sh
```

Build and sign the Apple helper:

```bash
export MOTLIE_VZ_CODESIGN_IDENTITY='Apple Development: Your Name (TEAMID)'
export MOTLIE_VZ_ENTITLEMENTS_FILE='libs/vfs/examples/v1.15/vz.entitlements'
./build-vz-runner.sh
```

Launch Alice and Bob in separate terminals:

```bash
./launch-vz.sh --guest alice --vm-name motlie-v1-15-alice-iter
./launch-vz.sh --guest bob --vm-name motlie-v1-15-bob-iter
```

Each launch performs:

1. fresh per-run clone from the generic base disk
2. native Apple Vz boot through the signed helper
3. guest IP discovery over the runner NAT
4. guest-specific provisioning over SSH
5. direct validation of the mounted guest view
6. clean teardown by default

## Lifecycle Contract

The intended default lifecycle is:

- fresh clone
- guest-specific provisioning
- validation
- clean teardown

That means a successful run should not leave behind:

- a live `vz-vsock-runner`
- the per-run VM clone
- stale pid files
- stale per-run serial logs being mistaken for a live run
- stale Unix socket ownership that blocks the next launch

`MOTLIE_VZ_KEEP_RUNNING=1` exists only for deliberate manual inspection.
`MOTLIE_VZ_REUSE_VM=1` remains a temporary local debug escape hatch and is not
part of the intended long-term contract.

## Validate

Alice VM:

```bash
mount | grep -E '/home/alice|/workspace'
ls -la /home/alice/.ssh
cat /home/alice/.env
cat /workspace/README.md
```

Bob VM:

```bash
mount | grep -E '/home/bob|/workspace'
ls -la /home/bob/.ssh
cat /home/bob/.env
cat /workspace/README.md
```

Expected shape:

- `/home/alice` is backed by the `alice-home` tag
- `/home/bob` is backed by the `bob-home` tag
- both guests mount `/workspace` through guest-specific tags and guest-specific
  host backings

## Failure Surfaces

`launch-vz.sh` now reports compact failure context when validation fails:

- runner log tail
- serial log tail
- guest IP when known

That is the review/debug parallel to the CH harness, where the operator should
not need to reconstruct the entire state manually to understand a launch
failure.
