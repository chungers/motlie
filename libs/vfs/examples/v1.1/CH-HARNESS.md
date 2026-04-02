# v1.1 Cloud Hypervisor Harness

`v1.1` demos the intended topology from the DESIGN:

- one `FsServer` per guest VM
- multiple mount tags inside each server
- one vsock socket per guest VM
- one guest-side FUSE mount thread per tag
- one shared `repl_host_v1_1` process managing all guest-scoped `FsServer` instances

## Original Requirements

This harness was reshaped around these requirements:

- build one generic shared image set
- do not bake Alice/Bob identity into the build artifact
- allow guest-local writes to `/`, including disposable package installs
- keep Alice and Bob isolated from each other's writable root state
- recreate that writable root state on the next launch

The harness therefore combines:

- one shared squashfs base built ahead of time
- one fresh per-guest ext4 runtime overlay created on every launch
- `vfs` subtree mounts layered on top of that merged guest root

## What Differs Vs v1

- `v1` runs one guest-oriented server flow at a time; `v1.1` is meant to run Alice and Bob concurrently from one host process
- `v1` uses the legacy single-tag `--tag` / `--dir` server startup path; `v1.1` starts `repl_host_v1_1 --empty` and provisions guests from REPL commands
- `repl_host_v1_1` still supports repeated `--guest <id>=<socket>` and guest-qualified `--mount` flags, but the `v1.1` harness now drives guest creation through `provision` and `mount`
- `v1` typically mounts one guest path; `v1.1` mounts two tags per guest in the demo
- `v1` is single-socket, single-CID, single-artifact-tree; `v1.1` uses separate sockets, CIDs, and runtime overlays per guest while still using one host process
- `v1.1` relies on the guest-side `TAG <name>` handshake so one server socket can route multiple tag connections for the same guest
- `v1.1` builds one generic base image set and synthesizes per-guest writable overlays at launch time

## Topology

```text
Host                                          Guest VMs
----                                          ---------
repl_host_v1_1 (one process)
  guest alice -> FsServer + socket /tmp/motlie-vfs-alice.vsock
    mounts:
      alice-home -> /tmp/.../alice-home       alice VM
      alice-workspace -> /tmp/.../alice-workspace

  guest bob -> FsServer + socket /tmp/motlie-vfs-bob.vsock
    mounts:
      bob-home -> /tmp/.../bob-home           bob VM
      bob-workspace -> /tmp/.../bob-workspace
```

Each guest mount connection goes to that guest's socket and immediately sends `TAG <name>\n`. `repl_host_v1_1` uses the socket/listener to select the guest `FsServer`, then uses the tag handshake to route within that guest's mounts.

## Prerequisites

Before following this harness:

- install the host package set:

```bash
sudo apt install \
  cloud-hypervisor \
  curl \
  debian-archive-keyring \
  e2fsprogs \
  libfuse3-dev \
  mmdebstrap \
  pkg-config \
  squashfs-tools-ng \
  uidmap
```

- build the shared base image set
- ensure a working Rust toolchain with `cargo` is installed for `build-guest.sh` and `cargo run --example repl_host_v1_1`
- ensure `/dev/vhost-vsock` exists or load it with `sudo modprobe vhost_vsock`
- use a shell that can successfully run the rootless `build-guest.sh` flow if you are rebuilding images
- ensure `cloud-hypervisor` is installed and on `PATH`
- ensure `mkfs.ext4` is installed for runtime overlay creation

## Build

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh
```

Artifacts land in `artifacts/base/`.

You do not need a prior `v1` build. This harness uses only `v1.1` artifacts.

## Start Host Server

```bash
cat setup-multiguest.sh.vfs - | \
  cargo run -p motlie-vfs --example repl_host_v1_1 --features vsock -- --empty
```

`repl_host` still supports the old `--tag` / `--dir` path for `v1`, but `v1.1` now uses the separate `repl_host_v1_1` binary and provisions guests from REPL commands instead of startup flags.

In this demo:

- the one `repl_host_v1_1` process owns Alice's and Bob's `FsServer` instances
- Alice's `FsServer` owns `alice-home` and `alice-workspace`
- Bob's `FsServer` owns `bob-home` and `bob-workspace`
- each guest VM opens one connection per mount tag back to its own server socket

The combined setup script uses:

- `provision alice /tmp/motlie-vfs-alice.vsock 1000 1000`
- `mount alice alice-home=/home/alice,/tmp/motlie-vfs-demo/alice-home ...`
- `provision bob /tmp/motlie-vfs-bob.vsock 1001 1001`
- `mount bob bob-home=/home/bob,/tmp/motlie-vfs-demo/bob-home ...`

Prototype helper:

- `launch alice` generates a helper script, starts it asynchronously, and returns the REPL prompt immediately
- `launch bob` does the same for Bob
- `launch -script alice` prints the helper script instead of executing it
- `shutdown alice` and `shutdown bob` request VM shutdown via the CH API socket
- direct `launch` writes helper and serial logs under `/tmp/motlie-vfs-launch/<guest>/`
- this helper requires a shared base rebuilt with the current `build-guest.sh` so cloud-init is present in the guest
- cloud-init writes guest config and directories, then queues `systemctl --no-block start motlie-vfs-guest.service` so the guest mounter starts after final-stage config without deadlocking `cloud-final.service`
- the helper does not require `cloud-localds`; `launch-ch.sh` seeds the NoCloud files into the per-launch guest overlay directly

Operator note:

- use `cat setup-multiguest.sh.vfs - | ...` so stdin stays attached to your terminal after the setup script is consumed
- that keeps `repl_host_v1_1` serving guest connections while the VMs boot

## Launch Guests

Minimal concurrent demo:

```bash
./launch-ch.sh --guest alice --no-net
./launch-ch.sh --guest bob --no-net
```

With TAP networking enabled:

```bash
./launch-ch.sh --guest alice
./launch-ch.sh --guest bob
```

For disposable package-install experiments, use a larger runtime overlay:

```bash
./launch-ch.sh --guest alice --overlay-size 2G
./launch-ch.sh --guest bob --overlay-size 2G
```

Defaults:

| Guest | CID | vsock socket | API socket | Guest IP |
|------|-----|--------------|------------|----------|
| alice | 3 | `/tmp/motlie-vfs-alice.vsock` | `/tmp/motlie-vfs-alice-api.sock` | `192.168.249.2` |
| bob | 4 | `/tmp/motlie-vfs-bob.vsock` | `/tmp/motlie-vfs-bob-api.sock` | `192.168.250.2` |

Launch uses:

- the shared base kernel and squashfs from `artifacts/base/`
- a fresh per-guest ext4 runtime overlay created under `/tmp/motlie-vfs-v11-runtime/<guest>/`
- `2G` as the default runtime overlay size unless `--overlay-size` or `OVERLAY_SIZE` overrides it

Console/boot note:

- `launch-ch.sh` uses `console=ttyS0` with Cloud Hypervisor `--serial tty --console off`
- this avoids the 90-second `dev-hvc0.device` / `serial-getty@hvc0.service` timeout caused by a mismatched `hvc0` console expectation
- the launch terminal is the guest console

## Manual Run Order

Use three terminals for the clearest flow:

1. Terminal 1: start the shared `repl_host_v1_1`
2. Terminal 2: launch Alice guest
3. Terminal 3: launch Bob guest

If you want SSH access for both guests, omit `--no-net` on both launch commands and connect to the guest IPs shown above.

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
- both guests mount `/workspace`, but each guest gets its own tag and own host backing directory
- `.env` and `.ssh` contents come from the in-memory overlay injected by the corresponding `setup-*.sh.vfs` file
- extra files placed under `overlay.d/common/` or `overlay.d/<guest>/` appear through the guest root overlay at boot
- Alice and Bob do not share the same writable ext4; each guest gets its own runtime overlay file

Mounted-home note:

- if an SSH session starts before `/home/<guest>` is mounted by `motlie-vfs-guest`, that shell can retain the pre-mount cwd inode
- in that case `.` may still show the old base-image home even though `/home/<guest>` and `~` resolve through the mounted VFS path
- `cd "$HOME"` or a fresh SSH session after the mount is active fixes the view

Disposable root-mutation check:

Inside Alice, with networking enabled:

```bash
apt update
apt install -y python3
python3 --version
```

Then confirm:

- Bob still does not have `python3` if it was not already in the shared base
- after shutting down Alice and launching her again, that `python3` install is gone because the runtime overlay was recreated

## Shut Down

```bash
curl --unix-socket /tmp/motlie-vfs-alice-api.sock -X PUT http://localhost/api/v1/vm.shutdown
curl --unix-socket /tmp/motlie-vfs-bob-api.sock -X PUT http://localhost/api/v1/vm.shutdown
```

## Standalone Expectation

This document is the `v1.1` runtime runbook. It does not require a `v1` harness run first. The only shared pieces are the repository binaries and libraries that both examples build from the same workspace.
