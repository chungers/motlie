# v1.1 Multi-Guest / Multi-Mount Example

This example extends the `v1` vertical slice to demo:

- two guest VMs: `alice` and `bob`
- one `FsServer` per guest VM
- multiple mount tags per guest VM
- one host socket per guest VM, with per-connection tag routing inside that socket

## Original Requirements

`v1.1` now targets these requirements explicitly:

- build one generic shared image set
- launch Alice and Bob from that same build artifact
- keep guest identity and mount config out of the build
- let a guest mutate `/` for that launch only
- ensure Alice's guest-local package installs do not affect Bob
- ensure those guest-local installs disappear on the next launch

That is why `v1.1` now separates:

1. build-time shared base artifacts
2. launch-time per-guest writable overlays

`vfs` still handles subtree mounts such as `/home/alice`, `/home/bob`, and
`/workspace`; the writable root behavior comes from the launch-time ext4
overlay, not from a `vfs` mount at `/`.

## Guest/Host Coordination Contract

`v1.1` depends on explicit coordination between the guest configs and the host
admin plane. The important parameters are:

- `guest id`: `alice` or `bob` in the host admin plane
- `socket`: the Unix socket file owned by that guest-scoped `FsServer`
- `uid/gid`: the guest identity the host should reflect in overlay ownership
- `tag`: the mount identifier sent in the `TAG <name>` handshake
- `guest_path`: where the guest mounts that tag inside `/`
- `host_path`: which host directory the host `FsServer` serves for that tag
- `uid/gid/mode`: ownership and mode for overlay-injected content inside the mounted subtree

In `v1.1`, those parameters are split across two sides:

- guest side: [mounts.alice.yaml](/tmp/vfs-v11-multiguest/libs/vfs/examples/v1.1/mounts.alice.yaml) and [mounts.bob.yaml](/tmp/vfs-v11-multiguest/libs/vfs/examples/v1.1/mounts.bob.yaml)
- host side: [setup-multiguest.sh.vfs](/tmp/vfs-v11-multiguest/libs/vfs/examples/v1.1/setup-multiguest.sh.vfs) and the `repl_host` provisioning commands

Important nuance:

- the host routes by `guest id + tag -> host_path`
- the guest decides `tag -> guest_path`
- the `guest_path` recorded in the host `mount` command is there to make that contract explicit for operators; the actual mount location still comes from the guest `mounts.yaml`

This is on the same level of importance as getting `uid/gid` correct. If the
tag, guest path, host path, or ownership values drift apart, the guest may
mount the wrong subtree or see files with unusable ownership.

## Files

| File | Purpose |
|------|---------|
| `mounts.alice.yaml` | Guest mount config for the Alice VM |
| `mounts.bob.yaml` | Guest mount config for the Bob VM |
| `setup-alice.sh.vfs` | Host REPL script for Alice's tags |
| `setup-bob.sh.vfs` | Host REPL script for Bob's tags |
| `setup-multiguest.sh.vfs` | Combined host REPL script that provisions both guests, assigns uid/gid, and defines their mounts |
| `build-guest.sh` | Builds one generic shared base image set under `artifacts/base/` |
| `launch-ch.sh` | Launches one guest VM at a time with guest-specific sockets/CID/IP |
| `overlay-init` | Boot-time init script |
| `overlay.d/` | Optional files copied into the guest runtime overlay at launch time |
| `IMAGE.md` | Guest-image build notes for v1.1 |
| `CH-HARNESS.md` | End-to-end commands and topology |

## Mount Layout

Alice VM:

```yaml
mounts:
  - tag: alice-home
    guest_path: /home/alice
    read_only: false
  - tag: alice-workspace
    guest_path: /workspace
    read_only: false
```

Bob VM:

```yaml
mounts:
  - tag: bob-home
    guest_path: /home/bob
    read_only: false
  - tag: bob-workspace
    guest_path: /workspace
    read_only: false
```

## Quick Start

### 1. Build the shared base once

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh
```

`launch-ch.sh` creates a fresh per-guest writable runtime overlay on each boot.
That is the layer that absorbs guest-local root changes such as temporary
package installs.

### 2. Start one host server process for both guest VMs

```bash
cd /tmp/vfs-v11-multiguest

cat libs/vfs/examples/v1.1/setup-multiguest.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- --empty
```

That one `repl_host` process owns:

- one `FsServer` instance for Alice
- one `FsServer` instance for Bob
- one listener/socket per guest

The provisioning script now uses REPL commands:

- `provision <guest> <socket> <uid> <gid>`
- `mount <guest> <tag>=<guest_path>,<host_path> ...`
- `launch <guest>`

You can inspect the control plane interactively with:

```text
help
help provision
help mount
help launch
guests
```

The `guest_path` portion is recorded for operator clarity and to match the
guest mount config, but the host-side `FsServer` still routes by `tag` to the
host path. The actual guest mount points still come from `mounts.alice.yaml`
and `mounts.bob.yaml`.

The guest mounter still connects once per tag and sends a small one-line `TAG <name>` handshake before FsOp/FsResult traffic begins. Guest selection happens at the socket/listener boundary, not in the wire protocol.

`launch <guest>` is a prototype control-plane helper. It renders a shell script
to stdout that embeds:

- guest-specific `mounts.yaml`
- guest-specific cloud-init `user-data`
- guest-specific cloud-init `meta-data`
- explicit `groupadd` / `useradd` commands in cloud-init `runcmd` for the provisioned uid/gid

The intended operator flow is:

1. `provision alice /tmp/... 1000 1000`
2. `mount alice alice-home=/home/alice,...`
3. `launch alice > /tmp/launch-alice-cloud-init.sh`
4. run that script outside the REPL

This is a prototype workflow for the control plane. `launch <guest>` now emits
real guest-specific cloud-init assets, but the guest-launch path is still a
shell-script helper rather than a VMM library API.

Current limitations:

- `launch <guest>` currently targets the demo guests `alice` and `bob`
- the generated script requires `cloud-localds` from `cloud-image-utils`
- the shared base must be rebuilt with the current `build-guest.sh` so the guest includes `cloud-init` and consumes the attached NoCloud seed at boot

The `.vfs` setup files are plain one-command-per-line REPL input, so any demo file content injected there must stay on one line as well.

### 3. Launch both guests

```bash
./launch-ch.sh --guest alice --no-net
./launch-ch.sh --guest bob --no-net
```

For SSH-enabled runs, drop `--no-net`. Alice uses `192.168.249.2`; Bob uses `192.168.250.2`.

For disposable root-mutation experiments such as `apt install python3`, use a
larger runtime overlay:

```bash
./launch-ch.sh --guest alice --overlay-size 2G
./launch-ch.sh --guest bob --overlay-size 2G
```

## Manual Validation On This Host

If you are running from the same host shape as this session, use these
manual prep steps before the `Quick Start` flow above.

### 1. Check your primary group

The generic base build uses `mmdebstrap --mode=unshare` by default, and that
path expects your login shell's primary gid to match the passwd entry for your
user.

```bash
id
getent passwd "$USER"
```

The important part is that the shell's primary gid matches the gid in the
passwd entry. In this session, `id` showed `gid=994(kvm)` while
`getent passwd dchung` showed primary gid `1000`, and that caused:

```text
newuidmap ... failed
failed to unshare the user namespace
```

If you see that mismatch, run the build from a normal login shell for the
user, use `exec newgrp`, or use a rootful fallback like:

```bash
MMDEBSTRAP_MODE=root ./build-guest.sh
```

### 2. Load the vsock kernel module

Cloud Hypervisor needs `/dev/vhost-vsock` for the guest vsock device:

```bash
sudo modprobe vhost_vsock
ls -l /dev/vhost-vsock
```

### 3. Build the shared base

Run these from a shell where step 1 is satisfied:

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh
```

### 4. Start the shared host server process

In terminal 1:

```bash
cd /tmp/vfs-v11-multiguest

cat libs/vfs/examples/v1.1/setup-multiguest.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- --empty
```

### 5. Launch both guests

In terminal 2:

```bash
cd libs/vfs/examples/v1.1
./launch-ch.sh --guest alice --no-net
```

In terminal 3:

```bash
cd libs/vfs/examples/v1.1
./launch-ch.sh --guest bob --no-net
```

### 6. Validate the mounted views

Inside Alice:

```bash
mount | grep -E '/home/alice|/workspace'
ls -la /home/alice/.ssh
cat /home/alice/.env
cat /workspace/README.md
```

Inside Bob:

```bash
mount | grep -E '/home/bob|/workspace'
ls -la /home/bob/.ssh
cat /home/bob/.env
cat /workspace/README.md
```

## What Changed Relative To v1

- `repl_host` now supports repeated `--mount <tag>=<dir>`
- `repl_host` now also supports repeated `--guest <id>=<socket>` and guest-qualified `--mount <id>:<tag>=<dir>` for one-process multi-guest demos
- `repl_host` now also supports REPL-driven `provision`, `mount`, and `launch` commands, so `v1.1` can define guest topology and render guest-specific cloud-init from stdin instead of process flags
- `motlie-vfs-guest` still reads a YAML mount list, but now each mount performs a tag-binding handshake on connect
- the demo scripts use separate sockets/CIDs/runtime overlays per guest so Alice and Bob can run at the same time
- `v1.1` now runs Alice and Bob through one host `repl_host` process while still keeping one `FsServer` per guest internally
- `v1.1` now builds one generic base image set and creates guest-specific writable overlays at launch time
- `v1.1` moved guest identity, hostname, and mount config out of `build-guest.sh` and into `launch-ch.sh`
- `v1.1` treats package-manager writes as disposable guest-local root mutations, not as `vfs` content

## Overlay Extras

`v1.1` can copy optional files from:

- `overlay.d/common/`
- `overlay.d/alice/`
- `overlay.d/bob/`

into the guest runtime overlay during `launch-ch.sh`.

That means you can add standalone scripts or self-contained binaries under
paths like `overlay.d/common/usr/local/bin/...` without rebuilding the
shared squashfs base.

For `/usr/local/bin`: yes, this works because the per-guest runtime ext4
overlay is the writable upper layer for `/`. The practical caveats are:

- ownership and mode come from the seed tree on the host
- scripts and self-contained binaries are the best fit
- package-managed software or binaries needing extra shared libraries should
  go into the shared base image instead

Rule of thumb:

- common coarse-grained software for every guest: shared base
- guest-local, disposable mutations for one run: launch-time overlay

## Notes

- The lower-level `FsOp`/`FsResult` framing did not change.
- `v1` still works with the legacy single-tag `--tag` / `--dir` path.
- See [CH-HARNESS.md](CH-HARNESS.md) for the full end-to-end flow.
