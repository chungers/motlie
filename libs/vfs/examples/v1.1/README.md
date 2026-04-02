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
- `socket`: the host listener socket for that guest-scoped `FsServer` (the demo uses `*.vsock_5000` for guest port 5000)
- `uid/gid`: the guest identity the host should reflect in overlay ownership
- `tag`: the mount identifier sent in the `TAG <name>` handshake
- `guest_path`: where the guest mounts that tag inside `/`
- `host_path`: which host directory the host `FsServer` serves for that tag
- `uid/gid/mode`: ownership and mode for overlay-injected content inside the mounted subtree

In `v1.1`, those parameters are split across two sides:

- guest side: [mounts.alice.yaml](/tmp/vfs-v11-multiguest/libs/vfs/examples/v1.1/mounts.alice.yaml) and [mounts.bob.yaml](/tmp/vfs-v11-multiguest/libs/vfs/examples/v1.1/mounts.bob.yaml)
- host side: [setup-multiguest.sh.vfs](/tmp/vfs-v11-multiguest/libs/vfs/examples/v1.1/setup-multiguest.sh.vfs) and the `repl_host_v1_1` provisioning commands

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
| `repl_host.rs` | `v1.1`-specific multi-guest host REPL example (`cargo run --example repl_host_v1_1`) |
| `build-guest.sh` | Builds one generic shared base image set under `artifacts/base/` |
| `launch-ch.sh` | Launches one guest VM at a time with guest-specific sockets/CID/IP |
| `overlay-init` | Boot-time init script |
| `overlay.d/` | Optional files copied into the guest runtime overlay at launch time |
| `IMAGE.md` | Guest-image build notes for v1.1 |
| `CH-HARNESS.md` | End-to-end commands and topology |

## Host Requirements

`v1.1` assumes a Linux host with these host-side packages installed:

```bash
sudo apt install \
  cloud-hypervisor \
  debian-archive-keyring \
  e2fsprogs \
  libfuse3-dev \
  mmdebstrap \
  pkg-config \
  squashfs-tools-ng \
  uidmap
```

What each host package is used for:

- `cloud-hypervisor`: runs the guest VMs
- `debian-archive-keyring`: keyring for `mmdebstrap`
- `e2fsprogs`: provides `mkfs.ext4` for per-launch runtime overlays
- `libfuse3-dev` and `pkg-config`: build dependencies for `motlie-vfs-guest-v1_1`
- `mmdebstrap`, `squashfs-tools-ng`, `uidmap`: shared base image build

You also need a working Rust toolchain with `cargo`, because:

- `build-guest.sh` compiles `motlie-vfs-guest-v1_1` from the workspace and installs it into the guest image as `motlie-vfs-guest`
- the runbooks use `cargo run -p motlie-vfs --example repl_host_v1_1 --features vsock`

You also need:

- outbound network access to download Debian packages and the CH kernel during image builds
- `/dev/vhost-vsock` available at guest-launch time, usually via `sudo modprobe vhost_vsock`
- a shell whose primary gid matches the passwd primary gid when using `mmdebstrap --mode=unshare`

Optional but useful:

- `tmux` for the guest login experience inside the built image
- `curl` on the host for the CH shutdown examples in this runbook

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

Preferred interactive startup:

```bash
cd /tmp/vfs-v11-multiguest

cargo run -p motlie-vfs --example repl_host_v1_1 --features vsock -- \
  --empty --script libs/vfs/examples/v1.1/setup-multiguest.sh.vfs
```

This keeps `repl_host_v1_1` attached directly to your terminal, so `rustyline`
history, arrow keys, and other control sequences work normally.

Pipe-based startup still works:

```bash
cd /tmp/vfs-v11-multiguest

cat libs/vfs/examples/v1.1/setup-multiguest.sh.vfs - | \
  cargo run -p motlie-vfs --example repl_host_v1_1 --features vsock -- --empty
```

That one `repl_host_v1_1` process owns:

- one `FsServer` instance for Alice
- one `FsServer` instance for Bob
- one listener/socket per guest

The provisioning script now uses REPL commands:

- `provision <guest> <socket> <uid> <gid>`
- `mount <guest> <tag>=<guest_path>,<host_path> ...`
- `launch <guest>`
- `launch -script <guest>`
- `shutdown <guest>`

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

The guest mounter still connects once per tag to guest vsock port `5000` and sends a small one-line `TAG <name>` handshake before FsOp/FsResult traffic begins. In the CH demo, the hypervisor owns `/tmp/motlie-vfs-<guest>.vsock` while `repl_host_v1_1` listens on `/tmp/motlie-vfs-<guest>.vsock_5000`.

`launch` is a prototype control-plane helper.

- `launch <guest>` writes the helper under `/tmp/motlie-vfs-launch/<guest>/`, starts it asynchronously, and immediately returns control to the REPL
- `launch -script <guest>` prints the same script to stdout

The helper embeds:

- guest-specific `mounts.yaml`
- guest-specific cloud-init `user-data`
- guest-specific cloud-init `meta-data`
- explicit `groupadd` / `useradd` commands in cloud-init `runcmd` for the provisioned uid/gid

Cloud-init in this flow writes guest config and creates mountpoint
directories, then queues a non-blocking `systemctl start motlie-vfs-guest.service`.
The helper avoids `restart` and uses `--no-block` so `cloud-final.service`
does not deadlock waiting on a unit that is ordered after `cloud-final.service`.

The intended operator flow is:

1. `provision alice /tmp/motlie-vfs-alice.vsock_5000 1000 1000`
2. `mount alice alice-home=/home/alice,...`
3. `launch alice`

If you want the raw helper instead:

3. `launch -script alice > /tmp/launch-alice-cloud-init.sh`
4. run that script outside the REPL

When `launch <guest>` runs directly from the REPL, logs go to:

- `/tmp/motlie-vfs-launch/<guest>/launch.log`
- `/tmp/motlie-vfs-launch/<guest>/serial.log`

That keeps guest boot output out of the REPL terminal.

This is a prototype workflow for the control plane. `launch <guest>` now emits
real guest-specific cloud-init assets, but the guest-launch path is still a
shell-script helper rather than a VMM library API.

Operator notes:

- prefer `--script <file>` for interactive operator use; that keeps stdin on the real TTY so `rustyline` history and non-printable keys work correctly
- use `cat ... - | cargo run ...` so the setup file is consumed first and stdin stays attached to your terminal
- that keeps the shared `repl_host_v1_1` process alive while guests boot and connect
- the generated helper does not require `cloud-localds`; `launch-ch.sh` copies the NoCloud files into the per-launch guest overlay directly
- when `repl_host_v1_1` is driven by scripted stdin, its status/help lines are emitted as shell comments (`# ...`) so `launch <guest>` output can be redirected directly into an executable helper script

Current limitations:

- `launch <guest>` currently targets the demo guests `alice` and `bob`
- the shared base must be rebuilt with the current `build-guest.sh` so the guest includes `cloud-init` and consumes the seeded NoCloud directory at boot

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

Console/boot note:

- `v1.1` now boots with `console=ttyS0` and Cloud Hypervisor `--serial tty --console off`
- this avoids the long `dev-hvc0.device` / `serial-getty@hvc0.service` timeout caused by a kernel cmdline that expects `hvc0`
- the terminal where `launch-ch.sh` runs is the guest console

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

cat libs/vfs/examples/v1.1/setup-multiguest.sh.vfs - | \
  cargo run -p motlie-vfs --example repl_host_v1_1 --features vsock -- --empty
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

Mounted-home note:

- if you SSH in before `motlie-vfs-guest` mounts `/home/<guest>`, your shell can retain the old cwd inode from the base image home
- in that case `.` may still show the old home even though `/home/<guest>` and `~` resolve through the mounted VFS path
- if `cat /home/alice/.env` works but `cat .env` does not, run `cd "$HOME"` or open a fresh SSH session after the mount is active

## What Changed Relative To v1

- `repl_host_v1_1` supports repeated `--mount <tag>=<dir>`
- `repl_host_v1_1` also supports repeated `--guest <id>=<socket>` and guest-qualified `--mount <id>:<tag>=<dir>` for one-process multi-guest demos
- `repl_host_v1_1` also supports REPL-driven `provision`, `mount`, and `launch` commands, so `v1.1` can define guest topology and render guest-specific cloud-init from stdin instead of process flags
- `motlie-vfs-guest` still reads a YAML mount list, but now each mount performs a tag-binding handshake on connect
- the demo scripts use separate sockets/CIDs/runtime overlays per guest so Alice and Bob can run at the same time
- `v1.1` now runs Alice and Bob through one host `repl_host_v1_1` process while still keeping one `FsServer` per guest internally
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
