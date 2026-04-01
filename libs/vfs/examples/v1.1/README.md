# v1.1 Multi-Guest / Multi-Mount Example

This example extends the `v1` vertical slice to demo:

- two guest VMs: `alice` and `bob`
- one `FsServer` per guest VM
- multiple mount tags per guest VM
- one host socket per guest VM, with per-connection tag routing inside that socket

## Files

| File | Purpose |
|------|---------|
| `mounts.alice.yaml` | Guest mount config for the Alice VM |
| `mounts.bob.yaml` | Guest mount config for the Bob VM |
| `setup-alice.sh.vfs` | Host REPL script for Alice's tags |
| `setup-bob.sh.vfs` | Host REPL script for Bob's tags |
| `build-guest.sh` | Builds guest-specific rootfs/overlay/kernel artifacts under `artifacts/<guest>/` |
| `launch-ch.sh` | Launches one guest VM at a time with guest-specific sockets/CID/IP |
| `overlay-init` | Boot-time init script |
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

### 1. Build both guest image sets

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh --guest alice
./build-guest.sh --guest bob
```

### 2. Start one host server per guest VM

Alice:

```bash
cat setup-alice.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- \
  --socket /tmp/motlie-vfs-alice.vsock_5000 \
  --mount alice-home=/tmp/motlie-vfs-demo/alice-home \
  --mount alice-workspace=/tmp/motlie-vfs-demo/alice-workspace
```

Bob:

```bash
cat setup-bob.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- \
  --socket /tmp/motlie-vfs-bob.vsock_5000 \
  --mount bob-home=/tmp/motlie-vfs-demo/bob-home \
  --mount bob-workspace=/tmp/motlie-vfs-demo/bob-workspace
```

Each `repl_host` process serves multiple tags for one guest VM. The guest mounter connects once per tag and sends a small one-line `TAG <name>` handshake before FsOp/FsResult traffic begins.

The `.vfs` setup files are plain one-command-per-line REPL input, so any demo file content injected there must stay on one line as well.

### 3. Launch both guests

```bash
./launch-ch.sh --guest alice --no-net
./launch-ch.sh --guest bob --no-net
```

For SSH-enabled runs, drop `--no-net`. Alice uses `192.168.249.2`; Bob uses `192.168.250.2`.

## Manual Validation On This Host

If you are running from the same host shape as this session, use these
manual prep steps before the `Quick Start` flow above.

### 1. Check your primary group

The rootless `mmdebstrap --mode=unshare` build path expects your login
shell's primary gid to match the passwd entry for your user.

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
user instead of this restricted session.

### 2. Load the vsock kernel module

Cloud Hypervisor needs `/dev/vhost-vsock` for the guest vsock device:

```bash
sudo modprobe vhost_vsock
ls -l /dev/vhost-vsock
```

### 3. Build both guest image sets

Run these from a shell where step 1 is satisfied:

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh --guest alice
./build-guest.sh --guest bob
```

### 4. Start one host server per guest

In terminal 1:

```bash
cd /tmp/vfs-v11-multiguest

cat libs/vfs/examples/v1.1/setup-alice.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- \
  --socket /tmp/motlie-vfs-alice.vsock_5000 \
  --mount alice-home=/tmp/motlie-vfs-demo/alice-home \
  --mount alice-workspace=/tmp/motlie-vfs-demo/alice-workspace
```

In terminal 2:

```bash
cd /tmp/vfs-v11-multiguest

cat libs/vfs/examples/v1.1/setup-bob.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- \
  --socket /tmp/motlie-vfs-bob.vsock_5000 \
  --mount bob-home=/tmp/motlie-vfs-demo/bob-home \
  --mount bob-workspace=/tmp/motlie-vfs-demo/bob-workspace
```

### 5. Launch both guests

In terminal 3:

```bash
cd libs/vfs/examples/v1.1
./launch-ch.sh --guest alice --no-net
```

In terminal 4:

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
- `motlie-vfs-guest` still reads a YAML mount list, but now each mount performs a tag-binding handshake on connect
- the demo scripts use separate sockets/CIDs/artifact trees per guest so Alice and Bob can run at the same time

## Notes

- The lower-level `FsOp`/`FsResult` framing did not change.
- `v1` still works with the legacy single-tag `--tag` / `--dir` path.
- See [CH-HARNESS.md](CH-HARNESS.md) for the full end-to-end flow.
