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

## What Changed Relative To v1

- `repl_host` now supports repeated `--mount <tag>=<dir>`
- `motlie-vfs-guest` still reads a YAML mount list, but now each mount performs a tag-binding handshake on connect
- the demo scripts use separate sockets/CIDs/artifact trees per guest so Alice and Bob can run at the same time

## Notes

- The lower-level `FsOp`/`FsResult` framing did not change.
- `v1` still works with the legacy single-tag `--tag` / `--dir` path.
- See [CH-HARNESS.md](CH-HARNESS.md) for the full end-to-end flow.
