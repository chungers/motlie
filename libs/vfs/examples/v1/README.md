# v1 End-to-End Example

Everything needed to run the motlie-vfs vertical slice: host server with
overlay injection → vsock transport → Cloud Hypervisor guest with FUSE mount.

## Files

| File | Purpose |
|------|---------|
| `setup-alice.sh.vfs` | REPL script: creates credentials layer, injects SSH keys + API tokens for alice |
| `mounts.yaml` | Guest config: tells `motlie-vfs-guest` which tags to mount and where |
| `build-guest.sh` | Builds Alpine squashfs root + ext4 overlay guest images |
| `launch-ch.sh` | Launches Cloud Hypervisor with vsock + TAP networking |
| `overlay-init` | Boot-time init script: squashfs + ext4 → overlayfs → pivot_root |
| `CH-HARNESS.md` | Detailed Cloud Hypervisor setup: kernel build, cross-compile, validation |
| `.gitignore` | Ignores `artifacts/` build output |

## Quick Start

### 1. Start the host server

```bash
# Interactive mode — type commands at the vfs> prompt
cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --tag alice-home --dir ~/alice

# Or scripted — pipe the setup script, server stays alive after EOF
cat examples/v1/setup-alice.sh.vfs | \
    cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --tag alice-home --dir ~/alice
```

The host server:
- Creates `FsServer` with tag `alice-home` backed by `~/alice`
- Listens on `/tmp/motlie-vfs.vsock_5000` for guest connections
- Accepts REPL commands to inject/whiteout/remove overlay files

### 2. Inject credentials (if not using the script)

```
vfs> layer credentials 0
vfs> putattr credentials alice-home /.ssh/authorized_keys 1000 1000 600 ssh-ed25519 AAAA... alice@dev
vfs> putattr credentials alice-home /.ssh/config 1000 1000 644 Host github.com
vfs> put credentials alice-home /.env ANTHROPIC_API_KEY=sk-ant-xxx
vfs> ls alice-home
```

### 3. Build guest images (requires Linux)

```bash
# Cross-compile the guest binary from macOS
cross build --release --target x86_64-unknown-linux-musl \
    -p motlie-vfs --bin motlie-vfs-guest --features vsock,client

# Build images on Linux
./examples/v1/build-guest.sh --guest-binary \
    target/x86_64-unknown-linux-musl/release/motlie-vfs-guest
```

See `CH-HARNESS.md` for kernel build instructions.

### 4. Boot the guest (requires Linux)

```bash
./examples/v1/launch-ch.sh
```

### 5. Validate (in the guest)

```bash
ssh alice@192.168.249.2    # password: testpass

# Inside the guest:
ls /home/alice/.ssh/       # → authorized_keys, config (from overlay)
cat /home/alice/.env       # → ANTHROPIC_API_KEY=sk-ant-xxx
ls /home/alice/projects/   # → README.md (from host disk)
```

## What the guest sees

```
/home/alice/                    ← FUSE mount, tag "alice-home" → ~/alice on host
├── projects/                   ← pass-through to ~/alice/projects (disk)
├── .bashrc                     ← pass-through to ~/alice/.bashrc (disk)
├── .ssh/                       ← IN-MEMORY OVERLAY (credentials layer)
│   ├── authorized_keys         ← uid=1000 gid=1000 mode=600
│   ├── config                  ← uid=1000 gid=1000 mode=644
│   ├── id_ed25519              ← uid=1000 gid=1000 mode=600
│   └── id_ed25519.pub          ← uid=1000 gid=1000 mode=644
└── .env                        ← IN-MEMORY OVERLAY (synthetic file)
```

Overlay files never touch the host disk. Dynamic injection and whiteouts
are visible to the guest on the next filesystem operation — no restart
or remount needed.

## Input Modes

The `repl_host` server detects how stdin is connected:

| Mode | How | After input |
|------|-----|-------------|
| Interactive | `repl_host --tag ...` | rustyline REPL until `quit` |
| Pipe + TTY | `cat script - \| repl_host` | script then interactive REPL |
| Pure pipe | `cat script \| repl_host` | script then serve until signaled |

Server never exits on pipe EOF — guest mounts stay alive.

## Config Files

**`mounts.yaml`** — read by `motlie-vfs-guest` inside the guest:
```yaml
mounts:
  - tag: alice-home
    guest_path: /home/alice
    read_only: false
```

**`setup-alice.sh.vfs`** — piped into `repl_host` on the host:
```
layer credentials 0
putattr credentials alice-home /.ssh/authorized_keys 1000 1000 600 ssh-ed25519 ...
put credentials alice-home /.env ANTHROPIC_API_KEY=sk-ant-xxx
```

## Architecture

```
Host                                      Guest (Cloud Hypervisor VM)
────                                      ─────
repl_host                                 motlie-vfs-guest
  FsServer + MemOverlay                     reads mounts.yaml
  vsock listener (:5000)  ←── vsock ──→     VsockClientTransport
  stdin REPL (admin)                        FuseClient → fuser::mount2
                                            /home/alice (FUSE mount)
```

One `FsServer` per guest VM. One vsock socket per VM. Tags route
mounted subtrees within a VM. Admin is in-process REPL only.
