# v1 End-to-End Example

Everything needed to run the motlie-vfs vertical slice: host server with
overlay injection → vsock transport → Cloud Hypervisor guest with FUSE mount.

## Files

| File | Purpose |
|------|---------|
| `setup-alice.sh.vfs` | REPL script: creates credentials layer, injects SSH keys + API tokens for alice |
| `mounts.yaml` | Guest config: tells `motlie-vfs-guest` which tags to mount and where |
| `build-guest.sh` | Builds Debian squashfs root + ext4 overlay guest images + kernel |
| `launch-ch.sh` | Launches Cloud Hypervisor with vsock + TAP networking |
| `overlay-init` | Boot-time init script: squashfs + ext4 → overlayfs → pivot_root |
| `CH-HARNESS.md` | Detailed Cloud Hypervisor setup: prerequisites, troubleshooting |
| `.gitignore` | Ignores `artifacts/` build output |

## Quick Start

### 1. Build guest images

```bash
# Builds guest binary, downloads CH kernel, creates squashfs + ext4 images
sudo ./build-guest.sh

# Or with a pre-built guest binary and existing kernel
sudo ./build-guest.sh --guest-binary /path/to/motlie-vfs-guest --kernel skip
```

### 2. Start the host server

```bash
# With overlay script (injects placeholder SSH keys + .env):
cat setup-alice.sh.vfs | \
    cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --socket /tmp/motlie-vfs.vsock_5000 \
    --tag alice-home --dir /tmp/alice-home

# With your real home dir as backing (no overlay):
echo "" | cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --socket /tmp/motlie-vfs.vsock_5000 \
    --tag alice-home --dir ~

# Interactive mode — type commands at the vfs> prompt:
cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --socket /tmp/motlie-vfs.vsock_5000 \
    --tag alice-home --dir /tmp/alice-home
```

The host server:
- Creates `FsServer` with tag `alice-home` backed by the specified directory
- Listens on `/tmp/motlie-vfs.vsock_5000` for guest vsock connections
- Accepts REPL commands to inject/whiteout/remove overlay files

Start the server **before** launching the VM.

### 3. Inject credentials (if not using the script)

```
vfs> layer credentials 0
vfs> putattr credentials alice-home /.ssh/authorized_keys 1000 1000 600 ssh-ed25519 AAAA... alice@dev
vfs> putattr credentials alice-home /.ssh/config 1000 1000 644 Host github.com
vfs> put credentials alice-home /.env ANTHROPIC_API_KEY=sk-ant-xxx
vfs> ls alice-home
```

### 4. Boot the guest

```bash
./launch-ch.sh            # with TAP networking (needs CAP_NET_ADMIN)
./launch-ch.sh --no-net   # vsock only, no SSH access
```

### 5. Validate

```bash
ssh alice@192.168.249.2    # password: testpass

# Inside the guest:
ls -la ~/.ssh/             # → authorized_keys, config, id_ed25519, id_ed25519.pub
cat ~/.env                 # → ANTHROPIC_API_KEY=sk-ant-xxx
```

### 6. Stop the guest

```bash
curl --unix-socket /tmp/motlie-vfs-api.sock -X PUT http://localhost/api/v1/vm.shutdown
# or from inside the guest: poweroff
```

## What the guest sees

```
/home/alice/                    ← FUSE mount, tag "alice-home" → host dir
├── .ssh/                       ← IN-MEMORY OVERLAY (credentials layer)
│   ├── authorized_keys         ← uid=1000 gid=1000 mode=600
│   ├── config                  ← uid=1000 gid=1000 mode=644
│   ├── id_ed25519              ← uid=1000 gid=1000 mode=600
│   └── id_ed25519.pub          ← uid=1000 gid=1000 mode=644
├── .env                        ← IN-MEMORY OVERLAY (synthetic file)
└── (any files in host dir)     ← pass-through from disk
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
