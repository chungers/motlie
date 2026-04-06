# v1.3 SSH Proxy + Programmatic Guest Control Plane

`v1.3` builds on the validated `v1.2` split-network flow and adds:

- **SSH proxy** (russh) for host-side ingress via CA-signed ephemeral certs **over vsock** — no TAP, fully userspace
- **Programmatic guest exec** (`exec <guest> <command>`) — run commands
  inside guests without human intervention
- **Automated validation** (`validate <guest>`) — v1.2 runbook as assertions
- **Fully parameterized launch** — no hardcoded guest names in shell scripts
- **Per-guest resource allocation** — CID, IPs, MACs allocated from a
  monotonic slot allocator, reused across shutdown/relaunch

## Success Outcomes

`v1.3` is considered successful when:

1. `exec alice uname -a` returns output from the guest via the SSH proxy
2. `validate alice` passes all checks (egress, DNS, writable root,
   agent-state symlinks, default route)
3. Multiple guests (alice, bob, arbitrary names) launch with unique
   resources — no collisions
4. The guest image only needs one rebuild (for sshd_config CA trust);
   the CA pubkey and principals are injected per-launch

## Non-Regression Constraints

These points were easy to get subtly wrong during `v1.3` bring-up and should be
treated as part of the design, not incidental implementation details.

### A. SSH overlay paths must not be group-writable

The runtime overlay built by [launch-ch.sh](./launch-ch.sh) must keep:

- intermediate directories at `755`, not `775`
- `/etc/ssh/auth_principals` at `root:root` and `755`
- `/etc/ssh/auth_principals/<user>` files at `644`
- `/etc/ssh/ca/user_ca.pub` at `644`

`fakeroot mkfs.ext4 -d ...` is intentional here. It ensures the injected CA and
principals files appear as `root:root` inside the guest. If ownership or any
parent directory mode is loosened, guest `sshd` can reject cert-based auth.

### B. The proxy's ephemeral SSH cert must include `permit-pty`

The proxy authenticates to guest `sshd` with an ephemeral user certificate from
[`SshCa::sign_ephemeral()`](../../src/ca.rs). That cert must include the
OpenSSH user-cert extension `permit-pty`.

Without `permit-pty`, the proxy can still authenticate and run plain `exec`
requests, but guest `sshd` rejects PTY allocation. The observed failure mode
was:

- `ssh -p 2222 alice@localhost` hung after MOTD
- `ssh -tt -p 2222 alice@localhost /bin/cat -v` printed raw `^M` / literal `^D`
- proxy debug output showed guest-side `Failure` followed by `Success`

After adding `permit-pty`, proxied interactive shells and PTY-backed exec paths
started working again.

### C. Compare direct guest SSH with proxied SSH when debugging PTY issues

Use these as the first differential test:

```bash
ssh -tt alice@192.168.249.2 /bin/cat -v
ssh -tt -p 2222 alice@localhost /bin/cat -v
```

If direct guest SSH is correct and proxied SSH is wrong, the bug is in the
proxy path, not the guest image or guest `sshd`.

### D. Agent-state redirection is a boot-time service contract

`motlie-agent-state.service` must keep:

- `~/.codex -> /agent-state/codex`
- `~/.claude -> /agent-state/claude`
- `~/.config/claude-code -> /agent-state/claude-code`

This is part of the validated `v1.2`/`v1.3` contract. If those symlinks are
replaced with home-local directories, tool auth/state silently stops landing in
the dedicated VFS-backed layer.

## Architecture Overview

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    repl_host_v1_3 (Rust, tokio)                 │
 │                                                                  │
 │  SshCa ─────────────────────┐                                    │
 │  (in-memory Ed25519 CA)     │                                    │
 │                             │                                    │
 │  AdminState                 │                                    │
 │  ├─ guests: FsServer+mounts│                                    │
 │  ├─ net_allocs: slot→alloc  │                                    │
 │  ├─ vnet_handles: per-guest │                                    │
 │  └─ launch_pids: per-guest  │                                    │
 │         │                   │                                    │
 │  ┌──────┴───────┐    ┌─────┴────────┐    ┌────────────────┐     │
 │  │ launch alice │    │ exec alice   │    │ validate alice │     │
 │  │              │    │ uname -a     │    │ (6 checks)     │     │
 │  └──────┬───────┘    └─────┬────────┘    └───────┬────────┘     │
 └─────────┼──────────────────┼─────────────────────┼──────────────┘
           │                  │                     │
           │           ┌──────┴──────────────────────┘
           │           │  ssh::exec_vsock()
           │           │  ├─ ca.sign_ephemeral("alice")
           │           │  ├─ VsockStream::connect(cid=3, port=2222)
           │           │  ├─ russh::client::connect_stream(vsock)
           │           │  ├─ authenticate_publickey(ephemeral_cert)
           │           │  ├─ channel.exec("uname -a")
           │           │  └─ collect stdout/stderr/exit_code
           │           │
           ▼           ▼ (vsock, AF_VSOCK)
   ┌─────────────┐  ┌──────────────────────────┐
   │ launch-ch.sh│  │  Guest VM (KVM/CH)        │
   │ (overlay +  │  │  ├─ socat vsock:2222→:22  │
   │  CH launch) │  │  ├─ sshd (CA trust)       │
   └──────┬──────┘  │  ├─ eth0: vnet egress     │
          │         │  └─ VFS mounts (vsock)     │
          ▼         └──────────────────────────┘
   cloud-hypervisor
```

## End-to-End Flow

### Phase 1: Image Build (one-time)

```
User                build-guest.sh                    Output
 │                       │                              │
 │  ./build-guest.sh     │                              │
 │──────────────────────>│                              │
 │                       │  mmdebstrap                  │
 │                       │──────────────────────>       │
 │                       │  Debian rootfs with:         │
 │                       │    openssh-server             │
 │                       │    cloud-init                 │
 │                       │    systemd-networkd           │
 │                       │    codex, claude-code, npm    │
 │                       │                              │
 │                       │  customize hooks:             │
 │                       │    sshd_config +=             │
 │                       │      TrustedUserCAKeys        │
 │                       │        /etc/ssh/ca/user_ca.pub│
 │                       │      AuthorizedPrincipalsFile │
 │                       │        ...auth_principals/%u  │
 │                       │    create users (alice, bob)  │
 │                       │    ssh-keygen -A              │
 │                       │    egress NIC service         │
 │                       │    agent-state service        │
 │                       │    overlay-init as /sbin/init │
 │                       │                              │
 │                       │  mksquashfs                   │
 │                       │──────────────────────>       │
 │                       │              artifacts/base/  │
 │                       │                Image          │
 │                       │                rootfs.squashfs│
```

The image is generic — no per-guest state. The sshd_config CA directives
point to paths that don't exist yet; they're populated per-launch via
the runtime overlay.

### Phase 2: REPL Startup + Provisioning

```
User                       repl_host_v1_3
 │                              │
 │  cargo run -p motlie-vmm    │
 │    --example repl_host_v1_3 │
 │    -- --empty               │
 │    --script setup-multi...  │
 │    --admin-net=tap          │
 │    --egress-net=vhost-user  │
 │─────────────────────────────>│
 │                              │  SshCa::new()
 │                              │    → Ed25519 CA keypair (in memory)
 │                              │
 │                              │  execute setup-multiguest.sh.vfs:
 │                              │    provision alice ... 1000 1000
 │                              │      → FsServer + overlay + vsock listener
 │                              │    mount alice alice-home=... alice-workspace=...
 │                              │    layer / mkdir / put (overlay content)
 │                              │    provision bob ... 1001 1001
 │                              │    mount bob ...
 │                              │
 │                              │  AdminState ready:
 │                              │    guests: {alice, bob}
 │                              │    ssh_ca: SshCa
 │                              │    net_allocs: {}  (empty until launch)
 │  vfs> _                      │
```

### Phase 3: Launch (per guest)

```
REPL         repl_host_v1_3       wrapper.sh        launch-ch.sh      cloud-hypervisor
 │                │                   │                  │                   │
 │ launch alice   │                   │                  │                   │
 │───────────────>│                   │                  │                   │
 │                │                   │                  │                   │
 │                │ ensure_vnet_backend(alice)            │                   │
 │                │   VnetBackend::builder()              │                   │
 │                │     .socket(motlie-vmm-alice.sock)    │                   │
 │                │     .build()?.start()?                │                   │
 │                │   → libslirp thread running           │                   │
 │                │                   │                  │                   │
 │                │ ensure_net_alloc(alice) → slot 0      │                   │
 │                │   cid=3, host_ip=192.168.249.1        │                   │
 │                │   guest_ip=192.168.249.2              │                   │
 │                │   admin_mac=52:54:00:ad:00:01         │                   │
 │                │   egress_mac=52:54:00:e9:00:01        │                   │
 │                │                   │                  │                   │
 │                │ ssh_ca.public_key_openssh()            │                   │
 │                │   → "ssh-ed25519 AAAA..."             │                   │
 │                │                   │                  │                   │
 │                │ render_launch_script()                 │                   │
 │                │ ├─ render_cloud_init()                 │                   │
 │                │ │    → "#cloud-config\n"               │                   │
 │                │ ├─ render_mounts_yaml()                │                   │
 │                │ │    → "mounts:\n  - tag: ..."         │                   │
 │                │ └─ emit wrapper.sh with all params     │                   │
 │                │                   │                  │                   │
 │                │ execute_launch_script()                │                   │
 │                │   write /tmp/motlie-vmm-launch/        │                   │
 │                │     alice/launch.sh                    │                   │
 │                │   /bin/bash launch.sh ────────────────>│                   │
 │                │                   │                  │                   │
 │                │                   │  mkdir $SEED_DIR  │                   │
 │                │                   │  write meta-data  │                   │
 │                │                   │  write mounts.yaml│                   │
 │                │                   │  write user-data  │                   │
 │                │                   │                  │                   │
 │                │                   │  launch-ch.sh \   │                   │
 │                │                   │    --guest alice \ │                   │
 │                │                   │    --cloud-init.. \│                   │
 │                │                   │    --cid 3 \      │                   │
 │                │                   │    --host-ip .. \  │                   │
 │                │                   │    --ssh-ca-pub.. \│                   │
 │                │                   │    ...            │                   │
 │                │                   │──────────────────>│                   │
 │                │                   │                  │                   │
 │                │                   │                  │ Build overlay:     │
 │                │                   │                  │ seed/upper/        │
 │                │                   │                  │  ├─ nocloud/       │
 │                │                   │                  │  │  user-data      │
 │                │                   │                  │  │  meta-data      │
 │                │                   │                  │  ├─ mounts.yaml    │
 │                │                   │                  │  ├─ hostname       │
 │                │                   │                  │  ├─ hosts          │
 │                │                   │                  │  ├─ ssh/ca/        │
 │                │                   │                  │  │  user_ca.pub    │
 │                │                   │                  │  └─ auth_principals│
 │                │                   │                  │     alice, root    │
 │                │                   │                  │                   │
 │                │                   │                  │ mkfs.ext4 → overlay│
 │                │                   │                  │                   │
 │                │                   │                  │ cloud-hypervisor   │
 │                │                   │                  │  --kernel Image    │
 │                │                   │                  │  --disk squashfs,  │
 │                │                   │                  │         overlay    │
 │                │                   │                  │  --net tap(admin)  │
 │                │                   │                  │  --net vhost(egress)
 │                │                   │                  │  --vsock cid=3     │
 │                │                   │                  │──────────────────>│
 │                │                   │                  │                   │
 │ ok: launch     │                   │                  │          VM boots │
 │ alice pid=...  │                   │                  │                   │
```

### Phase 4: Guest Boot

```
cloud-hypervisor          Guest kernel              Guest userspace
      │                       │                          │
      │  boot Image           │                          │
      │──────────────────────>│                          │
      │                       │                          │
      │                       │  overlay-init            │
      │                       │  mount squashfs (ro)     │
      │                       │  mount overlay.ext4 (rw) │
      │                       │  overlayfs merge         │
      │                       │  pivot_root              │
      │                       │  exec /sbin/init         │
      │                       │─────────────────────────>│
      │                       │                          │
      │                       │              systemd     │
      │                       │              ├─ sshd     │
      │                       │              │   reads:  │
      │                       │              │   TrustedUserCAKeys
      │                       │              │     /etc/ssh/ca/user_ca.pub ✓
      │                       │              │   AuthorizedPrincipalsFile
      │                       │              │     .../auth_principals/%u ✓
      │                       │              │                          │
      │                       │              ├─ systemd-networkd        │
      │                       │              │   eth0: static (admin TAP)
      │                       │              │   eth1: DHCP (egress)    │
      │                       │              │     → 10.0.2.15/24       │
      │                       │              │                          │
      │                       │              ├─ motlie-vmm-egress       │
      │                       │              │   default route → 10.0.2.2
      │                       │              │   DNS → 10.0.2.3         │
      │                       │              │                          │
      │                       │              ├─ motlie-agent-state      │
      │                       │              │   ~/.codex → /agent-state/codex
      │                       │              │   ~/.claude → /agent-state/claude
      │                       │              │                          │
      │                       │              └─ cloud-init              │
      │                       │                  reads /var/lib/cloud/  │
      │                       │                  seed/nocloud/user-data │
      │                       │                  processes #cloud-config│
      │                       │                                        │
      │                       │              Guest ready:               │
      │                       │                sshd on :22 (CA trust)   │
      │                       │                eth0: 192.168.249.2 (TAP)│
      │                       │                eth1: 10.0.2.15 (egress) │
```

### Phase 5: SSH Proxy Exec / Validate

```
REPL           repl_host_v1_3         SshCa          russh           Guest sshd
 │                  │                   │               │                │
 │ exec alice       │                   │               │                │
 │   uname -a       │                   │               │                │
 │─────────────────>│                   │               │                │
 │                  │                   │               │                │
 │                  │ guest_admin_ip()   │               │                │
 │                  │  → 192.168.249.2   │               │                │
 │                  │                   │               │                │
 │                  │ sign_ephemeral    │               │                │
 │                  │  ("alice")        │               │                │
 │                  │──────────────────>│               │                │
 │                  │                   │               │                │
 │                  │  EphemeralCert    │               │                │
 │                  │  { key, cert      │               │                │
 │                  │    principal=alice │               │                │
 │                  │    ttl=60s }      │               │                │
 │                  │<──────────────────│               │                │
 │                  │                   │               │                │
 │                  │ ssh::exec()       │               │                │
 │                  │──────────────────────────────────>│                │
 │                  │                   │  connect       │                │
 │                  │                   │  192.168.249.2:22              │
 │                  │                   │               │───────────────>│
 │                  │                   │               │                │
 │                  │                   │  authenticate  │                │
 │                  │                   │  publickey     │                │
 │                  │                   │  (ephemeral)   │                │
 │                  │                   │               │───────────────>│
 │                  │                   │               │                │
 │                  │                   │               │  cert signed   │
 │                  │                   │               │  by user_ca? ✓ │
 │                  │                   │               │  "alice" in    │
 │                  │                   │               │  principals? ✓ │
 │                  │                   │               │  not expired? ✓│
 │                  │                   │               │                │
 │                  │                   │               │  Auth::Accept  │
 │                  │                   │               │<───────────────│
 │                  │                   │               │                │
 │                  │                   │  channel_exec  │                │
 │                  │                   │  "uname -a"   │                │
 │                  │                   │               │───────────────>│
 │                  │                   │               │                │
 │                  │                   │               │  stdout:       │
 │                  │                   │               │  "Linux motlie │
 │                  │                   │               │   -alice 6.1.."│
 │                  │                   │               │  exit_code: 0  │
 │                  │                   │               │<───────────────│
 │                  │                   │               │                │
 │                  │  ExecOutput       │               │                │
 │                  │  { stdout, stderr,│               │                │
 │                  │    exit_code: 0 } │               │                │
 │                  │<─────────────────────────────────│                │
 │                  │                   │               │                │
 │ Linux motlie-    │                   │               │                │
 │ alice 6.1...     │                   │               │                │
 │ exit_code: 0     │                   │               │                │
 │<─────────────────│                   │               │                │
```

## Data Generation Ownership

All per-guest dynamic content is generated by Rust code in `repl_host_v1_3`.
Shell scripts receive everything via CLI flags — no hardcoded guest names.

| Artifact | Generated by | Consumed by |
|---|---|---|
| rootfs.squashfs, Image | `build-guest.sh` (one-time) | `launch-ch.sh` → CH `--kernel`/`--disk` |
| user-data, meta-data | `render_cloud_init()` + `render_launch_script()` (Rust) | wrapper → `$SEED_DIR` → `launch-ch.sh` → overlay → cloud-init |
| mounts.yaml | `render_mounts_yaml()` (Rust) | wrapper → `$SEED_DIR` → `launch-ch.sh` → overlay → guest agent |
| CID, IPs, MACs | `ensure_net_alloc()` (Rust, monotonic slot allocator) | wrapper → `launch-ch.sh` CLI flags → CH args |
| SSH CA pubkey | `SshCa::new()` (Rust, in-memory per session) | wrapper → `launch-ch.sh` `--ssh-ca-pubkey` → overlay → guest sshd |
| auth_principals | `launch-ch.sh` (from `--ssh-user` flag) | overlay → guest sshd |
| overlay.ext4 | `launch-ch.sh` (`mkfs.ext4 -d seed/`) | CH `--disk` (rw) |
| CH command line | `launch-ch.sh` (from CLI flags) | `cloud-hypervisor` process |
| Ephemeral SSH cert | `SshCa::sign_ephemeral()` (Rust, per-exec) | `ssh::exec()` → russh → guest sshd |

## Resource Allocation

Per-guest network resources are allocated from a monotonic slot counter
in `AdminState`. Once allocated, resources persist across shutdown/relaunch
within the same REPL session.

| Resource | Derivation | Example (slot 0) | Example (slot 1) |
|---|---|---|---|
| CID | `3 + slot` | 3 | 4 |
| Admin subnet | `192.168.(249+slot).0/24` | 192.168.249.0/24 | 192.168.250.0/24 |
| Host IP | `192.168.(249+slot).1` | 192.168.249.1 | 192.168.250.1 |
| Guest IP | `192.168.(249+slot).2` | 192.168.249.2 | 192.168.250.2 |
| Admin MAC | `52:54:00:ad:00:(slot+1)` | 52:54:00:ad:00:01 | 52:54:00:ad:00:02 |
| Egress MAC | `52:54:00:e9:00:(slot+1)` | 52:54:00:e9:00:01 | 52:54:00:e9:00:02 |

Admin and egress MACs use different OUI bytes (`ad` vs `e9`) to avoid
collisions. The slot counter never decrements — a shutdown guest keeps
its allocation.

## Runtime Overlay Contents

At launch time, `launch-ch.sh` assembles this overlay tree into ext4:

```
seed/upper/
  ├── var/lib/cloud/seed/nocloud/
  │     ├── user-data            ◄── render_cloud_init() (Rust)
  │     └── meta-data            ◄── render_launch_script() (Rust)
  ├── etc/motlie-vfs/
  │     └── mounts.yaml          ◄── render_mounts_yaml() (Rust)
  ├── etc/hostname               ◄── --hostname flag
  ├── etc/hosts                  ◄── launch-ch.sh (from hostname)
  ├── etc/ssh/ca/
  │     └── user_ca.pub          ◄── --ssh-ca-pubkey flag (v1.3)
  ├── etc/ssh/auth_principals/
  │     ├── <user>               ◄── --ssh-user flag (v1.3)
  │     └── root                 ◄── --ssh-user flag (v1.3)
  └── overlay.d/{common,<guest>}/ ◄── static overlay tree
```

## SSH Transport: vsock (Fully Userspace)

The SSH proxy reaches guest sshd over vsock — not TCP over a TAP NIC.
The guest connects OUT to the host (guest→host direction, same as VFS).
This eliminates TAP and `CAP_NET_ADMIN` entirely.

```
Guest VM (after boot)              CH vhost-vsock              Host process
─────────────────────              ──────────────              ────────────
socat VSOCK-CONNECT:2:2222 ──► /dev/vsock ──► $VSOCK_SOCKET_2222 (UDS)
  │                                                │
  TCP:localhost:22                          UnixListener::accept()
  │                                                │
  sshd (CA trust)                          russh::client::connect_stream()
                                                   │
                                           SSH Handle (multiplexed sessions)
```

**Key insight:** CH's vhost-vsock only supports guest→host connections.
The guest's socat bridge initiates the connection; the host accepts.
Multiple SSH sessions multiplex over this single connection via SSH
channels — no need for multiple vsock connections.

**Why vsock over libslirp hostfwd:**

| | vsock | hostfwd (libslirp) |
|---|---|---|
| Fault isolation | Independent of egress | Shared with egress |
| If egress breaks | SSH still works | SSH also breaks |
| Performance | Direct memory path | TCP through slirp stack |
| Guest changes | socat bridge service | None |
| Product alignment | Matches motlie-vmm.md | Not the target arch |

The guest runs a persistent `socat VSOCK-CONNECT:2:2222 TCP:127.0.0.1:22`
reconnect loop as the `motlie-vmm-vsock-ssh` systemd service, baked into the
image. This is intentional: early boot can race the host-side vsock listener,
so the guest bridge must retry instead of treating the first connect failure as
terminal.

## SSH Auth Model

```
Inbound (user/exec → repl_host):
  Localhost trust. russh binds 127.0.0.1 only.
  Username = guest identity. No key verification.

Outbound (repl_host → guest sshd, over vsock):
  CA-based ephemeral certs.
  ┌─────────────────────────────────────────┐
  │ SshCa (in-memory, per REPL session)     │
  │   signs: Ed25519, principal=<guest>,    │
  │          TTL=60s                        │
  │                                         │
  │ Guest sshd validates:                   │
  │   1. cert signed by user_ca.pub?    ✓   │
  │   2. principal in auth_principals?  ✓   │
  │   3. cert not expired?              ✓   │
  └─────────────────────────────────────────┘

The CA keypair is fresh each REPL session. The pubkey is injected into
each guest's overlay at launch. No image rebuild needed when the CA rotates.
```

## Mount Contract

Per-guest mounts are generated from the provisioned state by
`render_mounts_yaml()`. The typical layout:

```yaml
mounts:
  - tag: <guest>-home
    guest_path: /home/<guest>
    read_only: false
  - tag: <guest>-agent-state
    guest_path: /agent-state
    read_only: false
  - tag: <guest>-workspace
    guest_path: /workspace
    read_only: false
```

- `/home/<guest>` — host-disk-backed working tree
- `/agent-state` — dedicated VFS-backed tool-state layer
  (`~/.codex`, `~/.claude`, `~/.config/claude-code` symlinked here)
- `/workspace` — separate project tree

## Host Requirements

Same as v1.2. On Debian/Ubuntu:

```bash
sudo apt install \
  cloud-hypervisor \
  debian-archive-keyring \
  e2fsprogs \
  libfuse3-dev \
  mmdebstrap \
  pkg-config \
  squashfs-tools-ng \
  uidmap \
  wget \
  curl \
  git
```

Plus: Rust toolchain, `/dev/vhost-vsock` (`sudo modprobe vhost_vsock`).

## Quick Start

```bash
# 1. Build guest image (one-time, ~5min)
cd libs/vmm/examples/v1.3
./build-guest.sh

# 2. Start REPL
cargo run -p motlie-vmm --example repl_host_v1_3 -- \
  --empty \
  --script libs/vmm/examples/v1.3/setup-multiguest.sh.vfs \
  --admin-net=tap --egress-net=vhost-user

# 3. Launch a guest
launch alice

# 4. Exec into it (SSH proxy)
exec alice uname -a
exec alice curl -s https://example.com | head -5

# 5. Run automated validation
validate alice

# 6. Interactive SSH still works
#    ssh alice@192.168.249.2

# 7. Shutdown
shutdown alice
```

## Known Gaps

- The SSH proxy uses vsock (AF_VSOCK) for the host→guest path. This
  depends on the `vhost_vsock` kernel module being loaded.
- `render_cloud_init()` is minimal (`#cloud-config\n`). User account
  creation and package installation are baked into the image.
- `/agent-state` persistence is scoped to the REPL session lifetime.
- CH graceful shutdown returns HTTP 500; the REPL falls back to
  SIGTERM/SIGKILL.
