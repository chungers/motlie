# motlie-vmm — Architecture

A single Rust binary. Builds VM images from the running host.
Runs as a daemon that manages VMs, serves filesystems, signs
certificates, accepts SSH connections, and manages per-user
credentials. Firecracker and passt are embedded inside motlie-vmm
and extracted to memfd at runtime.

Zero host impact before, during, and after running.
Single binary to distribute. No dependencies to install.

## Usage

```bash
# Check platform support before anything else
motlie-vmm check

# Build an image (one-time)
motlie-vmm build \
    --include "openssh-server tmux git curl gh nodejs npm" \
    --output ./images/devbox

# Run the daemon
motlie-vmm daemon \
    --image ./images/devbox \
    --workspace ~/projects \
    --port 2222

# Users connect — each gets their own VM
ssh -p 2222 alice@localhost

# Add a mount to a running VM
motlie-vmm mount alice /data/dataset:/data:ro

# Remove a mount
motlie-vmm unmount alice /data

# View audit events
motlie-vmm events --since 1h
motlie-vmm events --vm alice --credential-only
motlie-vmm events --follow

# First session: tools prompt for auth (device code flow)
#   claude  → "Visit https://console.anthropic.com/device, enter code: ABCD-1234"
#   codex   → "Sign in with ChatGPT: https://..."
#   gh auth login → "Enter code 1A2B-3C4D at https://github.com/login/device"
#
# Tokens are saved. Every subsequent session: already authenticated.
```

## Principles

1. **Single binary.** Firecracker, passt, and the guest agent are
   embedded. One file to distribute. No external dependencies beyond
   `/dev/kvm` and a kernel ≥ 5.10.
1. **One process owns everything.** When motlie-vmm exits, all VMs,
   networking, and temporary state vanish.
1. **Zero host impact.** No iptables, no bridges, no ip_forward,
   no taps in the host namespace. No files in system directories.
1. **Never touch the host sshd.** Port 22 is sacred.
1. **Identity = SSH username = VM name.** `ssh alice@...` → VM “alice”,
   workspace `~/projects/alice`, credentials `~/.motlie-vmm/creds/alice`.
1. **Workspaces persist, VMs don’t.** Files in `/workspace` and
   tool credentials survive VM stop/restart. Installed packages
   are ephemeral.
1. **Authenticate once.** OAuth tokens and API keys are stored
   per-user on the host and re-mounted into every new VM session.
   Tools that support device code flows work through the SSH terminal.
1. **Nothing custom on disk.** The guest agent binary is delivered over
   vsock at boot into tmpfs — never touches persistent storage. CA keys,
   mount config, and env vars are injected into a per-VM overlay.ext4
   that is destroyed on VM stop. Images are generic Ubuntu + a tiny
   bootstrap binary. Updating the VMM binary updates the guest agent
   in every new VM without rebuilding images.
1. **No Docker.** Image built from the host using debootstrap or rsync.

## What’s inside the binary

```
motlie-vmm (single ELF, ~16MB)
  ├── motlie-vmm code                    ~3500 lines Rust
  ├── embedded: firecracker-aarch64   ~3MB static musl
  ├── embedded: passt-aarch64         ~300KB static
  └── embedded: motlie-vmm-guest      ~3MB static musl (FUSE agent)
      delivered to VM over vsock at boot, never written to host disk

squashfs image (built once, generic):
  └── motlie-vmm-bootstrap            ~100KB static musl (baked in)
      connects vsock, receives guest binary, execs it
```

## Process tree

```
Host sshd (port 22)                ← untouched, host netns

motlie-vmm.service                    ← systemd, KillMode=control-group
┌─────────────────── cgroup boundary (all children) ──────────────┐
│ motlie-vmm daemon                   ← host netns (binds :2222)  │
│       │                                                          │
│       ├── russh SSH server        ← in-process, port 2222       │
│       │                                                          │
│       │   ┌── netns-alice (created by unshare) ──────────────┐  │
│       ├── │ passt (from memfd)    userspace net stack         │  │
│       ├── │ firecracker (memfd)   KVM VM (joined via setns)  │  │
│       │   └──────────────────────────────────────────────────┘  │
│       │                                                          │
│       │   ┌── netns-bob (created by unshare) ────────────────┐  │
│       ├── │ passt (from memfd)                                │  │
│       ├── │ firecracker (from memfd)                          │  │
│       │   └──────────────────────────────────────────────────┘  │
│       │                                                          │
│       ├── vsock listener (alice)  ← tokio task                  │
│       │     vsock port 5000 (multiplexed)                       │
│       │       binary transfer, control, FS mounts               │
│       ├── vsock listener (bob)    ← tokio task                  │
│       ├── event bus               ← broadcast + motlie-db       │
│       ├── idle reaper             ← tokio task                  │
│       └── CA keys                 ← in memory                   │
└──────────────────────────────────────────────────────────────────┘

Cgroup: all children (every firecracker, every passt) are in the
  daemon's cgroup. systemd kills the entire tree on stop.

Netns: daemon stays in host netns. Each VM pair (passt + firecracker)
  lives in its own isolated netns. The daemon never touches the host's
  networking. When both processes in a netns die, kernel destroys it.
```

## Host impact: none

```
Host state while running:               After exit (any reason):
  ├── network interfaces: unchanged      identical
  ├── routing table: unchanged           identical
  ├── iptables: unchanged                identical
  ├── ip_forward: unchanged              identical
  ├── /usr, /etc, /lib: unchanged        identical
  └── processes: motlie-vmm + children      nothing
```

## Filesystem layout

```
~/.motlie-vmm/                              # persists
  config.toml
  ca/
    user_ca, user_ca.pub
    host_ca, host_ca.pub
  creds/                                 # per-user credentials (persist forever)
    alice/
      .config/gh/hosts.yml               # gh OAuth token
      .claude/                           # Claude Code session/tokens
      .codex/config.toml                 # Codex config/tokens
      .npmrc                             # npm auth token
    bob/
      ...

./images/devbox/                         # built once
  manifest.json
  kernel.img                             # ~16MB
  rootfs.squashfs                        # ~150-200MB

~/projects/                              # per-user workspaces (persist)
  alice/
  bob/

/tmp/motlie-vmm/                            # per-user scratch (persist across VM lifecycle)
  alice/
  bob/

/var/lib/motlie-vmm/vms/                    # ephemeral (destroyed on VM stop)
  alice/
    firecracker.pid, passt.pid
    firecracker.sock, vsock.sock
    overlay.ext4, vmconfig.json

/var/lib/motlie-vmm/db/                  # persistent (motlie-db / RocksDB)
                                         # event log, VM graph, resource edges

/var/lib/motlie-vmm/snapshots/           # persistent (snapshot/restore)
  alice/                                 # latest snapshot
    vmstate.bin, mem.bin
    overlay.ext4, meta.json
```

Nothing in `/usr/bin`, `/usr/local/bin`, `/etc`, or any system directory.

## Architecture diagram

```
                    ┌─────────────────────────────────────────────┐
                    │          motlie-vmm daemon (one process)        │
                    │                                             │
 ssh alice@:2222 ──►│  russh SSH server (:2222)                   │
                    │    │ 1. parse username → "alice"             │
                    │    │ 2. ensure_vm("alice")                   │
                    │    │ 3. sign ephemeral cert (60s TTL)        │
                    │    │ 4. connect to VM sshd, bridge channels  │
                    │    │                                        │
                    │  ┌─┴──────────┐  ┌───────────┐              │
                    │  │ VM Manager │  │  SSH CA    │              │
                    │  │ memfd exec │  │ (in mem)   │              │
                    │  └──┬─────┬──┘  └────────────┘              │
                    │     │     │                                   │
                    │     │  ┌──┴──────────────────┐               │
                    │     │  │ vsock FS Server      │               │
                    │     │  │ (per VM, per user)   │               │
                    │     │  │                      │               │
                    │     │  │ /workspace        rw │               │
                    │     │  │  → ~/projects/alice  │               │
                    │     │  │ /tmp              rw │               │
                    │     │  │  → /tmp/motlie-vmm/     │               │
                    │     │  │      alice           │               │
                    │     │  │ /root/.config/gh  rw │               │
                    │     │  │  → ~/.motlie-vmm/creds/ │               │
                    │     │  │      alice/.config/gh│               │
                    │     │  │ /root/.claude     rw │               │
                    │     │  │  → ~/.motlie-vmm/creds/ │               │
                    │     │  │      alice/.claude   │               │
                    │     │  │ /root/.codex      rw │               │
                    │     │  │  → ~/.motlie-vmm/creds/ │               │
                    │     │  │      alice/.codex    │               │
                    │     │  └─────────────────────┘               │
                    │     │                                        │
                    │  ┌──┴──── netns (alice) ───────────────────┐ │
                    │  │  passt (memfd) ── tap0 ── firecracker   │ │
                    │  │    translates          (memfd)           │ │
                    │  │    to host sockets      │               │ │
                    │  │    no iptables          sshd (CA)       │ │
                    │  │    no ip_forward        guest agent     │ │
                    │  │                         /workspace      │ │
                    │  │                         /tmp            │ │
                    │  │                         ~/.config/gh    │ │
                    │  │                         ~/.claude       │ │
                    │  │                         ~/.codex        │ │
                    │  │                         internet ✓      │ │
                    │  └─────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────┘
```

## Components

### 1. Platform Check (phase 1)

`motlie-vmm check` — verifies KVM, vsock, architecture, and kernel
version before anything else runs. Build this first so every subsequent
phase can use it for validation.

`motlie-vmm check` probes the host to verify all prerequisites before
running. It doesn’t just check file existence — it actually tries each
capability (opens `/dev/kvm`, issues `KVM_CREATE_VM` ioctl, verifies
vsock module is loaded).

**Example output:**

```
$ motlie-vmm check

Platform
  arch:        aarch64                         ✓
  kernel:      6.8.0-49-generic (≥ 5.10)       ✓

KVM
  /dev/kvm:    exists                          ✓
  accessible:  read/write (uid 1000)           ✓
  create VM:   ioctl(KVM_CREATE_VM) ok         ✓

vsock
  module:      vhost_vsock loaded              ✓
  /dev/vhost-vsock: exists                     ✓
  accessible:  read/write (uid 1000)           ✓

Image tools (for `motlie-vmm build` only)
  debootstrap: /usr/sbin/debootstrap           ✓
  mksquashfs:  /usr/bin/mksquashfs             ✓

Memory
  available:   121 GB                          ✓ (≥ 4 GB)

Ready to run.
```

**Implementation:**

```rust
pub fn check() -> Result<Report> {
    let mut report = Report::new();

    // Architecture
    let arch = std::env::consts::ARCH;
    report.check("arch", arch, arch == "aarch64");

    // Kernel version
    let uname = nix::sys::utsname::uname()?;
    let release = uname.release().to_str().unwrap_or("");
    let major_minor = parse_kernel_version(release);
    report.check("kernel", release, major_minor >= (5, 10));

    // KVM — don't just check existence, try to use it
    let kvm_exists = Path::new("/dev/kvm").exists();
    report.check("/dev/kvm exists", kvm_exists, kvm_exists);

    if kvm_exists {
        match std::fs::OpenOptions::new().read(true).write(true)
            .open("/dev/kvm")
        {
            Ok(fd) => {
                report.check("/dev/kvm accessible", "read/write", true);
                // Actually try creating a VM
                let ret = unsafe {
                    libc::ioctl(fd.as_raw_fd(), KVM_CREATE_VM, 0)
                };
                if ret >= 0 {
                    unsafe { libc::close(ret); }
                    report.check("KVM_CREATE_VM", "ok", true);
                } else {
                    report.check("KVM_CREATE_VM", "failed", false);
                }
            }
            Err(e) => {
                report.check("/dev/kvm accessible", e.to_string(), false);
            }
        }
    }

    // vsock — module must be loaded, not just available
    let vsock_mod = module_loaded("vhost_vsock");
    report.check("vhost_vsock module", vsock_mod, vsock_mod);

    let vhost_vsock = Path::new("/dev/vhost-vsock").exists();
    report.check("/dev/vhost-vsock", vhost_vsock, vhost_vsock);

    if !vsock_mod {
        report.hint("try: sudo modprobe vhost_vsock");
    }

    // Image build tools (optional — only needed for `build`)
    report.check_optional("debootstrap", which("debootstrap"));
    report.check_optional("mksquashfs", which("mksquashfs"));

    // Memory
    let mem_gb = available_memory_gb();
    report.check("available memory", format!("{} GB", mem_gb), mem_gb >= 4);

    report.print();
    Ok(report)
}

fn module_loaded(name: &str) -> bool {
    std::fs::read_to_string("/proc/modules")
        .map(|s| s.lines().any(|l| l.starts_with(name)))
        .unwrap_or(false)
}

fn available_memory_gb() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .unwrap_or_default();
    meminfo.lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0) / 1_048_576
}
```

`motlie-vmm daemon` runs this automatically at startup (via
`preflight_checks`) and refuses to start if any required check
fails. `motlie-vmm check` lets you diagnose issues before running.

**Common failure modes:**

|Symptom                             |Cause                                         |Fix                                               |
|------------------------------------|----------------------------------------------|--------------------------------------------------|
|`/dev/kvm` missing                  |KVM not enabled in kernel                     |Check BIOS virtualization settings                |
|`/dev/kvm` permission denied        |User not in `kvm` group                       |`sudo usermod -aG kvm $USER`                      |
|`KVM_CREATE_VM` fails               |Nested virt disabled or another VMM holds lock|Check `/sys/module/kvm/parameters/nested`         |
|`vhost_vsock` not loaded            |Module not auto-loaded                        |`sudo modprobe vhost_vsock`, add to `/etc/modules`|
|`/dev/vhost-vsock` permission denied|User not in `kvm` group                       |`sudo usermod -aG kvm $USER`                      |
|arch not `aarch64`                  |Wrong platform                                |motlie-vmm targets DGX Spark (aarch64)            |

**Prerequisites summary:**

```
/dev/kvm           (aarch64, kernel ≥ 5.10)
vhost_vsock        (kernel module, loaded)

Image building (one-time): debootstrap, mksquashfs

Run `motlie-vmm check` to verify all of the above.
```

### 2. VMM Backend Abstraction (phase 1)

The core abstraction that enables multiple hypervisor backends. All
components above this trait (SSH proxy, FUSE mounts, credential
management, event bus) are backend-agnostic. The initial implementation
is `FirecrackerBackend` for Linux/KVM. A future `VzBackend` targets
macOS with Apple’s Virtualization.framework.

```rust
pub trait VmmBackend: Send + Sync {
    type Vm: VmHandle;

    /// Verify platform prerequisites (KVM, vsock, etc.)
    fn check_platform(&self) -> Result<Report>;

    /// Build a guest image for this backend
    fn build_image(&self, config: &BuildConfig) -> Result<()>;

    /// Create and boot a VM
    fn create_vm(&self, config: &VmConfig) -> Result<Self::Vm>;

    /// Stop and destroy a VM
    fn stop_vm(&self, vm: &mut Self::Vm) -> Result<()>;

    /// Snapshot a running VM
    fn snapshot(&self, vm: &Self::Vm, path: &Path) -> Result<()>;

    /// Restore a VM from snapshot
    fn restore(&self, path: &Path, config: &VmConfig) -> Result<Self::Vm>;
}

pub trait VmHandle: Send + Sync {
    /// Get a vsock listener for the multiplexed port
    fn vsock_listener(&self) -> Result<VsockListener>;

    /// Guest IP for internal SSH connection
    fn guest_ip(&self) -> Ipv4Addr;

    /// CID for vsock
    fn cid(&self) -> u32;

    /// Check if VM is running
    fn is_running(&self) -> bool;
}
```

**FirecrackerBackend** (Linux):

```rust
pub struct FirecrackerBackend {
    firecracker: &'static EmbeddedBinary,
    passt: &'static EmbeddedBinary,
    guest_agent: &'static EmbeddedBinary,
    bootstrap: &'static EmbeddedBinary,
}

pub struct FirecrackerVm {
    pub name: String,
    pub firecracker: Child,
    pub passt: Child,
    pub guest_ip: Ipv4Addr,
    pub cid: u32,
    pub last_activity: Arc<Mutex<Instant>>,
    pub vsock_uds: PathBuf,
    _fc_fd: OwnedFd,
    _passt_fd: OwnedFd,
}

impl VmHandle for FirecrackerVm {
    fn vsock_listener(&self) -> Result<VsockListener> {
        vsock_listen(&self.vsock_uds, 5000)
    }
    fn guest_ip(&self) -> Ipv4Addr { self.guest_ip }
    fn cid(&self) -> u32 { self.cid }
    fn is_running(&self) -> bool { /* check PID */ }
}
```

The Daemon holds a `Box<dyn VmmBackend>` and all VM operations go
through the trait. The SSH server, FS server, control protocol, event
bus, and credential management never know which backend is running.

```rust
pub struct Daemon {
    backend: Box<dyn VmmBackend<Vm = Box<dyn VmHandle>>>,
    config: DaemonConfig,
    ca: SshCa,
    vms: Arc<Mutex<HashMap<String, Box<dyn VmHandle>>>>,
    event_bus: Arc<EventBus>,
    // ...
}
```

### 3. Embedded Binary Launcher (phase 1)

Writes an embedded binary to memfd, spawns it via `/proc/self/fd/{N}`.
Binary never touches disk.

```rust
pub struct EmbeddedBinary {
    name: &'static str,
    data: &'static [u8],
}

impl EmbeddedBinary {
    pub const fn new(name: &'static str, data: &'static [u8]) -> Self {
        Self { name, data }
    }

    fn prepare(&self) -> Result<(OwnedFd, String)> {
        let cname = CString::new(self.name)?;
        let fd = memfd_create(&cname, MemFdCreateFlag::MFD_CLOEXEC)?;
        let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
        file.write_all(self.data)?;
        let fd_path = format!("/proc/self/fd/{}", file.as_raw_fd());
        let owned = unsafe { OwnedFd::from_raw_fd(file.as_raw_fd()) };
        std::mem::forget(file);
        Ok((owned, fd_path))
    }

    pub fn spawn(&self, args: &[&str]) -> Result<(Child, OwnedFd)> {
        let (fd, path) = self.prepare()?;
        let child = Command::new(&path).args(args)
            .stdout(Stdio::null()).stderr(Stdio::null()).spawn()?;
        Ok((child, fd))
    }

    pub unsafe fn spawn_with_pre_exec<F>(
        &self, args: &[&str], pre_exec: F,
    ) -> Result<(Child, OwnedFd)>
    where F: FnMut() -> std::io::Result<()> + Send + Sync + 'static {
        let (fd, path) = self.prepare()?;
        let child = Command::new(&path).args(args)
            .stdout(Stdio::null()).stderr(Stdio::null())
            .pre_exec(pre_exec).spawn()?;
        Ok((child, fd))
    }
}

pub static FIRECRACKER: EmbeddedBinary = EmbeddedBinary::new(
    "firecracker", include_bytes!(concat!(env!("OUT_DIR"), "/firecracker")));
pub static PASST: EmbeddedBinary = EmbeddedBinary::new(
    "passt", include_bytes!(concat!(env!("OUT_DIR"), "/passt")));
pub static GUEST_AGENT: EmbeddedBinary = EmbeddedBinary::new(
    "motlie-vmm-guest", include_bytes!(concat!(env!("OUT_DIR"), "/motlie-vmm-guest")));
pub static BOOTSTRAP: EmbeddedBinary = EmbeddedBinary::new(
    "motlie-vmm-bootstrap", include_bytes!(concat!(env!("OUT_DIR"), "/motlie-vmm-bootstrap")));
```

### 4. Image Builder (phase 1)

Builds from the running host without Docker. The image contains NO
motlie-vmm binaries — only a tiny vsock bootstrap (~100KB, ~50 lines)
that fetches the real guest agent from the host at boot. The guest
agent never touches disk; it runs from tmpfs inside the VM.

```rust
pub fn build(config: BuildConfig) -> Result<()> {
    let rootfs = tempdir().join("rootfs");

    match config.strategy {
        BuildStrategy::Debootstrap => debootstrap(&rootfs, &config)?,
        BuildStrategy::HostCopy => rsync_host(&rootfs, &config)?,
    }

    // Bootstrap binary — the only motlie binary in the image.
    // Connects to host vsock, receives guest agent, execs it.
    // Essentially frozen — changes only if vsock port changes.
    install_bootstrap(&rootfs)?;
    install_bootstrap_service(&rootfs)?;

    configure_sshd_for_ca(&rootfs)?;
    write_overlay_init(&rootfs)?;
    install_credential_profile(&rootfs)?;
    configure_dns(&rootfs)?;
    strip_unnecessary(&rootfs)?;

    std::fs::create_dir_all(&config.output)?;
    make_squashfs(&rootfs, &config.output.join("rootfs.squashfs"))?;
    obtain_kernel(&config.output.join("kernel.img"), config.kernel_version.as_deref())?;
    write_manifest(&config)?;

    Ok(())
}

fn install_bootstrap(rootfs: &Path) -> Result<()> {
    // Bootstrap is a separate small static binary, also embedded
    // in the VMM via include_bytes! but baked into the image at
    // build time (not delivered over vsock — that would be circular).
    std::fs::write(
        rootfs.join("usr/local/bin/motlie-vmm-bootstrap"),
        BOOTSTRAP.data,
    )?;
    set_permissions(rootfs.join("usr/local/bin/motlie-vmm-bootstrap"), 0o755)?;
    Ok(())
}

fn install_credential_profile(rootfs: &Path) -> Result<()> {
    // Shell profile that sources API keys on login
    std::fs::write(rootfs.join("etc/profile.d/motlie-vmm-credentials.sh"), r#"
# motlie-vmm: load credentials injected by the daemon
if [ -f /etc/motlie-vmm/env ]; then
    set -a
    . /etc/motlie-vmm/env
    set +a
fi
"#)?;
    set_permissions(rootfs.join("etc/profile.d/motlie-vmm-credentials.sh"), 0o644)?;

    // Ensure credential target directories exist in the rootfs
    // (FUSE will mount over these)
    std::fs::create_dir_all(rootfs.join("root/.config/gh"))?;
    std::fs::create_dir_all(rootfs.join("root/.claude"))?;
    std::fs::create_dir_all(rootfs.join("root/.codex"))?;

    Ok(())
}
```

**motlie-vmm-bootstrap** — the only motlie binary baked into the
squashfs. ~50 lines of Rust, ~100KB static musl. It does exactly
one thing: fetch the real guest agent from the host over vsock,
write it to tmpfs, and exec it. It never changes unless the vsock
port number changes.

```rust
// motlie-vmm-bootstrap — entire source
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;

const HOST_CID: u32 = 2;
const VMM_PORT: u32 = 5000;         // single port for all vsock traffic
const GUEST_PATH: &str = "/tmp/motlie-vmm-guest";
const MAX_RETRIES: u32 = 30;

fn main() {
    // Retry loop — host vsock listener may not be ready yet
    let mut stream = None;
    for _ in 0..MAX_RETRIES {
        match vsock_connect(HOST_CID, VMM_PORT) {
            Ok(s) => { stream = Some(s); break; }
            Err(_) => std::thread::sleep(
                std::time::Duration::from_millis(100)),
        }
    }
    let mut stream = stream.expect("failed to connect to host vsock");

    // Handshake: identify this connection as a binary request
    let handshake = b"{\"type\":\"binary_request\"}\n";
    stream.write_all(handshake).expect("send handshake");

    // Read length-prefixed binary: [u64 len][bytes...]
    let mut len_buf = [0u8; 8];
    stream.read_exact(&mut len_buf).expect("read length");
    let len = u64::from_le_bytes(len_buf) as usize;

    let mut binary = vec![0u8; len];
    stream.read_exact(&mut binary).expect("read binary");

    // Write to tmpfs (RAM-backed, never hits persistent storage)
    std::fs::write(GUEST_PATH, &binary).expect("write binary");
    std::fs::set_permissions(
        GUEST_PATH,
        std::fs::Permissions::from_mode(0o755),
    ).expect("chmod");

    // Exec — replaces this process with the guest agent
    let err = exec::execvp(
        GUEST_PATH,
        &[GUEST_PATH, "--from-bootstrap"],
    );
    panic!("exec failed: {}", err);
}
```

**Bootstrap systemd unit** — baked into squashfs. Runs before sshd.
Starts the bootstrap, which fetches the guest agent and exec’s into it.

```ini
# /etc/systemd/system/motlie-vmm-guest.service
[Unit]
Description=motlie-vmm guest agent (vsock bootstrap)
After=network.target
Before=sshd.service

[Service]
Type=simple
ExecStart=/usr/local/bin/motlie-vmm-bootstrap
Restart=on-failure
RestartSec=1

[Install]
WantedBy=multi-user.target
```

Note: the unit is still called `motlie-vmm-guest.service` because
from systemd’s perspective the long-running process IS the guest agent
(bootstrap exec’s into it). The PID stays the same.

**overlay-init** — runs as PID 1 (`init=/sbin/overlay-init` on kernel
command line). Stacks the read-only squashfs with the writable ext4
overlay, pivot_roots into the merged filesystem, then execs systemd.
The overlay contains only config files (CA keys, mount config, env) —
no binaries.

```bash
#!/bin/sh
# /sbin/overlay-init — baked into squashfs at image build time
# Kernel boots with: init=/sbin/overlay-init overlay_root=vdb
set -e

# Parse overlay device from kernel cmdline
OVERLAY_DEV=""
for arg in $(cat /proc/cmdline); do
    case "$arg" in
        overlay_root=*) OVERLAY_DEV="/dev/${arg#overlay_root=}" ;;
    esac
done

# 1. Mount essential filesystems
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev

# 2. Create mount points
mkdir -p /mnt/lower /mnt/upper /mnt/work /mnt/merged

# 3. Mount the read-only squashfs (already the rootfs, bind it)
mount --bind / /mnt/lower

# 4. Mount the writable overlay ext4
mount -t ext4 "$OVERLAY_DEV" /mnt/upper

# 5. Create overlay work directory
mkdir -p /mnt/upper/upper /mnt/upper/work

# 6. Stack overlayfs: lower (squashfs) + upper (ext4)
mount -t overlay overlay \
    -o lowerdir=/mnt/lower,upperdir=/mnt/upper/upper,workdir=/mnt/upper/work \
    /mnt/merged

# Overlay now contains only config files:
#   /etc/ssh/ca/user_ca.pub, host cert, principals, mounts.json, env
# No binaries — guest agent arrives over vsock after systemd starts.

# 7. Move mount points into merged root
mkdir -p /mnt/merged/mnt/lower /mnt/merged/mnt/upper
mount --move /mnt/lower /mnt/merged/mnt/lower
mount --move /mnt/upper /mnt/merged/mnt/upper

# 8. Pivot into merged root
cd /mnt/merged
pivot_root . mnt/merged/mnt/old_root

# 9. Clean up old root mounts
umount -l /mnt/old_root/proc 2>/dev/null || true
umount -l /mnt/old_root/sys 2>/dev/null || true
umount -l /mnt/old_root/dev 2>/dev/null || true

# 10. Exec systemd as PID 1
exec /sbin/init
```

**VM boot sequence:**

```
Firecracker boots kernel with init=/sbin/overlay-init overlay_root=vdb
  │
  ├─ overlay-init (PID 1, shell script)
  │    mount squashfs (vda) as lower
  │    mount ext4 (vdb) as upper     ← config only: CA, principals, env
  │    mount -t overlay (lower + upper → merged)
  │    pivot_root into merged
  │    exec /sbin/init
  │
  ├─ systemd (PID 1, takes over)
  │    ├─ motlie-vmm-guest.service
  │    │    ExecStart=/usr/local/bin/motlie-vmm-bootstrap
  │    │      → connects to host vsock port 5000
  │    │      → sends handshake: { type: "binary_request" }
  │    │      → host sends guest agent binary (~3MB)
  │    │      → bootstrap writes to /tmp/motlie-vmm-guest (tmpfs)
  │    │      → bootstrap exec's into guest agent (same PID)
  │    │      → guest agent connects vsock:5000 { type: "control" }
  │    │      → guest agent connects vsock:5000 { type: "fs", tag: "workspace" }
  │    │      → guest agent connects vsock:5000 { type: "fs", tag: "scratch" }
  │    │      → ... (one connection per mount)
  │    │      → creates FUSE mounts
  │    │      → sends Ready on control connection
  │    │      → listens for dynamic mount commands
  │    │
  │    └─ sshd.service               ← stock OpenSSH, CA-based auth
  │         trusts /etc/ssh/ca/user_ca.pub
  │         host cert from overlay
  │         principals file from overlay
  │
  └─ VM ready (host receives "ready" signal, completes ensure_vm)
```

### 5. VM Manager + Networking (phase 2)

This is the `FirecrackerBackend` implementation of the `VmmBackend`
trait (§2). Each VM gets: passt (memfd) in its own netns, firecracker
(memfd) joining the same netns, vsock mounts for workspace + scratch

- credentials.

```rust
// FirecrackerBackend's Vm — implements VmHandle trait
pub struct Vm {
    pub name: String,
    pub firecracker: Child,
    pub passt: Child,
    pub guest_ip: Ipv4Addr,
    pub cid: u32,
    pub last_activity: Arc<Mutex<Instant>>,
    _fc_fd: OwnedFd,        // memfd, freed on drop
    _passt_fd: OwnedFd,     // memfd, freed on drop
}

impl Daemon {
    pub async fn ensure_vm(&self, username: &str) -> Result<Ipv4Addr> {
        // Return existing VM
        {
            let vms = self.vms.lock().await;
            if let Some(vm) = vms.get(username) { return Ok(vm.guest_ip); }
        }

        // Create per-user directories
        let workspace = self.config.workspace_base.join(username);
        let scratch = self.config.scratch_base.join(username);
        std::fs::create_dir_all(&workspace)?;
        std::fs::create_dir_all(&scratch)?;

        let cid = self.cid_counter.fetch_add(1, Ordering::SeqCst);
        let guest_ip: Ipv4Addr = format!("172.16.{}.2", cid).parse()?;
        let gateway: Ipv4Addr = format!("172.16.{}.1", cid).parse()?;
        let vm_dir = PathBuf::from(format!("/var/lib/motlie-vmm/vms/{}", username));
        std::fs::create_dir_all(&vm_dir)?;

        // ── 1. passt from memfd, unshare(CLONE_NEWNET) ──

        let (passt_child, passt_fd) = unsafe {
            PASST.spawn_with_pre_exec(
                &["--foreground", "--tap-name", "tap0",
                  "--address", &guest_ip.to_string(),
                  "--gateway", &gateway.to_string(),
                  "--netmask", "255.255.255.252",
                  "--dns", "8.8.8.8", "--dns", "1.1.1.1"],
                || {
                    nix::sched::unshare(nix::sched::CloneFlags::CLONE_NEWNET)?;
                    Ok(())
                },
            )?
        };
        let passt_pid = passt_child.id();
        std::fs::write(vm_dir.join("passt.pid"), passt_pid.to_string())?;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let netns_path = format!("/proc/{}/ns/net", passt_pid);

        // ── 2. Overlay ──

        let overlay = vm_dir.join("overlay.ext4");
        create_sparse_overlay(&overlay, "2G")?;

        // ── 3. Inject per-VM config into overlay ──
        //
        // Only config files go here — no binaries.
        // Guest agent arrives over vsock via the bootstrap.

        // SSH CA
        inject_into_overlay(&overlay, "/etc/ssh/ca/user_ca.pub",
            self.ca.user_ca_pubkey_openssh()?.as_bytes())?;

        // Host cert
        let host_cert = self.ca.sign_host_cert(
            &read_host_pubkey_from_image(&self.config.image)?,
            &[username.to_string(), guest_ip.to_string()],
        )?;
        inject_into_overlay(&overlay, "/etc/ssh/ssh_host_ed25519_key-cert.pub",
            host_cert.to_openssh()?.as_bytes())?;

        // Principal = username only
        inject_into_overlay(&overlay, "/etc/ssh/auth_principals/root",
            username.as_bytes())?;

        // ── 4. Inject API key environment variables ──

        self.inject_env_credentials(&overlay)?;

        // ── 5. Build mount list: workspace + scratch + credentials ──

        let mut mounts = vec![
            VsockMount {
                tag: "workspace".into(),
                host_path: workspace,
                guest_path: "/workspace".into(),
                read_only: false,
            },
            VsockMount {
                tag: "scratch".into(),
                host_path: scratch,
                guest_path: "/tmp".into(),
                read_only: false,
            },
        ];

        // Per-user credential mounts (read-write so tools can write tokens)
        let cred_mounts = self.setup_credential_mounts(username)?;
        mounts.extend(cred_mounts);

        // Write mount config for guest agent
        let mount_cfg: Vec<serde_json::Value> = mounts.iter().map(|m| {
            serde_json::json!({
                "tag": m.tag,
                "guest_path": m.guest_path,
                "read_only": m.read_only,
            })
        }).collect();
        inject_into_overlay(&overlay, "/etc/motlie-vmm/mounts.json",
            serde_json::to_string_pretty(&serde_json::json!({"mounts": mount_cfg}))?.as_bytes())?;

        // ── 6. Start vsock listener (single port, multiplexed) ──
        //
        // One listener on vsock port 5000. Each incoming connection
        // sends a handshake identifying its type. The handler routes:
        //   BinaryRequest → send guest agent binary, close
        //   Control       → bidirectional control loop
        //   Fs { tag }    → FS request/response loop for that mount

        let vsock_uds = vm_dir.join("vsock.sock");
        let uds = vsock_uds.clone();
        tokio::spawn(async move {
            let listener = vsock_listen(&uds, 5000).await.unwrap();
            loop {
                let stream = listener.accept().await.unwrap();
                let vm_ctx = vm_context.clone();
                tokio::spawn(async move {
                    let handshake: HandshakeMsg = recv_msg(&mut stream).await;
                    match handshake {
                        HandshakeMsg::BinaryRequest => {
                            let binary = GUEST_AGENT.data;
                            let len = (binary.len() as u64).to_le_bytes();
                            stream.write_all(&len).await.ok();
                            stream.write_all(binary).await.ok();
                            // Connection closes. Bootstrap has the binary.
                        }
                        HandshakeMsg::Control => {
                            control_loop(stream, &vm_ctx).await;
                        }
                        HandshakeMsg::Fs { tag } => {
                            fs_loop(stream, &vm_ctx, &tag).await;
                        }
                    }
                });
            }
        });

        // ── 7. Firecracker config ──

        let image_dir = &self.config.image;
        let boot_args = format!(
            "keep_bootcon console=ttyS0 reboot=k panic=1 pci=off \
             ip={}::{}:255.255.255.252::eth0:off \
             init=/sbin/overlay-init overlay_root=vdb",
            guest_ip, gateway
        );
        let fc_config = serde_json::json!({
            "boot-source": {
                "kernel_image_path": image_dir.join("kernel.img").to_str(),
                "boot_args": boot_args },
            "drives": [
                { "drive_id": "rootfs",
                  "path_on_host": image_dir.join("rootfs.squashfs").to_str(),
                  "is_root_device": true, "is_read_only": true },
                { "drive_id": "overlay",
                  "path_on_host": overlay.to_str(),
                  "is_root_device": false, "is_read_only": false }],
            "network-interfaces": [{ "iface_id": "eth0",
                "guest_mac": generate_mac(username), "host_dev_name": "tap0" }],
            "vsock": { "vsock_id": "vsock0", "guest_cid": cid,
                "uds_path": vsock_uds.to_str() },
            "machine-config": {
                "vcpu_count": self.config.vcpus_per_vm,
                "mem_size_mib": self.config.mem_per_vm_mib }
        });
        let config_path = vm_dir.join("vmconfig.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&fc_config)?)?;

        // ── 8. Firecracker from memfd, join passt's netns ──

        let api_sock = vm_dir.join("firecracker.sock").to_str().unwrap().to_string();
        let cfg = config_path.to_str().unwrap().to_string();
        let netns = netns_path.clone();

        let (fc_child, fc_fd) = unsafe {
            FIRECRACKER.spawn_with_pre_exec(
                &["--api-sock", &api_sock, "--config-file", &cfg],
                move || {
                    let fd = std::fs::File::open(&netns)?;
                    nix::sched::setns(
                        std::os::unix::io::AsRawFd::as_raw_fd(&fd),
                        nix::sched::CloneFlags::CLONE_NEWNET)?;
                    Ok(())
                },
            )?
        };
        std::fs::write(vm_dir.join("firecracker.pid"), fc_child.id().to_string())?;

        // ── 9. Wait for guest readiness ──
        //
        // Boot sequence inside VM:
        //   overlay-init → systemd → bootstrap connects vsock:5000
        //   → handshake: BinaryRequest → receives guest binary
        //   → writes to tmpfs → exec's into guest agent
        //   → guest agent connects vsock:5000 (Control + Fs)
        //   → mounts FUSE, sends Ready on control connection

        wait_for_guest_ready(&vsock_uds, Duration::from_secs(30)).await?;
        println!("  vm created: {} → {}", username, guest_ip);

        let vm = Vm {
            name: username.to_string(), firecracker: fc_child, passt: passt_child,
            guest_ip, cid,
            last_activity: Arc::new(Mutex::new(Instant::now())),
            _fc_fd: fc_fd, _passt_fd: passt_fd,
        };
        let ip = vm.guest_ip;
        self.vms.lock().await.insert(username.to_string(), vm);
        Ok(ip)
    }

    /// Create per-user credential directories and return vsock mounts.
    /// These are READ-WRITE so tools can write OAuth tokens.
    fn setup_credential_mounts(&self, username: &str) -> Result<Vec<VsockMount>> {
        let cred_base = dirs::home_dir().unwrap()
            .join(".motlie-vmm/creds")
            .join(username);

        let mut mounts = Vec::new();

        for cm in &self.config.credential_mounts {
            let host_path = cred_base.join(&cm.host_subpath);

            if cm.is_file {
                // Ensure parent dir and file exist
                if let Some(parent) = host_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                if !host_path.exists() {
                    std::fs::write(&host_path, "")?;
                }
            } else {
                std::fs::create_dir_all(&host_path)?;
            }

            mounts.push(VsockMount {
                tag: format!("cred-{}", cm.host_subpath.replace(['/', '.'], "-")),
                host_path,
                guest_path: cm.guest_path.clone(),
                read_only: false,
            });
        }

        Ok(mounts)
    }

    /// Inject API keys as environment variables into the overlay.
    /// These are sourced by /etc/profile.d/motlie-vmm-credentials.sh on login.
    fn inject_env_credentials(&self, overlay: &Path) -> Result<()> {
        let mut env_lines = Vec::new();

        // From motlie-vmm config
        for (key, value) in &self.config.credential_env {
            if !value.is_empty() {
                env_lines.push(format!("{}={}", key, value));
            }
        }

        // From host environment (fallback)
        let keys = [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
            "GITHUB_TOKEN", "GH_TOKEN",
        ];
        for key in keys {
            if let Ok(val) = std::env::var(key) {
                if !env_lines.iter().any(|l| l.starts_with(&format!("{}=", key))) {
                    env_lines.push(format!("{}={}", key, val));
                }
            }
        }

        if !env_lines.is_empty() {
            inject_into_overlay(overlay, "/etc/motlie-vmm/env",
                env_lines.join("\n").as_bytes())?;
        }

        Ok(())
    }

    async fn stop_vm(&self, username: &str) -> Result<()> {
        let mut vms = self.vms.lock().await;
        if let Some(mut vm) = vms.remove(username) {
            vm.firecracker.kill().ok(); vm.firecracker.wait().ok();
            vm.passt.kill().ok();       vm.passt.wait().ok();
            // _fc_fd, _passt_fd drop → memfds freed
            // passt + firecracker were the only processes in their netns
            // → kernel destroys netns, all networking state vanishes
            // → no bridges, taps, iptables, or routes to clean up
            std::fs::remove_dir_all(format!("/var/lib/motlie-vmm/vms/{}", username)).ok();
            // Credentials in ~/.motlie-vmm/creds/{user}/ persist
            // Workspace in ~/projects/{user}/ persists
            println!("  vm stopped: {}", username);
        }
        Ok(())
    }
}
```

### 6. SSH CA (phase 3)

```rust
pub struct SshCa { user_ca: PrivateKey, host_ca: PrivateKey }

impl SshCa {
    pub fn new(dir: &Path) -> Result<Self> {
        Ok(Self {
            user_ca: load_or_generate(&dir.join("user_ca"))?,
            host_ca: load_or_generate(&dir.join("host_ca"))?,
        })
    }

    pub fn user_ca_pubkey_openssh(&self) -> Result<String> {
        Ok(self.user_ca.public_key().to_openssh()?)
    }

    /// Throwaway keypair + 60s cert for internal connection.
    pub fn sign_ephemeral_for_vm(&self, username: &str) -> Result<(PrivateKey, Certificate)> {
        let key = PrivateKey::random(&mut rand::thread_rng(), Algorithm::Ed25519)?;
        let now = unix_timestamp();
        let cert = Builder::new_v01(key.public_key().clone(), CertType::User)?
            .serial(rand::random())?.key_id(&format!("motlie-vmm-{}-{}", username, now))?
            .valid_after(now - 300)?.valid_before(now + 60)?
            .principals(&[username.to_string()])?.sign(&self.user_ca)?;
        Ok((key, cert))
    }

    pub fn sign_host_cert(&self, pubkey: &PublicKey, names: &[String]) -> Result<Certificate> {
        let now = unix_timestamp();
        let cert = Builder::new_v01(pubkey.clone(), CertType::Host)?
            .serial(rand::random())?.key_id(&format!("motlie-vmm-host-{}", names[0]))?
            .valid_after(now - 300)?.valid_before(now + 365 * 86400)?
            .principals(names)?.sign(&self.host_ca)?;
        Ok(cert)
    }
}
```

### 7. SSH Configuration Inside the VM (phase 3)

The VM runs stock OpenSSH with CA-based authentication. No custom
sshd patches — only standard `sshd_config` directives (stable since
OpenSSH 5.4, 2010).

**sshd_config (baked into image at build time):**

```
TrustedUserCAKeys /etc/ssh/ca/user_ca.pub
AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
PubkeyAuthentication yes
AuthorizedKeysFile none
HostCertificate /etc/ssh/ssh_host_ed25519_key-cert.pub
PermitRootLogin yes
```

This config references three files that **don’t exist at build time**.
They’re injected per-VM into the overlay at runtime:

|File                                    |Contents                                |Injected by                             |
|----------------------------------------|----------------------------------------|----------------------------------------|
|`/etc/ssh/ca/user_ca.pub`               |motlie-vmm’s user CA public key         |`inject_into_overlay` during `ensure_vm`|
|`/etc/ssh/ssh_host_ed25519_key-cert.pub`|Host cert signed by motlie-vmm’s host CA|`ca.sign_host_cert` during `ensure_vm`  |
|`/etc/ssh/auth_principals/root`         |The username string (e.g. “alice”)      |`inject_into_overlay` during `ensure_vm`|

The host keypair (`ssh_host_ed25519_key` + `.pub`) is generated once
at image build time and lives in the read-only squashfs base. Same key
across all VMs — but each VM gets its own host *certificate* attesting
that motlie-vmm vouches for this host key for this VM’s IP and username.

**Auth flow in detail:**

```
1. ssh -p 2222 alice@localhost
2. russh accepts (auth_none or auth_publickey), extracts username "alice"
3. ensure_vm("alice") → creates VM if needed
4. motlie-vmm signs an ephemeral keypair:
     algorithm: Ed25519
     principal: "alice"
     TTL: 60 seconds
     signed by: motlie-vmm's user CA (in memory)
5. russh connects to VM sshd at 172.16.x.2:22 using ephemeral cert
6. VM sshd checks:
     a. Is the cert signed by /etc/ssh/ca/user_ca.pub?     → yes
     b. Is "alice" listed in /etc/ssh/auth_principals/root? → yes
     c. Is the cert still valid (within 60s window)?        → yes
     → access granted as root
7. russh bridges the external SSH channel to the VM SSH channel
8. Alice lands in a root shell inside the VM

Isolation check — if someone tried to connect directly to this VM
with a cert for "bob":
     a. Is the cert signed by the CA? → yes (same CA)
     b. Is "bob" in /etc/ssh/auth_principals/root? → NO (file contains "alice")
     → rejected
```

No user-side key management needed. motlie-vmm acts as both the CA and
the SSH client — it signs a throwaway cert and immediately uses it
to connect, all inside the same process. The 60-second TTL means the
cert is useless by the time anyone could intercept it.

### 8. Daemon (phase 4)

```rust
pub struct DaemonConfig {
    pub image: PathBuf,
    pub workspace_base: PathBuf,
    pub scratch_base: PathBuf,
    pub listen: SocketAddr,           // default 127.0.0.1:2222
    pub vcpus_per_vm: u32,            // default 2
    pub mem_per_vm_mib: u32,          // default 2048
    pub idle_timeout_secs: u64,       // default 1800
    pub credential_env: HashMap<String, String>,
    pub credential_mounts: Vec<CredentialMount>,
}

#[derive(Clone)]
pub struct CredentialMount {
    pub host_subpath: String,          // relative to ~/.motlie-vmm/creds/{user}/
    pub guest_path: String,            // absolute path inside VM
    pub is_file: bool,                 // single file vs directory
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            // ...
            credential_mounts: vec![
                CredentialMount {
                    host_subpath: ".config/gh".into(),
                    guest_path: "/root/.config/gh".into(),
                    is_file: false,
                },
                CredentialMount {
                    host_subpath: ".claude".into(),
                    guest_path: "/root/.claude".into(),
                    is_file: false,
                },
                CredentialMount {
                    host_subpath: ".codex".into(),
                    guest_path: "/root/.codex".into(),
                    is_file: false,
                },
                CredentialMount {
                    host_subpath: ".npmrc".into(),
                    guest_path: "/root/.npmrc".into(),
                    is_file: true,
                },
            ],
            credential_env: HashMap::new(),
        }
    }
}

impl Daemon {
    pub async fn run(config: DaemonConfig) -> Result<()> {
        preflight_checks(&config)?;
        cleanup_previous_run()?;

        let manifest = ImageManifest::load(&config.image.join("manifest.json"))?;
        let ca = SshCa::new(&dirs::home_dir().unwrap().join(".motlie-vmm/ca"))?;

        std::fs::create_dir_all(&config.workspace_base)?;
        std::fs::create_dir_all(&config.scratch_base)?;
        std::fs::create_dir_all("/var/lib/motlie-vmm/vms")?;

        let daemon = Arc::new(Daemon { config, manifest, ca, ... });

        setup_signal_handlers(daemon.vms.clone());
        tokio::spawn(daemon.clone().idle_reaper_loop());

        let server_key = generate_ed25519_key()?;
        let russh_config = Arc::new(russh::server::Config {
            keys: vec![server_key], ..Default::default()
        });

        println!("motlie-vmm listening on {}", daemon.config.listen);
        println!("  ssh -p {} <username>@localhost", daemon.config.listen.port());

        russh::server::run(russh_config, daemon.config.listen, daemon).await
    }
}

fn preflight_checks(config: &DaemonConfig) -> Result<()> {
    ensure!(config.listen.port() != 22, "refusing to bind to port 22");
    ensure!(Path::new("/dev/kvm").exists(),
        "/dev/kvm not found — run `motlie-vmm check` for details");
    ensure!(Path::new("/dev/vhost-vsock").exists(),
        "vsock not available — try `sudo modprobe vhost_vsock`");
    ensure!(config.image.join("manifest.json").exists(), "image not found");
    // No check for firecracker or passt — they're embedded
    Ok(())
}
```

### 9. SSH Server — russh, in-process (phase 4)

```rust
impl russh::server::Server for Daemon {
    type Handler = SshSession;
    fn new_client(&mut self, _: Option<SocketAddr>) -> SshSession {
        SshSession { daemon: self.clone(), username: None, vm_client: None }
    }
}

struct SshSession {
    daemon: Arc<Daemon>,
    username: Option<String>,
    vm_client: Option<russh::client::Handle<VmSshHandler>>,
}

impl russh::server::Handler for SshSession {
    type Error = anyhow::Error;

    async fn auth_none(&mut self, user: &str) -> Result<Auth, Self::Error> {
        self.username = Some(user.to_string());
        Ok(Auth::Accept)
    }

    async fn auth_publickey(&mut self, user: &str, _: &PublicKey) -> Result<Auth, Self::Error> {
        self.username = Some(user.to_string());
        Ok(Auth::Accept)
    }

    async fn channel_open_session(&mut self, ..) -> Result<bool, Self::Error> {
        let username = self.username.as_ref().unwrap().clone();
        let vm_ip = self.daemon.ensure_vm(&username).await?;
        let (key, cert) = self.daemon.ca.sign_ephemeral_for_vm(&username)?;
        let client = russh::client::connect(
            Arc::new(russh::client::Config::default()),
            (vm_ip, 22), VmSshHandler { cert, key },
        ).await?;
        self.vm_client = Some(client);
        Ok(true)
    }

    // data, pty_request, shell_request, exec_request, window_change_request
    // all forward to self.vm_client and touch idle timer
}
```

### 10. vsock Multiplexed Listener + Guest Agent (phases 5–6)

All VM↔host communication flows through a single vsock port (5000).
Each incoming connection sends a handshake message identifying its type:

```rust
#[derive(Serialize, Deserialize)]
pub enum HandshakeMsg {
    BinaryRequest,            // bootstrap: send me the guest agent binary
    Control,                  // guest agent: bidirectional control channel
    Fs { tag: String },       // guest agent: FS request/response for this mount
}
```

The host listener routes each connection:

```rust
async fn handle_connection(stream: VsockStream, vm: &VmContext) {
    let handshake: HandshakeMsg = recv_msg(&mut stream).await;
    match handshake {
        HandshakeMsg::BinaryRequest => {
            // Send guest agent binary, then close
            send_length_prefixed(&mut stream, GUEST_AGENT.data).await;
        }
        HandshakeMsg::Control => {
            // Bidirectional: Ready, AddMount, RemoveMount, Shutdown
            control_loop(stream, vm).await;
        }
        HandshakeMsg::Fs { tag } => {
            // FS request/response loop for one mount
            fs_loop(stream, vm, &tag).await;
        }
    }
}
```

The credential directories are just additional FS connections — the
guest agent opens one connection per mount at the tool-expected paths.
The tools read/write tokens normally, unaware they’re on a vsock-backed
FUSE mount.

### 11. Control Protocol + Dynamic Mounts (phase 7)

Mounts are not limited to VM creation time. Because the filesystem
is vsock-backed FUSE (not block devices), new mounts can be added
or removed while the VM is running. The guest agent’s control
connection (opened at startup with `{ type: "control" }` handshake)
is a bidirectional channel for lifecycle and dynamic mount commands.

**Control protocol:**

```rust
#[derive(Serialize, Deserialize)]
pub enum ControlMsg {
    // Guest → Host
    Ready,
    MountReady { tag: String },
    MountFailed { tag: String, error: String },

    // Host → Guest
    AddMount { tag: String, guest_path: String, read_only: bool },
    RemoveMount { tag: String },
    Shutdown,
}
```

**Dynamic mount flow:**

```
Host (motlie-vmm)                          Guest (motlie-vmm-guest)
     │                                       │
     │  ControlMsg::AddMount {               │
     │    tag: "data",                       │
     │    guest_path: "/data",               │
     │    read_only: true                    │
     │  }                                    │
     │──── control connection (vsock:5000) ─►│
     │                                       │  mkdir /data
     │                                       │  new vsock:5000 connection
     │                                       │    handshake: Fs { tag: "data" }
     │                                       │  fuser::mount2(...)
     │                                       │
     │             ControlMsg::MountReady {   │
     │               tag: "data"             │
     │◄── control connection (vsock:5000) ───│
     │                                       │
```

**Guest agent main loop:**

The guest agent opens multiple connections to the same vsock port,
each identified by its handshake:

```rust
const HOST_CID: u32 = 2;
const VMM_PORT: u32 = 5000;

fn main() {
    let config = read_json("/etc/motlie-vmm/mounts.json");

    // Open FS connections for initial mounts (one per mount)
    for mount in &config.mounts {
        spawn_fuse_mount(mount); // each opens vsock:5000 with Fs { tag } handshake
    }

    // Open control connection
    let mut control = vsock_connect(HOST_CID, VMM_PORT).unwrap();
    send_msg(&mut control, &HandshakeMsg::Control);

    // Signal ready
    send_msg(&mut control, &ControlMsg::Ready);

    // Listen for commands from host
    loop {
        match recv_msg(&mut control) {
            Ok(ControlMsg::AddMount { tag, guest_path, read_only }) => {
                let tag_clone = tag.clone();
                let result = std::thread::Builder::new()
                    .name(format!("fuse-{}", tag))
                    .spawn(move || {
                        std::fs::create_dir_all(&guest_path).ok();
                        let fs = VsockFuse::new(HOST_CID, VMM_PORT, &tag_clone, read_only);
                        let mut opts = vec![
                            fuser::MountOption::AutoUnmount,
                            fuser::MountOption::AllowRoot,
                        ];
                        if read_only { opts.push(fuser::MountOption::RO); }
                        fuser::mount2(fs, &guest_path, &opts).unwrap();
                    });

                match result {
                    Ok(_) => {
                        std::thread::sleep(Duration::from_millis(100));
                        send_msg(&mut control, &ControlMsg::MountReady { tag });
                    }
                    Err(e) => {
                        send_msg(&mut control, &ControlMsg::MountFailed {
                            tag, error: e.to_string(),
                        });
                    }
                }
            }

            Ok(ControlMsg::RemoveMount { tag }) => {
                let guest_path = find_mount_path(&tag);
                Command::new("fusermount").args(["-u", &guest_path]).status().ok();
            }

            Ok(ControlMsg::Shutdown) => {
                std::process::exit(0);
            }

            Err(_) => break, // Control channel closed — host is gone

            _ => {}
        }
    }
}
```

**Host-side dynamic mount API:**

```rust
impl Daemon {
    pub async fn add_mount(
        &self,
        username: &str,
        host_path: PathBuf,
        guest_path: String,
        read_only: bool,
    ) -> Result<()> {
        let vms = self.vms.lock().await;
        let vm = vms.get(username).ok_or(anyhow!("VM not found"))?;

        let tag = format!("dynamic-{}", sanitize(&guest_path));

        // 1. Register the new path with the vsock FS server
        vm.fs_server.add_mount(VsockMount {
            tag: tag.clone(),
            host_path,
            guest_path: guest_path.clone(),
            read_only,
        }).await?;

        // 2. Tell the guest agent to create the FUSE mount
        vm.send_control(ControlMsg::AddMount {
            tag: tag.clone(),
            guest_path,
            read_only,
        }).await?;

        // 3. Wait for confirmation
        match vm.recv_control_timeout(Duration::from_secs(5)).await? {
            ControlMsg::MountReady { tag: t } if t == tag => Ok(()),
            ControlMsg::MountFailed { error, .. } => Err(anyhow!("mount failed: {}", error)),
            _ => Err(anyhow!("unexpected response")),
        }
    }

    pub async fn remove_mount(&self, username: &str, guest_path: &str) -> Result<()> {
        let vms = self.vms.lock().await;
        let vm = vms.get(username).ok_or(anyhow!("VM not found"))?;
        let tag = format!("dynamic-{}", sanitize(guest_path));
        vm.send_control(ControlMsg::RemoveMount { tag: tag.clone() }).await?;
        vm.fs_server.remove_mount(&tag).await?;
        Ok(())
    }
}
```

**vsock FS server dynamic registration:**

```rust
impl FsServer {
    pub async fn add_mount(&self, mount: VsockMount) -> Result<()> {
        let mut mounts = self.mounts.write().await;
        let root_inode = self.allocate_inode(mount.host_path.clone());
        mounts.insert(mount.tag.clone(), MountInfo {
            host_path: mount.host_path,
            root_inode,
            read_only: mount.read_only,
        });
        Ok(())
    }

    pub async fn remove_mount(&self, tag: &str) -> Result<()> {
        let mut mounts = self.mounts.write().await;
        mounts.remove(tag);
        // Inodes for this mount become stale — guest gets ENOENT
        // on subsequent access, which is correct behavior
        Ok(())
    }
}
```

**CLI:**

```bash
# While alice's VM is running:
motlie-vmm mount alice /data/dataset:/data:ro
motlie-vmm mount alice /home/shared:/shared
motlie-vmm unmount alice /data
```

### 12. Event Bus + Audit (phase 8)

Every filesystem operation flows through the vsock FS server’s
`handle_request()` — motlie-vmm *is* the filesystem. Events are emitted
HOST-SIDE, before returning the response. The VM has no awareness
that events are being generated. No event socket exposed to the VM.

```
VM process → VM kernel → FUSE → guest agent → vsock → YOUR CODE → host filesystem
                                                         ↑
                                                    events emitted here
                                                    policy enforced here
                                                    audit logged here
```

**Event types:**

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FsEvent {
    pub timestamp: String,
    pub vm_name: String,
    pub operation: FsOp,
    pub mount_tag: String,
    pub path: String,
    pub bytes: Option<usize>,
    pub credential: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FsOp { Lookup, Open, Read, Write, Create, Delete }
```

**Audited FS server wrapper:**

```rust
pub struct AuditedFsServer {
    inner: FsServer,
    vm_name: String,
    events: broadcast::Sender<FsEvent>,
    credential_tags: HashSet<String>,
}

impl AuditedFsServer {
    pub fn handle_request(&mut self, req: FsRequest) -> FsResponse {
        let (mount_tag, is_cred) = self.classify_request(&req);

        if is_cred {
            let (op, path, bytes) = self.extract_event_info(&req);
            let _ = self.events.send(FsEvent {
                timestamp: Utc::now().to_rfc3339(),
                vm_name: self.vm_name.clone(),
                operation: op,
                mount_tag,
                path,
                bytes,
                credential: true,
            });
        }

        self.inner.handle_request(req)
    }
}
```

**Event bus: broadcast for live fan-out + motlie-db for persistent audit and graph:**

The persistent store uses `motlie-db` (`github.com/chungers/motlie/libs/db`),
a RocksDB-based engine with graph, fulltext, and vector subsystems.
Events are persisted to the DB. The graph subsystem models VMs and the
resources they have access to — mounts, credentials, network namespaces —
as a queryable graph.

```
Graph model:

  [vm:alice] ──mount──► [dir:~/projects/alice]
             ──mount──► [dir:/tmp/motlie-vmm/alice]
             ──cred───► [cred:alice/.config/gh]
             ──cred───► [cred:alice/.claude]
             ──cred───► [cred:alice/.codex]
             ──netns──► [netns:alice]
             ──event──► [event:1234]  (credential read, .claude, t=...)

  [vm:bob]   ──mount──► [dir:~/projects/bob]
             ──cred───► [cred:bob/.config/gh]
             ...
```

```rust
pub struct EventBus {
    /// In-memory fan-out to live subscribers
    broadcast: broadcast::Sender<FsEvent>,
    /// Persistent graph + event store (motlie-db)
    db: Option<motlie_db::Database>,
}

impl EventBus {
    pub fn emit(&self, event: FsEvent) {
        let _ = self.broadcast.send(event.clone());
        if let Some(ref db) = self.db {
            // Persist event and update graph edges
            // (implementation uses motlie-db graph + kv APIs)
            if let Err(e) = db.append_event(&event) {
                tracing::error!("event persist failed: {}", e);
            }
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<FsEvent> {
        self.broadcast.subscribe()
    }
}
```

**Daemon wiring:**

```rust
impl Daemon {
    pub async fn run(config: DaemonConfig) -> Result<()> {
        let db = motlie_db::Database::open("/var/lib/motlie-vmm/db")?;
        let (tx, _) = broadcast::channel::<FsEvent>(4096);
        let event_bus = Arc::new(EventBus { broadcast: tx, db: Some(db) });

        // Live audit logger
        let mut rx = event_bus.subscribe();
        tokio::spawn(async move {
            while let Ok(ev) = rx.recv().await {
                if ev.credential {
                    tracing::info!(
                        vm = %ev.vm_name,
                        op = ?ev.operation,
                        path = %ev.path,
                        "credential access"
                    );
                }
            }
        });

        // event_bus passed to each VM's AuditedFsServer
        // ...
    }
}
```

**Output:**

```
2026-03-22T15:30:01Z INFO  credential read  vm=alice file=config.toml mount=cred-codex
2026-03-22T15:30:01Z INFO  credential read  vm=alice file=hosts.yml mount=cred-config-gh
2026-03-22T15:35:12Z WARN  credential write vm=alice file=credentials.json mount=cred-claude bytes=1847
```

**CLI access:**

```bash
motlie-vmm events --since 1h          # recent events
motlie-vmm events --vm alice           # filtered by VM
motlie-vmm events --credential-only    # only credential access
motlie-vmm events --follow             # tail -f style (subscribe to broadcast)
```

**Optional policy enforcement (same interception point):**

```rust
// Rate limit credential reads
if is_cred && matches!(req, FsRequest::Read { .. }) {
    if self.cred_read_count.fetch_add(1, Ordering::Relaxed) > 100 {
        return FsResponse::Error(libc::EACCES);
    }
}

// Block writes to certain credential files
if is_cred && matches!(req, FsRequest::Write { .. }) {
    if path.contains("hosts.yml") && self.config.gh_readonly {
        return FsResponse::Error(libc::EROFS);
    }
}
```

### 13. Lifecycle: process isolation and cleanup (phase 9)

**Why cleanup is guaranteed:**

Two kernel mechanisms make orphan processes and leaked networking
impossible, regardless of how the daemon exits:

1. **cgroup** — every child process (firecracker, passt, across all VMs)
   inherits the daemon’s cgroup. systemd’s `KillMode=control-group`
   kills the entire tree. There is no way for a child to escape this.
1. **per-VM network namespaces** — each passt creates its own netns via
   `unshare(CLONE_NEWNET)`. The daemon stays in the host netns (it needs
   to bind port 2222). When both processes in a netns die (passt +
   firecracker), the kernel destroys the namespace automatically. No
   bridges, taps, iptables rules, or routes to clean up.

These are kernel guarantees, not application logic. Even if motlie-vmm
has a bug, even if it’s SIGKILL’d, the kernel cleans up both process
and network state.

**Three layers of defense:**

|Layer            |What it cleans up   |Trigger                    |Mechanism                                 |
|-----------------|--------------------|---------------------------|------------------------------------------|
|1. Signal handler|Processes (graceful)|SIGTERM/SIGINT             |Kill each child, wait, remove temp dirs   |
|2. systemd cgroup|Processes (force)   |Daemon doesn’t exit in 10s |`KillMode=control-group` kills entire tree|
|3. Kernel netns  |Networking          |Last process in netns exits|Namespace + all its state destroyed       |

Plus a startup sweep for any state that survived a hard crash (PID files,
overlay.ext4 files):

**Layer 1: Signal handler** — graceful shutdown

```rust
fn setup_signal_handlers(vms: Arc<Mutex<HashMap<String, Vm>>>) {
    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(SignalKind::terminate()).unwrap();
        let mut sigint = tokio::signal::unix::signal(SignalKind::interrupt()).unwrap();
        tokio::select! { _ = sigterm.recv() => {}, _ = sigint.recv() => {} }
        eprintln!("\nshutting down...");
        let mut vms = vms.lock().await;
        for (name, mut vm) in vms.drain() {
            eprintln!("  stopping: {}", name);
            vm.firecracker.kill().ok(); vm.firecracker.wait().ok();
            vm.passt.kill().ok();       vm.passt.wait().ok();
            // passt + firecracker dead → kernel destroys their netns
        }
        std::fs::remove_dir_all("/var/lib/motlie-vmm/vms").ok();
        std::process::exit(0);
    });
}
```

**Layer 2: systemd cgroup** — force kill if Layer 1 hangs

```ini
[Service]
ExecStart=/path/to/motlie-vmm daemon ...
ExecStopPost=/path/to/motlie-vmm cleanup
Restart=on-failure
RestartSec=3
KillMode=control-group
TimeoutStopSec=10
```

After 10 seconds, systemd sends SIGKILL to every process in the
cgroup. Every firecracker, every passt, the daemon itself — all dead.
Their netns instances are destroyed by the kernel immediately after.

**Layer 3: Startup sweep** — clean up leaked files from hard crashes

```rust
fn cleanup_previous_run() -> Result<()> {
    let vm_dir = Path::new("/var/lib/motlie-vmm/vms");
    if !vm_dir.exists() { return Ok(()); }
    for entry in std::fs::read_dir(vm_dir)? {
        let dir = entry?.path();
        kill_pid_file(&dir.join("firecracker.pid"));
        kill_pid_file(&dir.join("passt.pid"));
        std::fs::remove_dir_all(&dir).ok();
    }
    Ok(())
}
```

Note: Layer 3 only cleans up *files* (PID files, overlay.ext4). Process
and network cleanup is already guaranteed by Layers 1+2 and the kernel.
The sweep handles the case where the daemon was SIGKILL’d before it
could delete `/var/lib/motlie-vmm/vms/alice/`.

**Exit scenario walkthrough:**

```
Clean shutdown (SIGTERM):
  Layer 1 kills children → netns destroyed → files removed → exit 0

Daemon crash (panic, segfault):
  Daemon dies → systemd detects → Layer 2 kills cgroup
  → all children die → all netns destroyed
  → next startup: Layer 3 sweeps leftover files

Daemon SIGKILL:
  Same as crash. cgroup kill is immediate.

Daemon OOM:
  OOM killer targets daemon → same as SIGKILL path.
  If OOM kills a child instead, daemon detects and cleans up.

Host reboot:
  Everything dies. Kernel cleans up all namespaces.
  Temp files in /var/lib/motlie-vmm/vms/ may persist on disk.
  Next startup: Layer 3 sweeps them.
```

## Credential auth scenarios

### API key user

```bash
# On host, set once in ~/.bashrc:
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# motlie-vmm daemon reads these from host environment.
# Injects into /etc/motlie-vmm/env in every VM overlay.
# VM's /etc/profile.d/motlie-vmm-credentials.sh sources them.

ssh -p 2222 alice@localhost
$ claude "hello"          # ✓ uses ANTHROPIC_API_KEY immediately
$ codex "hello"           # ✓ uses OPENAI_API_KEY immediately
```

### Subscription/OAuth user, first session

```bash
ssh -p 2222 alice@localhost

# Claude Code — no API key set, does device code flow:
$ claude
  To sign in, visit: https://console.anthropic.com/device
  Enter code: ABCD-1234
  Waiting for authentication...

# User copies URL to their local browser, enters code.
# Claude Code gets token, writes to ~/.claude/
# ~/.claude/ is vsock-mounted from ~/.motlie-vmm/creds/alice/.claude/
# Token persists on host.

# Codex — same pattern:
$ codex
  Sign in with ChatGPT
  Visit: https://auth.openai.com/device
  Code: WXYZ-5678

# User enters code in browser. Token saved to ~/.codex/
# Persists via vsock mount.

# gh — device code flow:
$ gh auth login
  ! First copy your one-time code: 1A2B-3C4D
  Open https://github.com/login/device

# Token saved to ~/.config/gh/hosts.yml. Persists via vsock mount.
```

### Subscription/OAuth user, returning

```bash
ssh -p 2222 alice@localhost

# Even if the VM was destroyed and recreated:
# ~/.motlie-vmm/creds/alice/.claude/ is mounted at /root/.claude
# ~/.motlie-vmm/creds/alice/.config/gh is mounted at /root/.config/gh
# ~/.motlie-vmm/creds/alice/.codex is mounted at /root/.codex

$ claude "hello"          # ✓ token found, no auth prompt
$ gh repo list            # ✓ already authenticated
$ codex "hello"           # ✓ already authenticated
```

### Token refresh

```
OAuth tokens expire. When they do:
  1. Tool silently uses refresh token → gets new access token → works
  2. If refresh token also expired (30-90 days):
     Tool re-prompts with device code flow
     User enters code in browser again
     New tokens written → persist via vsock mount
Same experience as on a laptop. No motlie-vmm involvement needed.
```

### Mixed org — some users have API keys, some use OAuth

```toml
# ~/.motlie-vmm/config.toml
[credentials.env]
# Leave empty — each user authenticates themselves
ANTHROPIC_API_KEY = ""
OPENAI_API_KEY = ""
```

Each user’s first session, they run their tools and authenticate.
Tokens saved to their per-user cred directory. Every subsequent
session is automatic. Users with API keys can export them inside
the VM — they’ll be written to the cred mount and persist too.

## Per-user mount table

```
Tag              Host path                                    Guest path            Mode
workspace        ~/projects/{user}                            /workspace            rw
scratch          /tmp/motlie-vmm/{user}                          /tmp                  rw
cred-config-gh   ~/.motlie-vmm/creds/{user}/.config/gh           /root/.config/gh      rw
cred-claude      ~/.motlie-vmm/creds/{user}/.claude              /root/.claude         rw
cred-codex       ~/.motlie-vmm/creds/{user}/.codex               /root/.codex          rw
cred-npmrc       ~/.motlie-vmm/creds/{user}/.npmrc               /root/.npmrc          rw
```

## What runs inside the VM

**Custom code: motlie-vmm-guest (one binary, ~700 lines)**

Everything else is stock Ubuntu configured at image build time.
The guest agent binary is NOT on disk anywhere — the host sends it
over vsock at boot, the bootstrap writes it to tmpfs (RAM), and exec’s
into it. Updating the VMM binary automatically updates the guest agent
in every new VM without rebuilding images.

```
Processes:
  PID 1: /sbin/init (systemd)
  sshd                    ← stock OpenSSH, CA config
  motlie-vmm-guest        ← custom, FUSE+vsock+control loop
                             (running from /tmp/motlie-vmm-guest, tmpfs)

In squashfs (built once, same for all VMs, no motlie binaries except bootstrap):
  /usr/local/bin/motlie-vmm-bootstrap            ← ~100KB, fetches guest agent over vsock
  /etc/systemd/system/motlie-vmm-guest.service   ← starts bootstrap → exec's guest agent
  /etc/ssh/sshd_config                           ← CA trust config (see §7)
  /etc/ssh/ssh_host_ed25519_key                  ← host keypair (generated at build)
  /etc/ssh/ssh_host_ed25519_key.pub
  /etc/profile.d/motlie-vmm-credentials.sh       ← sources /etc/motlie-vmm/env
  /sbin/overlay-init                             ← overlayfs setup script

In overlay ext4 (injected by host at VM create time, config only — no binaries):
  /etc/ssh/ca/user_ca.pub                        ← CA pubkey
  /etc/ssh/ssh_host_ed25519_key-cert.pub         ← signed host cert
  /etc/ssh/auth_principals/root                  ← "alice"
  /etc/motlie-vmm/mounts.json                    ← mount config
  /etc/motlie-vmm/env                            ← API keys (if any)

In tmpfs (RAM only, received over vsock at boot, never touches disk):
  /tmp/motlie-vmm-guest                          ← guest agent binary (~3MB)

FUSE mounts (created by motlie-vmm-guest after bootstrap):
  /workspace              ← vsock → ~/projects/alice
  /tmp                    ← vsock → /tmp/motlie-vmm/alice
  /root/.config/gh        ← vsock → ~/.motlie-vmm/creds/alice/.config/gh
  /root/.claude           ← vsock → ~/.motlie-vmm/creds/alice/.claude
  /root/.codex            ← vsock → ~/.motlie-vmm/creds/alice/.codex
  /root/.npmrc            ← vsock → ~/.motlie-vmm/creds/alice/.npmrc

Dynamic FUSE mounts (added at runtime via control protocol):
  /data                   ← vsock → /data/dataset (example)
  /shared                 ← vsock → /home/shared (example)
```

The guest agent has no knowledge of events, auditing, credentials,
policies, VM lifecycle, or other VMs. It’s a dumb pipe between FUSE
and vsock. All intelligence is host-side.

## Every exit scenario

Networking is always pristine because the daemon never touches the host
netns. Each VM’s networking exists in an isolated kernel namespace that
the kernel destroys automatically when its processes die. Processes are
always cleaned up because they’re all in one cgroup that systemd owns.

|Event             |Processes             |Networking               |Host sshd   |Workspaces|Credentials|
|------------------|----------------------|-------------------------|------------|----------|-----------|
|Clean shutdown    |Layer 1 kills         |netns destroyed          |untouched   |persist   |persist    |
|motlie-vmm crash  |Layer 2 (cgroup) kills|netns destroyed          |untouched   |persist   |persist    |
|motlie-vmm SIGKILL|Layer 2 (cgroup) kills|netns destroyed          |untouched   |persist   |persist    |
|motlie-vmm OOM    |Layer 2 (cgroup) kills|netns destroyed          |untouched   |persist   |persist    |
|single VM stop    |daemon kills pair     |that VM’s netns destroyed|untouched   |persist   |persist    |
|host reboot       |everything dies       |kernel cleans all        |starts fresh|persist   |persist    |

## SSH connection flow

```
1.  ssh -p 2222 alice@localhost
2.  russh reads username "alice"
3.  auth_none("alice") → Accept
4.  channel_open_session:
5.    ensure_vm("alice")
6.      new? →
          mkdir ~/projects/alice
          mkdir /tmp/motlie-vmm/alice
          mkdir ~/.motlie-vmm/creds/alice/{.config/gh,.claude,.codex}
          allocate CID, guest IP
          PASST.spawn_with_pre_exec(unshare CLONE_NEWNET) → memfd
          create sparse overlay
          inject into overlay: CA pub, host cert, principals, env credentials
          start vsock listener (port 5000, multiplexed)
          build mount list: workspace + scratch + 4 credential dirs
          inject mount config into overlay
          start vsock FS server (6 mounts)
          FIRECRACKER.spawn_with_pre_exec(setns) → memfd
          wait for guest "ready"
      exists? → return IP
7.    sign_ephemeral_for_vm("alice") → throwaway key + 60s cert
8.    russh::client::connect to VM sshd
9.    bridge client ↔ VM channels
10. Alice lands in VM
      /workspace       = ~/projects/alice                  (rw)
      /tmp             = /tmp/motlie-vmm/alice                (rw)
      /root/.config/gh = ~/.motlie-vmm/creds/alice/.config/gh (rw)
      /root/.claude    = ~/.motlie-vmm/creds/alice/.claude    (rw)
      /root/.codex     = ~/.motlie-vmm/creds/alice/.codex     (rw)
      /root/.npmrc     = ~/.motlie-vmm/creds/alice/.npmrc     (rw)
      env: ANTHROPIC_API_KEY, OPENAI_API_KEY (if set on host)
      internet works (passt)
```

## Crate dependencies

```toml
[dependencies]
tokio              = { version = "1", features = ["full"] }
russh              = "0.46"
russh-keys         = "0.46"
ssh-key            = { version = "0.6", features = ["ed25519", "std", "crypto"] }
serde              = { version = "1", features = ["derive"] }
serde_json         = "1"
rmp-serde          = "1"
clap               = { version = "4", features = ["derive"] }
nix                = { version = "0.29", features = ["sched", "signal", "fs"] }
rand               = "0.8"
anyhow             = "1"
dirs               = "5"
tempfile           = "3"
tracing            = "0.1"
tracing-subscriber = "0.3"
chrono             = { version = "0.4", features = ["serde"] }
motlie-db          = { path = "../libs/db" }  # RocksDB-based, from motlie
```

Guest agent: `fuser`, `vsock`, `rmp-serde`, `serde`, `serde_json`

## Size

|Component                                 |Lines    |
|------------------------------------------|---------|
|CLI + main                                |200      |
|VmmBackend trait + FirecrackerBackend     |100      |
|Platform check (`motlie-vmm check`)       |100      |
|Embedded binary launcher                  |100      |
|Build script (build.rs)                   |150      |
|Image builder                             |400      |
|Daemon core                               |150      |
|SSH server (russh)                        |350      |
|VM manager + creds + memfd + netns        |550      |
|SSH CA                                    |200      |
|vsock FS server                           |800      |
|vsock multiplexed listener + binary server|100      |
|vsock control protocol (host side)        |150      |
|Event bus + audit                         |200      |
|Lifecycle (signals, cleanup)              |100      |
|Guest agent (FUSE + vsock + control loop) |700      |
|Bootstrap (separate crate, in squashfs)   |50       |
|**Total**                                 |**~4350**|

## Security

|Property                    |Mechanism                                                                      |
|----------------------------|-------------------------------------------------------------------------------|
|Single distributable        |motlie-vmm contains everything                                                 |
|No system modifications     |Nothing in /usr, /etc, etc.                                                    |
|User isolation              |Separate KVM VM + netns per user                                               |
|Network isolation           |passt in per-VM netns, no host changes                                         |
|No custom binaries on disk  |Guest agent delivered over vsock to tmpfs; never touches persistent storage    |
|No credentials in overlay   |Only config files (CA, principals, env); overlay.ext4 is small and ephemeral   |
|Guest agent version pinning |Always matches running VMM binary; no stale images                             |
|Ephemeral internal certs    |Throwaway keypair, 60s TTL                                                     |
|CA-based SSH auth           |Stock sshd_config, TrustedUserCAKeys + AuthorizedPrincipals                    |
|Per-user SSH principal      |VM sshd accepts only that username                                             |
|Per-user credentials        |Credential dirs scoped per user                                                |
|Mount isolation             |vsock FS scoped to that user’s directories                                     |
|Dynamic mounts              |Added at runtime via vsock control protocol                                    |
|Credential writeback        |OAuth tokens persist across VM lifecycle                                       |
|Credential audit trail      |Every read/write logged via event bus                                          |
|Policy enforcement          |Rate limiting, read-only overrides at FS interception point                    |
|No external VM access       |russh on 127.0.0.1 only                                                        |
|Host sshd untouched         |Port 22 always works                                                           |
|Zero host network impact    |Per-VM netns; daemon stays in host netns; kernel destroys netns on process exit|
|No orphan processes possible|All children in daemon’s cgroup; systemd `KillMode=control-group` kills tree   |
|memfd execution             |Binaries never written to disk                                                 |
|Clean process lifecycle     |Signal handler (graceful) + cgroup (force) + startup sweep (files)             |

## Advanced: Snapshot and Restore

Firecracker supports full VM snapshots (memory + vCPU + device state)
with ~28ms restore time. The challenge is not the VM itself — it’s
everything motlie-vmm manages around it: vsock connections, FUSE mounts,
passt networking, and network namespaces.

**Core principle:** snapshot preserves guest state (expensive). Host
state (netns, passt, vsock listener, FS server) is cheap to recreate.
The guest agent is made reconnection-aware so it survives the restore
gap.

### What gets snapshotted

```
Saved to disk:
  /var/lib/motlie-vmm/snapshots/alice/
    vmstate.bin          ← Firecracker VM memory + device state
    mem.bin              ← VM memory file (~size of VM RAM, compressed)
    overlay.ext4         ← copy of writable overlay layer
    meta.json            ← username, guest_ip, cid, timestamp

NOT saved (reconstructed on restore):
    network namespace    ← new one created
    passt process        ← new one spawned
    vsock listener       ← new one started
    FS server inode maps ← rebuilt from host directories
    control connection   ← guest agent reconnects

NOT in snapshot (always on host, always available):
    ~/projects/alice/    ← workspace, re-mounted via FUSE
    ~/.motlie-vmm/creds/ ← credentials, re-mounted via FUSE
```

### Snapshot flow

```rust
impl Daemon {
    pub async fn snapshot(&self, username: &str) -> Result<PathBuf> {
        let vms = self.vms.lock().await;
        let vm = vms.get(username).ok_or(anyhow!("VM not found"))?;

        let snap_dir = PathBuf::from(format!(
            "/var/lib/motlie-vmm/snapshots/{}", username));
        std::fs::create_dir_all(&snap_dir)?;

        // 1. Pause the VM
        firecracker_api(&vm.api_sock, "PATCH", "/vm",
            json!({"state": "Paused"}))?;

        // 2. Create snapshot
        firecracker_api(&vm.api_sock, "PUT", "/snapshot/create",
            json!({
                "snapshot_type": "Full",
                "snapshot_path": snap_dir.join("vmstate.bin").to_str(),
                "mem_file_path": snap_dir.join("mem.bin").to_str(),
            }))?;

        // 3. Copy overlay (VM's writable filesystem state)
        std::fs::copy(
            format!("/var/lib/motlie-vmm/vms/{}/overlay.ext4", username),
            snap_dir.join("overlay.ext4"),
        )?;

        // 4. Save metadata
        let meta = SnapshotMeta {
            username: username.to_string(),
            guest_ip: vm.guest_ip,
            cid: vm.cid,
            timestamp: Utc::now(),
        };
        std::fs::write(
            snap_dir.join("meta.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;

        // 5. Stop the VM (releases netns, passt, vsock listener)
        drop(vms);
        self.stop_vm(username).await?;

        Ok(snap_dir)
    }
}
```

### Restore flow

```rust
impl Daemon {
    pub async fn restore(&self, username: &str) -> Result<Ipv4Addr> {
        let snap_dir = PathBuf::from(format!(
            "/var/lib/motlie-vmm/snapshots/{}", username));
        let meta: SnapshotMeta = serde_json::from_str(
            &std::fs::read_to_string(snap_dir.join("meta.json"))?)?;

        let vm_dir = PathBuf::from(format!(
            "/var/lib/motlie-vmm/vms/{}", username));
        std::fs::create_dir_all(&vm_dir)?;

        // 1. Restore overlay
        std::fs::copy(
            snap_dir.join("overlay.ext4"),
            vm_dir.join("overlay.ext4"),
        )?;

        // 2. Fresh netns + passt (same IP as before)
        let (passt_child, passt_fd) = spawn_passt(meta.guest_ip)?;
        let netns_path = format!("/proc/{}/ns/net", passt_child.id());

        // 3. Start vsock listener (guest agent will reconnect)
        let vsock_uds = vm_dir.join("vsock.sock");
        start_vsock_listener(&vsock_uds, username, &self.mounts_for(username));

        // 4. Firecracker restore (joins new netns)
        let (fc_child, fc_fd) = spawn_firecracker_restore(
            &snap_dir, &vm_dir, &vsock_uds, &netns_path, meta.cid,
        )?;

        // 5. Resume — VM wakes up, guest agent reconnects
        firecracker_api(&vm_dir.join("firecracker.sock"),
            "PATCH", "/vm", json!({"state": "Resumed"}))?;

        // 6. Wait for guest agent to reconnect and signal ready
        wait_for_guest_ready(&vsock_uds, Duration::from_secs(30)).await?;

        // ... register VM in self.vms
        Ok(meta.guest_ip)
    }
}
```

### Guest agent reconnection

When the VM is restored, the guest agent wakes up mid-execution with
dead vsock connections. Every vsock connection is wrapped in a
reconnect loop:

```rust
struct ResilientVsockConnection {
    tag: String,
    stream: Option<VsockStream>,
}

impl ResilientVsockConnection {
    fn send(&mut self, msg: &[u8]) -> io::Result<()> {
        loop {
            match &mut self.stream {
                Some(s) => match s.write_all(msg) {
                    Ok(()) => return Ok(()),
                    Err(_) => { self.stream = None; }
                },
                None => { self.reconnect()?; }
            }
        }
    }

    fn reconnect(&mut self) -> io::Result<()> {
        loop {
            match vsock_connect(HOST_CID, VMM_PORT) {
                Ok(mut s) => {
                    send_msg(&mut s, &HandshakeMsg::Fs {
                        tag: self.tag.clone()
                    })?;
                    self.stream = Some(s);
                    return Ok(());
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }
}
```

FUSE operations that are in-flight during snapshot are lost — the
application’s syscall is paused in the kernel. On restore, the
kernel retries (FUSE has a request timeout), or the application gets
a transient error and retries. For dev workloads (editors, git, builds),
this is the same experience as resuming a laptop from sleep with a
network mount.

### Restore sequence diagram

```
Snapshot:                            Restore:
  host pauses VM                      host creates fresh netns + passt
  firecracker snapshot to disk        host starts vsock listener (:5000)
  copy overlay.ext4                   firecracker restore (joins netns)
  stop VM                             resume VM
  netns destroyed                       │
                                        ├─ guest agent wakes up
                                        ├─ vsock connections are dead
                                        ├─ reconnect loop kicks in
                                        ├─ connects vsock:5000 (Control)
                                        ├─ connects vsock:5000 (Fs per mount)
                                        ├─ FUSE mounts resume
                                        ├─ sends Ready
                                        └─ VM fully operational (~200ms gap)
```

### What survives restore

|State                          |Survives?          |How?                             |
|-------------------------------|-------------------|---------------------------------|
|Running processes inside VM    |Yes                |VM memory snapshot               |
|Open files, shell history      |Yes                |VM memory snapshot               |
|Installed packages             |Yes                |overlay.ext4 in snapshot         |
|/workspace files               |Yes                |Host directory, FUSE reconnects  |
|Credentials (gh, claude, codex)|Yes                |Host directory, FUSE reconnects  |
|Network connections (outbound) |No                 |TCP connections break, apps retry|
|FUSE mounts                    |~200ms interruption|Agent reconnects automatically   |
|tmux/screen sessions           |Yes                |In VM memory, terminals resume   |

### Snapshot size

Dominated by VM memory. A 2GB VM produces ~2GB snapshot (Firecracker
compresses memory). The overlay.ext4 adds a few MB. Credentials are
NOT in the snapshot — they live on the host and are re-mounted via
FUSE on reconnect.

### CLI

```bash
motlie-vmm snapshot alice            # pause, snapshot, stop
motlie-vmm restore alice             # restore from latest snapshot
motlie-vmm snapshots                 # list all snapshots
motlie-vmm snapshot alice --keep     # snapshot without stopping (clone)
```

### Filesystem layout for snapshots

```
/var/lib/motlie-vmm/snapshots/       # persistent
  alice/
    vmstate.bin
    mem.bin
    overlay.ext4
    meta.json
  alice-20260322T1830/                # named/timestamped snapshots
    ...
```

## Project Planning: Incremental Build Phases

Each phase produces a testable milestone. No phase requires the next
one to be useful. Phases 1–4 produce a working SSH-into-VM system
with zero custom guest binaries.

### Phase 1: Image builder + Firecracker boot

**Build:** `motlie-vmm check`, `motlie-vmm build`, embedded binary
launcher (firecracker only), overlay-init.

**Test:** Build an image, spawn firecracker manually, verify overlay-init
works (squashfs + ext4 stacking), VM boots to systemd prompt on serial
console.

**Dependencies:** debootstrap, mksquashfs, a firecracker binary, a kernel.
No custom guest binaries. No passt. No networking. No SSH.

**Components:** §1 Platform Check, §2 VMM Backend, §3 Embedded Binary Launcher, §4 Image Builder.

```bash
motlie-vmm build --include "openssh-server" --output ./images/devbox
motlie-vmm check
# manually spawn firecracker with serial console to verify boot
```

### Phase 2: Networking

**Build:** Embed passt, `unshare(CLONE_NEWNET)` + `setns()` logic,
IP allocation.

**Test:** Boot a VM with passt, verify internet access from serial
console (ping, curl).

**Dependencies:** Phase 1 + passt binary. Still no SSH, no vsock, no
custom guest binaries.

**Components:** §5 VM Manager (networking portions).

### Phase 3: SSH CA + overlay injection

**Build:** SSH CA (key generation, cert signing), `inject_into_overlay`
(loop mount ext4, write files, unmount), manual SSH connection from host.

**Test:** Inject CA pubkey + host cert + principals into overlay, boot VM,
SSH in from the host using a manually signed cert.

**Dependencies:** Phase 2 + ssh-key crate. Still no russh, no vsock, no
guest binaries.

**Components:** §6 SSH CA, §7 SSH Configuration Inside the VM.

```bash
# manually: ssh -i /tmp/ephemeral root@172.16.0.2
```

### Phase 4: russh proxy

**Build:** russh SSH server (in-process, port 2222), `ensure_vm`
orchestration (ties phases 1–3 together), ephemeral cert signing on connect.

**Test:** `ssh -p 2222 alice@localhost` lands in a VM. Full SSH flow
works. No FUSE mounts — `/workspace` is empty. But you have isolated,
internet-connected VMs per user.

**Dependencies:** Phase 3 + russh crate.

**Components:** §8 Daemon, §9 SSH Server.

**This is already a useful product** — isolated VMs with SSH access,
internet, per-user workspaces (via overlay). You can install tools,
run code, do everything except share host directories.

### Phase 5: Bootstrap binary + vsock listener

**Build:** `motlie-vmm-bootstrap` crate (~50 lines, separate cargo target),
bake it into squashfs during `motlie-vmm build`, vsock multiplexed listener
(just the `BinaryRequest` handler initially).

**Test:** Boot VM, bootstrap connects to vsock, host sends a test binary,
bootstrap writes to tmpfs and exec’s it. The test binary can be a
hello-world that prints and exits — doesn’t need to be the real guest agent.

**Dependencies:** Phase 4 + vsock crate. First time a custom binary is
baked into the image.

**Components:** §10 vsock Multiplexed Listener (BinaryRequest handler only).

### Phase 6: Guest agent + FUSE mounts

**Build:** `motlie-vmm-guest` crate (FUSE + vsock, ~700 lines), vsock FS
server (host side, ~800 lines), `Fs { tag }` handler in multiplexed
listener.

**Test:** Boot VM, bootstrap fetches real guest agent, agent mounts
`/workspace` via FUSE, files appear from host directory.

**Dependencies:** Phase 5 + fuser crate.

**Components:** §10 vsock Multiplexed Listener (Fs handler), Guest Agent.

### Phase 7: Control protocol + dynamic mounts

**Build:** `Control` handler in multiplexed listener, `AddMount` /
`RemoveMount` in guest agent, `motlie-vmm mount` / `unmount` CLI.

**Test:** `motlie-vmm mount alice /data:/data:ro` while VM is running.

**Dependencies:** Phase 6.

**Components:** §11 Control Protocol + Dynamic Mounts.

### Phase 8: Credentials + events + graph

**Build:** Credential mount setup (per-user cred dirs), env var injection,
event bus (broadcast + motlie-db), `motlie-vmm events` CLI.

**Test:** OAuth device flow inside VM, token persists across VM restart.
Event log records credential access.

**Dependencies:** Phase 7 + motlie-db.

**Components:** §12 Event Bus + Audit, credential portions of §5 VM Manager.

### Phase 9: Lifecycle hardening

**Build:** Signal handlers, idle reaper, startup sweep, systemd unit file.

**Test:** SIGKILL the daemon, verify no orphan processes or leaked netns.
Verify next startup cleans up leftover files.

**Dependencies:** Phase 8.

**Components:** §13 Lifecycle.

### Phase 10: Snapshot and restore (advanced)

**Build:** Firecracker snapshot/restore API integration, guest agent
reconnection resilience (`ResilientVsockConnection`), snapshot CLI,
snapshot filesystem layout.

**Test:** `motlie-vmm snapshot alice`, verify snapshot files on disk.
`motlie-vmm restore alice`, verify VM resumes with running processes
intact, FUSE mounts reconnect within ~200ms, workspace and credentials
available immediately.

**Dependencies:** Phase 9 + guest agent reconnection logic. Requires
modifying the guest agent to wrap all vsock connections in reconnect
loops. Firecracker snapshot API is the easy part; the guest agent
resilience is the work.

**Components:** Advanced: Snapshot and Restore section.

### Dependency graph

```
Phase 1: check + trait + build + firecracker boot
    │
Phase 2: + passt networking
    │
Phase 3: + SSH CA + overlay injection
    │
Phase 4: + russh proxy ←── usable product here
    │
Phase 5: + bootstrap + vsock listener
    │
Phase 6: + guest agent + FUSE mounts ←── host dir sharing
    │
Phase 7: + dynamic mounts
    │
Phase 8: + credentials + events ←── full feature set
    │
Phase 9: + lifecycle hardening ←── production ready (Linux)
    │
Phase 10: + snapshot/restore ←── advanced
    │
Phase 11: + macOS VzBackend ←── cross-platform
```

### Binary build targets

|Binary                |Crate         |When needed|Where it runs            |
|----------------------|--------------|-----------|-------------------------|
|`motlie-vmm`          |main          |Phase 1    |Host (memfd for children)|
|`firecracker`         |downloaded    |Phase 1    |Host (memfd)             |
|`passt`               |downloaded    |Phase 2    |Host (memfd)             |
|`motlie-vmm-bootstrap`|separate crate|Phase 5    |Guest (in squashfs)      |
|`motlie-vmm-guest`    |separate crate|Phase 6    |Guest (tmpfs via vsock)  |

## Advanced: macOS Port with Virtualization.framework

A future `VzBackend` implementation of the `VmmBackend` trait (§2)
targeting Apple Silicon Macs. The VM boundary changes completely
(Firecracker/KVM → Virtualization.framework), but everything above
the trait — SSH proxy, FUSE mounts, credential management, event
bus — is backend-agnostic and ports unchanged.

### What carries over unchanged

Everything above the `VmmBackend` trait:

- russh SSH server (pure Rust, cross-platform)
- SSH CA (ssh-key crate, cross-platform)
- vsock FS protocol, FUSE guest agent, bootstrap binary (run inside Linux VM)
- Control protocol, HandshakeMsg, multiplexed listener design
- Event bus, motlie-db integration
- Credential management (host-side directory structure)

The guest image (kernel + rootfs) is format-compatible — VZ’s
`VZLinuxBootLoader` takes a kernel + initrd + cmdline, same as
Firecracker. The guest agent and bootstrap run inside the Linux VM
and don’t know which hypervisor is underneath.

### What changes

|Linux (FirecrackerBackend)            |macOS (VzBackend)                           |Impact              |
|--------------------------------------|--------------------------------------------|--------------------|
|Firecracker + KVM                     |Virtualization.framework (VZ)               |VM lifecycle rewrite|
|`/dev/kvm`                            |`VZVirtualMachine` API (Obj-C FFI)          |New Rust bindings   |
|`memfd_create` for binary execution   |Not needed (VZ is a framework)              |Removed             |
|`unshare(CLONE_NEWNET)` + per-VM netns|VZ NAT (`VZNATNetworkDeviceAttachment`)     |Simpler             |
|passt (userspace network stack)       |Not needed (VZ provides NAT)                |Removed             |
|overlayfs (squashfs + ext4)           |APFS clone of base disk image               |Rewritten           |
|debootstrap                           |Download pre-built image or build externally|Simplified          |
|`vhost_vsock` kernel module           |`VZVirtioSocketDeviceConfiguration`         |Different API       |
|cgroup (process cleanup)              |Process groups + launchd                    |Weaker but adequate |
|`/proc/modules`, `/proc/meminfo`      |`sysctl`, IOKit                             |Small rewrite       |

### VzBackend implementation

```rust
pub struct VzBackend {
    // No embedded binaries — VZ is a linked framework
}

pub struct VzVm {
    vm: VzVirtualMachine,    // Rust wrapper around VZVirtualMachine
    guest_ip: Ipv4Addr,
    cid: u32,
    vsock_device: VzVsockDevice,
}

impl VmHandle for VzVm {
    fn vsock_listener(&self) -> Result<VsockListener> {
        // VZ vsock API — get listener from VZVirtioSocketDevice
        self.vsock_device.listen(5000)
    }
    fn guest_ip(&self) -> Ipv4Addr { self.guest_ip }
    fn cid(&self) -> u32 { self.cid }
    fn is_running(&self) -> bool { self.vm.state() == VzVmState::Running }
}

impl VmmBackend for VzBackend {
    type Vm = VzVm;

    fn check_platform(&self) -> Result<Report> {
        // Check: macOS ≥ 13, Apple Silicon, VZ entitlement signed
        // No KVM, no vhost_vsock
    }

    fn build_image(&self, config: &BuildConfig) -> Result<()> {
        // APFS clone of base image instead of overlayfs
        // No overlay-init needed — VM boots directly from writable clone
        // Bootstrap still baked into image (same as Linux)
    }

    fn create_vm(&self, config: &VmConfig) -> Result<VzVm> {
        // VZLinuxBootLoader with kernel + cmdline
        // VZDiskImageStorageDeviceAttachment for root + overlay
        // VZNATNetworkDeviceAttachment (replaces passt + netns)
        // VZVirtioSocketDeviceConfiguration (replaces vhost_vsock)
        // vm.start()
    }

    fn stop_vm(&self, vm: &mut VzVm) -> Result<()> {
        // vm.stop() — VZ cleans up networking automatically
    }

    fn snapshot(&self, vm: &VzVm, path: &Path) -> Result<()> {
        // vm.pause() + VZ saveMachineStateTo (mature API)
    }

    fn restore(&self, path: &Path, config: &VmConfig) -> Result<VzVm> {
        // VZ restoreMachineStateFrom + resume
    }
}
```

### VZ Rust bindings

Apple’s Virtualization.framework is Objective-C/Swift. The existing
Rust crates (`virtualization-rs`, `virt-fwk`) are incomplete and
unmaintained. A thin FFI layer (~600 lines) using the `objc2` crate
covers the needed API surface:

```rust
// Minimal VZ binding — what we need
pub struct VzVirtualMachine { /* objc2 wrapper */ }
pub struct VzVsockDevice { /* objc2 wrapper */ }

impl VzVirtualMachine {
    fn new(config: &VzMachineConfig) -> Result<Self>;
    fn start(&self) -> Result<()>;
    fn pause(&self) -> Result<()>;
    fn resume(&self) -> Result<()>;
    fn stop(&self) -> Result<()>;
    fn save_state(&self, path: &Path) -> Result<()>;
    fn restore_state(&self, path: &Path) -> Result<()>;
    fn vsock_device(&self) -> &VzVsockDevice;
}

impl VzVsockDevice {
    fn listen(&self, port: u32) -> Result<VsockListener>;
    fn connect(&self, port: u32) -> Result<VsockStream>;
}
```

### Disk image management (replaces overlayfs)

```
Linux:                                macOS:
  rootfs.squashfs (read-only base)      rootfs.img (raw/qcow2 base, read-only)
  + overlay.ext4 (per-VM writable)      → APFS clone → alice.img (CoW, instant)
  + overlay-init stacks them            VM boots directly from alice.img
                                        inject config by mounting + writing + unmounting
```

APFS clone-on-write is instant and space-efficient. The per-VM image
starts at ~0 bytes overhead and grows only as the guest writes.
No overlay-init script needed — the VM boots directly from the
writable image.

### macOS-specific constraints

1. **Entitlement required.** Binary must be signed with
   `com.apple.security.virtualization`. Distribute signed or user
   signs locally.
1. **macOS 13 (Ventura) minimum** for VZ Linux boot. macOS 14+ for
   better vsock support.
1. **Apple Silicon only** for Linux guest VMs. Intel Macs can’t run
   aarch64 Linux guests via VZ.
1. **No cgroup.** Process cleanup via `killpg()` + `launchd` policies.
   Less robust than Linux cgroup but adequate for single-user macOS.
1. **Not zero host impact.** VZ creates `vmnet` interfaces (managed
   by macOS, cleaned up when VM stops). Unlike Linux where passt +
   netns leaves the host untouched.

### Porting effort estimate

|Category                               |Lines|
|---------------------------------------|-----|
|Code unchanged (above VmmBackend trait)|~1750|
|Code rewritten (below VmmBackend trait)|~1300|
|New code (VZ bindings, disk management)|~800 |
|Code removed (memfd, passt, netns)     |-400 |

~50% of the codebase carries over unchanged. Estimated timeline:
~7-8 weeks after the Linux version is feature-complete (phases 1–9).

### Phase 11 in project planning

The macOS port is Phase 11, after snapshot/restore:

```
Phase 9:  + lifecycle hardening ←── production ready (Linux)
    │
Phase 10: + snapshot/restore ←── advanced
    │
Phase 11: + macOS VzBackend ←── cross-platform
```