# sparkvm — Architecture

A single Rust binary. Builds VM images from the running host.
Runs as a daemon that manages VMs, serves filesystems, signs
certificates, accepts SSH connections, and manages per-user
credentials. Firecracker and passt are embedded inside sparkvm
and extracted to memfd at runtime.

Zero host impact before, during, and after running.
Single binary to distribute. No dependencies to install.

## Usage

```bash
# Build an image (one-time)
sparkvm build \
    --include "openssh-server tmux git curl gh nodejs npm" \
    --output ./images/devbox

# Run the daemon
sparkvm daemon \
    --image ./images/devbox \
    --workspace ~/projects \
    --port 2222

# Users connect — each gets their own VM
ssh -p 2222 alice@localhost

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
1. **One process owns everything.** When sparkvm exits, all VMs,
   networking, and temporary state vanish.
1. **Zero host impact.** No iptables, no bridges, no ip_forward,
   no taps in the host namespace. No files in system directories.
1. **Never touch the host sshd.** Port 22 is sacred.
1. **Identity = SSH username = VM name.** `ssh alice@...` → VM “alice”,
   workspace `~/projects/alice`, credentials `~/.sparkvm/creds/alice`.
1. **Workspaces persist, VMs don’t.** Files in `/workspace` and
   tool credentials survive VM stop/restart. Installed packages
   are ephemeral.
1. **Authenticate once.** OAuth tokens and API keys are stored
   per-user on the host and re-mounted into every new VM session.
   Tools that support device code flows work through the SSH terminal.
1. **No credentials in images.** CA pubkey, API keys, and OAuth
   tokens are injected per-VM at runtime via overlay and vsock mounts.
1. **No Docker.** Image built from the host using debootstrap or rsync.

## What’s inside the binary

```
sparkvm (single ELF, ~16MB)
  ├── sparkvm code                    ~3500 lines Rust
  ├── embedded: firecracker-aarch64   ~3MB static musl
  ├── embedded: passt-aarch64         ~300KB static
  └── embedded: sparkvm-guest         ~3MB static musl (FUSE agent)
```

## Process tree

```
Host sshd (port 22)                ← untouched

sparkvm.service                    ← systemd, KillMode=control-group
  └── sparkvm daemon
        │
        ├── russh SSH server        ← in-process, port 2222
        │
        │   ┌── network namespace (alice) ──────────────┐
        ├── │ passt (from memfd)    userspace net stack  │
        ├── │ firecracker (memfd)   KVM VM               │
        │   └───────────────────────────────────────────┘
        │
        │   ┌── network namespace (bob) ────────────────┐
        ├── │ passt (from memfd)                         │
        ├── │ firecracker (from memfd)                   │
        │   └───────────────────────────────────────────┘
        │
        ├── vsock FS (alice)        ← tokio task
        │     workspace, scratch, credentials
        ├── vsock FS (bob)          ← tokio task
        ├── idle reaper             ← tokio task
        └── CA keys                 ← in memory
```

## Host impact: none

```
Host state while running:               After exit (any reason):
  ├── network interfaces: unchanged      identical
  ├── routing table: unchanged           identical
  ├── iptables: unchanged                identical
  ├── ip_forward: unchanged              identical
  ├── /usr, /etc, /lib: unchanged        identical
  └── processes: sparkvm + children      nothing
```

## Filesystem layout

```
~/.sparkvm/                              # persists
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

/tmp/sparkvm/                            # per-user scratch (persist across VM lifecycle)
  alice/
  bob/

/var/lib/sparkvm/vms/                    # ephemeral (destroyed on VM stop)
  alice/
    firecracker.pid, passt.pid
    firecracker.sock, vsock.sock
    overlay.ext4, vmconfig.json
```

Nothing in `/usr/bin`, `/usr/local/bin`, `/etc`, or any system directory.

## Architecture diagram

```
                    ┌─────────────────────────────────────────────┐
                    │          sparkvm daemon (one process)        │
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
                    │     │  │  → /tmp/sparkvm/     │               │
                    │     │  │      alice           │               │
                    │     │  │ /root/.config/gh  rw │               │
                    │     │  │  → ~/.sparkvm/creds/ │               │
                    │     │  │      alice/.config/gh│               │
                    │     │  │ /root/.claude     rw │               │
                    │     │  │  → ~/.sparkvm/creds/ │               │
                    │     │  │      alice/.claude   │               │
                    │     │  │ /root/.codex      rw │               │
                    │     │  │  → ~/.sparkvm/creds/ │               │
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

### 1. Embedded Binary Launcher

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
    "sparkvm-guest", include_bytes!(concat!(env!("OUT_DIR"), "/sparkvm-guest")));
```

### 2. Image Builder

Builds from the running host without Docker. Guest agent extracted
from embedded blob. debootstrap or rsync strategy.

```rust
pub fn build(config: BuildConfig) -> Result<()> {
    let rootfs = tempdir().join("rootfs");

    match config.strategy {
        BuildStrategy::Debootstrap => debootstrap(&rootfs, &config)?,
        BuildStrategy::HostCopy => rsync_host(&rootfs, &config)?,
    }

    // Guest agent from embedded binary
    std::fs::write(rootfs.join("usr/local/bin/sparkvm-guest"), GUEST_AGENT.data)?;
    set_permissions(rootfs.join("usr/local/bin/sparkvm-guest"), 0o755)?;

    install_guest_agent_service(&rootfs)?;
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

fn install_credential_profile(rootfs: &Path) -> Result<()> {
    // Shell profile that sources API keys on login
    std::fs::write(rootfs.join("etc/profile.d/sparkvm-credentials.sh"), r#"
# sparkvm: load credentials injected by the daemon
if [ -f /etc/sparkvm/env ]; then
    set -a
    . /etc/sparkvm/env
    set +a
fi
"#)?;
    set_permissions(rootfs.join("etc/profile.d/sparkvm-credentials.sh"), 0o644)?;

    // Ensure credential target directories exist in the rootfs
    // (FUSE will mount over these)
    std::fs::create_dir_all(rootfs.join("root/.config/gh"))?;
    std::fs::create_dir_all(rootfs.join("root/.claude"))?;
    std::fs::create_dir_all(rootfs.join("root/.codex"))?;

    Ok(())
}
```

### 3. Daemon

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
    pub host_subpath: String,          // relative to ~/.sparkvm/creds/{user}/
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
        let ca = SshCa::new(&dirs::home_dir().unwrap().join(".sparkvm/ca"))?;

        std::fs::create_dir_all(&config.workspace_base)?;
        std::fs::create_dir_all(&config.scratch_base)?;
        std::fs::create_dir_all("/var/lib/sparkvm/vms")?;

        let daemon = Arc::new(Daemon { config, manifest, ca, ... });

        setup_signal_handlers(daemon.vms.clone());
        tokio::spawn(daemon.clone().idle_reaper_loop());

        let server_key = generate_ed25519_key()?;
        let russh_config = Arc::new(russh::server::Config {
            keys: vec![server_key], ..Default::default()
        });

        println!("sparkvm listening on {}", daemon.config.listen);
        println!("  ssh -p {} <username>@localhost", daemon.config.listen.port());

        russh::server::run(russh_config, daemon.config.listen, daemon).await
    }
}

fn preflight_checks(config: &DaemonConfig) -> Result<()> {
    ensure!(config.listen.port() != 22, "refusing to bind to port 22");
    ensure!(Path::new("/dev/kvm").exists(), "/dev/kvm not found");
    ensure!(config.image.join("manifest.json").exists(), "image not found");
    // No check for firecracker or passt — they're embedded
    Ok(())
}
```

### 4. SSH Server (russh, in-process)

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

### 5. VM Manager + Networking + Credentials

Each VM gets: passt (memfd) in its own netns, firecracker (memfd)
joining the same netns, vsock mounts for workspace + scratch + credentials.

```rust
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
        let vm_dir = PathBuf::from(format!("/var/lib/sparkvm/vms/{}", username));
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
        inject_into_overlay(&overlay, "/etc/sparkvm/mounts.json",
            serde_json::to_string_pretty(&serde_json::json!({"mounts": mount_cfg}))?.as_bytes())?;

        // ── 6. Start vsock FS server ──

        let vsock_uds = vm_dir.join("vsock.sock");
        let uds = vsock_uds.clone();
        tokio::spawn(async move { vsock_file_server(&uds, mounts).await; });

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
            .join(".sparkvm/creds")
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
    /// These are sourced by /etc/profile.d/sparkvm-credentials.sh on login.
    fn inject_env_credentials(&self, overlay: &Path) -> Result<()> {
        let mut env_lines = Vec::new();

        // From sparkvm config
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
            inject_into_overlay(overlay, "/etc/sparkvm/env",
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
            // netns destroyed by kernel
            std::fs::remove_dir_all(format!("/var/lib/sparkvm/vms/{}", username)).ok();
            // Credentials in ~/.sparkvm/creds/{user}/ persist
            // Workspace in ~/projects/{user}/ persists
            println!("  vm stopped: {}", username);
        }
        Ok(())
    }
}
```

### 6. SSH CA

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
            .serial(rand::random())?.key_id(&format!("sparkvm-{}-{}", username, now))?
            .valid_after(now - 300)?.valid_before(now + 60)?
            .principals(&[username.to_string()])?.sign(&self.user_ca)?;
        Ok((key, cert))
    }

    pub fn sign_host_cert(&self, pubkey: &PublicKey, names: &[String]) -> Result<Certificate> {
        let now = unix_timestamp();
        let cert = Builder::new_v01(pubkey.clone(), CertType::Host)?
            .serial(rand::random())?.key_id(&format!("sparkvm-host-{}", names[0]))?
            .valid_after(now - 300)?.valid_before(now + 365 * 86400)?
            .principals(names)?.sign(&self.host_ca)?;
        Ok(cert)
    }
}
```

### 7. vsock FS Server + Guest Agent

Same as previous. Per-VM tokio task serves filesystem ops over vsock.
Guest agent runs FUSE mounts inside the VM. Msgpack wire protocol.

The credential directories are just additional vsock mounts — the
guest agent mounts them at the tool-expected paths. The tools
read/write tokens normally, unaware they’re on a vsock-backed FUSE mount.

### 8. Lifecycle: three layers of cleanup

**Layer 1: Signal handler**

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
        }
        std::fs::remove_dir_all("/var/lib/sparkvm/vms").ok();
        std::process::exit(0);
    });
}
```

**Layer 2: systemd cgroup**

```ini
[Service]
ExecStart=/path/to/sparkvm daemon ...
ExecStopPost=/path/to/sparkvm cleanup
Restart=on-failure
RestartSec=3
KillMode=control-group
TimeoutStopSec=10
```

**Layer 3: Startup sweep**

```rust
fn cleanup_previous_run() -> Result<()> {
    let vm_dir = Path::new("/var/lib/sparkvm/vms");
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

## Credential auth scenarios

### API key user

```bash
# On host, set once in ~/.bashrc:
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# sparkvm daemon reads these from host environment.
# Injects into /etc/sparkvm/env in every VM overlay.
# VM's /etc/profile.d/sparkvm-credentials.sh sources them.

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
# ~/.claude/ is vsock-mounted from ~/.sparkvm/creds/alice/.claude/
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
# ~/.sparkvm/creds/alice/.claude/ is mounted at /root/.claude
# ~/.sparkvm/creds/alice/.config/gh is mounted at /root/.config/gh
# ~/.sparkvm/creds/alice/.codex is mounted at /root/.codex

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
Same experience as on a laptop. No sparkvm involvement needed.
```

### Mixed org — some users have API keys, some use OAuth

```toml
# ~/.sparkvm/config.toml
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
scratch          /tmp/sparkvm/{user}                          /tmp                  rw
cred-config-gh   ~/.sparkvm/creds/{user}/.config/gh           /root/.config/gh      rw
cred-claude      ~/.sparkvm/creds/{user}/.claude              /root/.claude         rw
cred-codex       ~/.sparkvm/creds/{user}/.codex               /root/.codex          rw
cred-npmrc       ~/.sparkvm/creds/{user}/.npmrc               /root/.npmrc          rw
```

## Every exit scenario

|Event          |Networking|memfds |Host sshd|Workspaces|Credentials|
|---------------|----------|-------|---------|----------|-----------|
|Clean shutdown |pristine  |freed  |untouched|persist   |persist    |
|sparkvm crash  |pristine  |freed  |untouched|persist   |persist    |
|sparkvm SIGKILL|pristine  |freed  |untouched|persist   |persist    |
|sparkvm OOM    |pristine  |freed  |untouched|persist   |persist    |
|single VM stop |pristine  |freed  |untouched|persist   |persist    |
|host reboot    |pristine  |nothing|starts   |persist   |persist    |

## SSH connection flow

```
1.  ssh -p 2222 alice@localhost
2.  russh reads username "alice"
3.  auth_none("alice") → Accept
4.  channel_open_session:
5.    ensure_vm("alice")
6.      new? →
          mkdir ~/projects/alice
          mkdir /tmp/sparkvm/alice
          mkdir ~/.sparkvm/creds/alice/{.config/gh,.claude,.codex}
          allocate CID, guest IP
          PASST.spawn_with_pre_exec(unshare CLONE_NEWNET) → memfd
          create sparse overlay
          inject: CA pub, host cert, principals, env credentials
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
      /tmp             = /tmp/sparkvm/alice                (rw)
      /root/.config/gh = ~/.sparkvm/creds/alice/.config/gh (rw)
      /root/.claude    = ~/.sparkvm/creds/alice/.claude    (rw)
      /root/.codex     = ~/.sparkvm/creds/alice/.codex     (rw)
      /root/.npmrc     = ~/.sparkvm/creds/alice/.npmrc     (rw)
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
```

Guest agent: `fuser`, `vsock`, `rmp-serde`, `serde`, `serde_json`

## Size

|Component                         |Lines    |
|----------------------------------|---------|
|CLI + main                        |200      |
|Embedded binary launcher          |100      |
|Build script (build.rs)           |150      |
|Image builder                     |400      |
|Daemon core                       |150      |
|SSH server (russh)                |350      |
|VM manager + creds + memfd + netns|550      |
|SSH CA                            |200      |
|vsock FS server                   |800      |
|Lifecycle (signals, cleanup)      |100      |
|Guest agent                       |600      |
|**Total**                         |**~3600**|

## Security

|Property                |Mechanism                                 |
|------------------------|------------------------------------------|
|Single distributable    |sparkvm contains everything               |
|No system modifications |Nothing in /usr, /etc, etc.               |
|User isolation          |Separate KVM VM + netns per user          |
|Network isolation       |passt in per-VM netns, no host changes    |
|No credentials in image |All injected per-VM at runtime            |
|Ephemeral internal certs|Throwaway keypair, 60s TTL                |
|Per-user SSH principal  |VM sshd accepts only that username        |
|Per-user credentials    |Credential dirs scoped per user           |
|Mount isolation         |vsock FS scoped to that user’s directories|
|Credential writeback    |OAuth tokens persist across VM lifecycle  |
|No external VM access   |russh on 127.0.0.1 only                   |
|Host sshd untouched     |Port 22 always works                      |
|Zero host network impact|passt + netns, kernel-collected           |
|memfd execution         |Binaries never written to disk            |
|Clean process lifecycle |Signal handler + cgroup + sweep           |

## Prerequisites

```
/dev/kvm
kernel ≥ 5.10

Image building (one-time): debootstrap, mksquashfs
```

That’s it. One binary, zero install.