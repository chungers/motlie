//! Single-binary VM orchestrator with:
//!   - Operator REPL over stdin/tty
//!   - Internal CA for SSH guest identity
//!   - Cgroup v2 lifecycle management
//!   - vhost-user-net + libslirp networking
//!   - virtiofs filesystem backend
//!   - vsock guest ↔ host communication
//!   - russh SSH frontend with certificate-based host verification
//!   - cloud-init metadata via shared filesystem
//!
//! Guest image assumptions:
//!   - cloud-init installed, reads from ConfigDrive or NoCloud datasource
//!   - A systemd service mounts virtiofs tag “shared” at /mnt/shared
//!   - openssh-server configured to use host certificates from cloud-init
//!
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │  Orchestrator process                                                  │
//! │                                                                        │
//! │  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  │
//! │  │ Operator │  │ SSH     │  │ Internal │  │ Slirp   │  │ Virtiofs │  │
//! │  │ REPL     │  │ Frontend│  │ CA       │  │ + vhost │  │ + vsock  │  │
//! │  │ (tty)    │  │ (russh) │  │          │  │ net     │  │ backends │  │
//! │  └────┬─────┘  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘  │
//! │       │              │           │              │            │         │
//! │  ┌────▼──────────────▼───────────▼──────────────▼────────────▼──────┐  │
//! │  │                      GuestManager                               │  │
//! │  │  ┌─────────────┐                                                │  │
//! │  │  │ CgroupMgr   │  spawn_guest() → fork+exec worker             │  │
//! │  │  └─────────────┘  shutdown_guest() → control socket + cgroup    │  │
//! │  └─────────┬──────────────────┬──────────────────┬─────────────────┘  │
//! │            │                  │                  │                     │
//! └────────────┼──────────────────┼──────────────────┼─────────────────────┘
//!         ┌────▼────┐       ┌────▼────┐        ┌────▼────┐
//!         │ Worker  │       │ Worker  │        │ Worker  │
//!         │ guest-a │       │ guest-b │        │ guest-c │
//!         │ CID 3   │       │ CID 4   │        │ CID 5   │
//!         └─────────┘       └─────────┘        └─────────┘
//!
//! Identity flow for a new guest:
//!
//!   1. GuestManager::spawn_guest() called (via REPL or SSH)
//!   2. CA generates:
//!      - SSH host key + host certificate (signed by CA)
//!      - SSH user certificate for the connecting user
//!      - cloud-init metadata (hostname, network, CA trust anchor)
//!   3. All written to shared_dir/<guest_id>/cloud-init/
//!   4. Worker starts CH with virtiofs pointing at shared_dir/<guest_id>/
//!   5. Guest boots, cloud-init reads NoCloud datasource from /mnt/shared/cloud-init/
//!   6. cloud-init installs host cert, configures sshd
//!   7. SSH frontend connects to guest via vsock, verifies host cert against CA
//!
//! REPL commands:
//!   list                    — show running guests
//!   spawn [disk] [cpus] [mem]  — launch a new guest
//!   shutdown <id>           — graceful shutdown
//!   kill <id>               — force kill via cgroup
//!   ssh <id>                — open SSH to guest (proxy via vsock)
//!   inspect <id>            — show guest details + cgroup stats
//!   ca status               — show CA certificate info
//!   ca rotate               — rotate CA keypair
//!   help                    — show commands
//!   quit                    — shut down everything

use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use std::{env, fs, process};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════
// PART 1: Internal Certificate Authority
// ═══════════════════════════════════════════════════════════════════════════

/// The CA manages a root keypair and issues short-lived SSH certificates
/// for each guest. This eliminates TOFU (trust-on-first-use) problems
/// when the SSH frontend connects to a freshly spawned guest.
///
/// Certificate types issued:
///   - Host certificates: prove the guest’s sshd identity to the SSH frontend
///   - User certificates: prove the user’s identity to the guest’s sshd
///
/// In production, use the `ssh-key` or `russh-keys` crate for real
/// SSH certificate generation. This sketch shows the structure.
struct CertificateAuthority {
/// CA private key (ed25519) — signs all certificates
ca_private_key: Vec<u8>,
/// CA public key — distributed to guests and SSH frontend as trust anchor
ca_public_key: Vec<u8>,
/// Fingerprint for display
ca_fingerprint: String,
/// Where to store CA keys on disk
ca_dir: PathBuf,
/// Serial number counter for certificates
next_serial: u64,
/// Certificate validity duration
cert_validity_secs: u64,
}

/// Identity bundle for a single guest
#[derive(Debug, Clone)]
struct GuestIdentity {
/// Unique hostname assigned to this guest
hostname: String,
/// SSH host private key (ed25519)
host_private_key: Vec<u8>,
/// SSH host certificate (signed by CA)
host_certificate: Vec<u8>,
/// CA public key (for the guest to trust user certs)
ca_public_key: Vec<u8>,
/// Serial number of the host certificate
serial: u64,
}

/// User certificate for SSH authentication into a guest
#[derive(Debug, Clone)]
struct UserCertificate {
/// The user’s public key (provided during SSH auth)
user_public_key: Vec<u8>,
/// Signed certificate granting access to the guest
certificate: Vec<u8>,
/// Which principals (usernames) this cert is valid for
principals: Vec<String>,
/// Expiry timestamp
valid_until: SystemTime,
}

impl CertificateAuthority {
fn new(ca_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
fs::create_dir_all(ca_dir)?;

```
    let ca_key_path = ca_dir.join("ca_ed25519");
    let ca_pub_path = ca_dir.join("ca_ed25519.pub");

    let (ca_private_key, ca_public_key, ca_fingerprint) = if ca_key_path.exists() {
        // Load existing CA keys
        let priv_key = fs::read(&ca_key_path)?;
        let pub_key = fs::read(&ca_pub_path)?;
        let fingerprint = Self::compute_fingerprint(&pub_key);
        log::info!("[ca] Loaded existing CA: {}", fingerprint);
        (priv_key, pub_key, fingerprint)
    } else {
        // Generate new CA keypair
        //
        // In real code:
        //   let key = russh_keys::key::KeyPair::generate_ed25519().unwrap();
        //   let priv_bytes = key.to_bytes();
        //   let pub_bytes = key.public_key().to_bytes();
        //   fs::write(&ca_key_path, &priv_bytes)?;
        //   fs::set_permissions(&ca_key_path, Permissions::from_mode(0o600))?;
        //   fs::write(&ca_pub_path, &pub_bytes)?;

        let priv_key = vec![0u8; 64]; // placeholder
        let pub_key = vec![0u8; 32]; // placeholder
        let fingerprint = "SHA256:placeholder".to_string();

        fs::write(&ca_key_path, &priv_key)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&ca_key_path, fs::Permissions::from_mode(0o600))?;
        }
        fs::write(&ca_pub_path, &pub_key)?;

        log::info!("[ca] Generated new CA keypair: {}", fingerprint);
        (priv_key, pub_key, fingerprint)
    };

    // Load or initialize serial counter
    let serial_path = ca_dir.join("serial");
    let next_serial = if serial_path.exists() {
        fs::read_to_string(&serial_path)?.trim().parse().unwrap_or(1)
    } else {
        1
    };

    Ok(Self {
        ca_private_key,
        ca_public_key,
        ca_fingerprint,
        ca_dir: ca_dir.to_path_buf(),
        next_serial,
        cert_validity_secs: 86400, // 24 hours
    })
}

/// Issue a host identity for a new guest.
/// Generates a host keypair and signs a host certificate.
fn issue_host_identity(
    &mut self,
    guest_id: &str,
) -> Result<GuestIdentity, Box<dyn std::error::Error>> {
    let serial = self.next_serial;
    self.next_serial += 1;
    self.save_serial()?;

    let hostname = format!("guest-{}", guest_id);

    // Generate a fresh ed25519 keypair for this guest's sshd
    //
    // In real code:
    //   let host_key = russh_keys::key::KeyPair::generate_ed25519()?;
    //   let host_pub = host_key.public_key();
    //
    //   // Sign a host certificate
    //   let cert = ssh_key::certificate::Builder::new_host(
    //       host_pub,
    //       serial,
    //       &[hostname.clone()],      // principals (hostnames)
    //       valid_after,
    //       valid_before,
    //   )?
    //   .sign(&ca_private_key)?;

    let host_private_key = vec![0u8; 64]; // placeholder
    let host_certificate = vec![0u8; 256]; // placeholder

    log::info!(
        "[ca] Issued host cert for {} (serial {}, fingerprint {})",
        hostname, serial, "SHA256:host-placeholder"
    );

    Ok(GuestIdentity {
        hostname,
        host_private_key,
        host_certificate,
        ca_public_key: self.ca_public_key.clone(),
        serial,
    })
}

/// Issue a user certificate allowing SSH access to a specific guest.
fn issue_user_certificate(
    &mut self,
    user_public_key: &[u8],
    username: &str,
    guest_id: &str,
) -> Result<UserCertificate, Box<dyn std::error::Error>> {
    let serial = self.next_serial;
    self.next_serial += 1;
    self.save_serial()?;

    let valid_until =
        SystemTime::now() + std::time::Duration::from_secs(self.cert_validity_secs);

    // In real code:
    //   let user_pub = ssh_key::PublicKey::from_bytes(user_public_key)?;
    //   let cert = ssh_key::certificate::Builder::new_user(
    //       user_pub,
    //       serial,
    //       &[username.to_string()],
    //       valid_after,
    //       valid_before,
    //   )?
    //   .extension("permit-pty", "")?
    //   .extension("permit-port-forwarding", "")?
    //   .critical_option("source-address", &allowed_ips)?
    //   .sign(&ca_private_key)?;

    let certificate = vec![0u8; 256]; // placeholder

    log::info!(
        "[ca] Issued user cert for {}@guest-{} (serial {})",
        username, guest_id, serial
    );

    Ok(UserCertificate {
        user_public_key: user_public_key.to_vec(),
        certificate,
        principals: vec![username.to_string()],
        valid_until,
    })
}

/// Rotate the CA keypair. Existing guests keep working until their
/// certs expire; new guests get certs from the new CA.
fn rotate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    // Archive old keys
    let timestamp = chrono_placeholder();
    let archive_dir = self.ca_dir.join(format!("archive-{}", timestamp));
    fs::create_dir_all(&archive_dir)?;
    let _ = fs::rename(
        self.ca_dir.join("ca_ed25519"),
        archive_dir.join("ca_ed25519"),
    );
    let _ = fs::rename(
        self.ca_dir.join("ca_ed25519.pub"),
        archive_dir.join("ca_ed25519.pub"),
    );

    // Generate new keypair
    // (same as in new())
    self.ca_private_key = vec![0u8; 64]; // placeholder
    self.ca_public_key = vec![0u8; 32]; // placeholder
    self.ca_fingerprint = "SHA256:rotated-placeholder".to_string();

    fs::write(self.ca_dir.join("ca_ed25519"), &self.ca_private_key)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(
            self.ca_dir.join("ca_ed25519"),
            fs::Permissions::from_mode(0o600),
        )?;
    }
    fs::write(self.ca_dir.join("ca_ed25519.pub"), &self.ca_public_key)?;

    log::info!("[ca] Rotated CA keypair: {}", self.ca_fingerprint);
    Ok(())
}

fn fingerprint(&self) -> &str {
    &self.ca_fingerprint
}

fn save_serial(&self) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(
        self.ca_dir.join("serial"),
        self.next_serial.to_string(),
    )?;
    Ok(())
}

fn compute_fingerprint(_pub_key: &[u8]) -> String {
    // In real code: SHA256 hash of the public key, base64 encoded
    "SHA256:placeholder".to_string()
}
```

}

fn chrono_placeholder() -> String {
// In real code: chrono::Utc::now().format(”%Y%m%d%H%M%S”).to_string()
“20260402120000”.to_string()
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 2: Cloud-init metadata generator
// ═══════════════════════════════════════════════════════════════════════════

/// Writes cloud-init NoCloud datasource files into the guest’s shared dir.
/// The guest mounts the virtiofs tag “shared” and cloud-init finds:
///   /mnt/shared/cloud-init/meta-data
///   /mnt/shared/cloud-init/user-data
///   /mnt/shared/cloud-init/vendor-data
///   /mnt/shared/ssh/                     ← host keys + certs
struct CloudInitGenerator;

impl CloudInitGenerator {
/// Write all cloud-init and SSH identity files for a guest.
fn write_guest_config(
shared_dir: &Path,
guest_id: &str,
identity: &GuestIdentity,
network_config: &GuestNetworkConfig,
) -> Result<(), Box<dyn std::error::Error>> {
// ── cloud-init directory ──
let ci_dir = shared_dir.join(“cloud-init”);
fs::create_dir_all(&ci_dir)?;

```
    // meta-data (YAML)
    let meta_data = format!(
        r#"instance-id: {guest_id}
```

local-hostname: {hostname}
“#,
guest_id = guest_id,
hostname = identity.hostname,
);
fs::write(ci_dir.join(“meta-data”), &meta_data)?;

```
    // network-config (v2 format)
    let network_data = format!(
        r#"version: 2
```

ethernets:
eth0:
dhcp4: true
nameservers:
addresses:
- {dns}
“#,
dns = network_config.dns_server,
);
fs::write(ci_dir.join(“network-config”), &network_data)?;

```
    // user-data: configure sshd to use our host cert + trust our CA
    let user_data = format!(
        r#"#cloud-config
```

hostname: {hostname}

# Install the CA public key as a trusted authority for user certs

ssh_authorized_principals:

- name: root
  principals: [”*”]

write_files:

# CA public key — sshd trusts user certs signed by this CA

- path: /etc/ssh/ca.pub
  content: |
  {ca_public_key}
  permissions: ‘0644’

# Host private key (generated by our CA)

- path: /etc/ssh/ssh_host_ed25519_key
  content: |
  {host_private_key}
  permissions: ‘0600’

# Host certificate (signed by our CA)

- path: /etc/ssh/ssh_host_ed25519_key-cert.pub
  content: |
  {host_certificate}
  permissions: ‘0644’

runcmd:

# Configure sshd to present the host certificate

- |
  cat >> /etc/ssh/sshd_config.d/orchestrator.conf << ‘EOF’
  HostKey /etc/ssh/ssh_host_ed25519_key
  HostCertificate /etc/ssh/ssh_host_ed25519_key-cert.pub
  TrustedUserCAKeys /etc/ssh/ca.pub
  EOF
- systemctl restart sshd

# Signal readiness via vsock (port 9999 → orchestrator)

- |
  echo “READY” | socat - VSOCK-CONNECT:2:9999 || true
  “#,
  hostname = identity.hostname,
  ca_public_key = base64_placeholder(&identity.ca_public_key),
  host_private_key = base64_placeholder(&identity.host_private_key),
  host_certificate = base64_placeholder(&identity.host_certificate),
  );
  fs::write(ci_dir.join(“user-data”), &user_data)?;
  
  ```
  log::info!(
      "[cloud-init] Wrote config for guest {} at {}",
      guest_id,
      shared_dir.display()
  );
  
  Ok(())
  ```
  
  }
  }

fn base64_placeholder(data: &[u8]) -> String {
// In real code: base64::encode(data)
format!(”<base64:{} bytes>”, data.len())
}

#[derive(Clone)]
struct GuestNetworkConfig {
/// IP assigned via slirp DHCP
guest_ip: String,
/// Gateway (slirp host-side)
gateway: String,
/// DNS server (slirp forwarding)
dns_server: String,
}

impl Default for GuestNetworkConfig {
fn default() -> Self {
Self {
guest_ip: “10.0.2.15”.to_string(),
gateway: “10.0.2.2”.to_string(),
dns_server: “10.0.2.3”.to_string(),
}
}
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 3: Wire protocol
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Serialize, Deserialize, Debug)]
enum OrchestratorMsg {
Configure(GuestSpec),
Shutdown,
Ping,
}

#[derive(Serialize, Deserialize, Debug)]
enum WorkerMsg {
Ready { vsock_cid: u64 },
Exited { code: i32 },
Error(String),
Pong,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GuestSpec {
id: String,
kernel: String,
cmdline: String,
disk: String,
cpus: u8,
memory_mb: u64,
vsock_cid: u64,
shared_dir: String,
}

fn send_msg<T: Serialize>(stream: &mut UnixStream, msg: &T) -> io::Result<()> {
let data = serde_json::to_vec(msg)?;
let len = (data.len() as u32).to_ne_bytes();
stream.write_all(&len)?;
stream.write_all(&data)?;
stream.flush()
}

fn recv_msg<T: for<’de> Deserialize<’de>>(stream: &mut UnixStream) -> io::Result<T> {
let mut len_buf = [0u8; 4];
stream.read_exact(&mut len_buf)?;
let len = u32::from_ne_bytes(len_buf) as usize;
let mut buf = vec![0u8; len];
stream.read_exact(&mut buf)?;
serde_json::from_slice(&buf)
.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 4: Cgroup v2 manager (same as before, abbreviated)
// ═══════════════════════════════════════════════════════════════════════════

struct GuestResources {
memory_max: u64,
cpu_max: (u64, u64),
pids_max: u64,
}

struct CgroupManager {
root: PathBuf,
}

impl CgroupManager {
fn new(cgroup_root: &str) -> Result<Self, Box<dyn std::error::Error>> {
let root = PathBuf::from(cgroup_root);
fs::create_dir_all(&root)?;
log::info!(”[cgroup] Initialized at {}”, root.display());
Ok(Self { root })
}

```
fn create_guest_cgroup(
    &self,
    guest_id: &str,
    resources: &GuestResources,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = self.root.join(format!("guest-{}.scope", guest_id));
    fs::create_dir_all(&path)?;
    fs::write(path.join("memory.max"), resources.memory_max.to_string())?;
    fs::write(path.join("memory.swap.max"), "0")?;
    fs::write(
        path.join("cpu.max"),
        format!("{} {}", resources.cpu_max.0, resources.cpu_max.1),
    )?;
    fs::write(path.join("pids.max"), resources.pids_max.to_string())?;
    log::info!("[cgroup] Created {}", path.display());
    Ok(path)
}

fn assign_pid(cgroup_path: &Path, pid: u32) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(cgroup_path.join("cgroup.procs"), pid.to_string())?;
    Ok(())
}

fn destroy_guest_cgroup(&self, guest_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = self.root.join(format!("guest-{}.scope", guest_id));
    if !path.exists() {
        return Ok(());
    }
    if path.join("cgroup.kill").exists() {
        let _ = fs::write(path.join("cgroup.kill"), "1");
    }
    let start = std::time::Instant::now();
    loop {
        let procs = fs::read_to_string(path.join("cgroup.procs")).unwrap_or_default();
        if procs.trim().is_empty() || start.elapsed() > std::time::Duration::from_secs(5) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let _ = fs::remove_dir(&path);
    Ok(())
}

fn recover_stale_cgroups(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut recovered = Vec::new();
    if !self.root.exists() {
        return Ok(recovered);
    }
    for entry in fs::read_dir(&self.root)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("guest-") && name.ends_with(".scope") {
            let id = name
                .strip_prefix("guest-")
                .and_then(|s| s.strip_suffix(".scope"))
                .unwrap_or(&name)
                .to_string();
            self.destroy_guest_cgroup(&id)?;
            recovered.push(id);
        }
    }
    Ok(recovered)
}

/// Read current memory usage from cgroup stats
fn read_memory_usage(&self, guest_id: &str) -> Option<u64> {
    let path = self.root.join(format!("guest-{}.scope", guest_id));
    fs::read_to_string(path.join("memory.current"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Read current CPU usage (microseconds)
fn read_cpu_usage(&self, guest_id: &str) -> Option<u64> {
    let path = self.root.join(format!("guest-{}.scope", guest_id));
    let stat = fs::read_to_string(path.join("cpu.stat")).ok()?;
    for line in stat.lines() {
        if let Some(val) = line.strip_prefix("usage_usec ") {
            return val.trim().parse().ok();
        }
    }
    None
}

/// Read PID count
fn read_pid_count(&self, guest_id: &str) -> Option<u64> {
    let path = self.root.join(format!("guest-{}.scope", guest_id));
    fs::read_to_string(path.join("pids.current"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}
```

}

// ═══════════════════════════════════════════════════════════════════════════
// PART 5: Backend services (abbreviated, same structure)
// ═══════════════════════════════════════════════════════════════════════════

struct SlirpNetBackend {
guest_id: String,
vhost_socket_path: PathBuf,
}

impl SlirpNetBackend {
fn new(guest_id: &str, runtime_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
let path = runtime_dir.join(format!(“vhost-net-{}.sock”, guest_id));
Ok(Self { guest_id: guest_id.to_string(), vhost_socket_path: path })
}
fn vhost_socket(&self) -> &Path { &self.vhost_socket_path }
fn shutdown(&self) { let _ = fs::remove_file(&self.vhost_socket_path); }
}

struct FuseBackend {
guest_id: String,
virtiofs_socket_path: PathBuf,
}

impl FuseBackend {
fn new(guest_id: &str, shared_dir: &Path, runtime_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
let path = runtime_dir.join(format!(“virtiofs-{}.sock”, guest_id));
fs::create_dir_all(shared_dir)?;
Ok(Self { guest_id: guest_id.to_string(), virtiofs_socket_path: path })
}
fn virtiofs_socket(&self) -> &Path { &self.virtiofs_socket_path }
fn shutdown(&self) { let _ = fs::remove_file(&self.virtiofs_socket_path); }
}

struct VsockProxy {
guest_id: String,
cid: u64,
}

impl VsockProxy {
fn new(guest_id: &str, cid: u64) -> Result<Self, Box<dyn std::error::Error>> {
Ok(Self { guest_id: guest_id.to_string(), cid })
}
fn shutdown(&self) {}
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 6: GuestManager — combines CA, cgroups, backends, cloud-init
// ═══════════════════════════════════════════════════════════════════════════

struct GuestHandle {
pid: u32,
control: UnixStream,
spec: GuestSpec,
identity: GuestIdentity,
net_backend: SlirpNetBackend,
fuse_backend: FuseBackend,
vsock_proxy: VsockProxy,
cgroup_path: PathBuf,
started_at: std::time::Instant,
}

struct OrchestratorConfig {
ssh_listen_addr: String,
runtime_dir: PathBuf,
shared_dir: PathBuf,
cgroup_root: String,
ca_dir: PathBuf,
kernel_path: String,
default_disk: String,
default_cpus: u8,
default_memory_mb: u64,
}

impl Default for OrchestratorConfig {
fn default() -> Self {
Self {
ssh_listen_addr: “0.0.0.0:2222”.to_string(),
runtime_dir: PathBuf::from(”/var/run/orchestrator”),
shared_dir: PathBuf::from(”/var/lib/orchestrator/shared”),
cgroup_root: “/sys/fs/cgroup/orchestrator”.to_string(),
ca_dir: PathBuf::from(”/var/lib/orchestrator/ca”),
kernel_path: “/opt/clh/kernel/vmlinux”.to_string(),
default_disk: “/opt/clh/images/base.raw”.to_string(),
default_cpus: 2,
default_memory_mb: 1024,
}
}
}

struct GuestManager {
guests: HashMap<String, GuestHandle>,
ca: CertificateAuthority,
cgroups: CgroupManager,
runtime_dir: PathBuf,
base_shared_dir: PathBuf,
kernel_path: String,
default_disk: String,
default_cpus: u8,
default_memory_mb: u64,
next_cid: u64,
}

impl GuestManager {
fn new(config: &OrchestratorConfig) -> Result<Self, Box<dyn std::error::Error>> {
fs::create_dir_all(&config.runtime_dir)?;
fs::create_dir_all(&config.shared_dir)?;

```
    let ca = CertificateAuthority::new(&config.ca_dir)?;
    let cgroups = CgroupManager::new(&config.cgroup_root)?;

    // Recover stale guests from previous crash
    let stale = cgroups.recover_stale_cgroups()?;
    for id in &stale {
        let _ = fs::remove_file(config.runtime_dir.join(format!("vhost-net-{}.sock", id)));
        let _ = fs::remove_file(config.runtime_dir.join(format!("virtiofs-{}.sock", id)));
        let _ = fs::remove_dir_all(config.shared_dir.join(id));
    }

    Ok(Self {
        guests: HashMap::new(),
        ca,
        cgroups,
        runtime_dir: config.runtime_dir.clone(),
        base_shared_dir: config.shared_dir.clone(),
        kernel_path: config.kernel_path.clone(),
        default_disk: config.default_disk.clone(),
        default_cpus: config.default_cpus,
        default_memory_mb: config.default_memory_mb,
        next_cid: 3,
    })
}

fn alloc_cid(&mut self) -> u64 {
    let cid = self.next_cid;
    self.next_cid += 1;
    cid
}

fn spawn_guest(
    &mut self,
    disk: Option<&str>,
    cpus: Option<u8>,
    memory_mb: Option<u64>,
) -> Result<String, Box<dyn std::error::Error>> {
    let guest_id = Uuid::new_v4().to_string()[..8].to_string();
    let cid = self.alloc_cid();
    let cpus = cpus.unwrap_or(self.default_cpus);
    let memory_mb = memory_mb.unwrap_or(self.default_memory_mb);
    let disk = disk.unwrap_or(&self.default_disk).to_string();
    let shared_dir = self.base_shared_dir.join(&guest_id);

    // ── Step 1: Issue identity from CA ──
    let identity = self.ca.issue_host_identity(&guest_id)?;

    // ── Step 2: Write cloud-init config into shared dir ──
    let net_config = GuestNetworkConfig::default();
    CloudInitGenerator::write_guest_config(
        &shared_dir,
        &guest_id,
        &identity,
        &net_config,
    )?;

    // ── Step 3: Create cgroup ──
    let resources = GuestResources {
        memory_max: ((memory_mb as u64) + 128) << 20,
        cpu_max: ((cpus as u64) * 100_000, 100_000),
        pids_max: (cpus as u64) + 32,
    };
    let cgroup_path = self.cgroups.create_guest_cgroup(&guest_id, &resources)?;

    // ── Step 4: Create backend services ──
    let net_backend = SlirpNetBackend::new(&guest_id, &self.runtime_dir)?;
    let fuse_backend = FuseBackend::new(&guest_id, &shared_dir, &self.runtime_dir)?;
    let vsock_proxy = VsockProxy::new(&guest_id, cid)?;

    let spec = GuestSpec {
        id: guest_id.clone(),
        kernel: self.kernel_path.clone(),
        cmdline: "console=hvc0 root=/dev/vda1 rw quiet".to_string(),
        disk,
        cpus,
        memory_mb,
        vsock_cid: cid,
        shared_dir: shared_dir.to_string_lossy().to_string(),
    };

    // ── Step 5: Fork + cgroup + exec ──
    let (parent_sock, child_sock) = UnixStream::pair()?;
    let child_sock_fd = child_sock.as_raw_fd();
    let vhost_net_sock = net_backend.vhost_socket().to_string_lossy().to_string();
    let virtiofs_sock = fuse_backend.virtiofs_socket().to_string_lossy().to_string();
    let cgroup_path_str = cgroup_path.to_string_lossy().to_string();

    let pid = match unsafe { libc::fork() } {
        -1 => {
            self.cgroups.destroy_guest_cgroup(&guest_id)?;
            return Err("fork() failed".into());
        }
        0 => {
            drop(parent_sock);

            // Join cgroup before exec
            if let Err(e) = CgroupManager::assign_pid(
                &PathBuf::from(&cgroup_path_str),
                process::id(),
            ) {
                eprintln!("cgroup assign failed: {}", e);
                process::exit(1);
            }

            close_all_fds_except(&[child_sock_fd]);

            unsafe {
                libc::fcntl(child_sock_fd, libc::F_SETFD, 0);
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            }

            let exe = env::current_exe().expect("current_exe");
            let _err = std::process::Command::new(exe)
                .arg("--worker")
                .env("CH_CONTROL_FD", child_sock_fd.to_string())
                .env("CH_VHOST_NET_SOCK", &vhost_net_sock)
                .env("CH_VIRTIOFS_SOCK", &virtiofs_sock)
                .exec();
            process::exit(1);
        }
        pid => pid as u32,
    };

    drop(child_sock);

    let mut parent_sock = parent_sock;
    send_msg(&mut parent_sock, &OrchestratorMsg::Configure(spec.clone()))?;

    match recv_msg::<WorkerMsg>(&mut parent_sock)? {
        WorkerMsg::Ready { vsock_cid: _ } => {
            log::info!("Guest {} booted (pid {}, CID {})", guest_id, pid, cid);
        }
        WorkerMsg::Error(e) => {
            self.cleanup_failed_guest(&guest_id, &net_backend, &fuse_backend, &vsock_proxy);
            return Err(format!("Guest {} boot failed: {}", guest_id, e).into());
        }
        _ => {}
    }

    self.guests.insert(guest_id.clone(), GuestHandle {
        pid,
        control: parent_sock,
        spec,
        identity,
        net_backend,
        fuse_backend,
        vsock_proxy,
        cgroup_path,
        started_at: std::time::Instant::now(),
    });

    Ok(guest_id)
}

fn shutdown_guest(&mut self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(mut handle) = self.guests.remove(id) {
        let _ = send_msg(&mut handle.control, &OrchestratorMsg::Shutdown);

        let start = std::time::Instant::now();
        loop {
            let mut status: libc::c_int = 0;
            let ret = unsafe { libc::waitpid(handle.pid as i32, &mut status, libc::WNOHANG) };
            if ret > 0 { break; }
            if start.elapsed() > std::time::Duration::from_secs(10) {
                log::warn!("Guest {} timeout, killing cgroup", id);
                self.cgroups.destroy_guest_cgroup(id)?;
                unsafe { libc::waitpid(handle.pid as i32, &mut status, 0) };
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        handle.net_backend.shutdown();
        handle.fuse_backend.shutdown();
        handle.vsock_proxy.shutdown();
        let _ = self.cgroups.destroy_guest_cgroup(id);
        let _ = fs::remove_dir_all(self.base_shared_dir.join(id));
        log::info!("Guest {} cleaned up", id);
    }
    Ok(())
}

fn force_kill_guest(&mut self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(handle) = self.guests.remove(id) {
        self.cgroups.destroy_guest_cgroup(id)?;
        let mut status: libc::c_int = 0;
        unsafe { libc::waitpid(handle.pid as i32, &mut status, 0) };
        handle.net_backend.shutdown();
        handle.fuse_backend.shutdown();
        handle.vsock_proxy.shutdown();
        let _ = fs::remove_dir_all(self.base_shared_dir.join(id));
        log::info!("Guest {} force killed", id);
    }
    Ok(())
}

fn reap_finished(&mut self) {
    let mut exited = Vec::new();
    for (id, handle) in &self.guests {
        let mut status: libc::c_int = 0;
        let ret = unsafe { libc::waitpid(handle.pid as i32, &mut status, libc::WNOHANG) };
        if ret > 0 {
            log::info!("Guest {} exited spontaneously", id);
            exited.push(id.clone());
        }
    }
    for id in &exited {
        if let Some(handle) = self.guests.remove(id) {
            handle.net_backend.shutdown();
            handle.fuse_backend.shutdown();
            handle.vsock_proxy.shutdown();
            let _ = self.cgroups.destroy_guest_cgroup(id);
            let _ = fs::remove_dir_all(self.base_shared_dir.join(id));
        }
    }
}

fn shutdown_all(&mut self) {
    let ids: Vec<String> = self.guests.keys().cloned().collect();
    for id in ids {
        let _ = self.shutdown_guest(&id);
    }
}

fn cleanup_failed_guest(
    &self,
    id: &str,
    net: &SlirpNetBackend,
    fuse: &FuseBackend,
    vsock: &VsockProxy,
) {
    let _ = self.cgroups.destroy_guest_cgroup(id);
    net.shutdown();
    fuse.shutdown();
    vsock.shutdown();
    let _ = fs::remove_dir_all(self.base_shared_dir.join(id));
}
```

}

// ═══════════════════════════════════════════════════════════════════════════
// PART 7: SSH frontend (russh)
// ═══════════════════════════════════════════════════════════════════════════

struct SshSession {
username: String,
guest_id: Option<String>,
manager: Arc<Mutex<GuestManager>>,
}

impl SshSession {
fn new(manager: Arc<Mutex<GuestManager>>) -> Self {
Self { username: String::new(), guest_id: None, manager }
}

```
/// On auth: issue a user certificate for this session
fn on_auth(&mut self, username: &str, user_pubkey: &[u8]) -> Result<UserCertificate, Box<dyn std::error::Error>> {
    self.username = username.to_string();

    // Spawn guest if needed
    let guest_id = {
        let mut mgr = self.manager.lock().unwrap();
        // In production: look up user → guest mapping
        let id = mgr.spawn_guest(None, None, None)?;
        id
    };

    // Issue a user certificate
    let cert = {
        let mut mgr = self.manager.lock().unwrap();
        mgr.ca.issue_user_certificate(user_pubkey, username, &guest_id)?
    };

    self.guest_id = Some(guest_id);
    Ok(cert)
}

/// Proxy SSH traffic to the guest via vsock
fn proxy_to_guest(&self) -> Result<(), Box<dyn std::error::Error>> {
    let guest_id = self.guest_id.as_ref().ok_or("no guest")?;
    let mgr = self.manager.lock().unwrap();
    let handle = mgr.guests.get(guest_id).ok_or("guest gone")?;

    // The SSH frontend trusts the guest's host cert because it was
    // signed by our CA. The guest trusts the user cert for the same reason.
    //
    // Connect to guest:
    //   vsock_stream = VsockStream::connect(handle.spec.vsock_cid, 22)?;
    //
    // SSH handshake over vsock:
    //   1. Frontend verifies guest's host-cert against CA public key
    //   2. Frontend presents user-cert to guest's sshd
    //   3. Guest's sshd verifies user-cert against CA public key
    //   4. Bidirectional relay between external SSH channel and vsock

    log::info!(
        "[ssh] Proxying {} → guest {} (CID {})",
        self.username, guest_id, handle.spec.vsock_cid
    );

    Ok(())
}
```

}

async fn run_ssh_server(
config: Arc<OrchestratorConfig>,
manager: Arc<Mutex<GuestManager>>,
) -> Result<(), Box<dyn std::error::Error>> {
log::info!(”[ssh] Listening on {}”, config.ssh_listen_addr);

```
// In production with russh:
//
// let host_key = russh_keys::load_secret_key(&host_key_path, None)?;
// let russh_config = russh::server::Config {
//     keys: vec![host_key],
//     ..Default::default()
// };
// russh::server::run(Arc::new(russh_config), &addr, SshServerImpl { manager }).await

let listener = tokio::net::TcpListener::bind(&config.ssh_listen_addr).await?;
loop {
    let (_, addr) = listener.accept().await?;
    let mgr = manager.clone();
    tokio::spawn(async move {
        let mut session = SshSession::new(mgr);
        match session.on_auth("user", b"fake-pubkey") {
            Ok(_cert) => {
                log::info!("[ssh] {} connected from {}", session.username, addr);
                let _ = session.proxy_to_guest();
            }
            Err(e) => log::error!("[ssh] Auth failed for {}: {}", addr, e),
        }
    });
}
```

}

// ═══════════════════════════════════════════════════════════════════════════
// PART 8: Operator REPL
// ═══════════════════════════════════════════════════════════════════════════

fn run_repl(manager: Arc<Mutex<GuestManager>>) {
let stdin = io::stdin();
let mut stdout = io::stdout();

```
print_repl_banner();

loop {
    print!("\x1b[1;36morchestrator>\x1b[0m ");
    stdout.flush().unwrap();

    let mut line = String::new();
    if stdin.lock().read_line(&mut line).unwrap() == 0 {
        // EOF (ctrl-D)
        println!("\nShutting down...");
        manager.lock().unwrap().shutdown_all();
        process::exit(0);
    }

    let parts: Vec<&str> = line.trim().split_whitespace().collect();
    if parts.is_empty() {
        continue;
    }

    match parts[0] {
        "help" | "h" | "?" => print_help(),

        "list" | "ls" => {
            let mgr = manager.lock().unwrap();
            if mgr.guests.is_empty() {
                println!("  No running guests.");
            } else {
                println!(
                    "  {:<10} {:<8} {:<6} {:<8} {:<10} {:<10} {:<10}",
                    "ID", "PID", "CID", "CPUs", "Mem (MB)", "Uptime", "MemUsage"
                );
                println!("  {}", "-".repeat(70));
                for (id, handle) in &mgr.guests {
                    let uptime = handle.started_at.elapsed();
                    let mem_usage = mgr.cgroups
                        .read_memory_usage(id)
                        .map(|b| format!("{}MB", b >> 20))
                        .unwrap_or_else(|| "?".to_string());
                    println!(
                        "  {:<10} {:<8} {:<6} {:<8} {:<10} {:<10} {:<10}",
                        &id[..8.min(id.len())],
                        handle.pid,
                        handle.spec.vsock_cid,
                        handle.spec.cpus,
                        handle.spec.memory_mb,
                        format_duration(uptime),
                        mem_usage,
                    );
                }
            }
        }

        "spawn" => {
            let disk = parts.get(1).copied();
            let cpus = parts.get(2).and_then(|s| s.parse().ok());
            let mem = parts.get(3).and_then(|s| s.parse().ok());

            print!("  Spawning guest...");
            stdout.flush().unwrap();

            match manager.lock().unwrap().spawn_guest(disk, cpus, mem) {
                Ok(id) => println!(" \x1b[32m✓\x1b[0m guest {} ready", id),
                Err(e) => println!(" \x1b[31m✗\x1b[0m {}", e),
            }
        }

        "shutdown" | "stop" => {
            if let Some(id) = parts.get(1) {
                match manager.lock().unwrap().shutdown_guest(id) {
                    Ok(_) => println!("  \x1b[32m✓\x1b[0m guest {} shut down", id),
                    Err(e) => println!("  \x1b[31m✗\x1b[0m {}", e),
                }
            } else {
                println!("  Usage: shutdown <guest-id>");
            }
        }

        "kill" => {
            if let Some(id) = parts.get(1) {
                match manager.lock().unwrap().force_kill_guest(id) {
                    Ok(_) => println!("  \x1b[32m✓\x1b[0m guest {} killed", id),
                    Err(e) => println!("  \x1b[31m✗\x1b[0m {}", e),
                }
            } else {
                println!("  Usage: kill <guest-id>");
            }
        }

        "inspect" => {
            if let Some(id) = parts.get(1) {
                let mgr = manager.lock().unwrap();
                if let Some(handle) = mgr.guests.get(*id) {
                    let uptime = handle.started_at.elapsed();
                    let mem = mgr.cgroups.read_memory_usage(id);
                    let cpu = mgr.cgroups.read_cpu_usage(id);
                    let pids = mgr.cgroups.read_pid_count(id);

                    println!("  Guest: {}", id);
                    println!("  ├─ PID:        {}", handle.pid);
                    println!("  ├─ Hostname:   {}", handle.identity.hostname);
                    println!("  ├─ CID:        {}", handle.spec.vsock_cid);
                    println!("  ├─ vCPUs:      {}", handle.spec.cpus);
                    println!("  ├─ Memory:     {} MB configured", handle.spec.memory_mb);
                    println!("  ├─ Disk:       {}", handle.spec.disk);
                    println!("  ├─ Uptime:     {}", format_duration(uptime));
                    println!("  ├─ Cert serial:{}", handle.identity.serial);
                    println!("  ├─ Cgroup:     {}", handle.cgroup_path.display());
                    println!("  │  ├─ Memory:  {}",
                        mem.map(|b| format!("{} MB / {} MB", b >> 20, handle.spec.memory_mb))
                           .unwrap_or_else(|| "unavailable".into()));
                    println!("  │  ├─ CPU:     {}",
                        cpu.map(|us| format!("{:.2}s total", us as f64 / 1_000_000.0))
                           .unwrap_or_else(|| "unavailable".into()));
                    println!("  │  └─ PIDs:    {}",
                        pids.map(|p| p.to_string())
                            .unwrap_or_else(|| "unavailable".into()));
                    println!("  ├─ Net socket: {}",
                        handle.net_backend.vhost_socket().display());
                    println!("  ├─ FS socket:  {}",
                        handle.fuse_backend.virtiofs_socket().display());
                    println!("  └─ Shared dir: {}", handle.spec.shared_dir);
                } else {
                    println!("  Guest {} not found", id);
                }
            } else {
                println!("  Usage: inspect <guest-id>");
            }
        }

        "ca" => {
            match parts.get(1).copied() {
                Some("status") => {
                    let mgr = manager.lock().unwrap();
                    println!("  CA fingerprint: {}", mgr.ca.fingerprint());
                    println!("  Next serial:    {}", mgr.ca.next_serial);
                    println!("  Cert validity:  {}h", mgr.ca.cert_validity_secs / 3600);
                    println!("  CA dir:         {}", mgr.ca.ca_dir.display());
                }
                Some("rotate") => {
                    print!("  Rotating CA keypair...");
                    stdout.flush().unwrap();
                    match manager.lock().unwrap().ca.rotate() {
                        Ok(_) => {
                            let mgr = manager.lock().unwrap();
                            println!(
                                " \x1b[32m✓\x1b[0m new fingerprint: {}",
                                mgr.ca.fingerprint()
                            );
                            println!(
                                "  ⚠  Existing guests keep their certs until they expire."
                            );
                            println!(
                                "     New guests will get certs from the new CA."
                            );
                        }
                        Err(e) => println!(" \x1b[31m✗\x1b[0m {}", e),
                    }
                }
                _ => println!("  Usage: ca <status|rotate>"),
            }
        }

        "shutdown-all" => {
            println!("  Shutting down all guests...");
            manager.lock().unwrap().shutdown_all();
            println!("  \x1b[32m✓\x1b[0m all guests stopped");
        }

        "quit" | "exit" => {
            println!("  Shutting down all guests...");
            manager.lock().unwrap().shutdown_all();
            println!("  Goodbye.");
            process::exit(0);
        }

        other => {
            println!("  Unknown command: '{}'. Type 'help' for commands.", other);
        }
    }
}
```

}

fn print_repl_banner() {
println!();
println!(”  \x1b[1;34m╔══════════════════════════════════════════╗\x1b[0m”);
println!(”  \x1b[1;34m║\x1b[0m   VM Orchestrator                        \x1b[1;34m║\x1b[0m”);
println!(”  \x1b[1;34m║\x1b[0m   Cloud Hypervisor + SSH + Internal CA    \x1b[1;34m║\x1b[0m”);
println!(”  \x1b[1;34m╚══════════════════════════════════════════╝\x1b[0m”);
println!();
println!(”  Type ‘help’ for commands.”);
println!();
}

fn print_help() {
println!(”  \x1b[1mCommands:\x1b[0m”);
println!(”    list                          List running guests”);
println!(”    spawn [disk] [cpus] [mem_mb]  Launch a new guest”);
println!(”    shutdown <id>                 Graceful shutdown”);
println!(”    kill <id>                     Force kill via cgroup”);
println!(”    inspect <id>                  Show guest details + resource usage”);
println!(”    ca status                     Show CA certificate info”);
println!(”    ca rotate                     Rotate CA keypair”);
println!(”    shutdown-all                  Shut down all guests”);
println!(”    quit                          Shut down everything and exit”);
}

fn format_duration(d: std::time::Duration) -> String {
let secs = d.as_secs();
if secs < 60 {
format!(”{}s”, secs)
} else if secs < 3600 {
format!(”{}m{}s”, secs / 60, secs % 60)
} else {
format!(”{}h{}m”, secs / 3600, (secs % 3600) / 60)
}
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 9: Worker process (lean child, unchanged from before)
// ═══════════════════════════════════════════════════════════════════════════

fn run_worker() -> Result<(), Box<dyn std::error::Error>> {
unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) };

```
let ctrl_fd: i32 = env::var("CH_CONTROL_FD")?.parse()?;
let _vhost_net_sock = env::var("CH_VHOST_NET_SOCK")?;
let _virtiofs_sock = env::var("CH_VIRTIOFS_SOCK")?;

let mut ctrl = unsafe { UnixStream::from_raw_fd(ctrl_fd) };

let spec = match recv_msg::<OrchestratorMsg>(&mut ctrl)? {
    OrchestratorMsg::Configure(spec) => spec,
    _ => return Err("expected Configure".into()),
};

log::info!(
    "[worker][pid {}] Guest {} (CID {})",
    process::id(), spec.id, spec.vsock_cid
);

// Build VmConfig from spec + socket paths
// start_vmm_thread → VmCreate → VmBoot
// (same as previous examples)

send_msg(&mut ctrl, &WorkerMsg::Ready { vsock_cid: spec.vsock_cid })?;

// Control listener thread
let guest_id = spec.id.clone();
std::thread::spawn(move || {
    loop {
        match recv_msg::<OrchestratorMsg>(&mut ctrl) {
            Ok(OrchestratorMsg::Shutdown) => {
                log::info!("[worker] Guest {}: shutdown", guest_id);
                // vm_shutdown + vmm_shutdown
                break;
            }
            Ok(OrchestratorMsg::Ping) => {
                let _ = send_msg(&mut ctrl, &WorkerMsg::Pong);
            }
            Err(_) => {
                log::warn!("[worker] Control socket lost");
                break;
            }
            _ => {}
        }
    }
});

// Block on VMM thread
// vmm_handle.thread.join()

// In the real version, block here. For the sketch, just sleep.
loop {
    std::thread::sleep(std::time::Duration::from_secs(3600));
}
```

}

// ═══════════════════════════════════════════════════════════════════════════
// PART 10: Utilities
// ═══════════════════════════════════════════════════════════════════════════

fn close_all_fds_except(keep: &[RawFd]) {
if let Ok(entries) = fs::read_dir(”/proc/self/fd”) {
for entry in entries.flatten() {
if let Ok(fd) = entry.file_name().to_string_lossy().parse::<i32>() {
if fd > 2 && !keep.contains(&fd) {
unsafe { libc::close(fd) };
}
}
}
}
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 11: Main — run REPL + SSH + reaper concurrently
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
env_logger::init();

```
// ── Worker mode: no tokio, no REPL, no SSH, no backends ──
if env::args().any(|a| a == "--worker") {
    return run_worker();
}

// ── Orchestrator mode ──
let config = Arc::new(OrchestratorConfig::default());
let manager = Arc::new(Mutex::new(GuestManager::new(&config)?));

log::info!(
    "Orchestrator pid={}, SSH={}, cgroup={}, CA={}",
    process::id(),
    config.ssh_listen_addr,
    config.cgroup_root,
    config.ca_dir.display(),
);

// Background: periodic reaper
let reaper = manager.clone();
tokio::spawn(async move {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
    loop {
        interval.tick().await;
        reaper.lock().unwrap().reap_finished();
    }
});

// Background: SSH server
let ssh_mgr = manager.clone();
let ssh_config = config.clone();
tokio::spawn(async move {
    if let Err(e) = run_ssh_server(ssh_config, ssh_mgr).await {
        log::error!("[ssh] Server failed: {}", e);
    }
});

// Foreground: operator REPL on a blocking thread
// (stdin is blocking IO, can't run on tokio executor)
let repl_mgr = manager.clone();
let repl_handle = tokio::task::spawn_blocking(move || {
    run_repl(repl_mgr);
});

// If the REPL exits (quit or ctrl-D), shut everything down
repl_handle.await?;

Ok(())
```

}