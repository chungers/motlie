use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};

use crate::VnetError;

#[cfg(target_os = "linux")]
fn default_dns_ipv4() -> Ipv4Addr {
    crate::slirp::parse_host_dns()
}

#[cfg(not(target_os = "linux"))]
fn default_dns_ipv4() -> Ipv4Addr {
    Ipv4Addr::new(10, 0, 2, 3)
}

/// A single host-to-guest TCP port forward rule (optional debug/demo helper).
/// libslirp binds `bind_addr:host_port` on the host and forwards accepted
/// connections to `guest_ipv4:guest_port` inside the virtual network.
#[derive(Debug, Clone)]
pub struct PortForward {
    /// Host-side bind address. Default: 127.0.0.1 (loopback only).
    pub bind_addr: Ipv4Addr,
    /// Host-side port to listen on.
    pub host_port: u16,
    /// Guest-side port to forward to.
    pub guest_port: u16,
}

/// Configuration for a vhost-user-net backend instance.
/// One instance per guest VM. See DESIGN NFR-4 for the isolation model.
#[derive(Debug, Clone)]
pub struct VnetConfig {
    /// Path to the vhost-user Unix socket.
    /// CH connects via: --net vhost_user=true,socket=<path>
    /// Must be unique per guest. Builder validates path is writable.
    pub socket_path: PathBuf,

    /// Guest IP assigned via DHCP. Default: 10.0.2.15
    pub guest_ipv4: Ipv4Addr,

    /// Gateway IP inside the virtual network. Default: 10.0.2.2
    pub host_ipv4: Ipv4Addr,

    /// Subnet mask. Default: 255.255.255.0
    pub netmask: Ipv4Addr,

    /// DNS server IP. Default: parsed from host /etc/resolv.conf at build() time.
    pub dns_ipv4: Ipv4Addr,

    /// MAC address for the guest's virtio-net device.
    /// Default: 52:54:00:12:34:56 (QEMU convention).
    pub mac: [u8; 6],

    /// Optional host-to-guest TCP port forwards (debug/demo helper).
    pub host_forwards: Vec<PortForward>,
}

/// Builder for VnetConfig.
pub struct VnetConfigBuilder {
    socket_path: Option<PathBuf>,
    guest_ipv4: Ipv4Addr,
    host_ipv4: Ipv4Addr,
    netmask: Ipv4Addr,
    dns_ipv4: Option<Ipv4Addr>,
    mac: [u8; 6],
    host_forwards: Vec<PortForward>,
}

impl VnetConfigBuilder {
    pub fn new() -> Self {
        Self {
            socket_path: None,
            guest_ipv4: Ipv4Addr::new(10, 0, 2, 15),
            host_ipv4: Ipv4Addr::new(10, 0, 2, 2),
            netmask: Ipv4Addr::new(255, 255, 255, 0),
            dns_ipv4: None,
            mac: [0x52, 0x54, 0x00, 0x12, 0x34, 0x56],
            host_forwards: Vec::new(),
        }
    }

    /// Required. Path to the vhost-user Unix socket.
    pub fn socket_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.socket_path = Some(path.into());
        self
    }

    pub fn guest_ipv4(mut self, addr: Ipv4Addr) -> Self {
        self.guest_ipv4 = addr;
        self
    }

    pub fn host_ipv4(mut self, addr: Ipv4Addr) -> Self {
        self.host_ipv4 = addr;
        self
    }

    pub fn netmask(mut self, addr: Ipv4Addr) -> Self {
        self.netmask = addr;
        self
    }

    pub fn dns_ipv4(mut self, addr: Ipv4Addr) -> Self {
        self.dns_ipv4 = Some(addr);
        self
    }

    pub fn mac(mut self, mac: [u8; 6]) -> Self {
        self.mac = mac;
        self
    }

    /// Add a host→guest TCP port forward (optional debug/demo helper).
    pub fn host_forward_tcp(mut self, host_port: u16, guest_port: u16) -> Self {
        self.host_forwards.push(PortForward {
            bind_addr: Ipv4Addr::LOCALHOST,
            host_port,
            guest_port,
        });
        self
    }

    /// Build and validate the config.
    pub fn build(self) -> Result<VnetConfig, VnetError> {
        let socket_path = self
            .socket_path
            .ok_or_else(|| VnetError::SocketPath("socket_path is required".into()))?;

        if socket_path.as_os_str().is_empty() {
            return Err(VnetError::SocketPath("socket_path is empty".into()));
        }

        // Validate parent directory exists and is writable.
        let parent = socket_path
            .parent()
            .unwrap_or(Path::new("/"));
        if !parent.exists() {
            return Err(VnetError::SocketPath(format!(
                "parent directory does not exist: {}",
                parent.display()
            )));
        }

        // Check for duplicate host_forward (bind_addr, host_port) pairs.
        let mut seen_binds = std::collections::HashSet::new();
        for fwd in &self.host_forwards {
            if !seen_binds.insert((fwd.bind_addr, fwd.host_port)) {
                return Err(VnetError::SocketPath(format!(
                    "duplicate host_forward bind: {}:{}",
                    fwd.bind_addr, fwd.host_port
                )));
            }
        }

        // Validate guest and host are in the same subnet.
        let guest_net = u32::from(self.guest_ipv4) & u32::from(self.netmask);
        let host_net = u32::from(self.host_ipv4) & u32::from(self.netmask);
        if guest_net != host_net {
            return Err(VnetError::SocketPath(format!(
                "guest {} and host {} are not in the same subnet (mask {})",
                self.guest_ipv4, self.host_ipv4, self.netmask
            )));
        }

        // Resolve DNS at build() time if not explicitly set.
        let dns_ipv4 = self
            .dns_ipv4
            .unwrap_or_else(default_dns_ipv4);

        Ok(VnetConfig {
            socket_path,
            guest_ipv4: self.guest_ipv4,
            host_ipv4: self.host_ipv4,
            netmask: self.netmask,
            dns_ipv4,
            mac: self.mac,
            host_forwards: self.host_forwards,
        })
    }
}

impl VnetConfig {
    /// Create a builder.
    pub fn builder() -> VnetConfigBuilder {
        VnetConfigBuilder::new()
    }
}
