use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt;
use std::net::Ipv4Addr;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::Serialize;
use thiserror::Error;

use crate::spec::UNIX_SOCKET_PATH_MAX_BYTES;

const MAX_MAC_SLOT_CAPACITY: u32 = 1 << 20;
const DEFAULT_SOCKET_NAME_PREFIX: &str = "motlie-vmm";
const MAC_OUI_PREFIX: [u8; 3] = [0x52, 0x54, 0x00];

fn unix_socket_path_len(path: &Path) -> usize {
    #[cfg(unix)]
    {
        path.as_os_str().as_bytes().len()
    }

    #[cfg(not(unix))]
    {
        path.as_os_str().to_string_lossy().len()
    }
}

fn validate_unix_socket_path(path: &Path) -> Result<(), GuestNetAllocatorError> {
    let len = unix_socket_path_len(path);
    if len > UNIX_SOCKET_PATH_MAX_BYTES {
        return Err(GuestNetAllocatorError::SocketPathTooLong {
            path: path.to_path_buf(),
            len,
            max_len: UNIX_SOCKET_PATH_MAX_BYTES,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Ipv4Subnet {
    pub network: Ipv4Addr,
    pub prefix_len: u8,
}

impl Ipv4Subnet {
    pub fn new(network: Ipv4Addr, prefix_len: u8) -> Result<Self, GuestNetAllocatorError> {
        if prefix_len > 32 {
            return Err(GuestNetAllocatorError::InvalidPrefix { prefix_len });
        }
        let mask = prefix_mask(prefix_len);
        Ok(Self {
            network: Ipv4Addr::from(u32::from(network) & mask),
            prefix_len,
        })
    }

    pub fn child_capacity(&self, child_prefix_len: u8) -> Result<u32, GuestNetAllocatorError> {
        if child_prefix_len < self.prefix_len || child_prefix_len > 32 {
            return Err(GuestNetAllocatorError::InvalidChildPrefix {
                base_prefix_len: self.prefix_len,
                child_prefix_len,
            });
        }

        let shift = u32::from(child_prefix_len - self.prefix_len);
        Ok(1u32.checked_shl(shift).unwrap_or(0))
    }

    pub fn child_subnet(
        &self,
        child_prefix_len: u8,
        slot: u32,
    ) -> Result<Ipv4Subnet, GuestNetAllocatorError> {
        let capacity = self.child_capacity(child_prefix_len)?;
        if slot >= capacity {
            return Err(GuestNetAllocatorError::SubnetSlotExhausted { slot, capacity });
        }

        let subnet_size = subnet_size(child_prefix_len);
        let base = u32::from(self.network);
        let child_network = base + slot.saturating_mul(subnet_size);
        Ipv4Subnet::new(Ipv4Addr::from(child_network), child_prefix_len)
    }

    pub fn address_at_offset(&self, offset: u32) -> Result<Ipv4Addr, GuestNetAllocatorError> {
        let subnet_size = subnet_size(self.prefix_len);
        if offset >= subnet_size {
            return Err(GuestNetAllocatorError::AddressOffsetOutOfRange {
                subnet: *self,
                offset,
                subnet_size,
            });
        }

        if self.prefix_len <= 30 && (offset == 0 || offset == subnet_size - 1) {
            return Err(GuestNetAllocatorError::ReservedAddressOffset {
                subnet: *self,
                offset,
            });
        }

        Ok(Ipv4Addr::from(u32::from(self.network) + offset))
    }
}

impl fmt::Display for Ipv4Subnet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.network, self.prefix_len)
    }
}

impl FromStr for Ipv4Subnet {
    type Err = GuestNetAllocatorError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (network, prefix_len) =
            value
                .split_once('/')
                .ok_or_else(|| GuestNetAllocatorError::InvalidSubnetNotation {
                    value: value.to_string(),
                })?;
        let network = network.parse::<Ipv4Addr>().map_err(|_| {
            GuestNetAllocatorError::InvalidSubnetNotation {
                value: value.to_string(),
            }
        })?;
        let prefix_len = prefix_len.parse::<u8>().map_err(|_| {
            GuestNetAllocatorError::InvalidSubnetNotation {
                value: value.to_string(),
            }
        })?;
        Self::new(network, prefix_len)
    }
}

/// Typed admin-side IPv4 pair for a guest-specific ingress subnet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdminIpv4Pair {
    pub host: Ipv4Addr,
    pub guest: Ipv4Addr,
}

/// Typed egress-side IPv4 layout for a guest-specific userspace network backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EgressIpv4Layout {
    pub guest: Ipv4Addr,
    pub host: Ipv4Addr,
    pub dns: Ipv4Addr,
    pub netmask: Ipv4Addr,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ipv4SubnetPool {
    pub base: Ipv4Subnet,
    pub guest_prefix_len: u8,
    pub first_subnet_slot: u32,
    pub host_offset: u32,
    pub guest_offset: u32,
    pub dns_offset: Option<u32>,
}

impl Ipv4SubnetPool {
    pub fn capacity(&self) -> Result<u32, GuestNetAllocatorError> {
        let raw_capacity = self.base.child_capacity(self.guest_prefix_len)?;
        Ok(raw_capacity.saturating_sub(self.first_subnet_slot))
    }

    pub fn subnet_for_slot(&self, slot: u32) -> Result<Ipv4Subnet, GuestNetAllocatorError> {
        let raw_capacity = self.base.child_capacity(self.guest_prefix_len)?;
        let subnet_slot = self.first_subnet_slot.checked_add(slot).ok_or(
            GuestNetAllocatorError::SubnetSlotOverflow {
                first_subnet_slot: self.first_subnet_slot,
                slot,
            },
        )?;
        if subnet_slot >= raw_capacity {
            return Err(GuestNetAllocatorError::SubnetSlotExhausted {
                slot: subnet_slot,
                capacity: raw_capacity,
            });
        }
        self.base.child_subnet(self.guest_prefix_len, subnet_slot)
    }

    pub fn validate(&self, label: &'static str) -> Result<(), GuestNetAllocatorError> {
        let raw_capacity = self.base.child_capacity(self.guest_prefix_len)?;
        if self.first_subnet_slot >= raw_capacity {
            return Err(GuestNetAllocatorError::InvalidPool {
                label,
                reason: format!(
                    "first_subnet_slot {} exceeds raw pool capacity {}",
                    self.first_subnet_slot, raw_capacity
                ),
            });
        }
        let subnet = self.subnet_for_slot(0)?;
        subnet
            .address_at_offset(self.host_offset)
            .map_err(|source| GuestNetAllocatorError::InvalidPool {
                label,
                reason: source.to_string(),
            })?;
        subnet
            .address_at_offset(self.guest_offset)
            .map_err(|source| GuestNetAllocatorError::InvalidPool {
                label,
                reason: source.to_string(),
            })?;

        if self.host_offset == self.guest_offset {
            return Err(GuestNetAllocatorError::InvalidPool {
                label,
                reason: "host_offset and guest_offset must differ".to_string(),
            });
        }

        if let Some(dns_offset) = self.dns_offset {
            subnet.address_at_offset(dns_offset).map_err(|source| {
                GuestNetAllocatorError::InvalidPool {
                    label,
                    reason: source.to_string(),
                }
            })?;
            if dns_offset == self.host_offset || dns_offset == self.guest_offset {
                return Err(GuestNetAllocatorError::InvalidPool {
                    label,
                    reason: "dns_offset must not collide with host/guest offsets".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Stable per-guest network assignment.
///
/// This is intentionally broader than the current `examples/v1.3` allocator:
/// it captures both admin-ingress and egress identity so later consumers can
/// use one allocation table instead of assembling values from multiple helper
/// functions spread across a harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GuestNetAssignment {
    pub guest_name: String,
    pub slot: u32,
    pub cid: u32,
    pub admin_subnet: Ipv4Subnet,
    pub admin_ipv4: AdminIpv4Pair,
    pub admin_mac: [u8; 6],
    pub egress_subnet: Ipv4Subnet,
    pub egress_ipv4: EgressIpv4Layout,
    pub egress_mac: [u8; 6],
    pub vnet_socket_path: PathBuf,
}

/// Allocation policy for guest identity.
///
/// The reviewed policy is slot-derived: a guest slot deterministically maps to
/// CID, admin subnet/IP/MAC, egress subnet/IP/MAC, and socket paths.
#[derive(Debug, Clone, Serialize)]
pub struct GuestNetAllocatorConfig {
    /// First guest CID to assign. Current `v1.3` starts at 3.
    pub first_cid: u32,
    /// Optional explicit capacity clamp after the subnet pools are evaluated.
    pub max_guests: Option<u32>,
    /// Prefix used for vhost-user socket paths. Keep this directory short: it
    /// participates directly in AF_UNIX socket addresses, including delayed
    /// auto-provisioned boots that happen after the operator detaches.
    pub socket_dir: PathBuf,
    /// Namespace-sensitive socket name prefix used for vhost-user sockets. Keep
    /// the combined socket_dir + socket_name_prefix budget under sun_path.
    pub socket_name_prefix: String,
    /// Admin ingress subnet pool.
    pub admin_pool: Ipv4SubnetPool,
    /// Egress subnet pool used by `motlie-vnet`.
    pub egress_pool: Ipv4SubnetPool,
}

impl GuestNetAllocatorConfig {
    pub fn capacity(&self) -> Result<u32, GuestNetAllocatorError> {
        self.admin_pool.validate("admin_pool")?;
        self.egress_pool.validate("egress_pool")?;

        let admin_capacity = self.admin_pool.capacity()?;
        let egress_capacity = self.egress_pool.capacity()?;
        let cid_capacity = u32::MAX.saturating_sub(self.first_cid).saturating_add(1);
        let capacity = admin_capacity
            .min(egress_capacity)
            .min(cid_capacity)
            .min(MAX_MAC_SLOT_CAPACITY);
        Ok(self.max_guests.unwrap_or(capacity).min(capacity))
    }

    pub fn with_max_guests(mut self, max_guests: u32) -> Self {
        self.max_guests = Some(max_guests);
        self
    }

    pub fn validate(&self) -> Result<(), GuestNetAllocatorError> {
        let capacity = self.capacity()?;
        if capacity == 0 {
            return Err(GuestNetAllocatorError::InvalidPool {
                label: "allocator",
                reason: "computed capacity is zero".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for GuestNetAllocatorConfig {
    fn default() -> Self {
        Self {
            first_cid: 3,
            max_guests: None,
            socket_dir: std::env::temp_dir(),
            socket_name_prefix: DEFAULT_SOCKET_NAME_PREFIX.to_string(),
            admin_pool: Ipv4SubnetPool {
                base: Ipv4Subnet::new(Ipv4Addr::new(172, 20, 0, 0), 16)
                    .expect("static subnet is valid"),
                guest_prefix_len: 30,
                first_subnet_slot: 0,
                host_offset: 1,
                guest_offset: 2,
                dns_offset: None,
            },
            egress_pool: Ipv4SubnetPool {
                base: Ipv4Subnet::new(Ipv4Addr::new(10, 0, 0, 0), 8)
                    .expect("static subnet is valid"),
                guest_prefix_len: 24,
                first_subnet_slot: 2,
                host_offset: 2,
                guest_offset: 15,
                dns_offset: Some(3),
            },
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GuestNetAllocatorError {
    #[error("guest name cannot be empty")]
    EmptyGuestName,
    #[error("invalid IPv4 prefix length: {prefix_len}")]
    InvalidPrefix { prefix_len: u8 },
    #[error("invalid child prefix {child_prefix_len} for base prefix {base_prefix_len}")]
    InvalidChildPrefix {
        base_prefix_len: u8,
        child_prefix_len: u8,
    },
    #[error("invalid IPv4 subnet notation: {value}")]
    InvalidSubnetNotation { value: String },
    #[error("subnet slot space exhausted: slot {slot} exceeds capacity {capacity}")]
    SubnetSlotExhausted { slot: u32, capacity: u32 },
    #[error(
        "subnet slot overflow: first_subnet_slot {first_subnet_slot} + slot {slot} exceeds u32"
    )]
    SubnetSlotOverflow { first_subnet_slot: u32, slot: u32 },
    #[error(
        "address offset {offset} is out of range for subnet {subnet} (subnet size {subnet_size})"
    )]
    AddressOffsetOutOfRange {
        subnet: Ipv4Subnet,
        offset: u32,
        subnet_size: u32,
    },
    #[error("address offset {offset} is reserved in subnet {subnet}")]
    ReservedAddressOffset { subnet: Ipv4Subnet, offset: u32 },
    #[error("invalid {label}: {reason}")]
    InvalidPool { label: &'static str, reason: String },
    #[error(
        "guest slot space exhausted: next slot {next_slot} exceeds configured capacity {capacity}"
    )]
    Exhausted { next_slot: u32, capacity: u32 },
    #[error("unix socket path is too long ({len} bytes > {max_len}): {path}")]
    SocketPathTooLong {
        path: PathBuf,
        len: usize,
        max_len: usize,
    },
}

/// Stable guest network allocation table.
///
/// Intended direction:
/// - `examples/v1.3` stops owning `net_allocs` / `next_net_slot`
/// - `libs/vmm` asks this type for a stable assignment per guest name
/// - launch rendering and `motlie-vnet` backend startup consume `GuestNetAssignment`
#[derive(Debug, Clone)]
pub struct GuestNetAllocator {
    config: GuestNetAllocatorConfig,
    assignments: BTreeMap<String, GuestNetAssignment>,
    next_slot: u32,
}

impl GuestNetAllocator {
    pub fn new(config: GuestNetAllocatorConfig) -> Result<Self, GuestNetAllocatorError> {
        config.validate()?;
        Ok(Self {
            config,
            assignments: BTreeMap::new(),
            next_slot: 0,
        })
    }

    pub fn config(&self) -> &GuestNetAllocatorConfig {
        &self.config
    }

    pub fn capacity(&self) -> Result<u32, GuestNetAllocatorError> {
        self.config.capacity()
    }

    pub fn next_slot(&self) -> u32 {
        self.next_slot
    }

    pub fn remaining_capacity(&self) -> Result<u32, GuestNetAllocatorError> {
        Ok(self.capacity()?.saturating_sub(self.next_slot))
    }

    pub fn assignments(&self) -> &BTreeMap<String, GuestNetAssignment> {
        &self.assignments
    }

    pub fn get(&self, guest_name: &str) -> Option<&GuestNetAssignment> {
        self.assignments.get(guest_name)
    }

    pub fn ensure(
        &mut self,
        guest_name: &str,
    ) -> Result<&GuestNetAssignment, GuestNetAllocatorError> {
        if guest_name.is_empty() {
            return Err(GuestNetAllocatorError::EmptyGuestName);
        }
        let capacity = self.capacity()?;

        match self.assignments.entry(guest_name.to_string()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let slot = self.next_slot;
                if slot >= capacity {
                    return Err(GuestNetAllocatorError::Exhausted {
                        next_slot: slot,
                        capacity,
                    });
                }

                self.next_slot += 1;
                let assignment = Self::build_assignment(&self.config, guest_name, slot)?;
                Ok(entry.insert(assignment))
            }
        }
    }

    fn build_assignment(
        config: &GuestNetAllocatorConfig,
        guest_name: &str,
        slot: u32,
    ) -> Result<GuestNetAssignment, GuestNetAllocatorError> {
        let admin_subnet = config.admin_pool.subnet_for_slot(slot)?;
        let egress_subnet = config.egress_pool.subnet_for_slot(slot)?;
        let admin_host = admin_subnet.address_at_offset(config.admin_pool.host_offset)?;
        let admin_guest = admin_subnet.address_at_offset(config.admin_pool.guest_offset)?;
        let egress_host = egress_subnet.address_at_offset(config.egress_pool.host_offset)?;
        let egress_guest = egress_subnet.address_at_offset(config.egress_pool.guest_offset)?;
        let egress_dns = egress_subnet.address_at_offset(
            config
                .egress_pool
                .dns_offset
                .unwrap_or(config.egress_pool.host_offset),
        )?;
        let egress_netmask = Ipv4Addr::from(prefix_mask(egress_subnet.prefix_len));

        let vnet_socket_path = config
            .socket_dir
            .join(format!("{}-{guest_name}.sock", config.socket_name_prefix));
        validate_unix_socket_path(&vnet_socket_path)?;

        Ok(GuestNetAssignment {
            guest_name: guest_name.to_string(),
            slot,
            cid: config.first_cid + slot,
            admin_subnet,
            admin_ipv4: AdminIpv4Pair {
                host: admin_host,
                guest: admin_guest,
            },
            admin_mac: mac_from_slot(slot, 0xa0),
            egress_subnet,
            egress_ipv4: EgressIpv4Layout {
                guest: egress_guest,
                host: egress_host,
                dns: egress_dns,
                netmask: egress_netmask,
            },
            egress_mac: mac_from_slot(slot, 0xe0),
            vnet_socket_path,
        })
    }
}

fn prefix_mask(prefix_len: u8) -> u32 {
    if prefix_len == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix_len))
    }
}

fn subnet_size(prefix_len: u8) -> u32 {
    if prefix_len == 32 {
        1
    } else {
        1u32 << (32 - u32::from(prefix_len))
    }
}

fn mac_from_slot(slot: u32, plane_prefix: u8) -> [u8; 6] {
    [
        MAC_OUI_PREFIX[0],
        MAC_OUI_PREFIX[1],
        MAC_OUI_PREFIX[2],
        plane_prefix | ((slot >> 16) as u8 & 0x0f),
        (slot >> 8) as u8,
        slot as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subnet_child_capacity_and_offsets_work() {
        let subnet = Ipv4Subnet::new(Ipv4Addr::new(172, 20, 0, 0), 16).unwrap();
        assert_eq!(subnet.child_capacity(30).unwrap(), 16_384);
        let child = subnet.child_subnet(30, 2).unwrap();
        assert_eq!(child.to_string(), "172.20.0.8/30");
        assert_eq!(
            child.address_at_offset(1).unwrap(),
            Ipv4Addr::new(172, 20, 0, 9)
        );
        assert_eq!(
            child.address_at_offset(2).unwrap(),
            Ipv4Addr::new(172, 20, 0, 10)
        );
    }

    #[test]
    fn ensure_returns_stable_assignment_for_same_guest() {
        let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default()).unwrap();
        let first = alloc.ensure("alice").unwrap().clone();
        let second = alloc.ensure("alice").unwrap().clone();
        assert_eq!(first, second);
        assert_eq!(first.cid, 3);
        assert_eq!(first.admin_subnet.to_string(), "172.20.0.0/30");
        assert_eq!(first.admin_ipv4.guest, Ipv4Addr::new(172, 20, 0, 2));
        assert_eq!(first.egress_subnet.to_string(), "10.0.2.0/24");
        assert_eq!(first.egress_ipv4.host, Ipv4Addr::new(10, 0, 2, 2));
        assert_eq!(first.egress_ipv4.dns, Ipv4Addr::new(10, 0, 2, 3));
    }

    #[test]
    fn ensure_advances_without_collision_for_first_three_slots() {
        let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default()).unwrap();
        let alice = alloc.ensure("alice").unwrap().clone();
        let bob = alloc.ensure("bob").unwrap().clone();
        let carol = alloc.ensure("carol").unwrap().clone();

        assert_eq!(alice.cid, 3);
        assert_eq!(bob.cid, 4);
        assert_eq!(carol.cid, 5);
        assert_ne!(alice.admin_ipv4.guest, bob.admin_ipv4.guest);
        assert_ne!(bob.admin_ipv4.guest, carol.admin_ipv4.guest);
        assert_ne!(alice.admin_mac, bob.admin_mac);
        assert_ne!(bob.admin_mac, carol.admin_mac);
        assert_eq!(alice.egress_subnet.to_string(), "10.0.2.0/24");
        assert_eq!(bob.egress_subnet.to_string(), "10.0.3.0/24");
        assert_eq!(carol.egress_subnet.to_string(), "10.0.4.0/24");
        assert_eq!(alloc.capacity().unwrap(), 16_384);
    }

    #[test]
    fn ensure_returns_exhausted_once_configured_capacity_runs_out() {
        let mut alloc =
            GuestNetAllocator::new(GuestNetAllocatorConfig::default().with_max_guests(2)).unwrap();

        alloc.ensure("alice").unwrap();
        alloc.ensure("bob").unwrap();
        let err = alloc.ensure("carol").unwrap_err();

        assert_eq!(
            err,
            GuestNetAllocatorError::Exhausted {
                next_slot: 2,
                capacity: 2,
            }
        );
    }

    #[test]
    fn ensure_rejects_overlong_vnet_socket_paths() {
        let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig {
            socket_dir: PathBuf::from("/home/dchung/cdx-autopro/this/path/is/intentionally/much/longer/than/the-unix-domain-socket-budget/live/s"),
            socket_name_prefix: "v14-r2556482-9190".to_string(),
            ..GuestNetAllocatorConfig::default()
        })
        .unwrap();

        let err = alloc.ensure("alice").unwrap_err();
        assert!(matches!(
            err,
            GuestNetAllocatorError::SocketPathTooLong { .. }
        ));
    }

    #[test]
    fn invalid_pool_rejects_reserved_guest_offset() {
        let err = GuestNetAllocatorConfig {
            admin_pool: Ipv4SubnetPool {
                guest_offset: 3,
                ..GuestNetAllocatorConfig::default().admin_pool
            },
            ..GuestNetAllocatorConfig::default()
        }
        .validate()
        .unwrap_err();

        assert!(matches!(err, GuestNetAllocatorError::InvalidPool { .. }));
    }
}
