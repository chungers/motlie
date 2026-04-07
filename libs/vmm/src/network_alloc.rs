use std::collections::{BTreeMap, btree_map::Entry};
use std::net::Ipv4Addr;
use std::path::PathBuf;

use thiserror::Error;

/// Typed admin-side IPv4 pair for a guest-specific ingress subnet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdminIpv4Pair {
    pub host: Ipv4Addr,
    pub guest: Ipv4Addr,
}

/// Typed egress-side IPv4 layout for a guest-specific userspace network backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgressIpv4Layout {
    pub guest: Ipv4Addr,
    pub host: Ipv4Addr,
    pub dns: Ipv4Addr,
    pub netmask: Ipv4Addr,
}

/// Stable per-guest network assignment.
///
/// This is intentionally broader than the current `examples/v1.3` allocator:
/// it captures both admin-ingress and egress identity so later consumers can
/// use one allocation table instead of assembling values from multiple helper
/// functions spread across a harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestNetAssignment {
    pub guest_name: String,
    pub slot: u16,
    pub cid: u32,
    pub admin_ipv4: AdminIpv4Pair,
    pub admin_mac: [u8; 6],
    pub egress_ipv4: EgressIpv4Layout,
    pub egress_mac: [u8; 6],
    pub vnet_socket_path: PathBuf,
}

/// Allocation policy for guest identity.
///
/// TODO(vmm): wire this into a library-owned allocator used by the future
/// `boot` / `boot_and_wait` orchestration APIs. Today `examples/v1.3/repl_host.rs`
/// still owns the live allocation logic.
#[derive(Debug, Clone)]
pub struct GuestNetAllocatorConfig {
    /// First guest CID to assign. Current `v1.3` starts at 3.
    pub first_cid: u32,
    /// First admin subnet octet to assign in 192.168.X.0/24.
    pub first_admin_subnet: u16,
    /// Maximum distinct admin /24 subnets available.
    pub max_admin_subnets: u16,
    /// Prefix used for vhost-user socket paths.
    pub socket_dir: PathBuf,
}

impl Default for GuestNetAllocatorConfig {
    fn default() -> Self {
        Self {
            first_cid: 3,
            first_admin_subnet: 249,
            // 249..=255 gives 7 non-colliding /24s in the current v1.3 scheme.
            max_admin_subnets: 7,
            socket_dir: PathBuf::from("/tmp"),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GuestNetAllocatorError {
    #[error("guest name cannot be empty")]
    EmptyGuestName,
    #[error(
        "guest slot space exhausted: next slot {next_slot} exceeds configured admin subnet capacity {capacity}"
    )]
    Exhausted { next_slot: u16, capacity: u16 },
}

/// Stable guest network allocation table.
///
/// Intended direction:
/// - `examples/v1.3` stops owning `net_allocs` / `next_net_slot`
/// - `libs/vmm` asks this type for a stable assignment per guest name
/// - launch rendering and `motlie-vnet` backend startup consume `GuestNetAssignment`
///
/// TODO(vmm): this is a scaffold only. It does not yet replace the live
/// `ensure_net_alloc()` path in `examples/v1.3/repl_host.rs`.
#[derive(Debug, Clone)]
pub struct GuestNetAllocator {
    config: GuestNetAllocatorConfig,
    assignments: BTreeMap<String, GuestNetAssignment>,
    next_slot: u16,
}

impl GuestNetAllocator {
    pub fn new(config: GuestNetAllocatorConfig) -> Self {
        Self {
            config,
            assignments: BTreeMap::new(),
            next_slot: 0,
        }
    }

    pub fn config(&self) -> &GuestNetAllocatorConfig {
        &self.config
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

        match self.assignments.entry(guest_name.to_string()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let slot = self.next_slot;
                if slot >= self.config.max_admin_subnets {
                    return Err(GuestNetAllocatorError::Exhausted {
                        next_slot: slot,
                        capacity: self.config.max_admin_subnets,
                    });
                }

                self.next_slot += 1;
                let assignment = Self::build_assignment(&self.config, guest_name, slot);
                Ok(entry.insert(assignment))
            }
        }
    }

    fn build_assignment(
        config: &GuestNetAllocatorConfig,
        guest_name: &str,
        slot: u16,
    ) -> GuestNetAssignment {
        let subnet = config.first_admin_subnet + slot;
        let host = Ipv4Addr::new(192, 168, subnet as u8, 1);
        let guest = Ipv4Addr::new(192, 168, subnet as u8, 2);
        let suffix = (slot + 1) as u8;

        GuestNetAssignment {
            guest_name: guest_name.to_string(),
            slot,
            cid: config.first_cid + slot as u32,
            admin_ipv4: AdminIpv4Pair { host, guest },
            admin_mac: [0x52, 0x54, 0x00, 0xad, 0x00, suffix],
            egress_ipv4: EgressIpv4Layout {
                guest: Ipv4Addr::new(10, 0, 2, 15),
                host: Ipv4Addr::new(10, 0, 2, 2),
                dns: Ipv4Addr::new(10, 0, 2, 3),
                netmask: Ipv4Addr::new(255, 255, 255, 0),
            },
            egress_mac: [0x52, 0x54, 0x00, 0xe9, 0x00, suffix],
            vnet_socket_path: config
                .socket_dir
                .join(format!("motlie-vmm-{guest_name}.sock")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_returns_stable_assignment_for_same_guest() {
        let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default());
        let first = alloc.ensure("alice").unwrap().clone();
        let second = alloc.ensure("alice").unwrap().clone();
        assert_eq!(first, second);
        assert_eq!(first.cid, 3);
        assert_eq!(first.admin_ipv4.guest, Ipv4Addr::new(192, 168, 249, 2));
    }

    #[test]
    fn ensure_advances_without_collision_for_first_three_slots() {
        let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig::default());
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
    }

    #[test]
    fn ensure_returns_exhausted_once_configured_admin_subnets_run_out() {
        let mut alloc = GuestNetAllocator::new(GuestNetAllocatorConfig {
            max_admin_subnets: 2,
            ..GuestNetAllocatorConfig::default()
        });

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
}
