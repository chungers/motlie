//! VMM-owned guest runtime components for product guest images.
//!
//! Keep guest image entrypoints under `libs/vmm/bins/` and reusable guest-side
//! logic here. VFS and VNET remain libraries consumed by this runtime; do not
//! create VFS/VNET-owned v1.5 bin or example trees for the convergence line.

pub mod vfs_mount;
