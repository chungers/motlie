//! Canonical v1.5 VMM-owned guest mounter binary.
//!
//! Source lives under `libs/vmm/bins/v1.5` by contract. The binary composes
//! VFS library APIs but is not owned by `libs/vfs`, and v1.5 must not grow a
//! `libs/vfs/bins/v1.5` or `libs/vnet/bins/v1.5` tree.

fn main() -> anyhow::Result<()> {
    motlie_vmm::guest::vfs_mount::main()
}
