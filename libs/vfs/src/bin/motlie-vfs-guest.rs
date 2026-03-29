//! motlie-vfs-guest: v1 guest-side mounter binary.
//!
//! This is a thin binary over the public guest APIs in `motlie_vfs::client::guest`.
//! It reads a mount config, obtains vsock streams via VMM-owned handshake logic,
//! and delegates to `GuestMountRunner` for the actual FUSE mounts.

fn main() {
    // Placeholder — Phase 4.2 implements the real guest binary.
    eprintln!("motlie-vfs-guest: not yet implemented");
    std::process::exit(1);
}
