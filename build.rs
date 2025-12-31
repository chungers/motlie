//! Build script for root workspace package
//!
//! SIMD feature detection has been moved to libs/core/build.rs.
//! This build.rs is kept minimal for the root package.

fn main() {
    // SIMD detection is now handled by motlie-core's build.rs
    // This file is kept for any future root-level build configuration
}
