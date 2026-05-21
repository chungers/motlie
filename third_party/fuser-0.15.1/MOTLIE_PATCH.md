# Motlie fuser Patch

This is a minimal vendor of `fuser` 0.15.1 for v1.5 guest image assembly.

The published 0.15.1 `build.rs` decides whether the pure-rust Linux mount
implementation is available with build-host `#[cfg(target_os = "linux")]`.
That breaks the v1.5 contract because macOS must cross-compile Linux guest
binaries before injecting them into CH/VZ images. The local patch keeps the
crate version and source intact except for `build.rs`, which reads Cargo's
`CARGO_CFG_TARGET_OS` and therefore permits the pure-rust path when the target
OS is Linux.

Remove this vendor when upstream `fuser` supports macOS-hosted Linux-target
cross-compilation without requiring a target libfuse sysroot.
