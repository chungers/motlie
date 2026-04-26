//! Apple Virtualization.framework backend family.
//!
//! This is the intended home for simple standard macOS guest boot and resource
//! realization once the portable `VmSpec` + `GuestBackends` API converges.

use std::path::PathBuf;

use crate::spec::GuestRuntimePaths;

pub mod shell;

pub fn artifacts_dir(runtime_paths: &GuestRuntimePaths) -> PathBuf {
    runtime_paths.runtime_dir.join("vz-artifacts")
}

pub fn vm_name(runtime_paths: &GuestRuntimePaths, guest_id: &str) -> String {
    let runtime_stem = runtime_paths
        .runtime_dir
        .parent()
        .and_then(|path| path.file_name())
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "runtime".into());
    sanitize_name(&format!("motlie-v1-45-{runtime_stem}-{guest_id}"))
}

fn sanitize_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for byte in raw.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' => out.push(byte as char),
            _ => out.push('-'),
        }
    }
    out
}
