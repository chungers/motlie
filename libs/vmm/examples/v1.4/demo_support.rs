use std::path::PathBuf;

use motlie_vmm::spec::RuntimeNamespace;

pub const DEMO_GUEST_VSOCK_PORT: u32 = 5000;

pub fn demo_guest_ids(guest_id: &str) -> Result<(u32, u32), String> {
    match guest_id {
        "alice" => Ok((1000, 1000)),
        "bob" => Ok((1001, 1001)),
        _ => Err(format!("unknown demo guest '{guest_id}'")),
    }
}

pub fn demo_guest_socket_path(
    namespace: &RuntimeNamespace,
    guest_id: &str,
) -> Result<PathBuf, String> {
    namespace
        .guest_vsock_port_socket(guest_id, DEMO_GUEST_VSOCK_PORT)
        .map_err(|err| err.to_string())
}
