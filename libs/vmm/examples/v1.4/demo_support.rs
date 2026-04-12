use std::path::PathBuf;

use motlie_vmm::spec::RuntimeNamespace;

pub const DEMO_GUEST_VSOCK_PORT: u32 = 5000;

pub fn demo_guest_ids(guest_id: &str, slot: u32) -> Result<(u32, u32), String> {
    let builtin_uid = match guest_id {
        "alice" => Some(1000u32),
        "bob" => Some(1001u32),
        _ => None,
    };
    let uid = builtin_uid.unwrap_or(
        2000u32
        .checked_add(slot)
        .ok_or_else(|| format!("guest slot {slot} exceeds supported uid/gid range"))?,
    );
    Ok((uid, uid))
}

pub fn demo_guest_socket_path(
    namespace: &RuntimeNamespace,
    guest_id: &str,
) -> Result<PathBuf, String> {
    namespace
        .guest_vsock_port_socket(guest_id, DEMO_GUEST_VSOCK_PORT)
        .map_err(|err| err.to_string())
}
