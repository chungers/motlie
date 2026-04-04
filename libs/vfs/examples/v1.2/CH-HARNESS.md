# v1.2 Cloud Hypervisor Harness

`v1.2` is the split-network successor to `v1.1`:

- TAP remains the short-term admin/SSH path
- `motlie-vnet` is the target outbound internet path
- `/agent-state` is a dedicated read/write VFS-backed layer for Codex + Claude state

This harness document is intentionally short until the composed host flow in
Phase 3.5 is validated.

Current expectations:

- build the guest image with [build-guest.sh](./build-guest.sh)
- use [launch-ch.sh](./launch-ch.sh) for direct launches
- use [repl_host.rs](./repl_host.rs) via `cargo run -p motlie-vfs --example repl_host_v1_2 --features vsock -- ...` as the only host-side runtime entry point

Supported launcher modes today:

- `--admin-net=none --egress-net=none`
- `--admin-net=tap --egress-net=tap`
- `--admin-net=tap --egress-net=vhost-user`

The detailed success criteria and remaining gaps live in [README.md](./README.md).
