# motlie-vnet Examples

Proof-of-concept programs for the `motlie-vnet` user-mode networking stack,
starting from the validated `motlie-vfs` v1.1 multi-guest workflow.

`motlie-vnet` examples begin at `v1.2` because Phases 1 and 2 (libslirp wrapper
and vhost-user-net backend) are library-only with no standalone demo harness.
The `v1.2` example is where `motlie-vnet` first integrates into a running
Cloud Hypervisor guest flow.

## v1.2 — Split-Network / Agent-State POC

`v1.2` adds `motlie-vnet` egress networking on top of the `v1.1` multi-guest
VFS flow:

- TAP remains the admin/SSH ingress path (inherited from `v1.1`)
- `motlie-vnet` provides outbound internet via a second vhost-user-net NIC
- `/home/<user>` stays disk-backed from the host
- `/agent-state` is a dedicated read/write VFS-backed layer for Codex + Claude state

The host REPL binary is `repl_host_v1_2`:

```bash
cargo run -p motlie-vfs --example repl_host_v1_2 --features vsock -- \
  --empty --script libs/vnet/examples/v1.2/setup-multiguest.sh.vfs \
  --admin-net=tap --egress-net=vhost-user
```

See [v1.2/README.md](./v1.2/README.md) for the full workflow, validation
runbook, and known gaps.

### Relationship to libs/vfs/examples

The `v1.2` example set here is forked from `libs/vfs/examples/v1.2` because
`v1.2` is the first demo that exercises `motlie-vnet`. The two copies are
intentionally kept in sync:

- `libs/vfs/examples/v1.2` is the canonical copy from the VFS perspective
- `libs/vnet/examples/v1.2` is the canonical copy from the vnet perspective

Both contain the same guest image builder, launcher, REPL host, and mount
configs. The docs, scripts, and harness files are identical.

### Key files

| File | Purpose |
|------|---------|
| `v1.2/README.md` | Full v1.2 workflow, success criteria, mount contract, validated runbook |
| `v1.2/CH-HARNESS.md` | Cloud Hypervisor harness setup, host preflight, launcher modes |
| `v1.2/IMAGE.md` | Guest image build notes |
| `v1.2/build-guest.sh` | Shared base image builder (squashfs + kernel) |
| `v1.2/launch-ch.sh` | Per-guest launcher with `--admin-net` / `--egress-net` / `--vnet-socket` |
| `v1.2/repl_host.rs` | Host REPL binary (`repl_host_v1_2`) |
| `v1.2/setup-multiguest.sh.vfs` | Combined host REPL script for alice + bob |

### Network modes

`v1.2` supports three launcher configurations:

- `--admin-net=none --egress-net=none` — no networking
- `--admin-net=tap --egress-net=tap` — both NICs on TAP (no vnet)
- `--admin-net=tap --egress-net=vhost-user` — SSH on TAP, internet on motlie-vnet

The third mode is the target architecture for `motlie-vnet` integration.

### What v1.2 validates for motlie-vnet

1. Guest reaches the internet over the `motlie-vnet` vhost-user-net egress NIC
2. DNS resolution works through the libslirp built-in DNS forwarder (`10.0.2.3`)
3. The egress NIC coexists with the TAP admin NIC without route conflicts
4. The `VnetBackend` / `VnetHandle` API from Phase 2 is usable from the REPL host
