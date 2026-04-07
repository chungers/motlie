# v1.3 Integration

This directory owns the repeatable `v1.3` smoke harness for future `motlie`
development. It intentionally stays under `examples/v1.3` so it can evolve
with the harness without becoming a workspace-wide default test.

## Rootless Smoke

Run:

```bash
cd /tmp/vmm-v1.3/libs/vmm/examples/v1.3/integration
./rootless-smoke.sh
```

What it validates:

- builds `repl_host_v1_3` and `russh_pty_probe`
- rebuilds the `v1.3` guest image
- starts the rootless/userspace harness with:
  - `--admin-net=none`
  - `--egress-net=vhost-user`
- boots `alice`
- waits for `SSH bridge ready for guest 'alice'`
- validates:
  - SSH exec through the proxy
  - PTY behavior through `russh_pty_probe`
  - VFS-backed home/workspace/agent-state visibility
  - rootless default route inside the guest
  - outbound HTTPS from the guest
  - the REPL's own `validate alice` checklist

On failure, the script dumps:

- the tmux pane containing `repl_host_v1_3`
- `/tmp/motlie-vmm-launch/alice/launch.log`
- `/tmp/motlie-vmm-launch/alice/serial.log`

This is the preferred manual regression check for `v1.3` until the harness
logic is extracted from `repl_host.rs` into reusable `motlie-vmm` library code.
