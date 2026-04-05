# motlie-vnet

`motlie-vnet` is the user-mode networking crate used by the `v1.2` example to
provide outbound guest egress over a vhost-user net device backed by libslirp.

## Build Features

`motlie-vnet` keeps its normal build quiet by default. Optional live packet and
queue diagnostics are available behind a Cargo feature:

- `debug-trace`
  - Enables temporary stderr diagnostics from the vhost-user backend
  - Prints queue kicks, TX/RX packet summaries, and TX completion failures
  - Intended for live bring-up and protocol debugging, not normal runs

Build the crate directly with tracing enabled:

```bash
cargo check -p motlie-vnet --features debug-trace
```

## Using The Flag From `motlie-vfs`

The `motlie-vfs` crate forwards this feature as `vnet-debug-trace`, so the
`v1.2` host REPL can enable the same diagnostics without building
`motlie-vnet` separately.

Normal `v1.2` host run:

```bash
cargo run -p motlie-vfs --example repl_host_v1_2 --features vsock -- \
  --empty \
  --script libs/vfs/examples/v1.2/setup-multiguest.sh.vfs \
  --admin-net=tap \
  --egress-net=vhost-user
```

Debug `v1.2` host run with backend traces enabled:

```bash
cargo run -p motlie-vfs --example repl_host_v1_2 --features vsock,vnet-debug-trace -- \
  --empty \
  --script libs/vfs/examples/v1.2/setup-multiguest.sh.vfs \
  --admin-net=tap \
  --egress-net=vhost-user
```

## Related Docs

- [DESIGN.md](/home/dchung/codex-vfs/motlie-vnet/libs/vnet/docs/DESIGN.md)
- [PLAN.md](/home/dchung/codex-vfs/motlie-vnet/libs/vnet/docs/PLAN.md)
