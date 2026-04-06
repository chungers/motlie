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

## Using The Flag In `v1.2`

The `v1.2` harness is now owned by `motlie-vnet`, so debug tracing is enabled
directly on the `motlie-vnet` example binary.

Normal `v1.2` host run:

```bash
cargo run -p motlie-vnet --example repl_host_v1_2 -- \
  --empty \
  --script libs/vnet/examples/v1.2/setup-multiguest.sh.vfs \
  --admin-net=tap \
  --egress-net=vhost-user
```

Debug `v1.2` host run with backend traces enabled:

```bash
cargo run -p motlie-vnet --example repl_host_v1_2 --features debug-trace -- \
  --empty \
  --script libs/vnet/examples/v1.2/setup-multiguest.sh.vfs \
  --admin-net=tap \
  --egress-net=vhost-user
```

## Related Docs

- [DESIGN.md](/home/dchung/codex-vfs/motlie-vnet/libs/vnet/docs/DESIGN.md)
- [PLAN.md](/home/dchung/codex-vfs/motlie-vnet/libs/vnet/docs/PLAN.md)
- [examples/README.md](/home/dchung/codex-vfs/motlie-vnet/libs/vnet/examples/README.md)
