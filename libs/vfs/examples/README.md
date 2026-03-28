# motlie-vfs Examples

Proof-of-concept programs for the v1 Cloud Hypervisor guest workflow.

## simple_host

Host-side server with a `rustyline` REPL for overlay mutation testing.

```bash
cargo run -p motlie-vfs --example simple_host
```

See `libs/vfs/docs/DESIGN.md` for the full Cloud Hypervisor proof-of-concept
setup and validation procedure.
