# motlie-vfs Examples

Proof-of-concept programs for the v1 Cloud Hypervisor guest workflow.

> **Note:** These are scaffolds only in the current phase. The examples will
> gain real behavior in later phases (REPL in Phase 5.1, CH harness in Phase 5.1).

## simple_host

Host-side server with overlay mutation testing. Will include a `rustyline`
REPL once the server core and overlay are implemented.

```bash
# Requires the vsock feature (needed for transport setup)
cargo run -p motlie-vfs --example simple_host --features vsock
```

Currently exits immediately with a "not yet implemented" message.

See `libs/vfs/docs/DESIGN.md` for the full Cloud Hypervisor proof-of-concept
setup and validation procedure.
