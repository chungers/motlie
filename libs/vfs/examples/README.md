# motlie-vfs Examples

Proof-of-concept programs for the v1 Cloud Hypervisor guest workflow.

## simple_host

Host-side server with FsServer + MemOverlay, listening on a Unix socket
(simulating the vsock host-side path), with a stdin command loop for
overlay mutation.

```bash
# Start with a temp host directory (seeded with sample data)
cargo run -p motlie-vfs --example simple_host --features vsock

# Or with an explicit host directory
cargo run -p motlie-vfs --example simple_host --features vsock -- /path/to/dir
```

**Commands:**
- `put <layer> <tag> <path> <content>` — inject a file
- `whiteout <layer> <tag> <path>` — hide a lower-layer file
- `rm <layer> <tag> <path>` — remove an overlay entry
- `ls <tag>` — list effective overlay entries
- `quit` — shut down

The server listens on `/tmp/motlie-vfs.vsock_5000` (the CH vsock
convention for guest→host connections on port 5000).

See `libs/vfs/image/README.md` for the full Cloud Hypervisor
proof-of-concept setup and validation procedure.
