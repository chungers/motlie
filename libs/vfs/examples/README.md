# motlie-vfs Examples

Proof-of-concept programs for the v1 Cloud Hypervisor guest workflow.

## simple_host

Host-side server: one `FsServer` + `MemOverlay` per guest VM, listening on a
parameterized Unix socket (the vsock host-side path), with a stdin command
loop for overlay mutation. No network admin — admin is in-process only.

```bash
# Default: socket /tmp/motlie-vfs.vsock_5000, tag alice-home, temp host dir
cargo run -p motlie-vfs --example simple_host --features vsock

# Explicit parameters:
cargo run -p motlie-vfs --example simple_host --features vsock -- \
    --socket /tmp/motlie-vfs-alice.vsock_5000 \
    --tag alice-home \
    --dir /path/to/host/dir

# Multi-guest: run separate instances with different socket/tag/dir
```

**Commands (stdin — in-process admin, no network):**
- `put <layer> <tag> <path> <content>` — inject a file
- `whiteout <layer> <tag> <path>` — hide a lower-layer file
- `rm <layer> <tag> <path>` — remove an overlay entry
- `ls <tag>` — list effective overlay entries
- `quit` — shut down

Each guest VM gets its own `FsServer` instance and vsock socket. Tags identify
mounted subtrees within that VM's server. See `libs/vfs/docs/DESIGN.md` FR-4
for the isolation model.

See `libs/vfs/image/README.md` for the full Cloud Hypervisor setup and
validation procedure.
