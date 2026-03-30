# motlie-vfs Examples

Proof-of-concept programs for the v1 Cloud Hypervisor guest workflow.

## repl_host

Host-side filesystem server: one `FsServer` + `MemOverlay` per guest VM,
listening on a parameterized Unix socket (the vsock host-side path), with a
rustyline REPL exposing every `MemOverlay` API operation. Admin is in-process
only — no network admin connections.

```bash
# Default: socket /tmp/motlie-vfs.vsock_5000, tag alice-home, temp host dir
cargo run -p motlie-vfs --example repl_host --features vsock

# Explicit parameters:
cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --socket /tmp/motlie-vfs-alice.vsock_5000 \
    --tag alice-home \
    --dir /path/to/host/dir

# Multi-guest: run separate instances with different socket/tag/dir
```

### REPL Commands

**Layer management:**
- `layer <name> <priority>` — create or update a named layer
- `rmlayer <name>` — remove layer and all its entries
- `layers` — list all layers with priority and entry count

**Content injection:**
- `put <layer> <tag> <path> <content>` — inject file with default attrs
- `putattr <layer> <tag> <path> <uid> <gid> <mode> <content>` — inject with explicit ownership/mode (mode is octal)
- `mkdir <layer> <tag> <path> [mode]` — create synthetic directory

**Suppression / removal:**
- `whiteout <layer> <tag> <path>` — hide a lower-layer entry
- `rm <layer> <tag> <path>` — remove an overlay entry

**Inspection:**
- `get <layer> <tag> <path>` — read content from a specific layer
- `ls <tag>` — list effective overlay entries (highest-priority wins)
- `lslayer <layer> <tag>` — list entries in a specific layer

**Other:**
- `help` — show all commands
- `quit` — shut down

Each guest VM gets its own `FsServer` instance and vsock socket. Tags identify
mounted subtrees within that VM's server. See `libs/vfs/docs/DESIGN.md` FR-4
for the isolation model.

See `libs/vfs/image/README.md` for the full Cloud Hypervisor setup and
validation procedure.
