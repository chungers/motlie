# motlie-vfs Examples

Proof-of-concept programs for the v1 Cloud Hypervisor guest workflow.

## repl_host

Host-side filesystem server: one `FsServer` + `MemOverlay` per guest VM,
listening on a parameterized Unix socket, with an admin command interface
exposing every `MemOverlay` API operation. Admin is in-process only — no
network admin connections.

### Input modes

The server detects how stdin is connected and adapts:

**Interactive (stdin is a TTY):**
```bash
cargo run -p motlie-vfs --example repl_host --features vsock -- --tag alice-home
# → rustyline REPL with line editing, history, ^C handling
```

**Pipe then interactive (`cat script - | ...`):**
```bash
cat setup-alice.sh.vfs - | cargo run -p motlie-vfs --example repl_host --features vsock -- --tag alice-home
# → executes script, then drops into interactive REPL
# → server keeps serving throughout
```

**Pure pipe (`cat script | ...`):**
```bash
cat setup-alice.sh.vfs | cargo run -p motlie-vfs --example repl_host --features vsock -- --tag alice-home
# → executes script, then server keeps running until SIGTERM/SIGINT
# → use this for automated/agent-driven setups
```

### Options

- `--socket <path>` — vsock socket path (default: `/tmp/motlie-vfs.vsock_5000`)
- `--tag <name>` — mount tag (default: `alice-home`)
- `--dir <path>` — host backing directory (default: temp dir with sample data)

### Commands

**Layer management:**
- `layer <name> <priority>` — create or update a named layer
- `rmlayer <name>` — remove layer and all its entries
- `layers` — list all layers with priority and entry count

**Content injection:**
- `put <layer> <tag> <path> <content>` — inject file with default attrs
- `putattr <layer> <tag> <path> <uid> <gid> <mode> <content>` — inject with explicit ownership/mode
- `mkdir <layer> <tag> <path> [mode]` — create synthetic directory

**Suppression / removal:**
- `whiteout <layer> <tag> <path>` — hide a lower-layer entry
- `rm <layer> <tag> <path>` — remove an overlay entry

**Inspection:**
- `get <layer> <tag> <path>` — read content from a specific layer
- `ls <tag>` — list effective overlay entries
- `lslayer <layer> <tag>` — list entries in a specific layer

**Other:**
- `help` — show all commands
- `quit` — shut down server

### Script files

Script files are plain text, one command per line. Lines starting with
`#` are comments. Empty lines are skipped. See `v1/setup-alice.sh.vfs`
for an example.

See `v1/README.md` for the full end-to-end flow and `v1/CH-HARNESS.md`
for the Cloud Hypervisor setup.
