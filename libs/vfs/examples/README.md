# motlie-vfs Examples

Proof-of-concept programs for the `v1` and `v1.1` Cloud Hypervisor guest workflows.

## repl_host

Stable `v1` host-side filesystem server: one single-guest `FsServer` +
`MemOverlay`, listening on one Unix socket, with an in-process admin REPL.

`repl_host` intentionally stays on the `v1` single-guest surface so later
roadmap slices do not drift or break the original example.

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
# → executes script, then may reopen /dev/tty if one is available
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

## repl_host_v1_1

`repl_host_v1_1` is the `v1.1`-specific multi-guest host harness. It owns the
admin-plane features that were added after `v1`:

- `--empty`, `--script`
- `provision`, `mount`, `guests`, `use`
- `launch`, `launch -script`, `shutdown`
- one `FsServer` per guest in one shared host process

Run it with:

```bash
cargo run -p motlie-vfs --example repl_host_v1_1 --features vsock -- \
  --empty --script libs/vfs/examples/v1.1/setup-multiguest.sh.vfs
```

See [v1.1/README.md](./v1.1/README.md) for the full multi-guest workflow.

### Coordination Contract

The guest and host must agree on several parameters. This is important enough
to treat as an explicit contract for both binaries:

- `socket`: the host-side Unix socket created by `repl_host` or `repl_host_v1_1` must match the socket path that the guest connects to
- `guest id`: only the admin control plane uses this; it selects which guest-scoped `FsServer` owns a socket and a set of mounts
- `tag`: the guest sends `TAG <name>` on connect, and the host must have provisioned that same tag in the target guest `FsServer`
- `guest_path`: the guest mount config decides where a tag is mounted inside the guest, for example `/home/alice` or `/workspace`
- `host_path`: the host REPL binds each tag to a host backing directory
- `uid/gid/mode`: overlay-injected files such as `.ssh/authorized_keys` or `.env` must use values that make sense for the guest user, and `provision` now records the intended guest uid/gid explicitly

Important nuance:

- the host-side `FsServer` routes by `tag -> host_path`
- the guest decides `tag -> guest_path`
- the `guest_path` written in the REPL `mount` command is documentation and operator coordination data; the runtime mount location still comes from the guest `mounts.yaml`

Example:

```text
guest config:  tag=alice-home guest_path=/home/alice
host config:   guest=alice uid=1000 gid=1000 tag=alice-home host_path=/tmp/motlie-vfs-demo/alice-home
overlay attrs: uid=1000 gid=1000 mode=0600 for /home/alice/.ssh/authorized_keys
```

If any of those parameters drift apart, mounts may connect to the wrong place,
overlay ownership may be wrong, or the guest may fail to see the expected data.

### Script files

Script files are plain text, one command per line. Lines starting with
`#` are comments. Empty lines are skipped. See `v1/setup-alice.sh.vfs`
for the single-guest shape and `v1.1/setup-multiguest.sh.vfs` for the
multi-guest shape.

See `v1/README.md` for the full end-to-end flow and `v1/CH-HARNESS.md`
for the single-guest Cloud Hypervisor setup. See `v1.1/README.md` for
the multi-guest, multi-mount variant.
