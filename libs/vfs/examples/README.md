# motlie-vfs Examples

Proof-of-concept programs for the v1 Cloud Hypervisor guest workflow.

## repl_host

Host-side filesystem server: one `FsServer` + `MemOverlay` per guest VM,
listening on a parameterized Unix socket, with an admin command interface
exposing every `MemOverlay` API operation. Admin is in-process only — no
network admin connections.

`repl_host` supports three provisioning styles:

- legacy single-guest `--tag` / `--dir` for `v1`
- startup-flag multi-guest `--guest` / guest-qualified `--mount`
- REPL-driven `--empty` plus `provision` / `mount` / `launch` for `v1.1`

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
# → for v1.1 operator-driven flows, prefer `cat script - | ...`
# → status/help lines emitted during scripted stdin are prefixed with `# `
#   so generated helper scripts remain executable shell scripts
```

### Options

- `--empty` — start with no guest and provision from REPL commands
- `--socket <path>` — vsock socket path (default: `/tmp/motlie-vfs.vsock_5000`)
- `--guest <id=socket>` — add one guest-scoped `FsServer` at startup
- `--mount <tag=dir>` — add a mount; repeat for multi-tag servers
- `--mount <id:tag=dir>` — add one mount to one guest at startup
- `--tag <name>` — mount tag (default: `alice-home`)
- `--dir <path>` — host backing directory (default: temp dir with sample data)

### Commands

**Provisioning / targeting:**
- `guests` — list provisioned guests, sockets, and mount counts
- `use <guest>` — set the default target guest
- `provision <guest> <socket> <uid> <gid>` — create one guest-scoped `FsServer`, record guest identity, and listener
- `mount <guest> <tag>=<guest_path>,<host_path> [more...]` — add one or more mounts to a guest
- `launch <guest>` — generate and start a prototype shell script asynchronously; logs land under `/tmp/motlie-vfs-launch/<guest>/`
- `launch -script <guest>` — print that helper shell script to stdout without executing it
- `shutdown <guest>` — request guest shutdown through `/tmp/motlie-vfs-<guest>-api.sock`

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
- `help <command>` — show detailed usage for one command, for example `help provision`
- `quit` — shut down server

The `launch` commands are prototype workflow helpers.

- `launch <guest>` writes the helper to `/tmp/motlie-vfs-launch/<guest>/launch.sh`, starts it asynchronously, and returns the REPL prompt immediately
- `launch -script <guest>` prints the same script to stdout

The helper embeds generated `mounts.yaml`, cloud-init `user-data`, and
`meta-data` for that guest, including explicit identity setup commands in
cloud-init `runcmd`. The intent is that a future VMM library can reuse the
same rendering helpers programmatically.

When `launch <guest>` executes the helper, it redirects:

- helper stdout/stderr to `/tmp/motlie-vfs-launch/<guest>/launch.log`
- guest serial console output to `/tmp/motlie-vfs-launch/<guest>/serial.log`

That avoids Cloud Hypervisor taking over the REPL terminal.

Current prototype limitation:

- the helper currently targets the demo guests `alice` and `bob`
- it seeds a NoCloud directory through `launch-ch.sh --cloud-init-dir`
- the shared `v1.1` base image must be rebuilt with the current `build-guest.sh` so the guest consumes the seeded NoCloud directory
- the generated helper does not require `cloud-localds`; `launch-ch.sh` seeds the NoCloud files into the guest runtime overlay directly

### Coordination Contract

The guest and host must agree on several parameters. This is important enough
to treat as an explicit contract:

- `socket`: the host-side Unix socket created by `repl_host` must match the socket path that the guest connects to
- `guest id`: only the admin control plane uses this; it selects which guest-scoped `FsServer` owns a socket and a set of mounts
- `tag`: the guest sends `TAG <name>` on connect, and the host must have provisioned that same tag in the target guest `FsServer`
- `guest_path`: the guest mount config decides where a tag is mounted inside the guest, for example `/home/alice` or `/workspace`
- `host_path`: `repl_host` binds each tag to a host backing directory
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
for an example.

See `v1/README.md` for the full end-to-end flow and `v1/CH-HARNESS.md`
for the single-guest Cloud Hypervisor setup. See `v1.1/README.md` for
the multi-guest, multi-mount variant.
