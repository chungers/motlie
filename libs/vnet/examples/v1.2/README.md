# v1.2 Split-Network / Agent-State Example

`v1.2` starts from the validated `v1.1` multi-guest VFS flow and adds the
Phase 3 migration toward `motlie-vnet`:

- keep inbound SSH on the existing TAP admin path
- move outbound internet to a separate `motlie-vnet` egress NIC
- preserve the per-launch writable root overlay from `v1.1`
- keep `/home/<user>` disk-backed from the host
- add one dedicated read/write VFS-backed layer at `/agent-state` for Codex + Claude state during the lifetime of the host REPL

This directory is the source of truth for the `v1.2` example / validation
harness. `v1` and `v1.1` remain in `libs/vfs/examples/`; beginning with
`v1.2`, the composed host flow is owned here in `motlie-vnet`.

## Success Outcomes

`v1.2` is considered successful when one guest can demonstrate all of these:

1. the guest reaches the internet over the new egress path
2. the user still connects via the `v1.1` TAP-admin SSH path
3. the user installs `python3` and other Debian packages with `apt-get`,
   proving the writable `/` overlay is stable
4. the user installs Codex CLI and Claude Code CLI inside the guest
5. the user completes auth/login for both tools inside the guest
6. the user runs the agent CLIs over SSH and verifies their state is written
   and read from the dedicated read/write agent-state layer for the lifetime of the running host REPL

## Feasibility Assessment

The current `feature/vnet` state is enough to start `v1.2` now.

What is already in place:

- `v1.1` proved the multi-guest VFS mount model, launch-time writable root
  overlay, and TAP-admin SSH path
- `motlie-vnet` Phase 2 is merged, including the usable `VnetBackend` /
  `VnetHandle` API and the `vhost-user-net + libslirp` backend

What follows directly from that:

- Goal 1 is feasible once the guest image can DHCP on a second NIC and the
  launcher can attach a `motlie-vnet` socket
- Goal 2 should remain feasible because `v1.2` keeps the existing TAP-admin
  path instead of changing SSH ingress
- Goal 3 is already aligned with `v1.1`: package installs belong in the
  per-launch writable root overlay, not in VFS-backed trees
- Goals 4-5 are feasible once outbound internet works and the guest has the
  packages needed to install Node/npm and talk to remote auth endpoints
- Goal 6 is feasible if agent state is isolated from the writable root overlay
  and from the general home mount

The one scope change agreed during planning is important:

- do not define success as only `~/.claude` and `~/.codex`
- instead define success as all tool state living under one dedicated read/write
  VFS-backed layer, with home/config paths redirected into that layer

That is why `v1.2` mounts `/agent-state` separately and treats it as the single
dedicated read/write VFS-backed home for:

- `~/.codex`
- `~/.claude`
- `~/.config/claude-code`

The home-directory symlinks are set up by the guest image at boot via the
`motlie-agent-state` systemd service. The backing directories live in the
dedicated VFS mount, not in the writable `/` overlay and not in the general
home mount.

## Host Requirements

`v1.2` does not yet remove the inherited `v1.1` host requirements for the
admin SSH path. The current example still needs:

- `cloud-hypervisor`
- `mmdebstrap`
- `squashfs-tools-ng`
- `e2fsprogs`
- `uidmap`
- `debian-archive-keyring`
- `libfuse3-dev`
- `pkg-config`
- `wget`
- `git`
- `curl`
- Rust toolchain with `cargo`
- `/dev/vhost-vsock` or the ability to load `vhost_vsock`

On Debian/Ubuntu:

```bash
sudo apt install \
  cloud-hypervisor \
  debian-archive-keyring \
  e2fsprogs \
  libfuse3-dev \
  mmdebstrap \
  pkg-config \
  squashfs-tools-ng \
  uidmap \
  wget \
  curl \
  git
```

If `build-guest.sh` runs in the default rootless `MMDEBSTRAP_MODE=unshare`,
the host also needs:

- `newuidmap` and `newgidmap`
- `/etc/subuid` entry for `$USER`
- `/etc/subgid` entry for `$USER`
- user namespaces allowed by the local AppArmor policy

If `/dev/vhost-vsock` is missing, the current guest launch path will not work
until you load the module:

```bash
sudo modprobe vhost_vsock
ls -l /dev/vhost-vsock
```

If that still fails, inspect:

```bash
modinfo vhost_vsock
dmesg | tail -n 50
```

## Current Phase Mapping

This tree now covers the first concrete slice of Phase 3:

- Phase 3.1 guest-image changes:
  - `systemd-networkd` is enabled in the image
  - validation packages (`curl`, `dnsutils`, `ca-certificates`) are present
  - baked image tools now include `sudo`, `python3`, `npm`, `bubblewrap`,
    `@openai/codex`, and `@anthropic-ai/claude-code`
  - the image MOTD shows the image build SHA and UTC build timestamp
  - the `motlie-agent-state` boot service redirects Codex/Claude state into
    `/agent-state`
- Phase 3.2 launcher migration:
  - `launch-ch.sh` now accepts `--admin-net`, `--egress-net`, and `--vnet-socket`
  - supported modes are:
    - `--admin-net=none --egress-net=none`
    - `--admin-net=tap --egress-net=tap`
    - `--admin-net=tap --egress-net=vhost-user`
  - the launcher seeds `/etc/hostname` and `/etc/hosts` into the runtime
    overlay so `sudo` and hostname-based tools work normally
- Phase 3.3/3.4 validation now stays inside the canonical `v1.2` host flow:
  - `repl_host_v1_2` is the only host-side binary entry point
  - `launch-ch.sh` remains the only guest launcher script entry point
  - `repl_host_v1_2` prints the host binary build SHA and UTC build timestamp
    at startup

Validated in this branch:

- TAP-admin SSH ingress still works
- outbound internet works over the `motlie-vnet` egress NIC
- `apt-get` works against the disposable writable `/` overlay
- `codex` and `claude` run in-guest
- Codex and Claude auth/state resolve into `/agent-state`
- `shutdown <guest>` in the REPL is deterministic, with fallback to process
  termination when Cloud Hypervisor API shutdown fails

Still pending:

- Phase 3.5: host-side composed launch flow where the `v1.2` control plane
  starts `motlie-vnet` automatically and wires it into `launch <guest>`

## Mount Contract

Alice:

```yaml
mounts:
  - tag: alice-home
    guest_path: /home/alice
  - tag: alice-agent-state
    guest_path: /agent-state
  - tag: alice-workspace
    guest_path: /workspace
```

Bob:

```yaml
mounts:
  - tag: bob-home
    guest_path: /home/bob
  - tag: bob-agent-state
    guest_path: /agent-state
  - tag: bob-workspace
    guest_path: /workspace
```

The split is deliberate:

- `/home/<user>` stays the user-visible working tree
- `/agent-state` is the dedicated read/write VFS-backed tool-state layer
- `/workspace` remains the separate working/project tree

## Implementation Notes

### Agent-state layout

The combined dedicated read/write layer is mounted at `/agent-state`. In the
current POC, `/home/<user>` is backed by real host directories, while
`/agent-state` is intentionally used as a REPL-overlay-backed tool-state area.
That means:

- normal home-directory behavior should look more like a usual disk-backed home
- Codex / Claude auth and state survive guest relaunch while the same host REPL
  process remains alive
- restarting the host REPL can discard `/agent-state` overlay state in the
  current POC

The setup scripts pre-create these directories per guest:

- `/agent-state/codex`
- `/agent-state/claude`
- `/agent-state/claude-code`

The guest image then redirects:

- `~/.codex -> /agent-state/codex`
- `~/.claude -> /agent-state/claude`
- `~/.config/claude-code -> /agent-state/claude-code`

`CODEX_HOME` and `CODEX_SQLITE_HOME` are also exported from the login profile
so Codex uses the dedicated read/write layer even if its path conventions
evolve.

### Home vs tool-state backing

For the current `v1.2` POC:

- `/home/alice` and `/home/bob` are meant to be host-disk-backed mounts
- home bootstrap files such as `.ssh`, `.env`, `.bashrc`, `.profile`, and
  `.config` are seeded onto that host-backed home path
- `~/.codex`, `~/.claude`, and `~/.config/claude-code` are redirected into the
  dedicated `/agent-state` mount

This split is intentional. It keeps ordinary home-directory behavior closer to
a normal disk-backed Linux home while still isolating agent state into one
separate area for the demo.

### Network model

Short-term `v1.2` keeps the validated `v1.1` ingress path:

- admin NIC: TAP, static `ip=` kernel cmdline, host-reachable SSH

New in `v1.2`:

- egress NIC: `vhost-user-net`, matched in-guest by a stable MAC and
  configured by a guest boot service with the slirp defaults:
  - `10.0.2.15/24` on `eth1`
  - default route via `10.0.2.2`
  - DNS server `10.0.2.3`

The intent is:

- TAP remains the management/SSH path
- `motlie-vnet` owns default route, DNS, and outbound internet

### Why this is safe to build incrementally

The VFS stack and the writable root overlay do not depend on the network mode.
That lets `v1.2` move the egress side first without destabilizing the already
validated `v1.1` SSH/VFS path.

## Validated Runbook

The current branch has been validated with this host flow:

1. run the `v1.2` REPL host:
   - `cargo run -p motlie-vnet --example repl_host_v1_2 -- --empty --script libs/vnet/examples/v1.2/setup-multiguest.sh.vfs --admin-net=tap --egress-net=vhost-user`
2. launch a guest from the REPL:
   - `launch alice`
3. connect over TAP-admin SSH:
   - `ssh alice@192.168.249.2`
4. verify split networking:
   - `ip route`
   - `cat /etc/resolv.conf`
   - `curl -I https://example.com`
   - `nslookup example.com 10.0.2.3`
5. verify baked image tools:
   - `which sudo python3 npm codex claude`
6. verify disposable writable `/` still behaves correctly:
   - `sudo apt-get update`
7. verify agent-state redirection:
   - `ls -ald ~/.codex ~/.claude ~/.config/claude-code`
   - `readlink ~/.codex ~/.claude ~/.config/claude-code`
   - `find /agent-state -maxdepth 4 -type f | sort`
8. verify auth persistence for the lifetime of the running host REPL:
   - sign in to `codex` and `claude`
   - confirm their auth files appear under `/agent-state`
   - `shutdown alice`
   - `launch alice`
   - confirm the auth files remain available

## Known Gaps / TODO

The current `v1.2` POC is intentionally not a production-grade general-purpose
filesystem. The most important known gaps are:

- `/agent-state` persistence is scoped to the lifetime of the running host
  REPL, not durable across host REPL restart
- Cloud Hypervisor graceful shutdown still returns HTTP `500` for this launch
  shape; the REPL relies on a `TERM`/`KILL` fallback
- Codex still warns `could not update PATH: Not supported (os error 95)`
  because `motlie-vfs` does not yet implement the full set of filesystem
  operations that developer tools expect on a normal home directory

The next VFS compatibility work should focus on:

1. `access`
2. file locking
3. xattrs
4. rename / setattr / fsync completeness for more editor and CLI workflows
