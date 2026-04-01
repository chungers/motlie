# v1.1 Cloud Hypervisor Harness

`v1.1` demos the intended topology from the DESIGN:

- one `FsServer` per guest VM
- multiple mount tags inside each server
- one vsock socket per guest VM
- one guest-side FUSE mount thread per tag

## What Differs Vs v1

- `v1` runs one guest-oriented server flow at a time; `v1.1` is meant to run Alice and Bob concurrently
- `v1` uses the legacy single-tag `--tag` / `--dir` server startup path; `v1.1` uses repeated `--mount <tag>=<dir>`
- `v1` typically mounts one guest path; `v1.1` mounts two tags per guest in the demo
- `v1` is single-socket, single-CID, single-artifact-tree; `v1.1` uses separate sockets, CIDs, and artifact trees per guest
- `v1.1` relies on the guest-side `TAG <name>` handshake so one server socket can route multiple tag connections for the same guest

## Topology

```text
Host                                          Guest VMs
----                                          ---------
repl_host (alice server)                      alice VM
  socket /tmp/motlie-vfs-alice.vsock_5000       tags: alice-home, alice-workspace
  mounts:
    alice-home -> /tmp/.../alice-home         repl_host (bob server)
    alice-workspace -> /tmp/.../alice-workspace

repl_host (bob server)                        bob VM
  socket /tmp/motlie-vfs-bob.vsock_5000         tags: bob-home, bob-workspace
  mounts:
    bob-home -> /tmp/.../bob-home
    bob-workspace -> /tmp/.../bob-workspace
```

Each guest mount connection goes to the guest's one socket and immediately sends `TAG <name>\n`. `repl_host` uses that handshake to route the stream to the right tag in its `FsServer`.

## Prerequisites

Before following this harness:

- build both `v1.1` guest image sets
- ensure `/dev/vhost-vsock` exists or load it with `sudo modprobe vhost_vsock`
- use a shell that can successfully run the rootless `build-guest.sh` flow if you are rebuilding images
- ensure `cloud-hypervisor` is installed and on `PATH`

## Build

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh --guest alice
./build-guest.sh --guest bob
```

Artifacts land in:

- `artifacts/alice/`
- `artifacts/bob/`

You do not need a prior `v1` build. This harness uses only `v1.1` artifacts.

## Start Host Servers

Alice server:

```bash
cat setup-alice.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- \
  --socket /tmp/motlie-vfs-alice.vsock_5000 \
  --mount alice-home=/tmp/motlie-vfs-demo/alice-home \
  --mount alice-workspace=/tmp/motlie-vfs-demo/alice-workspace
```

Bob server:

```bash
cat setup-bob.sh.vfs | \
  cargo run -p motlie-vfs --example repl_host --features vsock -- \
  --socket /tmp/motlie-vfs-bob.vsock_5000 \
  --mount bob-home=/tmp/motlie-vfs-demo/bob-home \
  --mount bob-workspace=/tmp/motlie-vfs-demo/bob-workspace
```

`repl_host` still supports the old `--tag` / `--dir` path for `v1`, but `v1.1` uses repeated `--mount`.

In this demo:

- Alice server owns `alice-home` and `alice-workspace`
- Bob server owns `bob-home` and `bob-workspace`
- each `repl_host` process serves exactly one guest VM
- each guest VM opens one connection per mount tag back to its own server socket

## Launch Guests

Minimal concurrent demo:

```bash
./launch-ch.sh --guest alice --no-net
./launch-ch.sh --guest bob --no-net
```

With TAP networking enabled:

```bash
./launch-ch.sh --guest alice
./launch-ch.sh --guest bob
```

Defaults:

| Guest | CID | vsock socket | API socket | Guest IP |
|------|-----|--------------|------------|----------|
| alice | 3 | `/tmp/motlie-vfs-alice.vsock` | `/tmp/motlie-vfs-alice-api.sock` | `192.168.249.2` |
| bob | 4 | `/tmp/motlie-vfs-bob.vsock` | `/tmp/motlie-vfs-bob-api.sock` | `192.168.250.2` |

## Manual Run Order

Use four terminals for the clearest flow:

1. Terminal 1: start Alice `repl_host`
2. Terminal 2: start Bob `repl_host`
3. Terminal 3: launch Alice guest
4. Terminal 4: launch Bob guest

If you want SSH access for both guests, omit `--no-net` on both launch commands and connect to the guest IPs shown above.

## Validate

Alice VM:

```bash
mount | grep -E '/home/alice|/workspace'
ls -la /home/alice/.ssh
cat /home/alice/.env
cat /workspace/README.md
```

Bob VM:

```bash
mount | grep -E '/home/bob|/workspace'
ls -la /home/bob/.ssh
cat /home/bob/.env
cat /workspace/README.md
```

Expected shape:

- `/home/alice` is backed by the `alice-home` tag
- `/home/bob` is backed by the `bob-home` tag
- both guests mount `/workspace`, but each guest gets its own tag and own host backing directory
- `.env` and `.ssh` contents come from the in-memory overlay injected by the corresponding `setup-*.sh.vfs` file

## Shut Down

```bash
curl --unix-socket /tmp/motlie-vfs-alice-api.sock -X PUT http://localhost/api/v1/vm.shutdown
curl --unix-socket /tmp/motlie-vfs-bob-api.sock -X PUT http://localhost/api/v1/vm.shutdown
```

## Standalone Expectation

This document is the `v1.1` runtime runbook. It does not require a `v1` harness run first. The only shared pieces are the repository binaries and libraries that both examples build from the same workspace.
