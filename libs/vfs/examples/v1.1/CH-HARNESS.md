# v1.1 Cloud Hypervisor Harness

`v1.1` demos the intended topology from the DESIGN:

- one `FsServer` per guest VM
- multiple mount tags inside each server
- one vsock socket per guest VM
- one guest-side FUSE mount thread per tag

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

## Build

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh --guest alice
./build-guest.sh --guest bob
```

Artifacts land in:

- `artifacts/alice/`
- `artifacts/bob/`

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

## Shut Down

```bash
curl --unix-socket /tmp/motlie-vfs-alice-api.sock -X PUT http://localhost/api/v1/vm.shutdown
curl --unix-socket /tmp/motlie-vfs-bob-api.sock -X PUT http://localhost/api/v1/vm.shutdown
```
