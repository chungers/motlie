# `tests/vmm/` — Golden Behavior Suite

Slice-agnostic, harness-agnostic Linux VM behavior tests for the `motlie-vmm`
layer. They assert **invariants the vmm contract must uphold** — not invariants
of any specific `v1.x` example or any specific runtime backend.

## Why these live here (not under `libs/vmm/examples/v1.x/`)

Each `v1.x` slice (CH, Apple Vz, future backends) currently re-creates its own
ad-hoc `verify-*.sh` scripts. That pattern produces drift and lets the same
regression slip through different slices. Centralizing the assertions in
`tests/vmm/`:

- Pins the **spec** of "what working means" once
- Lets every slice (`v1.2`, `v1.25`, `v1.3`, `v1.35`, ...) gate against the
  same suite
- Catches drift the moment a slice diverges from the contract
- Decouples the validation gate from whichever harness shape (repl_host,
  vsock, ssh forward) happens to be in flight

## Harness assumptions (intentional minimum)

The tests do **not** depend on `repl_host`, `vz-vsock-runner`, vsock,
host-forward TCP ports, or any motlie Rust binary inside the guest. The only
things the tests assume:

- A way to execute a single bash command inside a named guest. `driver.sh`
  abstracts this via the `--exec-cmd` template (replace literal `GUEST` token).
- Two guest user accounts named `alice` and `bob` (motlie convention).
- Mountpoints at `/workspace` and `/home/<user>`.
- Standard Debian/Ubuntu base with: `bash`, `curl`, `getent`, `sed`, `stat`,
  `findmnt`, `setfattr`, `socat`, `sudo`, `apt`.
- Internet egress configured *somehow* (NAT, slirp, file-handle helper — the
  test doesn't care which).

## Layout

```
tests/vmm/
├── README.md          # this file
├── driver.sh          # host-side dispatcher; harness-agnostic via --exec-cmd
├── shared/
│   └── result.sh      # `pass`/`fail`/`skip` contract
├── vnet/              # networking behavior
│   ├── vnet-01-dns-resolves.sh
│   ├── vnet-02-https-egress.sh
│   ├── vnet-03-apt-update-timing.sh
│   ├── vnet-04-multi-guest-egress.sh
│   ├── vnet-05-egress-backpressure.sh
│   ├── vnet-06-dns-latency.sh
│   ├── vnet-07-cloud-endpoints.sh
│   ├── vnet-08-mtu-large-transfer.sh
│   ├── vnet-09-concurrent-egress.sh
│   └── vnet-10-egress-stability.sh
└── vfs/               # filesystem behavior
    ├── vfs-01-mounts-present.sh
    ├── vfs-02-atomic-save-attrs.sh
    ├── vfs-03-open-handle-survives-rename.sh
    ├── vfs-04-xattr-contract.sh
    ├── vfs-05-per-guest-isolation.sh
    ├── vfs-06-sparse-file.sh
    ├── vfs-07-hardlink.sh
    ├── vfs-08-symlink.sh
    ├── vfs-09-concurrent-writes.sh
    └── vfs-10-mtime-preserved.sh
```

## Output contract

Every test script prints **exactly one** result line on stdout:

```
TEST=<name> RESULT=pass|fail|skip [DETAIL=<short message>]
```

and exits `0` on `pass`/`skip`, `1` on `fail`. The driver parses this and
writes a JSON summary at `tests/vmm/results/results.json`.

`skip` is for tests whose precondition isn't met on the current guest (e.g.
`socat` not installed). Skips are tallied separately and don't fail the suite.

## Running the suite

The driver takes an `--exec-cmd` template; the literal token `GUEST` is
replaced with each guest name in turn, and the test script body is piped on
stdin via `bash -s`.

### Example: ssh into a host-forwarded guest

```bash
tests/vmm/driver.sh \
  --guests alice,bob \
  --suite vnet,vfs \
  --exec-cmd 'ssh -p 2226 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null GUEST@127.0.0.1 -- bash -s'
```

### Example: tart exec into a Tart-managed VM

```bash
tests/vmm/driver.sh \
  --exec-cmd 'tart exec motlie-v1-35-GUEST-iter -- bash -s'
```

### Example: future repl_host exec

When the repl_host stabilizes a `--exec-into-guest` mode, only `--exec-cmd`
changes. Test files don't.

## Slice integration

Each `libs/vmm/examples/v1.x/launch-vz.sh` (or its CH equivalent) should call
the driver as the final validation step and persist results next to the slice
artifacts:

```bash
tests/vmm/driver.sh \
  --guests alice,bob --suite vnet,vfs \
  --exec-cmd "$EXEC_CMD" \
  --out-dir libs/vmm/examples/v1.x/artifacts
```

A slice is considered green when the driver exits `0` and `results.json`
contains `"result":"fail"` zero times.

## Adding a test

1. Create `vnet/vnet-NN-<short-name>.sh` or `vfs/vfs-NN-<short-name>.sh`.
2. Number sequentially within the suite. Don't reuse numbers.
3. Source the result helper and set `TEST_NAME`:
   ```bash
   #!/usr/bin/env bash
   set -u
   TEST_NAME=vnet-11-my-new-test
   . "$(dirname "$0")/../shared/result.sh"
   # ... your check ...
   pass "detail-on-success"   # or fail "why-it-broke"  or skip "missing-tool"
   ```
4. Make it idempotent and self-cleaning. No external state may persist.
5. `chmod +x` the new script.
6. Update this README's layout table.

## Test inventory

### `vnet/`

| # | name | what it asserts |
|---|---|---|
| 01 | dns-resolves | `getent` returns a non-loopback IPv4 for a real hostname |
| 02 | https-egress | TLS egress reaches a known sentinel and gets the expected body |
| 03 | apt-update-timing | `apt-get update` completes in `< 60s` (catches egress slowness) |
| 04 | multi-guest-egress | the running guest can hit a public IP-echo endpoint |
| 05 | egress-backpressure | sustained UDP queries: response rate ≥ 90% (positive evidence; catches silent helper drops) |
| 06 | dns-latency | first-resolution round-trip `< 2s` (catches DNS forwarder hangs) |
| 07 | cloud-endpoints | `github.com`, `registry.npmjs.org`, `pypi.org` all reachable |
| 08 | mtu-large-transfer | curl 5 MB blob, sha256 matches a control hash |
| 09 | concurrent-egress | 10 parallel HTTPS requests all succeed |
| 10 | egress-stability | 5 sequential requests succeed with no latency growth |

### `vfs/`

| # | name | what it asserts |
|---|---|---|
| 01 | mounts-present | `/workspace` + `/home/<user>` mounted and writable |
| 02 | atomic-save-attrs | `sed -i` (write-tmp + rename) preserves uid/gid/mode |
| 03 | open-handle-survives-rename | open fd stays valid across rename of the underlying file |
| 04 | xattr-contract | capability-aware: passes if xattrs work end-to-end **OR** are cleanly unsupported (`EOPNOTSUPP`/`EPERM`) |
| 05 | per-guest-isolation | peer guest's `/home/<peer>` is absent or unlistable from this guest (uses `id -un`, no `$USER` dependency) |
| 06 | sparse-file | `truncate`+offset write keeps holes; size != allocated |
| 07 | hardlink | hardlink yields shared inode; both paths resolve to same data |
| 08 | symlink | symlink resolves correctly; readlink returns expected target |
| 09 | concurrent-writes | two appender processes don't interleave-corrupt small lines |
| 10 | mtime-preserved | `touch -m -d` sets mtime correctly; `stat -c %Y` matches |
