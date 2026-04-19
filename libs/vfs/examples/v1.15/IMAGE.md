# v1.15 Guest Image Notes

`build-guest.sh` in `v1.15` builds one generic Apple Vz base guest.

Like `v1.1`, the artifact boundary is:

1. one shared base image
2. one fresh per-guest runtime clone per launch

Guest identity is not baked into the shared base image.

## Artifact Contract

### Build-Time Shared Base

The shared base is common to every guest:

- Ubuntu Tart base disk
- common package set
- shared system libraries
- common build prerequisites needed to compile `motlie-vfs-guest-v1_15`

### Launch-Time Per-Guest Runtime Clone

Each launch creates a fresh per-guest clone:

- one cloned `disk.img` per guest run
- guest-specific user and hostname written after native boot
- guest-specific `mounts.yaml`
- guest-local package or filesystem mutations discarded on the next fresh run

That keeps Alice and Bob isolated from one another and matches the high-level
`v1.1` contract even though the VMM implementation is different.

## What The Builder Does

`build-guest.sh` currently produces a reusable Tart-managed base VM:

- `motlie-v1-15-base`

The builder intentionally keeps the base generic:

- no Alice/Bob identity
- no guest-specific mount config
- no guest-scoped validation state

Those guest-specific changes happen at launch time after the native Vz boot.

## Runtime Guest State

At launch time, `launch-vz.sh`:

1. clones the generic base VM into a fresh per-run guest VM
2. boots that guest through the signed Apple helper
3. discovers the NAT IP
4. provisions the live guest over SSH:
   - user/home setup
   - `motlie-vfs-guest-v1_15` build/install
   - `mounts.yaml`
   - systemd units
5. validates the mounted guest view
6. tears the run down by default

This is heavier than the CH `v1.1` image flow, but it preserves the same
logical contract:

- shared base
- guest-specific runtime state
- disposable per-run writable state

## What Tart Still Does

Tart is retained only as base-image scaffolding:

- source of the reusable Ubuntu guest disk
- VM clone/container for the generic base artifact

Tart is not the runtime VMM path for `v1.15`:

- the runtime launch goes through the signed Apple helper
- guestfs transport goes through Apple virtio-socket
- mounted-view validation happens on the native Apple Vz path

## Intended Cleanup Contract

Fresh per-run clones are the default and the intended operator path.

The goal is to leave no lingering host traces after a successful run:

- no per-run VM clone
- no lingering runner process
- no stale pid/socket/log state interfering with the next launch

That cleanup expectation is part of the VMM-host contract, not just local
script hygiene.
