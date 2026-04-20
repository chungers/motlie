# v1.2 Cloud Hypervisor Harness

`v1.2` is the split-network successor to `v1.1`:

- TAP remains the short-term admin/SSH path
- `motlie-vnet` is the target outbound internet path
- `/agent-state` is a dedicated read/write VFS-backed layer for Codex + Claude state

This harness document is intentionally short until the composed host flow in
Phase 3.5 is validated.

Current expectations:

- build the guest image with [build-guest.sh](./build-guest.sh)
- use [launch-ch.sh](./launch-ch.sh) for direct launches
- use [repl_host.rs](./repl_host.rs) via `cargo run -p motlie-vnet --example repl_host_v1_2 -- ...` as the only host-side runtime entry point

## Host Preflight

`v1.2` currently inherits the `v1.1` host requirements for the admin/TAP/vsock
path and adds the userspace `motlie-vnet` egress backend in-process. Before
trying the example, make sure the host has:

### Required host packages

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

What they are used for:

- `cloud-hypervisor`: guest launch
- `debian-archive-keyring`: `mmdebstrap` Debian keyring
- `e2fsprogs`: `mkfs.ext4` for the writable runtime overlay
- `libfuse3-dev`, `pkg-config`: build `motlie-vfs` guest/client pieces
- `mmdebstrap`, `squashfs-tools-ng`, `uidmap`: guest image build
- `wget`, `git`: kernel download/build paths in `build-guest.sh`
- `curl`: host-side guest shutdown path and general validation

### Required host runtime state

- working Rust toolchain with `cargo`
- `/dev/vhost-vsock` available, or ability to load `vhost_vsock`
- ability to create/use the admin TAP path used by the inherited `v1.1` SSH flow
- outbound network access on the host for:
  - Debian package download during image build
  - CH kernel download during image build
  - guest outbound internet validation later

### Unattended preflight checklist

These checks should all pass before the example is expected to work:

```bash
command -v cargo rustc cloud-hypervisor mmdebstrap tar2sqfs mkfs.ext4 pkg-config wget curl git ssh newuidmap newgidmap
test -f /usr/share/keyrings/debian-archive-keyring.gpg
test -e /dev/vhost-vsock
```

If `MMDEBSTRAP_MODE=unshare` is used, also check:

```bash
getent passwd "$USER"
grep "^$USER:" /etc/subuid
grep "^$USER:" /etc/subgid
cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns
```

Interpretation:

- missing `newuidmap` / `newgidmap`, `/etc/subuid`, or `/etc/subgid` breaks
  rootless `mmdebstrap --mode=unshare`
- missing `/dev/vhost-vsock` breaks the current guest vsock launch path unless
  the module can be loaded
- missing host outbound internet breaks guest-image build and later guest egress
  validation

### If `/dev/vhost-vsock` is missing

Load the module and verify the device:

```bash
sudo modprobe vhost_vsock
ls -l /dev/vhost-vsock
```

If the device still does not appear, inspect:

```bash
modinfo vhost_vsock
dmesg | tail -n 50
```

If you want the module loaded automatically on boot:

```bash
echo vhost_vsock | sudo tee /etc/modules-load.d/vhost_vsock.conf
```

Supported launcher modes today:

- `--admin-net=none --egress-net=none`
- `--admin-net=tap --egress-net=tap`
- `--admin-net=tap --egress-net=vhost-user`

The detailed success criteria and remaining gaps live in [README.md](./README.md).
