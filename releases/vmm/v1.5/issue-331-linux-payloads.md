# Issue 331 Linux Payload Producer Evidence

Recorded by `gpt55-ch-aarch64-258-331-og` on 2026-05-24 PDT and updated
on 2026-05-25 PDT (2026-05-25 UTC) after issue 331 review comments.
The stable code/config validation below was produced from source commit
`f7d5e5b381f9f3b86b2a12ac524ab64b9a50c2d6` on the issue 331 PR branch
that targets `feature/vmm-vz`.

Generated rootfs tarballs, disk images, OCI blobs, and VM images are not
committed. The run root for this local producer pass is:

```text
/tmp/mbuild-331-linux-payloads
```

## Host

```text
Linux spark-2f6e 6.17.0-1018-nvidia #18-Ubuntu SMP PREEMPT_DYNAMIC Tue May 5 21:28:33 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux
```

Installed Rust targets included `aarch64-unknown-linux-musl` and
`x86_64-unknown-linux-musl`. Host-side Alpine staging used
`MOTLIE_MBUILD_APK_STATIC=/tmp/mbuild-tools/apk.static`.

## Comment Follow-up

`immutable_payloads[].source` is now target-aware and authoritative. Release
configs use `target/{rust_target}/release/...` for both immutable guest
payloads, and mbuild resolves that template for every payload instead of
special-casing known guest paths. Executable payload sources must exist before
rootfs assembly and must be ELF64 little-endian binaries whose `e_machine`
matches the configured guest `rust_target`.

The two v1.5 guest binaries are cross-compiled for each configured guest arch:

```text
target/{rust_target}/release/motlie-vfs-guest-v1_5 -> /opt/motlie/v1.5/guest/bin/motlie-vfs-guest
target/{rust_target}/release/motlie-vsock-ssh-bridge-v1_5 -> /opt/motlie/v1.5/guest/bin/motlie-vsock-ssh-bridge
```

Producer scope is split deliberately: guest binary cross-compilation is the
portable Rust step, while full OCI image production still requires a Linux
producer because rootfs assembly and package staging operate on Linux rootfs
contents. macOS/VZ can consume the produced artifacts, but it cannot be the
only full-image producer for this external OCI rootfs path.

## Produced Payloads

### Alpine 3.22, linux/arm64

Status: produced after the target-aware payload source fix, descriptor/blob
readback validated, and dry-run GHCR push plan written with native mbuild OCI
code. No Docker, skopeo, oras, or other registry tooling was used.

Config:

```text
releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml
```

Commands:

```bash
MOTLIE_MBUILD_APK_STATIC=/tmp/mbuild-tools/apk.static \
./target/debug/mbuild build \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild-331-linux-payloads/alpine-arm64-final/ch-src

./target/debug/mbuild validate \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64-final/ch-src \
  --require-executed

./target/debug/mbuild oci export \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64-final/ch-src \
  --out /tmp/mbuild-331-linux-payloads/alpine-arm64-final/oci \
  --tag motlie-guest:v1.5-alpine-arm64-issue331

./target/debug/mbuild oci validate \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64-final/ch-src \
  --layout /tmp/mbuild-331-linux-payloads/alpine-arm64-final/oci

./target/debug/mbuild oci evidence \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64-final/ch-src \
  --layout /tmp/mbuild-331-linux-payloads/alpine-arm64-final/oci \
  --publish-ref ghcr.io/chungers/motlie-guest:v1.5-alpine-arm64-issue331-local

./target/debug/mbuild oci push --dry-run \
  --layout /tmp/mbuild-331-linux-payloads/alpine-arm64-final/oci \
  --image ghcr.io/chungers/motlie-guest:v1.5-alpine-arm64-issue331
```

Evidence files:

```text
/tmp/mbuild-331-linux-payloads/alpine-arm64-final/ch-src/mbuild-manifest.json
/tmp/mbuild-331-linux-payloads/alpine-arm64-final/ch-src/mbuild-rootfs-assembly.json
/tmp/mbuild-331-linux-payloads/alpine-arm64-final/oci/mbuild-oci-export.json
/tmp/mbuild-331-linux-payloads/alpine-arm64-final/oci/mbuild-release-evidence.json
/tmp/mbuild-331-linux-payloads/alpine-arm64-final/oci/mbuild-oci-push.json
```

Digest ledger:

```text
source commit:            f7d5e5b381f9f3b86b2a12ac524ab64b9a50c2d6
source image index:       sha256:310c62b5e7ca5b08167e4384c68db0fd2905dd9c7493756d356e893909057601
source platform manifest: sha256:a46b5c913cad8b1038883ec9aff6003b4a11fdae3229a8e9e3a68f757d724cef
assembled rootfs:         sha256:9ab70cadb544fe2fe2735df8af6119d42e859713fe9867f961f5ac8b7e4d6d4f
assembled rootfs bytes:   897320960
OCI image config:         sha256:68dbe825af9b8621e714d15ef64cde0dccf8bb08ae8659e95ea00d133c7c6cbf
OCI image manifest:       sha256:50b99a4504ddb3e2ea3e8d9bb5eec46665d3f307209dd794b85211eab8c47249
OCI image index/layout:   sha256:1d79ad7c5d124a431c77697215f72998f603c6469482c687f00460cadad55f8c
```

The rootfs assembly manifest records the immutable payload sources as:

```text
/home/dchung/sessions/vmm-vz/motlie/target/aarch64-unknown-linux-musl/release/motlie-vfs-guest-v1_5
/home/dchung/sessions/vmm-vz/motlie/target/aarch64-unknown-linux-musl/release/motlie-vsock-ssh-bridge-v1_5
```

Both are ELF64 little-endian AArch64 executables.

`mbuild oci push --dry-run` prepared a native registry upload plan for
`ghcr.io/chungers/motlie-guest:v1.5-alpine-arm64-issue331` with two blobs and
one image manifest. No registry was contacted.

## Cross-Target Binary Validation

The x86_64 musl guest binary path was validated independently on this aarch64
Linux producer host:

```bash
CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=/home/dchung/.rustup/toolchains/stable-aarch64-unknown-linux-gnu/lib/rustlib/aarch64-unknown-linux-gnu/bin/rust-lld \
cargo build \
  --manifest-path libs/vmm/Cargo.toml \
  --release \
  --target x86_64-unknown-linux-musl \
  --no-default-features \
  --features guest-vfs \
  --bin motlie-vfs-guest-v1_5 \
  --bin motlie-vsock-ssh-bridge-v1_5
```

`file`/`readelf` confirmed both outputs are ELF64 little-endian x86-64:

```text
target/x86_64-unknown-linux-musl/release/motlie-vfs-guest-v1_5: ELF 64-bit LSB pie executable, x86-64, static-pie linked
target/x86_64-unknown-linux-musl/release/motlie-vsock-ssh-bridge-v1_5: ELF 64-bit LSB pie executable, x86-64, static-pie linked
Machine: Advanced Micro Devices X86-64
```

## Host-Gated Payloads

### Alpine 3.22, linux/amd64

Status: full image not produced on this aarch64 host. After the payload source
fix, the build resolved the pinned source digest, imported the linux/amd64
rootfs, cross-compiled and validated the x86_64 guest payloads, and assembled
the rootfs compatibility layer. It then stopped at the expected host execution
preflight:

```text
cross-architecture host-apk contract check requires qemu-x86_64-static; install qemu-user-static or build on native linux/amd64 hardware
```

This payload needs either a native linux/amd64 producer host or this aarch64
host configured with qemu-user-static/binfmt for x86_64.

### Ubuntu 24.04, linux/arm64 and linux/amd64

Status: not produced on this host. Both Ubuntu configs select
`package_stage_strategy=rootless-chroot` and fail preflight before OCI import.
This host has `newuidmap`, `newgidmap`, and subordinate ID ranges for `dchung`,
but AppArmor blocks the rootless namespace probe:

```text
unshare: cannot open /proc/self/setgroups: Permission denied
```

`/proc/sys/kernel/apparmor_restrict_unprivileged_userns` is `1` on this host.
These payloads need a Linux producer host where unprivileged user namespaces
and the rootless chroot preflight pass, or a future non-rootless apt/systemd
package staging strategy.

## Validation Commands

```text
cargo fmt --package mbuild -- --check
cargo check -p mbuild
cargo test -p mbuild
cargo build -p mbuild
git diff --check -- bins/mbuild/src/main.rs releases/vmm/v1.5/configs
```

`cargo test -p mbuild` passed 41/41 tests, including target-aware payload source
rendering, wrong-arch ELF rejection, and unknown payload source placeholder
rejection.

## Remaining Issue 331 Work

The implementation-side blockers from the issue 331 comments are addressed in
mbuild. The remaining producer matrix gaps are host/environment gated:

- Alpine `linux/amd64` full OCI layout on native linux/amd64 or aarch64 with `qemu-x86_64-static`/binfmt.
- Ubuntu `linux/arm64` full OCI layout on a Linux host where rootless user namespaces pass preflight.
- Ubuntu `linux/amd64` full OCI layout on a Linux host with both rootless user namespaces and x86_64 guest execution support, or native linux/amd64.
- A complete multi-arch publish-ready index after the missing layouts exist.
