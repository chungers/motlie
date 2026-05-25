# Issue 331 Linux Payload Producer Evidence

Recorded by `gpt55-ch-aarch64-258-331-og` on 2026-05-24 PDT
(2026-05-25 UTC). Source commit: `2ad8ead08d85c1e4e311d66a9e5bd2e735600fc3`
(`feature/vmm-vz` after PR 280 merge).

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

## Produced Payloads

### Alpine 3.22, linux/arm64

Status: produced, descriptor/blob readback validated, and dry-run push plan
written.

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
  --out /tmp/mbuild-331-linux-payloads/alpine-arm64/ch-src

./target/debug/mbuild validate \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64/ch-src \
  --require-executed

./target/debug/mbuild oci export \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64/ch-src \
  --out /tmp/mbuild-331-linux-payloads/alpine-arm64/oci \
  --tag motlie-guest:v1.5-alpine-arm64-issue331

./target/debug/mbuild oci validate \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64/ch-src \
  --layout /tmp/mbuild-331-linux-payloads/alpine-arm64/oci

./target/debug/mbuild oci evidence \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --artifact /tmp/mbuild-331-linux-payloads/alpine-arm64/ch-src \
  --layout /tmp/mbuild-331-linux-payloads/alpine-arm64/oci \
  --publish-ref ghcr.io/chungers/motlie-guest:v1.5-alpine-arm64-issue331-local

./target/debug/mbuild oci push --dry-run \
  --layout /tmp/mbuild-331-linux-payloads/alpine-arm64/oci \
  --image ghcr.io/chungers/motlie-guest:v1.5-alpine-arm64-issue331
```

Evidence files:

```text
/tmp/mbuild-331-linux-payloads/alpine-arm64/ch-src/mbuild-manifest.json
/tmp/mbuild-331-linux-payloads/alpine-arm64/oci/mbuild-oci-export.json
/tmp/mbuild-331-linux-payloads/alpine-arm64/oci/mbuild-release-evidence.json
/tmp/mbuild-331-linux-payloads/alpine-arm64/oci/mbuild-oci-push.json
```

Digest ledger:

```text
source image index:       sha256:310c62b5e7ca5b08167e4384c68db0fd2905dd9c7493756d356e893909057601
source platform manifest: sha256:a46b5c913cad8b1038883ec9aff6003b4a11fdae3229a8e9e3a68f757d724cef
assembled rootfs:         sha256:4fd938cd684224b2180f52690f978e4a3f2da3856477105678ea5262acd0847c
assembled rootfs bytes:   897320960
OCI image config:         sha256:add9130169f78b0c9144cd7acd37b56a75dfdef691b770ea7cc4ffe350687eb3
OCI image manifest:       sha256:cacc4704276b562d9bcd72e4423e0360b002246bf1acaecffbe32c4580ae74d6
OCI image index/layout:   sha256:fe0f8adce4e7891f90e1393dc180a8822e4fb2ee4de455e0646d52d17210d3dc
```

`mbuild oci push --dry-run` prepared a native registry upload plan for
`ghcr.io/chungers/motlie-guest:v1.5-alpine-arm64-issue331` with two blobs and
one image manifest. No registry was contacted.

## Host-Gated Payloads

### Alpine 3.22, linux/amd64

Status: not produced on this host. The build resolved the pinned source digest
and imported the rootfs, then stopped before package staging with:

```text
cross-architecture host-apk contract check requires qemu-x86_64-static; install qemu-user-static or build on native linux/amd64 hardware
```

This payload needs either a native linux/amd64 producer host or this aarch64
host configured with qemu-user-static/binfmt for x86_64.

### Ubuntu 24.04, linux/arm64 and linux/amd64

Status: not produced on this host. Both Ubuntu configs select
`package_stage_strategy=rootless-chroot` and fail PR 280 preflight before OCI
import. This host has `newuidmap`, `newgidmap`, and subordinate ID ranges for
`dchung`, but AppArmor blocks the rootless namespace probe:

```text
unshare: cannot open /proc/self/setgroups: Permission denied
```

`/proc/sys/kernel/apparmor_restrict_unprivileged_userns` is `1` on this host.
These payloads need a Linux producer host where unprivileged user namespaces
and the rootless chroot preflight pass, or a future non-rootless apt/systemd
package staging strategy.

## Remaining Issue 331 Work

The final issue 331 producer set still needs:

- Alpine `linux/amd64` local OCI layout and readback validation.
- Ubuntu `linux/arm64` local OCI layout and readback validation.
- Ubuntu `linux/amd64` local OCI layout and readback validation.
- A complete multi-arch publish-ready index after the missing layouts exist.

