# Motlie VMM v1.5 Release Inputs

This directory stores checked-in release inputs and worked-example notes for
the v1.5 VMM image flow. Generated VM images, rootfs tarballs, OCI blobs, and
harness logs are intentionally not checked in here.

## Configs

The current worked example is Ubuntu 24.04 with the `ubuntu-systemd` rootfs
profile:

```text
configs/motlie-image.ubuntu-24.04.linux-arm64.yaml
configs/motlie-image.ubuntu-24.04.linux-amd64.yaml
configs/motlie-image.ubuntu-24.04.default-arm64.yaml
```

The explicit per-platform files are the release-facing inputs. The
`default-arm64` file preserves the former v1.5 example default for traceability;
new build and release evidence should use the explicit platform config.

To add another rootfs, add a config named:

```text
configs/motlie-image.<rootfs-id>.<platform>.yaml
```

Update at least:

- `source.image`, `source.profile`, `source.platform`, and pinned OCI digests.
- `package_stage.manager`, packages, and npm globals.
- `immutable_payloads`, `services`, and `immutable_files` if the rootfs has a
  different init or filesystem layout.
- `emitters` only when backend adapter contracts change.

The rootfs profile must be implemented by the classifier/assembler before a
new config is accepted as a working image contract.

## Worked Example Artifacts

A full v1.5 run produces these artifact groups under an operator-selected run
root such as `/tmp/mbuild/v1.5-ubuntu-arm64`:

```text
ch-src/
  mbuild-manifest.json
  mbuild-common-rootfs.json
  mbuild-rootfs-assembly.json
  mbuild-oci-import.json
  mbuild-ch-emitter.log
  assembled-rootfs.tar
  base/Image
  base/rootfs.squashfs
  base/guest-contract.json
  ch-build-result.json

oci-arm64/
  oci-layout
  index.json
  blobs/sha256/<config-digest>
  blobs/sha256/<manifest-digest>
  blobs/sha256/<rootfs-layer-digest>
  mbuild-oci-export.json

ch-from-oci/
  mbuild-manifest.json
  mbuild-common-rootfs.json
  mbuild-validation-manifest.json
  mbuild-validation.log
  mbuild-release-evidence.json
  base/Image
  base/rootfs.squashfs
  base/guest-contract.json
  ch-build-result.json

oci-index/
  oci-layout
  index.json
  blobs/sha256/<per-platform-manifest-digests>
  mbuild-oci-index.json
```

The generated JSON files are the machine-readable contract for CI, release
coordination, and future agents. Large binary artifacts should live in the run
root, registry, or release artifact store, with only evidence JSON copied into
release coordination if needed.

## Commands

Arm64 CH source build from the pinned Ubuntu OCI rootfs:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/v1.5-ubuntu-arm64/ch-src
```

Export the common rootfs to a local OCI layout:

```bash
cargo run -p mbuild -- oci export \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/v1.5-ubuntu-arm64/ch-src \
  --out /tmp/mbuild/v1.5-ubuntu-arm64/oci-arm64 \
  --tag motlie-guest:v1.5-arm64
```

Build CH from that OCI payload:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/v1.5-ubuntu-arm64/ch-from-oci \
  --oci-layout /tmp/mbuild/v1.5-ubuntu-arm64/oci-arm64
```

Build VZ from the same OCI payload on Apple Silicon macOS:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target vz \
  --out /tmp/mbuild/v1.5-ubuntu-arm64/vz-from-oci \
  --oci-layout /tmp/mbuild/v1.5-ubuntu-arm64/oci-arm64
```

Validate a CH artifact with the v1.5 harness:

```bash
cargo run -p mbuild -- validate \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/v1.5-ubuntu-arm64/ch-from-oci \
  --require-executed \
  --scenario libs/vmm/examples/v1.5/scenarios/multiguest-validate.json
```
