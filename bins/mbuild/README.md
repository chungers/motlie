# mbuild

`mbuild` is the top-level Motlie v1.5 image builder CLI. It consumes the
Dockerfile-like image contract used by the v1.5 examples, drives the current
backend image adapters, regenerates per-guest seed artifacts, and emits
machine-readable manifests for CI and harness consumption.

Status as of 2026-05-18 (`@vmm-cdx`): `mbuild build --target ch` is the
durable Linux/CH image-builder entrypoint for the v1.5 demo. It resolves the
pinned Ubuntu or Alpine OCI source, imports rootfs layers, runs the selected
apt/npm or apk/npm package stage,
applies the native v1.5 Motlie compatibility layer, emits the common
`assembled-rootfs.tar` handoff before CH-specific boot adaptations, and emits CH
artifacts plus machine-readable manifests. The CH path no longer assumes the
guest platform is the same as the builder host platform: `source.platform`
selects the guest architecture, guest binaries are built as static musl
payloads with `rust-lld`, and cross-arch package staging requires qemu-user
binfmt or a native builder for that guest architecture. `mbuild seed`
regenerates per-guest seed overlays without rebuilding the immutable image.
Checked-in worked-example configs now live under `releases/vmm/v1.5/configs/`.
`mbuild validate` validates manifests, can require execution evidence, and can
delegate live guest conformance to the v1.5 harness. `mbuild oci export`
converts an executed artifact's `assembled-rootfs.tar` into a local OCI image
layout for issue #258 per-arch payload work. Adapter-backed targets such as VZ
can consume that local OCI layout with `--oci-layout`; `mbuild` validates the
layout and passes the canonical rootfs layer through the current adapter rootfs
handoff. The durable Apple VZ boot-container emitter is still transitional.
v1.5 is greenfield for this product contract: pre-v1.5 VZ source VMs/cached
disks are unsupported, and per-guest users are seed/runtime state rather than
image content.

## Commands

Build CH artifacts through the native external-OCI CH path:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/ch
```

Priority #258 targets are Apple Silicon VZ (`vz-darwin-arm64`) and
DGX/aarch64 Linux CH (`ch-linux-arm64`) first. Both consume/build the
`linux/arm64` guest payload. CH Linux amd64 (`ch-linux-amd64`) follows on a
coordinated x86_64/amd64 Linux host and consumes/builds the `linux/amd64`
guest payload.

Build CH artifacts for a non-native guest platform by setting
`source.platform` in the image config to the desired OCI platform. For example,
an arm64 Linux builder can build a `linux/amd64` CH guest when the host has:

- Rust target `x86_64-unknown-linux-musl`
- toolchain `rust-lld`
- `qemu-x86_64-static`
- enabled `/proc/sys/fs/binfmt_misc/qemu-x86_64`

The matching arm64 guest requirements on an amd64 Linux builder are:

- Rust target `aarch64-unknown-linux-musl`
- toolchain `rust-lld`
- `qemu-aarch64-static`
- enabled `/proc/sys/fs/binfmt_misc/qemu-aarch64`

Without qemu-user/binfmt, run the per-arch CH build on native hardware for that
guest platform. `mbuild` fails before OCI layer import if cross-arch package
staging cannot execute guest rootfs binaries.

### CH Package Stage Modes

2026-05-22, gpt55-ch-aarch64-258=280 -- CH external-OCI builds run
package-manager and npm installation inside the imported guest rootfs. By
default, `mbuild build` uses `--package-stage-mode auto`, which probes the
rootless user/mount namespace path and fails early with operator guidance when
the host blocks rootless chroot or bind mounts. It does not silently escalate.

Use explicit rootless mode on hosts known to support unprivileged user
namespaces, subordinate uid/gid maps, mount namespaces, bind mounts, and chroot:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/ch-alpine-arm64 \
  --package-stage-mode rootless
```

Use sudo mode on CH builder hosts that permit KVM/CH work but restrict rootless
mount or chroot operations. `mbuild` remains user-owned and runs only the
rootfs mutation/readback subprocesses through `sudo -n`; if passwordless sudo is
not already authorized for the operator, the build fails before package staging:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/ch-alpine-arm64 \
  --package-stage-mode sudo
```

`--package-stage-mode root` is for controlled CI/container builders where the
entire `mbuild` process is already uid 0. Operators should prefer sudo mode over
running `sudo mbuild` so Cargo caches, OCI caches, manifests, and most artifact
writes stay owned by the invoking user.

The same value can be supplied with `MOTLIE_MBUILD_PACKAGE_STAGE_MODE` when a
builder host has a fixed policy.

Build VZ artifacts through the current VZ adapter on an Apple Silicon macOS
host. The current VZ path consumes a local `linux/arm64` OCI layout and passes
its validated rootfs layer through the adapter rootfs handoff:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/ch

cargo run -p mbuild -- oci export \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/ch \
  --out /tmp/mbuild/oci-arm64 \
  --tag motlie-guest:v1.5-arm64

cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target vz \
  --out /tmp/mbuild/vz \
  --oci-layout /tmp/mbuild/oci-arm64
```

The first command may run on Linux/DGX to produce the arm64 common rootfs
handoff and OCI layout. The VZ build command must run on the macOS VZ builder.
`--rootfs-tarball` remains available as an explicit low-level adapter handoff,
but the issue #258 path should prefer `--oci-layout` so CH and VZ consume the
same OCI payload contract.

Plan without running backend adapters:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/plan/ch \
  --plan-only
```

Plan Alpine configs without running backend adapters:

```bash
cargo run -p mbuild -- build --plan-only \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/plan/alpine-arm64

cargo run -p mbuild -- build --plan-only \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-amd64.yaml \
  --target ch \
  --out /tmp/mbuild/plan/alpine-amd64
```

Build Alpine arm64 CH artifacts from the pinned Alpine OCI rootfs:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.alpine-3.22.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/ch-alpine-arm64
```

Regenerate per-guest seed artifacts:

```bash
cargo run -p mbuild -- seed \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --guest alice \
  --uid 2001 \
  --gid 2001 \
  --out /tmp/mbuild/seed/alice
```

Validate an emitted build manifest:

```bash
cargo run -p mbuild -- validate \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/ch \
  --require-executed
```

Delegate live conformance to the v1.5 harness and write a validation record:

```bash
cargo run -p mbuild -- validate \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/ch \
  --require-executed \
  --scenario libs/vmm/examples/v1.5/scenarios/multiguest-validate.json
```

Export the assembled rootfs handoff as a local OCI image layout:

```bash
cargo run -p mbuild -- oci export \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/ch \
  --out /tmp/mbuild/oci-arm64 \
  --tag motlie-guest:v1.5-arm64
```

Validate that the exported layout still matches the build config and artifact:

```bash
cargo run -p mbuild -- oci validate \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/ch \
  --layout /tmp/mbuild/oci-arm64
```

Consume a validated local OCI payload as the CH emitter input:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild/ch-from-oci \
  --oci-layout /tmp/mbuild/oci-arm64
```

Create a local multi-arch OCI image index after both per-platform payloads
exist:

```bash
cargo run -p mbuild -- oci index \
  --out /tmp/mbuild/oci-index \
  --image ghcr.io/chungers/motlie-guest:v1.5 \
  --layout /tmp/mbuild/oci-amd64 \
  --layout /tmp/mbuild/oci-arm64
```

Emit release-manifest-ready evidence for a VM image artifact target:

```bash
cargo run -p mbuild -- oci evidence \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild/ch \
  --layout /tmp/mbuild/oci-arm64 \
  --publish-ref ghcr.io/chungers/motlie-guest:v1.5-arm64
```

Resolve registry pins with the same OCI client used by the builder:

```bash
cargo run -p mbuild -- oci resolve \
  --image docker.io/library/ubuntu:24.04 \
  --platform linux/amd64
```

The build command writes:

```text
<out>/mbuild-manifest.json
```

For the native CH external-OCI path, the build also writes the cross-backend
rootfs handoff before applying CH boot adaptations:

```text
<out>/assembled-rootfs.tar
<out>/mbuild-common-rootfs.json
```

The seed command writes:

```text
<out>/mbuild-seed-manifest.json
```

When `validate --scenario` is used, validation also writes:

```text
<artifact>/mbuild-validation-manifest.json
<artifact>/mbuild-validation.log
```

When `oci export` is used, the output directory contains:

```text
<out>/oci-layout
<out>/index.json
<out>/blobs/sha256/<config-digest>
<out>/blobs/sha256/<manifest-digest>
<out>/blobs/sha256/<rootfs-layer-digest>
<out>/mbuild-oci-export.json
```

`mbuild oci validate` reads the layout back and verifies blob digests, blob
sizes, `index.json` platform annotations, image-manifest config/layer
descriptors, image-config platform fields, and the rootfs diff ID. It also
rejects stale layouts when the source digest, contract version, selected
platform, or input rootfs evidence no longer matches the current build config
and artifact manifest.

`mbuild oci index` writes a local OCI layout containing one multi-arch
`index.json` assembled from validated per-platform mbuild layouts. Registry
push remains an operator/release step that requires credentials and registry
policy; the checked-in builder produces the immutable local layouts, multi-arch
index, and release evidence consumed by that step.

Linux/CH validation evidence from 2026-05-14 (`@vmm-cdx`) used:

```bash
cargo run -p mbuild -- build \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --target ch \
  --out /tmp/mbuild-pr270-oci-ch-7

cargo run -p mbuild -- validate \
  --config releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml \
  --artifact /tmp/mbuild-pr270-oci-ch-7 \
  --require-executed
```

Then the same artifact passed these CH scenarios:

```text
libs/vmm/examples/v1.5/scenarios/multiguest-validate.json
libs/vmm/examples/v1.5/scenarios/auto-provision-ssh.json
libs/vmm/examples/v1.5/scenarios/agent-bootstrap.json
libs/vmm/examples/v1.5/scenarios/pty-agent-validation.json
libs/vmm/examples/v1.5/scenarios/pty-login.json
```

## Image Config Contract

The current config schema is intentionally explicit:

- `version`: must match the Motlie v1.5 image contract version.
- Unknown fields are rejected at every config level. Typos and unsupported
  directives must fail schema loading instead of being ignored.
- `source`: source kind, image/profile/platform identity, and digest policy.
  The checked-in v1.5 config uses `external-oci` pinned to
  `docker.io/library/ubuntu:24.04` for the native CH path.
- `package_stage`: package manager intent for mutable package installation.
- `immutable_payloads`: Motlie guest payloads copied into immutable paths.
- `sshd_policy`: guest SSH policy files and optional forced-command policy.
- `services`: systemd services expected to be enabled.
- `immutable_files`: files expected to exist in the immutable image layer.
- `seed_files`: per-guest seed or overlay files expected after emission.
- `seed`: per-guest seed topology templates. The checked-in config renders the
  validation user home, SSH principal, and VFS mount set from `{guest}` instead
  of hardcoding `/home/<guest>`, `/workspace`, or `/agent-state` in CLI code.
- `emitters`: registered backend targets. Each emitter declares its backend ID,
  optional `materialized_source`, adapter command, adapter env-var contract,
  seed backend values, and harness validation env-var contract. The adapter env
  includes package manager, update/install/clean intent so shell emitters do not
  rediscover config shape. The checked-in config registers `ch` and `vz`, but
  the CLI accepts any target ID declared here.
- `validation`: post-boot behavior checks the produced image must satisfy.

Checked-in v1.5 release configs:

```text
releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml
releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-amd64.yaml
releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.default-arm64.yaml
```

The explicit per-platform configs are the release-facing inputs. The
`default-arm64` file preserves the former v1.5 example default for traceability;
new #258 acceptance work should prefer the explicit per-platform paths so
release evidence can distinguish priority `vz-darwin-arm64` and
`ch-linux-arm64` targets from the later coordinated `ch-linux-amd64` target. In
Motlie docs, `amd64` is the OCI/Debian platform name and `x86_64` is the
Linux/Rust host architecture spelling.

`package_stage.manager` currently supports only `apt` in executable adapters.
APT package entries are validated with APT-aware syntax, including `+`, arch
qualifiers such as `foo:amd64`, and pinned specs such as `foo=version`. The
parser reserves `apk`, `dnf`, `zypper`, and `pacman`, but rejects them until a
concrete package strategy exists. This keeps the config schema forward-shaped
without pretending the current adapters implement non-apt roots.

## Manifest Contract

`mbuild-manifest.json` records the declared stages in execution order:

```text
source
import
classify
package
immutable-layer
policy
seed
backend-emitter
validation
```

The build manifest records the config source kind, target backend, package
intent, stage status, adapter log path, adapter materialized source, artifact
digests, immutable files, seed files, and pending runtime requirements.
`mbuild validate` compares those manifest fields back against the current
config so stale manifests fail when the build file changes. The seed manifest
records the generated NoCloud seed, backend env, VFS mount config, SSH
CA/principal seed files, guest ownership metadata, and artifact digests. The
validation manifest records the delegated harness command, scenario, target,
log path, and exit status.

The manifests are deliberately machine-readable so harnesses and CI can verify
what stage was produced without rediscovering output paths or inferring backend
intent from directory names.

`mbuild-oci-export.json` records the exported OCI layout descriptors, source
image-index digest, selected platform-manifest digest, contract version,
selected platform, input rootfs size/sha, and ref-name annotation. This is the
first #258 handoff format: publish/multi-arch work consumes this layout instead
of rediscovering rootfs tarball paths or rebuilding backend artifacts from shell
defaults. `mbuild-release-evidence.json` records the same data in the
`kind = "motlie.vm-image-artifact"` shape used by release manifests.

`mbuild` emits structured tracing logs. Use `RUST_LOG=debug` when debugging OCI
fetch/import, rootfs classification, backend adapter delegation, or harness
validation.

## Local Verification

```bash
cargo build -p mbuild
cargo test -p mbuild -- --nocapture
cargo clippy -p mbuild -- -D warnings
```
