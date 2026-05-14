# mbuild

`mbuild` is the top-level Motlie v1.5 image builder CLI. It consumes the
Dockerfile-like image contract used by the v1.5 examples, drives the current
backend image adapters, regenerates per-guest seed artifacts, and emits
machine-readable manifests for CI and harness consumption.

Status as of 2026-05-14 (`@vmm-cdx`): `mbuild build --target ch` is the
durable Linux/CH image-builder entrypoint for the v1.5 demo. It resolves the
pinned Ubuntu OCI source, imports rootfs layers, runs the apt/npm package stage,
applies the native v1.5 Motlie compatibility layer, and emits CH artifacts plus
machine-readable manifests. `mbuild seed` regenerates per-guest seed overlays
without rebuilding the immutable image. `mbuild validate` validates manifests,
can require execution evidence, and can delegate live guest conformance to the
v1.5 harness. The VZ target still records its current macOS adapter source until
the VZ emitter consumes the same assembled OCI rootfs path.

## Commands

Build CH artifacts through the native external-OCI CH path:

```bash
cargo run -p mbuild -- build \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --target ch \
  --out /tmp/mbuild/ch
```

Build VZ artifacts through the current VZ adapter:

```bash
cargo run -p mbuild -- build \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --target vz \
  --out /tmp/mbuild/vz
```

Plan without running backend adapters:

```bash
cargo run -p mbuild -- build \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --target ch \
  --out /tmp/mbuild/plan/ch \
  --plan-only
```

Regenerate per-guest seed artifacts:

```bash
cargo run -p mbuild -- seed \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --target ch \
  --guest alice \
  --uid 2001 \
  --gid 2001 \
  --out /tmp/mbuild/seed/alice
```

Validate an emitted build manifest:

```bash
cargo run -p mbuild -- validate \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --artifact /tmp/mbuild/ch \
  --require-executed
```

Delegate live conformance to the v1.5 harness and write a validation record:

```bash
cargo run -p mbuild -- validate \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --artifact /tmp/mbuild/ch \
  --require-executed \
  --scenario libs/vmm/examples/v1.5/scenarios/multiguest-validate.json
```

The build command writes:

```text
<out>/mbuild-manifest.json
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

Linux/CH validation evidence from 2026-05-14 (`@vmm-cdx`) used:

```bash
cargo run -p mbuild -- build \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --target ch \
  --out /tmp/mbuild-pr270-oci-ch-7

cargo run -p mbuild -- validate \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
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

`mbuild` emits structured tracing logs. Use `RUST_LOG=debug` when debugging OCI
fetch/import, rootfs classification, backend adapter delegation, or harness
validation.

## Local Verification

```bash
cargo build -p mbuild
cargo test -p mbuild -- --nocapture
cargo clippy -p mbuild -- -D warnings
```
