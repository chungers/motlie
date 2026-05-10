# mbuild

`mbuild` is the top-level Motlie v1.5 image builder CLI. It consumes the
Dockerfile-like image contract used by the v1.5 examples, drives the current
backend image adapters, regenerates per-guest seed artifacts, and emits
machine-readable manifests for CI and harness consumption.

Status as of 2026-05-09 (`@vmm-cdx`): `mbuild build` is the durable entrypoint
for the v1.5 image build. It consumes `motlie-image.yaml` and delegates
backend-specific artifact emission to the current `examples/v1.5` adapters.
`mbuild seed` regenerates per-guest seed overlays without rebuilding the
immutable image. `mbuild validate` validates manifests and can require adapter
execution evidence; live guest conformance still runs through the v1.5 harness.

## Commands

Build CH artifacts through the current CH adapter:

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

## Image Config Contract

The current config schema is intentionally explicit:

- `version`: must match the Motlie v1.5 image contract version.
- `source`: foundation OCI image, profile, platform, and digest policy.
- `package_stage`: package manager intent for mutable package installation.
- `immutable_payloads`: Motlie guest payloads copied into immutable paths.
- `sshd_policy`: guest SSH policy files and optional forced-command policy.
- `services`: systemd services expected to be enabled.
- `immutable_files`: files expected to exist in the immutable image layer.
- `seed_files`: per-guest seed or overlay files expected after emission.
- `emitters`: backend targets supported by the config, currently `ch` and `vz`.
- `validation`: post-boot behavior checks the produced image must satisfy.

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

The build manifest records the config source, target backend, package intent,
stage status, adapter log path, artifact digests, immutable files, seed files,
and pending runtime requirements. The seed manifest records the generated
NoCloud seed, backend env, VFS mount config, SSH CA/principal seed files, and
artifact digests. The validation manifest records the delegated harness command,
scenario, target, log path, and exit status.

The manifests are deliberately machine-readable so harnesses and CI can verify
what stage was produced without rediscovering output paths or inferring backend
intent from directory names.

## Local Verification

```bash
cargo build -p mbuild
cargo test -p mbuild -- --nocapture
cargo clippy -p mbuild -- -D warnings
```
