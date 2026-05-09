# mbuild

`mbuild` is the top-level Motlie v1.5 image builder CLI. It consumes the
Dockerfile-like image contract used by the v1.5 examples and emits a
machine-readable manifest that records the planned rootfs, policy, seed, and
backend-emitter stages.

Status as of 2026-05-08 (`@vmm-cdx`): this first slice validates the config and
emits or validates a declared-stage manifest. It does not yet install packages,
assemble the rootfs, emit CH/VZ boot artifacts, or run live guest validation.
Those stages remain the follow-on implementation work for the v1.5 demo path.

## Commands

Build a CH manifest:

```bash
cargo run -p mbuild -- build \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --target ch \
  --out /tmp/mbuild/ch
```

Validate the emitted manifest:

```bash
cargo run -p mbuild -- validate \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --artifact /tmp/mbuild/ch
```

Build a VZ manifest:

```bash
cargo run -p mbuild -- build \
  --config libs/vmm/examples/v1.5/motlie-image.yaml \
  --target vz \
  --out /tmp/mbuild/vz
```

The build command writes:

```text
<out>/mbuild-manifest.json
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

The manifest is deliberately machine-readable so harnesses and CI can verify
what stage was produced without rediscovering output paths or inferring backend
intent from directory names.

## Local Verification

```bash
cargo build -p mbuild
cargo test -p mbuild -- --nocapture
cargo clippy -p mbuild -- -D warnings
```
