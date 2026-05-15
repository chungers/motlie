# macOS Signing

Use this reference when validating Darwin artifacts or installer behavior.

Parameterize examples by the current per-binary manifest, not by workspace-level binary fields:

```text
BINARY_MANIFEST=releases/<bin>.toml
RELEASE_BRANCH=release/<YYYY-MM-codename>
[identity].binary=<installed command name>
[build].cargo_package=<cargo package name>
[build].cargo_bin=<cargo binary name>
[install].default_path=<final installed path, e.g. /usr/local/bin/<bin>>
[[target]].rust_target=<Darwin target.rust_target from manifest>
```

Minimum first-release requirement:

- ad-hoc sign with `codesign --force --sign -`
- verify with `codesign --verify --strict --verbose=2`
- execute from the final installed path

Build-path verification:

```sh
gh pr checkout <release-pr-number>
git switch "${RELEASE_BRANCH}"
git pull --ff-only
cargo build --release --locked --target "<[[target]].rust_target>" -p "<[build].cargo_package>" --bin "<[build].cargo_bin>"
codesign --force --sign - "target/<[[target]].rust_target>/release/<[identity].binary>"
codesign --verify --strict --verbose=2 "target/<[[target]].rust_target>/release/<[identity].binary>"
"target/<[[target]].rust_target>/release/<[identity].binary>" --version
```

Final-path verification:

```sh
sudo install -m 755 "target/<[[target]].rust_target>/release/<[identity].binary>" "<[install].default_path>"
sudo codesign --force --sign - "<[install].default_path>"
codesign --verify --strict --verbose=2 "<[install].default_path>"
"<[install].default_path>" --version
```

Why this gate exists:

- Apple Silicon validates executable signatures at startup.
- A copied Mach-O may fail at the final path even if the build-path binary runs.
- Failure can appear as immediate `SIGKILL`.

Release rule:

- Linux may cross-compile Darwin artifacts.
- Darwin staging evidence should be recorded in `BINARY_MANIFEST` through a sub-PR to the release branch.
- Darwin artifacts are not final until the macOS signing gate signs, verifies, repacks, and executes artifacts built from the final source tag.
- If the final tag commit differs from the staged signing commit, rebuild or revalidate before publishing.
- Record `rustc -Vv`, `cargo -V`, `codesign -dv`, target triple, and signing identity in the gate evidence.

Developer ID signing and notarization are later hardening steps, not the minimum first-release gate.
