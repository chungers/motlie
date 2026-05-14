# macOS Signing

Use this reference when validating Darwin artifacts or installer behavior.

Parameterize examples by the release target:

```text
BIN=<installed command name>
INSTALL_PATH=<final installed path, e.g. /usr/local/bin/<bin>>
BINARY_MANIFEST=releases/<bin>-<version>.toml
RELEASE_BRANCH=release/<YYYY-MM-codename>
RUST_TARGET=<target.rust_target from manifest>
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
cargo build --release --locked --target "${RUST_TARGET}" -p "<cargo-package>" --bin "${BIN}"
codesign --force --sign - "target/${RUST_TARGET}/release/${BIN}"
codesign --verify --strict --verbose=2 "target/${RUST_TARGET}/release/${BIN}"
"target/${RUST_TARGET}/release/${BIN}" --version
```

Final-path verification:

```sh
sudo install -m 755 "target/${RUST_TARGET}/release/${BIN}" "${INSTALL_PATH}"
sudo codesign --force --sign - "${INSTALL_PATH}"
codesign --verify --strict --verbose=2 "${INSTALL_PATH}"
"${INSTALL_PATH}" --version
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
