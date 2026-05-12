# Release Distribution Plan

## Changelog

- 2026-04-29, @gpt55-dgx: Initial phased implementation plan for Motlie release distribution.
- 2026-04-29, @gpt55-dgx: Updated Homebrew tap target to `motlie/homebrew-tap` and added installer script hosting/source-mode tasks.
- 2026-04-29, @gpt55-dgx: Added macOS signing and installed-path verification tasks for npm, direct installer, and Homebrew release flows.
- 2026-05-12, @gpt55-dgx: Generalized plan tasks around a selected binary target; `mmux` remains the first worked validation target.

## Status

Draft implementation plan for `docs/DESIGN_RELEASES.md` and issue #234.

## Phase 1: Release Metadata

- [ ] 1.1 Confirm the selected binary target fields: `BIN`, `CARGO_PACKAGE`, `CARGO_BIN`, `VERSION`, channels, targets, and whether it is host/SSH-safe. The first worked validation target is `mmux`. Reference: `docs/DESIGN_RELEASES.md#release-target-model`.
- [ ] 1.2 Confirm whether Linux musl targets are in v0.1 or deferred. Reference: `docs/DESIGN_RELEASES.md#open-questions`.
- [ ] 1.3 Fix workspace and package metadata needed by release tooling: version, repository, license, authors, and package descriptions. Reference: `docs/DESIGN_RELEASES.md#distribution-channels`.
- [ ] 1.4 Decide whether the release manifest lives at `release/motlie-release.toml` or `motlie-release.toml`. Reference: `docs/DESIGN_RELEASES.md#release-manifest`.

## Phase 2: Release Manifest and Validation

- [ ] 2.1 Add the shared release manifest with the selected binary target's archive, npm, installer, and Homebrew channel metadata. Reference: `docs/DESIGN_RELEASES.md#release-manifest`.
- [ ] 2.2 Add a release validation command in `xtask` or `release/` that checks manifest shape, target names, and accelerator suffix rules. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`.
- [ ] 2.3 Validate that CPU/default builds omit accelerator suffixes and CUDA builds include explicit suffixes. Reference: `docs/DESIGN_RELEASES.md#npm-package-naming`.
- [ ] 2.4 Validate that host/SSH-safe targets such as `mmux` are marked `force_command_safe = true` and install as native binaries. Reference: `docs/DESIGN_RELEASES.md#user-experience`.

## Phase 3: Native Build Matrix

- [ ] 3.1 Add CI jobs for the selected `BIN` on Linux x64 glibc. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`.
- [ ] 3.2 Add CI jobs for the selected `BIN` on Linux arm64 glibc. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`.
- [ ] 3.3 Add CI jobs for the selected `BIN` on macOS Apple Silicon. Reference: `docs/DESIGN_RELEASES.md#homebrew`.
- [ ] 3.4 Add CI jobs for the selected `BIN` on macOS Intel. Reference: `docs/DESIGN_RELEASES.md#homebrew`.
- [ ] 3.5 Run smoke tests on every build: `<bin> --help` and `<bin> --version`. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.

## Phase 4: GitHub Release Archives

- [ ] 4.1 Package each native build as `motlie-{bin}-v{version}-{os}-{arch}[-{libc}][-{accelerator}].tar.gz`. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`.
- [ ] 4.2 Include `bin/<bin>`, README, and license in each archive. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`.
- [ ] 4.3 Ad-hoc sign Darwin binaries before archive packaging. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 4.4 Verify Darwin binaries with `codesign --verify --strict --verbose=2` and execute `<bin> --version` from the packaged path. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 4.5 Generate checksums for all archives. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 4.6 Upload archives, checksums, and release notes to the `chungers/motlie` GitHub Release. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.

## Phase 5: npm Native Packages

- [ ] 5.1 Add npm package templates for native packages under `release/npm/`. Reference: `docs/DESIGN_RELEASES.md#npm`.
- [ ] 5.2 Generate Darwin npm packages for the selected `BIN`, for example `@motlie/<bin>-darwin-arm64` and `@motlie/<bin>-darwin-x64`. Reference: `docs/DESIGN_RELEASES.md#npm-package-naming`.
- [ ] 5.3 Generate Linux npm packages for the selected `BIN`, for example `@motlie/<bin>-linux-x64-gnu` and `@motlie/<bin>-linux-arm64-gnu`. Reference: `docs/DESIGN_RELEASES.md#npm-package-naming`.
- [ ] 5.4 Ensure each npm package exposes `<bin> = bin/<bin>` directly, without a Node launcher. Reference: `docs/DESIGN_RELEASES.md#npm`.
- [ ] 5.5 Ensure Darwin npm package binaries are signed before `npm pack`. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 5.6 Run `npm pack --dry-run` and inspect package contents. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 5.7 Install Darwin npm packages in CI and execute `<bin> --version` from the npm-installed path. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 5.8 Publish native npm packages to the `@motlie` org after release artifacts pass verification. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.

## Phase 6: Direct Installer

- [ ] 6.1 Add installer script sources under `release/install/`, including `install-<bin>.sh` for the selected target and shared detection helpers. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`.
- [ ] 6.2 Implement OS, architecture, and Linux libc detection. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 6.3 Implement archive selection from the release manifest. Reference: `docs/DESIGN_RELEASES.md#release-manifest`.
- [ ] 6.4 Download archive and checksum from a release-pinned GitHub Release URL. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 6.5 Verify checksum before installing. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 6.6 Install the selected `BIN` to its configured `INSTALL_PATH` by default. Reference: `docs/DESIGN_RELEASES.md#user-experience`.
- [ ] 6.7 Document the `curl -fsSL ... | sh` path and the safer audit-before-run path. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 6.8 Upload installer scripts to version-pinned GitHub Releases as canonical hosted artifacts. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`.
- [ ] 6.9 Optionally publish GitHub Pages latest convenience entrypoints under `https://motlie.github.io/install/`. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`.
- [ ] 6.10 Support installer `--source archive` and `--source npm` modes, with host/SSH-safe binaries defaulting to archive mode. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 6.11 Scope CUDA detection to binaries that publish CUDA-capable artifacts or packages. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`.
- [ ] 6.12 On macOS, re-sign the final installed binary after copying into the prefix. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 6.13 On macOS, verify and execute `${INSTALL_PATH} --version` from the final installed path. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.

## Phase 7: Homebrew Tap

- [ ] 7.1 Create or confirm the tap repository `motlie/homebrew-tap`. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`.
- [ ] 7.2 Add `Formula/<formula>.rb` that builds the selected binary from the Motlie source tag. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`.
- [ ] 7.3 Add Homebrew test that runs `<bin> --help` or `<bin> --version`. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`.
- [ ] 7.4 Configure tap CI for macOS Apple Silicon and Intel bottle builds. Reference: `docs/DESIGN_RELEASES.md#homebrew`.
- [ ] 7.5 Upload bottles using the tap's bottle workflow and update the formula bottle block if required. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 7.6 Verify `brew install motlie/tap/<formula>` on macOS. Reference: `docs/DESIGN_RELEASES.md#user-experience`.
- [ ] 7.7 Re-sign `bin/"<bin>"` in the formula after `bin.install` on macOS. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 7.8 Ensure formula and bottle tests execute `#{bin}/<bin> --version` from the installed path. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.

## Phase 8: End-to-End Publishing Workflow

- [ ] 8.1 Create and push the Motlie source tag. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 8.2 Build and test all native binaries from the tag. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 8.3 Sign and verify Darwin binaries before publishing. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 8.4 Upload GitHub Release archives, checksums, and installer scripts to `chungers/motlie`. Reference: `docs/DESIGN_RELEASES.md#github-releases`.
- [ ] 8.5 Publish generated native npm packages to the npm registry under `@motlie`. Reference: `docs/DESIGN_RELEASES.md#npm`.
- [ ] 8.6 Update the Homebrew tap repository and publish bottles for macOS only. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`.
- [ ] 8.7 Run post-publish installation verification for each channel from final installed paths. Reference: `docs/DESIGN_RELEASES.md#user-experience`.

## Phase 9: Verification Matrix

- [ ] 9.1 Verify Linux x64 glibc npm install: `npm install -g @motlie/<bin>-linux-x64-gnu`. Reference: `docs/DESIGN_RELEASES.md#npm`.
- [ ] 9.2 Verify Linux arm64 glibc npm install: `npm install -g @motlie/<bin>-linux-arm64-gnu`. Reference: `docs/DESIGN_RELEASES.md#npm`.
- [ ] 9.3 Verify macOS Apple Silicon npm install: `npm install -g @motlie/<bin>-darwin-arm64`. Reference: `docs/DESIGN_RELEASES.md#npm`.
- [ ] 9.4 Verify macOS Intel npm install: `npm install -g @motlie/<bin>-darwin-x64`. Reference: `docs/DESIGN_RELEASES.md#npm`.
- [ ] 9.5 Verify direct installer on Linux x64 and arm64. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 9.6 Verify direct installer on macOS Apple Silicon and Intel. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.
- [ ] 9.7 Verify Homebrew install on macOS Apple Silicon and Intel. Reference: `docs/DESIGN_RELEASES.md#homebrew`.
- [ ] 9.8 For host/SSH-safe binaries, verify SSH `ForceCommand ${INSTALL_PATH}` on Linux after direct installer install. Reference: `docs/DESIGN_RELEASES.md#user-experience`.
- [ ] 9.9 Verify macOS installed binaries with `codesign --verify --strict --verbose=2` and installed-path `<bin> --version`. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.

## Phase 10: Documentation

- [ ] 10.1 Add release installation docs for npm, installer, and Homebrew. Reference: `docs/DESIGN_RELEASES.md#user-experience`.
- [ ] 10.2 Add release operator docs that distinguish where artifacts are hosted: `chungers/motlie`, npm `@motlie`, and `motlie/homebrew-tap`. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 10.3 Add SSH `ForceCommand` docs for Linux and macOS install paths. Reference: `docs/DESIGN_RELEASES.md#user-experience`.
- [ ] 10.4 Add upgrade and uninstall docs for the direct installer. Reference: `docs/DESIGN_RELEASES.md#direct-installer`.

## Commit Readiness

A release-distribution implementation commit is ready only after:

- `cargo fmt` passes for touched Rust code.
- Rust release helper tests pass if helper code is added.
- Shell installer syntax and behavior tests pass.
- npm package dry-runs show only intended files.
- macOS installed-path signing checks pass for npm, direct installer, and Homebrew paths.
- Homebrew formula audit/test passes in the tap.
- Docs match the implemented release behavior.
