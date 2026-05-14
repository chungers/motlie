# Release Distribution Plan

## Changelog

- 2026-04-29, @gpt55-dgx: Initial phased implementation plan for Motlie release distribution.
- 2026-04-29, @gpt55-dgx: Updated Homebrew tap target to `motlie/homebrew-tap` and added installer script hosting/source-mode tasks.
- 2026-04-29, @gpt55-dgx: Added macOS signing and installed-path verification tasks for npm, direct installer, and Homebrew release flows.
- 2026-05-12, @gpt55-dgx: Generalized plan tasks around a selected binary target; `mmux` remains the first worked validation target.
- 2026-05-12, @gpt55-dgx: Reworked the plan as a manual v0 release process with explicit release PR, manifest, tag, artifact, signing, npm, and Homebrew steps; CI job creation is deferred.
- 2026-05-12, @gpt55-dgx: Aligned the plan to per-binary release manifests under `releases/<bin>/<version>.toml` and a long-running release coordination PR.
- 2026-05-12, @gpt55-dgx: Added skill-guided operator handoff requirements so different operators can pick up gates from manifest state.
- 2026-05-13, @gpt55-dgx: Added target-specific gate tracking, cargo-zigbuild toolchain evidence, merge-commit strategy, and disabled-channel deferral requirements.
- 2026-05-13, @gpt55-dgx: Added detached-tag build command and manifest-tracked installer validation gates.
- 2026-05-13, @gpt55-dgx: Made static musl the default Linux artifact policy when feasible, with glibc-floor evidence only for gnu fallback/CUDA targets.
- 2026-05-14, @gpt55-dgx: Split release evidence by universal, Darwin cross, Linux musl, and Linux gnu categories and pinned the default musl build toolchain.

## Status

Manual v0 release plan for `docs/DESIGN_RELEASES.md`, `docs/RELEASES.md`, and issue #234. This plan intentionally does not create CI jobs yet. The release coordination PR is the work queue and ledger for humans or agents working on different hosts.

Worked manifest:

```text
releases/mmux/0.1.0.toml
```

Worked release branch:

```text
release/mmux-v0.1.0
```

## Skill-Guided Operator Handoffs

Each phase below may be performed by a different operator on a different platform. Before acting, the release skill must read `releases/<bin>/<version>.toml`, identify the next incomplete gate, and prompt the human with:

- current manifest state;
- next gate and required host/platform;
- branch or PR to pull;
- command group or files to update;
- manifest fields to update;
- whether explicit approval is required.

Prompt templates live in `.agents/skills/release/references/operator-prompts.md`. The operator must update manifest gate state through the coordination PR or a sub-PR so the next operator can resume from repo state rather than chat context.

## Phase 1: Release Target Intake

- [ ] 1.1 Confirm the selected binary target, version, and release branch with the user. For `mmux`: `BIN=mmux`, `VERSION=0.1.0`, `CARGO_PACKAGE=motlie-mmux`, `CARGO_BIN=mmux`, `INSTALL_PATH=/usr/local/bin/mmux`, and `release/mmux-v0.1.0`. Reference: `docs/DESIGN_RELEASES.md#release-target-model`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 1.2 Confirm the release is manual v0 and that CI job creation is out of scope for the release coordination PR. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 1.3 Confirm channel scope: GitHub Release archives, direct installer, native npm packages, and Homebrew tap. Reference: `docs/DESIGN_RELEASES.md#distribution-channels`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 1.4 Confirm whether this target has accelerator variants. CPU/default artifacts omit accelerator suffixes; CUDA artifacts use explicit suffixes such as `cuda-12-4`. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`. Skill support: `.agents/skills/release/SKILL.md`.

## Phase 2: Open the Release Coordination PR

Branch from current `main` and open a long-running PR back to `main`.

```sh
git switch main
git pull --ff-only
git switch -c release/mmux-v0.1.0
```

- [ ] 2.1 Bump the workspace release version in `Cargo.toml` under `[workspace.package].version`. For `mmux`, `bins/mmux/Cargo.toml` inherits `version.workspace = true`; verify package name, bin name, and description there. Fix placeholder workspace metadata such as `authors = ["Your Name <your.email@example.com>"]` before the first real release. Reference: `docs/DESIGN_RELEASES.md#distribution-channels`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 2.2 Add `releases/<bin>/<version>.toml`. For `mmux`, add `releases/mmux/0.1.0.toml`. This file captures release intent, explicit non-derived names, target matrix, structured per-target status, `(id, target_id)` gates, and mutable status. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 2.3 Add `releases/<bin>/<version>.md` as the release-note source. For `mmux`, add `releases/mmux/0.1.0.md`. Reference: `docs/DESIGN_RELEASES.md#github-releases`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 2.4 Add source-side installer, npm, or Homebrew templates under `releases/<bin>/` only when needed by the release. The live Homebrew formula still belongs in `motlie/homebrew-tap`. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`. Skill support: `.agents/skills/release/references/homebrew-tap.md`.
- [ ] 2.5 Open the coordination PR: `release/<bin>-v<version> -> main`. The PR should explain that platform-specific work lands through sub-PRs targeting the release branch. Mark disabled-channel gates `deferred` at PR-open time. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/references/release-checklist.md`.

Release PR scope:

- `Cargo.toml` version and release metadata.
- `bins/<bin>/Cargo.toml` verification or package-specific metadata.
- `releases/<bin>/<version>.toml` with structured target status and target-specific gates.
- `releases/<bin>/<version>.md`.
- Optional source-side templates under `releases/<bin>/install/`, `releases/<bin>/npm/`, and `releases/<bin>/homebrew/`.
- Release docs and release skill updates.

The coordination PR must not publish npm packages, create a stable GitHub Release, or merge live Homebrew tap changes.

## Phase 3: Land Platform Sub-PRs Into the Release Branch

Platform work should use short branches targeting the release branch, not `main`.

```text
release/mmux-v0.1.0-linux-x64 -> release/mmux-v0.1.0
release/mmux-v0.1.0-darwin-arm64 -> release/mmux-v0.1.0
release/mmux-v0.1.0-npm -> release/mmux-v0.1.0
```

- [ ] 3.1 Each sub-PR builds or validates one scoped `(id, target_id)` gate and updates `releases/<bin>/<version>.toml` with state, source commit, timestamp, actor, target id, channel, and evidence links. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 3.2 Status updates are staging evidence only. Final artifacts must still be rebuilt or revalidated from the final source tag if the final tag commit differs from the staging commit. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 3.3 Build outputs are not committed to git. Store only deterministic names, checksums, source commits, and validation evidence in the manifest. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`. Skill support: `.agents/skills/release/references/release-checklist.md`.

## Phase 4: Pull the Coordination Branch on macOS

A macOS operator can pick up the release PR or a platform sub-PR and perform signing validation.

```sh
gh pr checkout <release-pr-number>
git switch release/mmux-v0.1.0
git pull --ff-only
cargo build --release --locked --target aarch64-apple-darwin -p motlie-mmux --bin mmux
codesign --force --sign - target/aarch64-apple-darwin/release/mmux
codesign --verify --strict --verbose=2 target/aarch64-apple-darwin/release/mmux
target/aarch64-apple-darwin/release/mmux --version
sudo install -m 755 target/aarch64-apple-darwin/release/mmux /usr/local/bin/mmux
sudo codesign --force --sign - /usr/local/bin/mmux
codesign --verify --strict --verbose=2 /usr/local/bin/mmux
/usr/local/bin/mmux --version
```

- [ ] 4.1 Record staged macOS signing evidence in `releases/<bin>/<version>.toml`, including `rust_target`, signing identity, `rustc -Vv`, `cargo -V`, and codesign evidence. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`. Skill support: `.agents/skills/release/references/macos-signing.md`.
- [ ] 4.2 Open a sub-PR back to the release branch with the manifest status update. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/macos-signing.md`.

## Phase 5: Complete the Coordination PR

- [ ] 5.1 Confirm every required gate in `releases/<bin>/<version>.toml` is complete or explicitly deferred. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 5.2 Confirm all sub-PRs have merged into the release branch and the coordination PR is up to date with `main`. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 5.3 Merge the coordination PR to `main` with a merge commit. Do not squash or rebase the release branch. If the merge commit changes the source commit used for staging evidence, final artifacts must be rebuilt or revalidated from the final tag. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.

## Phase 6: Tag and Publish From Main

Create the final source tag only after the coordination PR is merged.

```sh
git switch main
git pull --ff-only
rg -n 'binary = "mmux"|version = "0.1.0"|tag = "v0.1.0"' releases/mmux/0.1.0.toml Cargo.toml
git tag v0.1.0
git push origin v0.1.0
git switch --detach v0.1.0
git status --short --branch
```

- [ ] 6.1 Build final artifacts from `v<VERSION>`, not from a dirty worktree. Confirm `Cargo.lock` is committed and unchanged. Always record `rustc -Vv` and `cargo -V`. For Darwin-from-Linux, use the v0 default `cargo-zigbuild` and record `cargo zigbuild -V` plus `zig version`. For pure-Rust `linux-*-musl`, use the v0 default `rustup + cargo build --target`; use `cargo-zigbuild` for musl only when C dependencies need a musl-aware linker. Record `file <binary>`, `ldd <binary>`, and `readelf -d <binary>` evidence. For any enabled `linux-*-gnu` fallback, record both `ldd --version` plus `objdump -T <binary> | grep GLIBC_ | sort -u`; update `glibc_build_host_version` and `glibc_min_version` in the target block. Reference: `docs/DESIGN_RELEASES.md#linux-libc-policy`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 6.2 Use the manifest's explicit `archive_asset`, `archive_binary_path`, `npm_package`, `npm.bin_path`, and installer names. Do not derive names when the manifest provides them. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 6.3 Sign and verify final Darwin artifacts from the final tag. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`. Skill support: `.agents/skills/release/references/macos-signing.md`.
- [ ] 6.4 Create the GitHub Release from `releases/<bin>/<version>.md` and upload final archives, checksums, and installer assets. Reference: `docs/DESIGN_RELEASES.md#github-releases`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 6.5 Validate the release-pinned direct installer on each supported target and update the target-specific `installer-validated` gates. If GitHub Pages convenience installer URLs are enabled, update the Pages repository in a separate PR after the release-pinned installer exists. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`. Skill support: `.agents/skills/release/references/release-checklist.md`.

## Phase 7: Publish npm and Homebrew

- [ ] 7.1 Generate native npm packages from final artifacts. For `mmux`, the manifest requires `runner = "native-binary"` and `node_launcher = false`; do not create `mmux.sh` or `mmux.js` as the npm runtime entrypoint. Reference: `docs/DESIGN_RELEASES.md#npm`. Skill support: `.agents/skills/release/references/npm-auth.md`.
- [ ] 7.2 Run `npm pack --dry-run`, install generated packages on matching platforms, and execute `<bin> --version` from the npm-installed path. Reference: `docs/DESIGN_RELEASES.md#npm`. Skill support: `.agents/skills/release/references/npm-auth.md`.
- [ ] 7.3 Publish npm packages only after choosing the auth path. Trusted publishing is preferred; a scoped `NPM_TOKEN` is only needed at `npm publish` time if trusted publishing is unavailable. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/references/npm-auth.md`.
- [ ] 7.4 Open a separate PR in `motlie/homebrew-tap` for `Formula/<formula>.rb`; build from the final source tag and re-sign installed binaries on macOS. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`. Skill support: `.agents/skills/release/references/homebrew-tap.md`.

## Phase 8: Post-Release Ledger PR

Final GitHub Release URLs, uploaded asset URLs, npm links, Homebrew tap commits, and final checksums are not known before publication. Record them in a small post-release ledger PR after the release is complete.

- [ ] 8.1 Update `releases/<bin>/<version>.toml` to `state = "published"`. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 8.2 Add final `published` metadata: tag, GitHub Release URL, release notes URL, asset URLs, checksums, npm package URLs, Homebrew tap PR/commit, and install verification evidence. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 8.3 Merge the ledger PR to `main`. Do not move the release tag to include ledger-only metadata. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.

## Phase 9: Future Automation

These tasks are intentionally deferred. They should not be part of the manual v0 release PR unless the user explicitly reopens automation scope.

- [ ] 9.1 Add a manifest validation helper that checks schema, explicit names, status transitions, and accelerator suffix rules. Reference: `docs/DESIGN_RELEASES.md#core-motlie-work`.
- [ ] 9.2 Add CI jobs for Linux builds and Darwin cross-build staging. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 9.3 Add a manually approved macOS signing workflow. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 9.4 Add npm trusted-publishing workflows after package names and release assets are proven manually. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 9.5 Add Homebrew bottle automation in `motlie/homebrew-tap`. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`.

## Commit Readiness

A release coordination PR is ready to merge only after:

- The target is represented in `releases/<bin>/<version>.toml`.
- The manifest distinguishes immutable release intent from mutable, structured target/gate status.
- `Cargo.toml` contains the intended workspace version and non-placeholder release metadata.
- `Cargo.lock` policy and toolchain evidence requirements are captured in the manifest.
- Universal evidence commands are captured in `[toolchain].required_evidence_universal`.
- Darwin cross-build evidence commands are captured in `[toolchain].darwin_cross_required_evidence`.
- `linux-*-musl` targets have `linux_musl_toolchain` and static-link evidence commands captured in `[toolchain].linux_musl_required_evidence`.
- Any enabled `linux-*-gnu` fallback targets have `glibc_build_host_version` and `glibc_min_version` fields, with required evidence commands captured in `[toolchain].linux_gnu_required_evidence`.
- `releases/<bin>/<version>.md` exists and matches the release target.
- Any installer, npm, or Homebrew templates included in the PR match the manifest.
- If direct installer distribution is enabled, target-specific `installer-validated` gates exist in the manifest.
- Docs and release skill match this coordination-PR workflow.
- No CI workflow files are added unless separately approved.
- No packages, stable GitHub Release assets, or Homebrew tap changes have been published from untrusted PR context.
