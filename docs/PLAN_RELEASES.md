# Release Distribution Plan

## Changelog

- 2026-04-29, @gpt55-dgx: Initial phased implementation plan for Motlie release distribution.
- 2026-04-29, @gpt55-dgx: Updated Homebrew tap target to `motlie/homebrew-tap` and added installer script hosting/source-mode tasks.
- 2026-04-29, @gpt55-dgx: Added macOS signing and installed-path verification tasks for npm, direct installer, and Homebrew release flows.
- 2026-05-12, @gpt55-dgx: Generalized plan tasks around a selected binary target; `mmux` remains the first worked validation target.
- 2026-05-12, @gpt55-dgx: Reworked the plan as a manual v0 release process with explicit release PR, manifest, tag, artifact, signing, npm, and Homebrew steps; CI job creation is deferred.
- 2026-05-12, @gpt55-dgx: Aligned the plan to per-binary release manifests and a long-running release coordination PR.
- 2026-05-12, @gpt55-dgx: Added skill-guided operator handoff requirements so different operators can pick up gates from manifest state.
- 2026-05-13, @gpt55-dgx: Added target-specific gate tracking, cargo-zigbuild toolchain evidence, merge-commit strategy, and disabled-channel deferral requirements.
- 2026-05-13, @gpt55-dgx: Added detached-tag build command and manifest-tracked installer validation gates.
- 2026-05-13, @gpt55-dgx: Made static musl the default Linux artifact policy when feasible, with glibc-floor evidence only for gnu fallback/CUDA targets.
- 2026-05-14, @gpt55-dgx: Split release evidence by universal, Darwin cross, Linux musl, and Linux gnu categories and pinned the default musl build toolchain.
- 2026-05-14, @gpt55-dgx: Reworked the plan around branch-local calver-codename release branches that never merge to `main`; fixes are cherry-picked back separately.
- 2026-05-14, @gpt55-dgx: Clarified installer template copy flow and made the macOS signing example fully parameterized.
- 2026-05-14, @gpt55-dgx: Added explicit release-note draft, review, and finalization tasks.
- 2026-05-14, @gpt55-dgx: Changed the plan to discover stable `releases/<bin>.toml` binary manifests by scanning `releases/`, with versions stored in schema and aggregate GitHub notes built from per-binary notes.

## Status

Manual v0 release plan for `docs/DESIGN_RELEASES.md`, `docs/RELEASES.md`, and issue #234. This plan intentionally does not create CI jobs yet. A release branch is the work queue and ledger for humans or agents working on different hosts.

Worked release event:

```text
RELEASE_NAME=2026-05-amber-aardvark
RELEASE_BRANCH=release/2026-05-amber-aardvark
RELEASE_TAG=2026-05-amber-aardvark
```

Worked branch-local manifests:

```text
releases/manifest.toml
releases/notes.md
releases/mmux.toml
releases/mmux.md
```

## Skill-Guided Operator Handoffs

Each phase below may be performed by a different operator on a different platform. Before acting, the release skill must read `releases/manifest.toml`, scan `releases/*.toml` for per-binary manifests with `kind = "motlie.binary-release"`, identify the next incomplete workspace or binary gate, and prompt the human with:

- current release and manifest state;
- next workspace gate or `(binary, gate, target_id)`;
- required host/platform;
- branch or PR to pull;
- command group or files to update;
- manifest fields to update;
- whether explicit approval is required.

Prompt templates live in `.agents/skills/release/references/operator-prompts.md`. The operator must update manifest gate state on the release branch or through a sub-PR targeting the release branch so the next operator can resume from repo state rather than chat context. Release branches never merge to `main`; source or process fixes made there must be cherry-picked into separate `main` PRs.

## Phase 1: Release Event and Binary Intake

- [ ] 1.1 Confirm the release event name, release branch, and tag with the user. For the worked release: `RELEASE_NAME=2026-05-amber-aardvark`, `RELEASE_BRANCH=release/2026-05-amber-aardvark`, and `RELEASE_TAG=2026-05-amber-aardvark`. Reference: `docs/DESIGN_RELEASES.md#release-target-model`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 1.2 Confirm every binary included in the release event and create or update one stable manifest per binary. For `mmux`: `BINARY_MANIFEST=releases/mmux.toml`, `[identity].binary=mmux`, `[identity].version=0.1.0`, `[build].cargo_package=motlie-mmux`, `[build].cargo_bin=mmux`, and `[install].default_path=/usr/local/bin/mmux`. Reference: `docs/DESIGN_RELEASES.md#release-target-model`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 1.3 Confirm the release is manual v0 and that CI job creation is out of scope for the release branch. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 1.4 Confirm channel scope: GitHub Release archives, direct installer, native npm packages, and Homebrew tap. Reference: `docs/DESIGN_RELEASES.md#distribution-channels`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 1.5 Confirm whether any binary has accelerator variants. CPU/default artifacts omit accelerator suffixes; CUDA artifacts use explicit suffixes such as `cuda-12-4`. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`. Skill support: `.agents/skills/release/SKILL.md`.

## Phase 2: Open the Release Branch

Branch from current `main`, push the branch, and use that branch as the release ledger. Do not open a PR to merge the release branch back to `main`.

```sh
git switch main
git pull --ff-only
git switch -c release/2026-05-amber-aardvark
git push -u origin release/2026-05-amber-aardvark
```

- [ ] 2.1 Bump release-branch versions in `Cargo.toml` or per-binary manifests as needed. If a binary inherits `version.workspace = true`, the release branch can carry the workspace version used for this release. Fix placeholder workspace metadata such as `authors = ["Your Name <your.email@example.com>"]` before the first real release. Reference: `docs/DESIGN_RELEASES.md#distribution-channels`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 2.2 Add branch-local `releases/manifest.toml` and `releases/notes.md`. The workspace manifest owns release-event identity, branch, tag, GitHub Release URL, global defaults, workspace gates, discovery policy, and final per-binary completion summaries. It does not enumerate the build fan-out. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 2.3 Add one stable per-binary manifest and notes file for each binary, for example `releases/mmux.toml` and `releases/mmux.md`. Per-binary manifests capture version, explicit non-derived names, target matrix, structured per-target status, `(id, target_id)` gates, and mutable status. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 2.4 Draft every per-binary note referenced by `[release].notes_path`, then aggregate `releases/notes.md`. The workspace notes must list all binaries, versions, distribution channels, install commands, user-visible changes, verification/checksum guidance, and known issues. Per-binary notes must cover binary-specific changes and compatibility notes. Reference: `docs/DESIGN_RELEASES.md#release-notes`. Skill support: `.agents/skills/release/references/release-notes.md`.
- [ ] 2.5 Add source-side installer, npm, or Homebrew templates under branch-local `releases/install/`, `releases/npm/`, or `releases/homebrew/` only when needed by the release. Installer scripts should be copied from canonical templates on `main`, normally `bins/<bin>/install-template.sh`, into `releases/install/install-<bin>.sh`; release-specific values may be patched in the branch-local copy. The live Homebrew formula still belongs in `motlie/homebrew-tap`. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`. Skill support: `.agents/skills/release/references/homebrew-tap.md`.
- [ ] 2.6 Mark disabled-channel gates `deferred` when the release branch opens. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/references/release-checklist.md`.

Release branch scope:

- `Cargo.toml` version and release metadata for the release branch.
- `bins/<bin>/Cargo.toml` verification or package-specific metadata for each binary.
- `releases/manifest.toml` and `releases/notes.md`.
- `releases/<bin>.toml` and `releases/<bin>.md` for every binary in scope.
- Optional source-side templates under branch-local `releases/install/`, `releases/npm/`, and `releases/homebrew/`.

Reusable docs, release skill changes, tooling improvements, and real source fixes should land on `main` through normal PRs, usually by cherry-picking from the release branch if they were discovered there. The release branch must not publish npm packages, create a stable GitHub Release, or merge live Homebrew tap changes until the relevant gates are approved.

## Phase 3: Land Platform Sub-PRs Into the Release Branch

Platform work should use short branches targeting the release branch, not `main`.

```text
release/2026-05-amber-aardvark-mmux-linux-x64 -> release/2026-05-amber-aardvark
release/2026-05-amber-aardvark-mmux-darwin-arm64 -> release/2026-05-amber-aardvark
release/2026-05-amber-aardvark-mmux-npm -> release/2026-05-amber-aardvark
```

- [ ] 3.1 Each sub-PR builds or validates one scoped `(binary, id, target_id)` gate and updates the relevant per-binary manifest with state, source commit, timestamp, actor, target id, channel, and evidence links. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 3.2 Status updates are staging evidence only. Final artifacts must still be rebuilt or revalidated from the final source tag if the final tag commit differs from the staging commit. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 3.3 Build outputs are not committed to git. Store only deterministic names, checksums, source commits, and validation evidence in the manifest. Reference: `docs/DESIGN_RELEASES.md#artifact-naming`. Skill support: `.agents/skills/release/references/release-checklist.md`.

## Phase 4: Pull the Release Branch on macOS

A macOS operator can pick up the release branch or a platform sub-PR and perform signing validation.

```sh
gh pr checkout <sub-pr-number>
git switch release/2026-05-amber-aardvark
git pull --ff-only
cargo build --release --locked --target <rust-target> -p <cargo-package> --bin <cargo-bin>
codesign --force --sign - target/<rust-target>/release/<bin>
codesign --verify --strict --verbose=2 target/<rust-target>/release/<bin>
target/<rust-target>/release/<bin> --version
sudo install -m 755 target/<rust-target>/release/<bin> <install-path>
sudo codesign --force --sign - <install-path>
codesign --verify --strict --verbose=2 <install-path>
<install-path> --version
```

- [ ] 4.1 Record staged macOS signing evidence in the relevant per-binary manifest, including `rust_target`, signing identity, `rustc -Vv`, `cargo -V`, and codesign evidence. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`. Skill support: `.agents/skills/release/references/macos-signing.md`.
- [ ] 4.2 Open a sub-PR back to the release branch with the manifest status update. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/macos-signing.md`.

## Phase 5: Finalize the Release Branch

- [ ] 5.1 Confirm every required workspace gate in `releases/manifest.toml` and every required per-binary gate is complete or explicitly deferred. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 5.2 Confirm all sub-PRs have merged into the release branch. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 5.3 Identify any reusable source, doc, skill, or tooling fixes made on the release branch. Cherry-pick those fixes into `main` through normal PRs if they matter outside the release. Do not merge the release branch to `main`. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.

## Phase 6: Tag and Publish From the Release Branch

Create the final source tag from the final release branch commit after all required gates are complete or deferred.

```sh
git switch release/2026-05-amber-aardvark
git pull --ff-only
rg -n 'name = "2026-05-amber-aardvark"|tag = "2026-05-amber-aardvark"|binary = "mmux"|version = "0.1.0"' releases Cargo.toml
git tag 2026-05-amber-aardvark
git push origin 2026-05-amber-aardvark
git switch --detach 2026-05-amber-aardvark
git status --short --branch
```

- [ ] 6.1 Build final artifacts from `RELEASE_TAG`, not from a dirty worktree. Confirm `Cargo.lock` is committed and unchanged. Always record `rustc -Vv` and `cargo -V`. For Darwin-from-Linux, use the v0 default `cargo-zigbuild` and record `cargo zigbuild -V` plus `zig version`. For pure-Rust `linux-*-musl`, use the v0 default `rustup + cargo build --target`; use `cargo-zigbuild` for musl only when C dependencies need a musl-aware linker. Record `file <binary>`, `ldd <binary>`, and `readelf -d <binary>` evidence. For any enabled `linux-*-gnu` fallback, record both `ldd --version` plus `objdump -T <binary> | grep GLIBC_ | sort -u`; update `glibc_build_host_version` and `glibc_min_version` in the target block. Reference: `docs/DESIGN_RELEASES.md#linux-libc-policy`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 6.2 Use the per-binary manifest's explicit `archive_asset`, `archive_binary_path`, `npm_package`, `npm.bin_path`, and installer names. Do not derive names when the manifest provides them. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 6.3 Sign and verify final Darwin artifacts from the final tag. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`. Skill support: `.agents/skills/release/references/macos-signing.md`.
- [ ] 6.4 Finalize and human-approve `releases/notes.md` and per-binary notes. Confirm the notes match final manifest names, target matrix, install commands, known issues, and checksums. Reference: `docs/DESIGN_RELEASES.md#release-notes`. Skill support: `.agents/skills/release/references/release-notes.md`.
- [ ] 6.5 Create the GitHub Release from `releases/notes.md` and upload final archives, checksums, installers, workspace manifest, per-binary manifests, and per-binary notes. Reference: `docs/DESIGN_RELEASES.md#github-releases`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 6.6 Validate the release-pinned direct installer on each supported target and update the target-specific `installer-validated` gates. If GitHub Pages convenience installer URLs are enabled, update the Pages repository in a separate PR after the release-pinned installer exists. Reference: `docs/DESIGN_RELEASES.md#installer-script-hosting`. Skill support: `.agents/skills/release/references/release-checklist.md`.

## Phase 7: Publish npm and Homebrew

- [ ] 7.1 Generate native npm packages from final artifacts. For `mmux`, the manifest requires `runner = "native-binary"` and `node_launcher = false`; do not create `mmux.sh` or `mmux.js` as the npm runtime entrypoint. Reference: `docs/DESIGN_RELEASES.md#npm`. Skill support: `.agents/skills/release/references/npm-auth.md`.
- [ ] 7.2 Run `npm pack --dry-run`, install generated packages on matching platforms, and execute `<bin> --version` from the npm-installed path. Reference: `docs/DESIGN_RELEASES.md#npm`. Skill support: `.agents/skills/release/references/npm-auth.md`.
- [ ] 7.3 Publish npm packages only after choosing the auth path. Trusted publishing is preferred; a scoped `NPM_TOKEN` is only needed at `npm publish` time if trusted publishing is unavailable. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/references/npm-auth.md`.
- [ ] 7.4 Open a separate PR in `motlie/homebrew-tap` for `Formula/<formula>.rb`; build from the final source tag and re-sign installed binaries on macOS. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`. Skill support: `.agents/skills/release/references/homebrew-tap.md`.

## Phase 8: Finalize the Release Ledger

Final GitHub Release URLs, uploaded asset URLs, npm links, Homebrew tap commits, and final checksums are not known before publication. Record them on the retained release branch after the release is complete.

- [ ] 8.1 Update `releases/manifest.toml` and per-binary manifests to `state = "published"`. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/SKILL.md`.
- [ ] 8.2 Add final `published` metadata: tag, GitHub Release URL, release notes URL, asset URLs, checksums, npm package URLs, Homebrew tap PR/commit, and install verification evidence. Reference: `docs/DESIGN_RELEASES.md#release-manifest`. Skill support: `.agents/skills/release/references/release-checklist.md`.
- [ ] 8.3 Push the final ledger commit to the retained release branch and upload the final manifest set to the GitHub Release if assets need to reflect post-publication URLs. Do not move the release tag to include ledger-only metadata. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`. Skill support: `.agents/skills/release/SKILL.md`.

## Phase 9: Future Automation

These tasks are intentionally deferred. They should not be part of the manual v0 release branch unless the user explicitly reopens automation scope.

- [ ] 9.1 Add a manifest validation helper that scans `releases/*.toml`, filters `kind = "motlie.binary-release"`, enforces `releases/<identity.binary>.toml`, rejects duplicate binaries, validates `[identity].version` and `[release].notes_path`, checks explicit names, checks status transitions, and checks accelerator suffix rules. Reference: `docs/DESIGN_RELEASES.md#core-motlie-work`.
- [ ] 9.2 Add CI jobs for Linux builds and Darwin cross-build staging. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 9.3 Add a manually approved macOS signing workflow. Reference: `docs/DESIGN_RELEASES.md#macos-code-signing`.
- [ ] 9.4 Add npm trusted-publishing workflows after package names and release assets are proven manually. Reference: `docs/DESIGN_RELEASES.md#upload-and-publishing-workflow`.
- [ ] 9.5 Add Homebrew bottle automation in `motlie/homebrew-tap`. Reference: `docs/DESIGN_RELEASES.md#homebrew-tap`.

## Commit Readiness

A release branch is ready for final tagging only after:

- The release event is represented in `releases/manifest.toml`.
- Every binary in scope is represented in one stable `releases/<bin>.toml` file with `[identity].version`.
- The workspace and per-binary manifests distinguish immutable release intent from mutable, structured target/gate status.
- `Cargo.toml` contains the intended workspace version and non-placeholder release metadata.
- `Cargo.lock` policy and toolchain evidence requirements are captured in the relevant per-binary manifests.
- Universal evidence commands are captured in `[toolchain].required_evidence_universal`.
- Darwin cross-build evidence commands are captured in `[toolchain].darwin_cross_required_evidence`.
- `linux-*-musl` targets have `linux_musl_toolchain` and static-link evidence commands captured in `[toolchain].linux_musl_required_evidence`.
- Any enabled `linux-*-gnu` fallback targets have `glibc_build_host_version` and `glibc_min_version` fields, with required evidence commands captured in `[toolchain].linux_gnu_required_evidence`.
- `releases/notes.md` exists and aggregates all per-binary notes.
- Each binary manifest's `[release].notes_path`, for example `releases/<bin>.md`, exists and matches that release target.
- Release notes have been checked for placeholders, final tag, final install commands, target matrix, known issues, and human approval.
- Any installer, npm, or Homebrew templates included in the branch match the manifest.
- If direct installer distribution is enabled, target-specific `installer-validated` gates exist in the manifest.
- Any fixes needed on `main` have been cherry-picked through separate `main` PRs or explicitly documented as release-branch-only.
- Docs and release skill match this release-branch workflow.
- No CI workflow files are added unless separately approved.
- No packages, stable GitHub Release assets, or Homebrew tap changes have been published from untrusted PR context.
- The release branch will not be merged to `main`.
