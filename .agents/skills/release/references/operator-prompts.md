# Operator Prompts

Use this reference when a release task may be handled by different operators on different hosts.

First inspect:

```text
WORKSPACE_MANIFEST=releases/manifest.toml
RELEASE_BRANCH=release/<YYYY-MM-adjective-codename>
BINARY_MANIFEST=releases/<bin>-<version>.toml
```

Every prompt should include:

- current branch and git cleanliness
- current workspace and binary manifest `state`
- next incomplete workspace gate or `(binary, gate, target_id)`
- `target_id` when the gate is target-specific
- host/platform required
- exact branch or PR to pull
- command group to run or files to update
- manifest fields that will be updated
- whether explicit approval is required

## Intake

Prompt:

```text
@<identity> <datetime> -- I found WORKSPACE_MANIFEST=<path> with state=<state>. Please confirm RELEASE_NAME=<release-name>, RELEASE_BRANCH=<branch>, RELEASE_TAG=<tag>, binaries=<binaries>, enabled channels=<channels>, and targets=<targets>. I will only edit release metadata and docs until these are confirmed.
```

Action:

- Fill missing release-event and per-binary target fields.
- If a channel is disabled, mark that channel's gates `deferred` when the release branch opens with `deferred_reason = "channel disabled"`.
- Do not create tags, releases, registry packages, or tap changes.

## Release Branch

Prompt:

```text
@<identity> <datetime> -- Next gate is release-branch-created. I will create/update <release-branch>, add releases/manifest.toml, add per-binary manifests and notes, and push the release branch. This does not publish artifacts and will not merge to main.
```

Action:

- Update or create `releases/manifest.toml` and `releases/notes.md`.
- Update or create one `releases/<bin>-<version>.toml` and `releases/<bin>-<version>.md` for every binary in scope.
- Record branch URL and source commit in the workspace manifest gate after the branch exists.
- Record `main_merge_policy = "never-merge-release-branch"` and `main_fix_policy = "cherry-pick-source-fixes-only"` in the workspace manifest.

## Linux Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is <linux-gate> for binary=<bin>, target_id=<target-id>. This requires a Linux host. Pull <release-branch>, build <cargo-package>/<cargo-bin> for <rust-target>, verify artifact names, record toolchain and Linux linkage evidence, and update the per-binary manifest evidence through a sub-PR to <release-branch>.
```

Action:

- Use explicit manifest names such as `archive_asset` and `archive_binary_path`.
- Record source commit, checksums if generated, command notes, `rustc -Vv`, `cargo -V`, and target status evidence.
- For pure-Rust `linux-*-musl`, use the manifest default `rustup + cargo build --target`; use `cargo-zigbuild` only when C dependencies need a musl-aware linker. Record `file <binary>`, `ldd <binary>`, and `readelf -d <binary>` static-link evidence.
- Generate `linux-*-gnu` targets only when the manifest enables gnu fallback/CUDA targets; for those targets, record `ldd --version`, `objdump -T <binary> | grep GLIBC_ | sort -u`, `glibc_build_host_version`, and `glibc_min_version`.
- Do not commit build outputs.

## macOS Signing

Prompt:

```text
@<identity> <datetime> -- Next gate is <darwin-gate> for binary=<bin>, target_id=<target-id>. This requires macOS. Pull <release-branch>, build with --target <rust-target>, run build-path and installed-path codesign checks, then update the per-binary manifest evidence through a sub-PR to <release-branch>.
```

Action:

- Use `.agents/skills/release/references/macos-signing.md`.
- Record `completed_at`, `completed_by`, `source_commit`, `rust_target`, signing identity, and evidence.
- If the final tag commit later differs, repeat or revalidate signing from the final tag.

## npm Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is npm staging for binary=<bin>, target_id=<target-id>. I will generate the native npm package candidate using the manifest package name and bin path, run npm pack dry-run, and update manifest evidence. I will not publish to npm without explicit approval.
```

Action:

- Use manifest `npm_package`, `bin_path`, `runner`, and `node_launcher`.
- If `node_launcher = false`, do not create `<bin>.js` or `<bin>.sh` as the runtime entrypoint.
- Generate native package directories from final artifacts or staging artifacts as appropriate for the gate, run `npm pack --dry-run`, and install-test the generated `.tgz` before publish.
- Record dry-run output, package checksum, install-test output, package name, version, source tag, and provenance/auth mode when available.

## Homebrew Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is Homebrew staging. I will prepare or validate the tap PR shape for motlie/homebrew-tap and record evidence. I will not merge the tap PR or publish bottles before the final source tag exists.
```

Action:

- Use `.agents/skills/release/references/homebrew-tap.md`.
- Keep live formula changes in `motlie/homebrew-tap`.
- Record tap PR URL or formula evidence in the manifest.

## Release Branch Finalization

Prompt:

```text
@<identity> <datetime> -- All required gates appear complete or deferred. Please confirm release-branch finalization. I will identify any source, doc, skill, or tooling fixes that need cherry-picks to main. I will not merge the release branch to main.
```

Action:

- Do not tag or publish without human approval.
- Call out any `planned` or `failed` gates.
- Cherry-pick reusable fixes to normal `main` PRs when needed. Never merge the release branch to `main`.

## Final Tag and GitHub Release

Prompt:

```text
@<identity> <datetime> -- Ready for final tag <release-name> from <release-branch>. This is a tag-centric GitHub Release step. Please approve tag creation and GitHub Release publication from <workspace-manifest>.
```

Action:

- Verify `Cargo.toml`, workspace manifest tag, and per-binary versions.
- Verify `Cargo.lock` is committed and unchanged at the final tag.
- Create and push the tag only after approval.
- Build final artifacts from a detached checkout of the tag, for example `git switch --detach <release-name>`.
- Use manifest asset names for upload.
- Because tags are calver-codenames, explicitly mark the intended stable release with `gh release edit <release-name> --latest` when publishing.

## Installer Validation

Prompt:

```text
@<identity> <datetime> -- Next gate is installer validation for binary=<bin>, target_id=<target-id>. This requires the matching target host. Install from the release-pinned installer URL, execute <bin> --version from the installed path, and update the target-specific installer-validated gate in the per-binary manifest.
```

Action:

- Use the release-pinned GitHub Release installer URL, not a moving Pages URL, for required validation.
- Record installed path, command output, source release tag, installer URL, and checksum evidence.
- If GitHub Pages convenience installer URLs are enabled, update the Pages repository only after the release-pinned installer exists and record Pages verification in the retained release-branch ledger.

## npm Publish

Prompt:

```text
@<identity> <datetime> -- Ready to publish npm packages from final artifacts. Please approve npm auth mode: trusted publishing or temporary NPM_TOKEN. I will publish only packages listed in the per-binary manifests.
```

Action:

- Use `.agents/skills/release/references/npm-auth.md`.
- Publish only after final artifacts and package install checks pass.

## Final Ledger

Prompt:

```text
@<identity> <datetime> -- Release publication is complete. I will update the retained release branch manifests to state=published with final URLs, checksums, npm links, Homebrew tap commit, and install evidence. I will not move the release tag or merge the release branch to main.
```

Action:

- Update only release ledger metadata on the retained release branch.
- Do not move tags or republish unless a separate fix is approved.
