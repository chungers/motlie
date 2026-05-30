# Operator Prompts

Use this reference when a release task may be handled by different operators on different hosts.

First inspect:

```text
WORKSPACE_MANIFEST=releases/manifest.toml
RELEASE_BRANCH=release/<YYYY-MM-codename>
MASTER_ISSUE=<GitHub issue URL, when created>
BINARY_MANIFESTS=releases/*.toml excluding releases/manifest.toml, requiring kind = "motlie.binary-release"
```

Every prompt should include:

- current branch and git cleanliness
- current workspace and binary manifest `state`
- next incomplete workspace gate or `(binary, gate, target_id)`
- `target_id` when the gate is target-specific
- host/platform required
- exact branch or PR to pull
- master issue and sub-issue links when they exist
- command group to run or files to update
- manifest fields that will be updated
- whether explicit approval is required

## Codename Suggestion

Prompt:

```text
@<identity> <datetime> -- I will start release intake for binaries=<binaries>. I will suggest 3-5 <YYYY-MM-codename> names, check remote branch/tag conflicts, and wait for your selected release name before creating a branch or issues.
```

Action:

- Use the current month unless the human provides a release month.
- Suggest 3-5 codenames without assuming every codename must be adjective-animal.
- Check remote branches and tags for conflicts before presenting the final candidate set.
- Do not create a branch, issue, tag, registry package, or tap change until the human confirms the release name and binary list.

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
@<identity> <datetime> -- Next gate is release-branch-created. I will create/update <release-branch>, add releases/manifest.toml, add stable releases/<bin>.toml manifests and their referenced notes, and push the release branch. This does not publish artifacts and will not merge to main.
```

Action:

- Update or create `releases/manifest.toml` and `releases/notes.md`.
- Update or create one `releases/<bin>.toml` for every binary in scope, with `[identity].version` and `[release].notes_path`.
- Draft per-binary release notes from discovered manifests and release owner input, then aggregate `releases/notes.md`. Ask for missing user-visible summary, notable changes, breaking changes, known issues, and install guidance.
- Copy installer scripts from canonical templates such as `bins/<bin>/install-template.sh` into branch-local `releases/install/install-<bin>.sh` when installer distribution is in scope.
- Record branch URL and source commit in the workspace manifest gate after the branch exists.
- Record `main_merge_policy = "never-merge-release-branch"` and `main_fix_policy = "cherry-pick-source-fixes-only"` in the workspace manifest.

## Master Tracking Issue

Prompt:

```text
@<identity> <datetime> -- Release branch <release-branch> is pushed. I will create the master tracking issue for <release-name>, link manifests and binaries=<binaries>, record the issue URL in releases/manifest.toml, and push the ledger update. The issue coordinates the release, but manifests remain authoritative.
```

Action:

- Create the master issue after release branch and initial manifests exist.
- Include release name, release branch, tag, binaries, manifest files, target matrix, enabled channels, current gates, and current next step.
- State that release branch manifests are authoritative if issue/PR state disagrees.
- Record the master issue URL in workspace manifest tracking metadata and push that commit to the release branch.

## Sub-Issues and Sub-PRs

Prompt:

```text
@<identity> <datetime> -- Next gate requires <host-platform>: binary=<bin>, gate=<gate>, target_id=<target-id>. I will create or update a scoped sub-issue instructing the operator to branch from <release-branch>, update <binary-manifest>, and open a PR back to <release-branch> that closes the sub-issue.
```

Action:

- Create one or more sub-issues for platform/channel/gate work when another operator or host is needed.
- Each sub-issue must name the release branch, binary manifest, gate id, target id, required host/platform, commands or reference docs, expected evidence, and PR base branch.
- Instruct the operator to commit only manifest/status/evidence changes and release-scoped artifacts to the sub-PR; never build outputs.
- Sub-PRs target the release branch, not `main`, and should close their sub-issue on merge.
- Update the master issue with the sub-issue and sub-PR links.

## Release Notes

Prompt:

```text
@<identity> <datetime> -- I will draft or validate release notes for <release-name>. Please provide or approve the user-visible summary, notable changes, breaking changes, known issues, and install guidance for binaries=<binaries>. I will derive names, versions, targets, packages, asset names, and per-binary note paths from discovered manifests.
```

Action:

- Use `.agents/skills/release/references/release-notes.md`.
- Create or update every binary manifest's `[release].notes_path`, then aggregate `releases/notes.md`.
- Check notes against manifests for binary names, versions, targets, package names, install commands, asset names, checksums when final, and known issues.
- Do not publish the GitHub Release if notes still contain placeholders or unapproved claims.

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
@<identity> <datetime> -- All required gates appear complete or deferred in manifests. I will cross-check the master issue, sub-issues, and merged sub-PRs, then summarize any disagreement. Please confirm release-branch finalization. I will not merge the release branch to main.
```

Action:

- Do not tag or publish without human approval.
- Call out any `planned` or `failed` gates.
- Treat manifests as authoritative when issue/PR state disagrees, and comment on the master issue with the correction.
- Cherry-pick reusable fixes to normal `main` PRs when needed. Never merge the release branch to `main`.

## Final Tag and GitHub Release

Prompt:

```text
@<identity> <datetime> -- Ready for final tag <release-name> from <release-branch>. This is a tag-centric GitHub Release step. Please approve tag creation and GitHub Release publication from <workspace-manifest>.
```

Action:

- Verify `Cargo.toml`, workspace manifest tag, and per-binary versions.
- Verify `releases/notes.md` is human-approved and references every per-binary note.
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
- Close the master issue only after the GitHub Release is live, final ledger state is pushed, and required package/install gates are complete or explicitly deferred.
