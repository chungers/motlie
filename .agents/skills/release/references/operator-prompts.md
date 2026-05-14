# Operator Prompts

Use this reference when a release task may be handled by different operators on different hosts.

First inspect:

```text
MANIFEST=releases/<bin>/<version>.toml
RELEASE_BRANCH=release/<bin>-v<version>
```

Every prompt should include:

- current branch and git cleanliness
- current manifest `state`
- next incomplete gate
- `target_id` when the gate is target-specific
- host/platform required
- exact branch or PR to pull
- command group to run or files to update
- manifest fields that will be updated
- whether explicit approval is required

## Intake

Prompt:

```text
@<identity> <datetime> -- I found MANIFEST=<path> with state=<state>. Please confirm BIN=<bin>, VERSION=<version>, RELEASE_BRANCH=<branch>, enabled channels=<channels>, and targets=<targets>. I will only edit release metadata and docs until these are confirmed.
```

Action:

- Fill missing release target fields.
- If a channel is disabled, mark that channel's gates `deferred` at coordination-PR-open time with `deferred_reason = "channel disabled"`.
- Do not create tags, releases, registry packages, or tap changes.

## Coordination PR

Prompt:

```text
@<identity> <datetime> -- Next gate is release-pr-opened. I will create/update <release-branch>, add <manifest>, add release notes, and open a coordination PR to main. This does not publish artifacts.
```

Action:

- Update or create `releases/<bin>/<version>.toml`.
- Update or create `releases/<bin>/<version>.md`.
- Record PR URL and source commit in the manifest gate after the PR exists.
- Use `merge_strategy = "merge-commit"` in the manifest so sub-PR evidence remains visible after the coordination PR merges.

## Linux Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is <linux-gate> for target_id=<target-id>. This requires a Linux host. Pull <release-branch>, build <cargo-package>/<cargo-bin> for <rust-target>, verify artifact names, record toolchain and Linux linkage evidence, and update manifest evidence through a sub-PR to <release-branch>.
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
@<identity> <datetime> -- Next gate is <darwin-gate> for target_id=<target-id>. This requires macOS. Pull <release-branch>, build with --target <rust-target>, run build-path and installed-path codesign checks, then update manifest evidence through a sub-PR to <release-branch>.
```

Action:

- Use `.agents/skills/release/references/macos-signing.md`.
- Record `completed_at`, `completed_by`, `source_commit`, `rust_target`, signing identity, and evidence.
- If the final tag commit later differs, repeat or revalidate signing from the final tag.

## npm Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is npm staging for target_id=<target-id>. I will generate the native npm package candidate using the manifest package name and bin path, run npm pack dry-run, and update manifest evidence. I will not publish to npm without explicit approval.
```

Action:

- Use manifest `npm_package`, `bin_path`, `runner`, and `node_launcher`.
- If `node_launcher = false`, do not create `<bin>.js` or `<bin>.sh` as the runtime entrypoint.
- Record dry-run evidence.

## Homebrew Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is Homebrew staging. I will prepare or validate the tap PR shape for motlie/homebrew-tap and record evidence. I will not merge the tap PR or publish bottles before the final source tag exists.
```

Action:

- Use `.agents/skills/release/references/homebrew-tap.md`.
- Keep live formula changes in `motlie/homebrew-tap`.
- Record tap PR URL or formula evidence in the manifest.

## Coordination Merge

Prompt:

```text
@<identity> <datetime> -- All required gates appear complete or deferred. Please confirm whether to merge the coordination PR to main. After merge, final artifacts must be built from the final tag, not merely from staging evidence.
```

Action:

- Do not merge without human approval.
- Call out any `planned` or `failed` gates.
- Use a merge commit. Do not squash or rebase the release coordination branch.

## Final Tag and GitHub Release

Prompt:

```text
@<identity> <datetime> -- Ready for final tag v<version> from main. This is a tag-centric GitHub Release step. Please approve tag creation and GitHub Release publication from <manifest>.
```

Action:

- Verify `Cargo.toml`, manifest version, and manifest tag.
- Verify `Cargo.lock` is committed and unchanged at the final tag.
- Create and push the tag only after approval.
- Build final artifacts from a detached checkout of the tag, for example `git switch --detach v<VERSION>`.
- Use manifest asset names for upload.

## Installer Validation

Prompt:

```text
@<identity> <datetime> -- Next gate is installer validation for target_id=<target-id>. This requires the matching target host. Install from the release-pinned installer URL, execute <bin> --version from the installed path, and update the target-specific installer-validated gate in <manifest>.
```

Action:

- Use the release-pinned GitHub Release installer URL, not a moving Pages URL, for required validation.
- Record installed path, command output, source release tag, installer URL, and checksum evidence.
- If GitHub Pages convenience installer URLs are enabled, update the Pages repository only after the release-pinned installer exists and record Pages verification in the post-release ledger PR.

## npm Publish

Prompt:

```text
@<identity> <datetime> -- Ready to publish npm packages from final artifacts. Please approve npm auth mode: trusted publishing or temporary NPM_TOKEN. I will publish only packages listed in <manifest>.
```

Action:

- Use `.agents/skills/release/references/npm-auth.md`.
- Publish only after final artifacts and package install checks pass.

## Post-Release Ledger

Prompt:

```text
@<identity> <datetime> -- Release publication is complete. I will open a ledger PR updating <manifest> to state=published with final URLs, checksums, npm links, Homebrew tap commit, and install evidence. I will not move the release tag.
```

Action:

- Update only release ledger metadata.
- Do not move tags or republish unless a separate fix is approved.
