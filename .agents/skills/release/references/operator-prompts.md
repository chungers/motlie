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

## Linux Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is <linux-gate>. This requires a Linux host. Pull <release-branch>, build <cargo-package>/<cargo-bin>, verify artifact names from the manifest, and update manifest evidence through a sub-PR to <release-branch>.
```

Action:

- Use explicit manifest names such as `archive_asset` and `archive_binary_path`.
- Record source commit, checksums if generated, command notes, and evidence.
- Do not commit build outputs.

## macOS Signing

Prompt:

```text
@<identity> <datetime> -- Next gate is <darwin-gate>. This requires macOS. Pull <release-branch>, run build-path and installed-path codesign checks, then update manifest evidence through a sub-PR to <release-branch>.
```

Action:

- Use `.agents/skills/release/references/macos-signing.md`.
- Record `completed_at`, `completed_by`, `source_commit`, and evidence.
- If the final tag commit later differs, repeat or revalidate signing from the final tag.

## npm Staging

Prompt:

```text
@<identity> <datetime> -- Next gate is npm staging. I will generate native npm package candidates using manifest package names and bin paths, run npm pack dry-runs, and update manifest evidence. I will not publish to npm without explicit approval.
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

## Final Tag and GitHub Release

Prompt:

```text
@<identity> <datetime> -- Ready for final tag v<version> from main. This is a tag-centric GitHub Release step. Please approve tag creation and GitHub Release publication from <manifest>.
```

Action:

- Verify `Cargo.toml`, manifest version, and manifest tag.
- Create and push the tag only after approval.
- Build final artifacts from the tag.
- Use manifest asset names for upload.

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
