---
name: release
description: Execute or plan Motlie binary releases across GitHub Releases, npm @motlie native packages, direct installers, and the motlie/homebrew-tap Homebrew tap. Use when asked to release, stage, publish, package, codesign, create release PRs, update release docs, or verify release artifacts.
---

# Motlie Release

Use this skill for Motlie binary release work. The release target is parameterized by the binary the human specifies. Do not assume `mmux` unless the human says `mmux` or the task is explicitly the first worked release validation.

Before changing anything:

- identify yourself as required by `AGENTS.md`
- check `git status --short --branch`
- confirm the target branch, release version, and release target
- do not publish npm packages, GitHub Releases, or Homebrew tap changes without explicit human approval
- do not commit unrelated files

Release target fields:

```text
BIN=<installed command name>
CARGO_PACKAGE=<cargo package name>
CARGO_BIN=<cargo binary name>
VERSION=<release version>
INSTALL_PATH=<default absolute install path, if any>
FORMULA=<Homebrew formula name, if enabled>
NPM_PREFIX=@motlie/<package-prefix>
INSTALLER=install-<bin>.sh
FORCE_COMMAND_SAFE=true|false
```

Worked example:

```text
BIN=mmux
CARGO_PACKAGE=motlie-mmux
CARGO_BIN=mmux
INSTALL_PATH=/usr/local/bin/mmux
FORMULA=mmux
NPM_PREFIX=@motlie/mmux
INSTALLER=install-mmux.sh
FORCE_COMMAND_SAFE=true
```

Primary references:

- release playbook: `docs/RELEASES.md`
- design: `docs/DESIGN_RELEASES.md`
- plan: `docs/PLAN_RELEASES.md`
- issue: `https://github.com/chungers/motlie/issues/234`

Default release sequence:

1. Prepare a release PR against `main`.
2. Tag from current `main`.
3. Stage binaries on a GitHub prerelease.
4. Cross-compile Linux and Darwin target archives from Linux.
5. Run a separate macOS signing gate for Darwin artifacts.
6. Finalize GitHub Release assets and checksums.
7. Publish native npm packages under `@motlie`.
8. Update `motlie/homebrew-tap` by PR and publish bottles.
9. Verify installs from final installed paths.

Read references only when needed:

- npm publishing or token handling: `references/npm-auth.md`
- macOS `codesign` or Darwin artifact validation: `references/macos-signing.md`
- Homebrew formula/tap/bottle work: `references/homebrew-tap.md`
- end-to-end checklist: `references/release-checklist.md`

Hard requirements:

- native npm packages expose the requested binary directly; do not add a Node boot script.
- direct installers for host/SSH-safe binaries default to archive mode, not npm mode.
- Darwin binaries must be signed and verified from their installed path.
- npm auth is needed only at `npm publish` time unless trusted publishing is configured.
- release artifacts, npm packages, installer scripts, and Homebrew formulae must all trace back to the same source tag.

Use this response shape for release status:

```text
Release state:
- source tag:
- GitHub Release:
- artifacts:
- macOS signing:
- npm:
- Homebrew:
- remaining gates:
```
