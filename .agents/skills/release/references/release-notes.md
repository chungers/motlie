# Release Notes

Use this reference when drafting or validating branch-local release notes.

Inputs:

```text
WORKSPACE_MANIFEST=releases/manifest.toml
WORKSPACE_NOTES=releases/notes.md
BINARY_MANIFEST=releases/<bin>-<version>.toml
BINARY_NOTES=releases/<bin>-<version>.md
```

Ask the release owner for:

- one-line release summary
- notable user-visible changes per binary
- breaking changes or migration notes
- known issues and workarounds
- install guidance for each enabled channel
- audience-specific warnings such as CUDA, glibc fallback, SSH ForceCommand safety, or macOS signing

Draft `releases/notes.md` with:

```markdown
# <release-name>

## Summary

## Binaries

| Binary | Version | Channels | Targets |
| --- | --- | --- | --- |

## Install

## Changes

## Verification

## Known Issues

## Assets
```

Draft each `releases/<bin>-<version>.md` with:

```markdown
# <bin> <version>

## Summary

## Changes

## Install

## Targets

## Compatibility

## Known Issues
```

Validation rules:

- Derive binary names, versions, targets, package names, asset names, and install commands from manifests.
- Use `git log` only as supporting evidence; do not invent user-facing claims from commit subjects.
- Notes must not contain placeholders before `gh release create`.
- Workspace notes must reference every per-binary note.
- Final notes should mention checksum verification and the release-pinned installer URL when installer distribution is enabled.
- Record human approval and the notes path in the `github-release-published` gate evidence.
