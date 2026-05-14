# Release Notes

Use this reference when drafting or validating branch-local release notes.

Inputs:

```text
WORKSPACE_MANIFEST=releases/manifest.toml
WORKSPACE_NOTES=releases/notes.md
BINARY_MANIFEST=releases/<bin>.toml
BINARY_NOTES=<binary manifest [release].notes_path, usually releases/<bin>.md>
```

Ask the release owner for:

- one-line release summary
- notable user-visible changes per binary
- breaking changes or migration notes
- known issues and workarounds
- install guidance for each enabled channel
- audience-specific warnings such as CUDA, glibc fallback, SSH ForceCommand safety, or macOS signing

Draft each binary manifest's `[release].notes_path` first, then aggregate `releases/notes.md` from those files and release-event summary text.

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

Draft each `releases/<bin>.md` with:

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

- Derive binary names, versions, targets, package names, asset names, note paths, and install commands from discovered per-binary manifests.
- Use `git log` only as supporting evidence; do not invent user-facing claims from commit subjects.
- Notes must not contain placeholders before `gh release create`.
- Workspace notes must reference or summarize every per-binary note discovered from `[release].notes_path`.
- Final notes should mention checksum verification and the release-pinned installer URL when installer distribution is enabled.
- Record human approval and the notes path in the `github-release-published` gate evidence.
