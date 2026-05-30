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

Draft each binary manifest's `[release].notes_path` first, then aggregate `releases/notes.md` from those files, release-event summary text, final manifest state, and relevant sub-issue/sub-PR evidence.

**All links in `releases/notes.md` must be absolute URLs.** GitHub renders release-notes Markdown as the release body but does **not** rewrite relative links to point at uploaded assets or repo files — a relative `(mmux.md)` in the rendered body resolves to a path under `https://github.com/<owner>/<repo>/releases/tag/<tag>/` that does not exist and 404s for viewers. Use:

- `https://github.com/<owner>/<repo>/blob/<release-tag>/releases/<bin>.md` for per-binary notes — renders as Markdown on github.com.
- `https://github.com/<owner>/<repo>/releases/download/<release-tag>/<asset>` for raw release assets (archives, checksums, installers).
- `https://github.com/<owner>/<repo>/issues/<N>` and `.../pull/<N>` for cross-references.

Per-binary notes files (`releases/<bin>.md`) follow the same rule because they are also uploaded as release assets and rendered through the same release surface; relative links to other repo files break for the same reason.

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
- **All links must be absolute URLs**; relative Markdown links 404 in the rendered release body. Per-binary notes use `https://github.com/<owner>/<repo>/blob/<release-tag>/releases/<bin>.md`. Reject any link whose URL portion does not start with `https://`. Verify by previewing the rendered body (`gh release view <tag> --json body --jq .body | grep -E '\\((?!https://)'`) before approving for `gh release create`.
- Final notes should mention checksum verification and the release-pinned installer URL when installer distribution is enabled.
- Final notes should be regenerated after required sub-PRs merge so target/package/checksum/install details match the manifest ledger.
- If the release body needs correction after publication, use `gh release edit <tag> --notes-file <updated>.md` and `gh release upload <tag> <updated>.md --clobber`; do NOT move the release tag.
- Record human approval and the notes path in the `github-release-published` gate evidence.
