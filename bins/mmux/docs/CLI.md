# mmux CLI

## Status

Implemented CLI contract for the initial `mmux` binary under `bins/mmux/`.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-01 | @codex | Updated the Session Tags modal layout to show up to five scrollable key/value rows styled like the session list, with a visually distinct edit row; `Tab` cycles Key ↔ Value, Enter submits from either edit field, and `c` marks the focused tag row with `✓`. |
| 2026-05-01 | @codex | Added session rename and tag management modals: `r` renames the highlighted session from list focus, `t` opens the unified tag list/add/update/delete modal, and `i` remains unassigned. |
| 2026-04-29 | @opus47-macos-tmux | Updated recency-display semantics: activity column is observer-relative ("time since mmux last saw `session.activity` advance"); age column is `local_now − session.created` under an NTP-synced clock assumption. Wildly skewed host clocks produce mildly inaccurate age text but no functional regression. (Earlier drafts probed the host clock at fleet-connect via `#{epoch}` / `run-shell 'date +%s'`; that approach was abandoned because `run-shell` on tmux ≤ 3.4 corrupts the operator's attached pane.) |
| 2026-04-29 | @opus47-macos-tmux | Added Multi-host Mode section (issue #235): synopsis accepts multiple SSH URIs, mode auto-activates when 2+ hosts are listed, top status reads `mmux - multi-host mode (n)`, session rows insert a hostname column between attached marker and session name, sort is global by activity, all command keys dispatch by highlighted row's host, MOTD pane is hidden in multi-host. |
| 2026-04-28 | @gpt55-dgx | Clarified ForceCommand bypass requires exactly `MOTLIE_MMUX_BYPASS=1` and cross-referenced issue #232 for env-gated SSH integration tests. |
| 2026-04-28 | @gpt55-dgx | Consolidated session-list refresh to a single one-second `list_sessions_now()` poller for both activity ordering and structural reconciliation. |
| 2026-04-28 | @gpt55-dgx | Added a one-second visible-row refresh using `list_sessions_now()` so activity-sorted rows reorder even when no structural tmux event occurs. |
| 2026-04-28 | @gpt55-dgx | Sorted session-list rows by `SessionInfo.activity` descending so the most recently active sessions appear first. |
| 2026-04-28 | @gpt55-dgx | Changed session-list recency rows to unlabeled `<active> / <age>` text with a day bucket and right-side margin. |
| 2026-04-28 | @gpt55-dgx | Documented that session-list recency tolerates tmux empty-epoch expansion through the library fallback clock. |
| 2026-04-28 | @gpt55-dgx | Added aligned session-list recency columns for `active:<elapsed> / age:<elapsed>` using the target tmux server clock. |
| 2026-04-28 | @gpt55-dgx | Changed bottom status command hints from parenthesized mnemonics to underlined shortcut letters, e.g. underlined `h` in `help`. |
| 2026-04-28 | @gpt55-dgx | Documented review cleanup that keeps the CLI contract unchanged while moving CLI/state/detail/render/input/terminal/host helpers out of `main.rs` and hiding internal session ids from list rows. |
| 2026-04-27 | @gpt55-dgx | Updated modal layout: padded content, separator above button bar, bordered New Session input, and Help build metadata before key functions. |
| 2026-04-27 | @gpt55-dgx | Reordered bottom status commands and added `l` to toggle portrait/landscape layout at runtime. |
| 2026-04-27 | @gpt55-dgx | Changed main-view pane cycling from plain Left/Right to the `p` key and updated status hints. |
| 2026-04-27 | @gpt55-dgx | Documented in-memory selector UI state retention across default attach/detach re-entry. |
| 2026-04-27 | @gpt55-dgx | Split resize bounds by layout mode: landscape remains 25/75, portrait becomes 15/85. |
| 2026-04-27 | @gpt55-dgx | Added build date to Help and shortened the displayed git SHA to the last 8 characters. |
| 2026-04-27 | @gpt55-dgx | Shortened bottom status direction hints to `↑/↓ sel` and `←/→ pane`. |
| 2026-04-27 | @gpt55-dgx | Changed top status host/IP separator to `|` and reordered bottom command hints with `(h)elp` first. |
| 2026-04-27 | @gpt55-dgx | Added a top status bar for bold host/IP and right-justified time; Sessions title is now count-only. |
| 2026-04-27 | @gpt55-dgx | Changed plain Left/Right from one-way list/detail focus movement to cyclic pane focus movement. |
| 2026-04-27 | @gpt55-dgx | Renamed the selector command to `mmux` and updated CLI/ForceCommand examples. |
| 2026-04-27 | @gpt55-dgx | Updated Sessions title format to `Sessions [n] @ <hostname>, <ip address>` and removed the `keys` status-bar label. |
| 2026-04-27 | @gpt55-dgx | Moved the host label from the status bar into the Sessions pane title. |
| 2026-04-27 | @gpt55-dgx | Replaced directional words in status hints with arrow symbols and expanded the `h` help modal with key functions. |
| 2026-04-27 | @gpt55-dgx | Changed portrait mode default T/B split from 40:60 to 30:70. |
| 2026-04-26 | @gpt55-dgx | Added the `h` key for an About modal showing the motlie logo and build git SHA; Enter or Esc closes it. |
| 2026-04-26 | @gpt55-dgx | Finalized the CLI mode contract: default mode is attach-and-reenter selector behavior, and `--script` replaces `--print-session` / `--dashboard` for shell integration. |
| 2026-04-26 | @gpt55-dgx | Added `--portrait/-p` and `--landscape/-l` force flags and changed auto-detection to `columns / rows <= 4.0`, making 66x30 portrait. |
| 2026-04-26 | @gpt55-dgx | Set portrait auto-detection to `columns / rows <= 2.0` and embedded the `/tmp/motlie-TOP-CHOICE.txt` glyph as the MOTD-absent fallback icon. |
| 2026-04-26 | @gpt55-dgx | Replaced short mode with portrait mode: `--portrait` is the explicit override, default startup auto-detects PTY aspect ratio, the old `-s` flag is rejected, and the MOTD fallback logo uses the requested Claude artifact ASCII art. |
| 2026-04-26 | @gpt55-dgx | Updated selector keymap for attach on `a`, arrow-key pane focus, Shift-arrow resize behavior on macOS iTerm2, ANSI color in detail mode, polling-backed session refresh, and compact MOTD fallback graphics. |
| 2026-04-26 | @gpt55-dgx | Updated key and dashboard semantics after validation: resize accepts modified-arrow fallbacks when terminals remap Ctrl-arrow, monitor mode mirrors rendered screen content, and dashboard detach is protected against stopped selector jobs. |
| 2026-04-26 | @gpt55-dgx | Updated CLI reference to match implemented binary behavior: `-s`, `--print-session`, `--dashboard`, optional SSH URI, ForceCommand rejection/bypass, stdout/stderr split, and exit semantics. |
| 2026-04-26 | @gpt55-dgx | Initial CLI contract for issue #226 and PR #227: modes, arguments, keymap, stdout/stderr behavior, ForceCommand usage, and exit semantics. |
| 2026-04-26 | @gpt55-dgx | Addressed PR #227 round-3 keymap feedback by marking `Ctrl-Left`/`Ctrl-Right` resize as normal-mode-only in the table. |

## Synopsis

```text
mmux [OPTIONS] [SSH_URI]...
```

Examples:

```bash
mmux                                              # local host
mmux --portrait
mmux -p
mmux --landscape
mmux -l
mmux --script
mmux ssh://user@host                              # single SSH host
mmux 'ssh://user@host?identity-file=/home/user/.ssh/id_ed25519'
mmux ssh://a.example.com ssh://b.example.com      # multi-host (issue #235)
mmux ssh://user@host1 ssh://user@host2 ssh://user@host3
```

## Arguments and Options

| Form | Behavior |
|------|----------|
| no `ssh-uri` | Operate on the local host. |
| `[ssh-uri]` | Operate on the remote SSH target for MOTD, session list, sampling, monitor, create, kill, and attach. |
| `--portrait`, `-p` | Force portrait layout. Body is split into `T` session list and `B` detail pane. MOTD is omitted. Mutually exclusive with `--landscape` / `-l`. |
| `--landscape`, `-l` | Force landscape/normal layout. Body is split into `L`/`R`, with `L` split into MOTD and session list. Mutually exclusive with `--portrait` / `-p`. |
| `--script` | Select a session, print exactly `<name>\n` to stdout, and exit 0 without attaching. Cancel exits non-zero with empty stdout. |
| `--portrait --landscape` | Invalid. Startup error. |

The v1 target form is positional only. There is no `--target` flag in v1.

## Default Mode

```bash
mmux
```

Default mode opens the TUI against the local host. Pressing Enter or `a`
attaches the user's current PTY to the highlighted tmux session. After detach,
the selector re-enters the TUI if the attach child exited successfully or the
selected session still exists. If the child exits non-zero and the selected
session is gone, the selector exits with the child status. `q` / `Ctrl-C` exits
without attach.

During default attach/re-entry, the parent `mmux` process keeps a small
in-memory UI snapshot. On return from tmux detach, the selector restores the
last selected session when it still exists, falls back to the same list index
when it does not, and keeps the user's layout mode, pane split, and focused
pane. This state does not persist across separate `mmux` runs.

For remote targets:

```bash
mmux ssh://user@host
```

The TUI runs locally, but all tmux operations target the SSH host. Attach opens
an interactive SSH PTY and runs remote tmux attach against the selected session.

## Script Mode

```bash
tmux attach -t "$(mmux --script)"
```

`--script` is for shell composition. On selection:

- stdout: selected session name plus one newline
- stderr: TUI, status, and diagnostics
- exit status: 0

On cancel:

- stdout: empty
- stderr: diagnostics if any
- exit status: non-zero

This mode is not appropriate for SSH `ForceCommand` deployments because the
user has no shell to consume stdout.

## Portrait Mode

```bash
mmux --portrait
mmux -p
mmux --portrait --script
mmux --portrait ssh://user@host
```

Portrait mode is optimized for 32 row by roughly 65 column terminals. It uses a
vertical split:

- `T`: session list, default focus
- `B`: detail pane
- one-row top status bar and one-row bottom command/status bar

The initial T/B ratio is 30:70, giving the detail pane more vertical space by
default. `Ctrl-Up` / `Ctrl-Down` can resize the split after startup. Portrait
mode clamps the split so both `T` and `B` keep at least 15% height.

MOTD and the motlie placeholder are omitted in portrait mode. Use
`--landscape` / `-l` to force the normal `L`/`R` layout even when the PTY is
auto-detected as portrait.

Without `--portrait` / `-p` or `--landscape` / `-l`, startup calls
`crossterm::terminal::size()` on the connecting PTY and chooses portrait layout
when the character-cell aspect ratio is `columns / rows <= 4.0`. This treats
66x30, 80x24, 100x30, 160x40, and square-ish terminals as portrait layout.
Terminals wider than the 4.0 threshold use landscape layout. If the size cannot
be read, startup defaults to landscape layout.

## Multi-host Mode (issue #235)

Pass two or more SSH URIs on the command line to enter multi-host mode:

```bash
mmux ssh://a.example.com ssh://b.example.com
mmux ssh://user@host1 ssh://user@host2 ssh://user@host3
```

**Activation rule:**

| `len(ssh_uris)` | Mode |
|---|---|
| `0` | Single-host, target = localhost |
| `1` | Single-host, target = the SSH host |
| `≥ 2` | Multi-host, targets = all listed SSH hosts |

**UX differences in multi-host mode:**

- Top status bar reads `mmux - multi-host mode (n)` (where `n` is the host
  count) instead of the usual `<hostname> | <ip>`.
- Session list rows insert the host's label between the attached marker and
  the session name:

  ```
  > * a.example.com  dev          1m / 12d
    * b.example.com  jarvis       4h / 19d
      a.example.com  build        2d / 5d
      b.example.com  logs         3d / 7d
  ```

  Hostname column width is sized to the widest configured label (capped to
  keep the row readable; longer labels truncated with `…`).
- Sort is `SessionInfo.activity` descending, applied to the **merged** list
  across all hosts.
- All command keys (`Up`/`Down`, `PgUp`/`PgDn`, `Home`/`End`, `Enter`/`a`,
  `m`, `n`, `k`, `Ctrl-C`/`q`, `l`, `p`, `Ctrl-←/→`, `Ctrl-↑/↓`) behave the
  same as single-host. Each applies to the **highlighted row** and dispatches
  against that row's host.
- Attach uses the highlighted row's host: an interactive
  `ssh -t <host> tmux attach-session -t <name>` for SSH targets.
- New session and kill modals act on the highlighted row's host.
- MOTD pane is **hidden** in multi-host mode (per-host MOTD is not
  meaningful when multiple hosts coexist). Layout reflows accordingly.
- Layout flags (`-p`/`-l`) compose with multi-host as expected.

**Resilience:** if one host goes unreachable at refresh time, its rows
disappear from the list and a status banner indicator shows the failure;
other hosts continue to refresh. The failing host re-appears automatically
when it recovers.

**Recency math.** The display has two columns separated by ` / `:

* **Activity** (`active`, the left value) is **observer-relative**. Rather
  than comparing host time to local time, mmux remembers the moment it last
  saw the host's `session.activity` advance and renders "time since that
  moment." Insensitive to operator-vs-host clock skew by construction.
* **Age** (the right value) is `local_now − session.created` under the
  NTP-synced clock assumption. We do not probe the host's clock — there is
  no portable, side-effect-free way to do so on older tmux (`run-shell` on
  tmux ≤ 3.4 corrupts the operator's attached pane). Wildly skewed host
  clocks produce mildly inaccurate age text but no functional regression.

Cross-host *sorting* uses absolute `SessionInfo.activity` timestamps
directly; this is correct as long as host clocks are within typical NTP
drift (seconds).

**Empty case** is unchanged: zero SSH URIs uses localhost; single-host UX
is identical to pre-multi-host behavior.

## TUI Keymap

Normal mode main-view keys:

| Key | `MOTD` focused | `Lb` focused | `R` focused |
|-----|----------------|--------------|-------------|
| Up / Down | No-op | Move highlighted session | Scroll detail one line |
| PgUp / PgDn | No-op | Page session list | Page detail buffer |
| Home / End | No-op | First / last session | Top / bottom detail; End resumes monitor tail |
| `p` | Focus session list | Focus detail pane | Focus MOTD pane |
| Left / Right | No-op | No-op | No-op |
| `Esc` | Focus session list | Focus session list | Focus session list |
| `Ctrl-Left` / `Ctrl-Right`, `Shift-Left` / `Shift-Right`, Alt Left / Right, or terminal word-left/word-right fallback | Resize L/R split (normal mode only) | Resize L/R split (normal mode only) | Resize L/R split (normal mode only) |
| `l` | Toggle portrait/landscape layout | Toggle portrait/landscape layout | Toggle portrait/landscape layout |
| `m` | Monitor highlighted session | Monitor highlighted session | Monitor highlighted session |
| `n` | Open New Session modal | Open New Session modal | Open New Session modal |
| `k` | Open Kill Session modal | Open Kill Session modal | Open Kill Session modal |
| `r` | No-op | Open Rename Session modal | No-op |
| `t` | Open Session Tags modal | Open Session Tags modal | Open Session Tags modal |
| `h` | Open Help modal | Open Help modal | Open Help modal |
| Enter / `a` | Attach highlighted session | Attach highlighted session | Attach highlighted session |
| `q` / `Ctrl-C` | Exit without attach | Exit without attach | Exit without attach |

Portrait mode maps `T` to `Lb` and `B` to `R`; because MOTD is omitted, `p`
cycles between `T` and `B`. It uses
`Ctrl-Up` / `Ctrl-Down` to resize `T` / `B`; Alt/Shift modified arrows are
accepted as compatibility fallbacks. Normal mode L/R resize stays clamped to
25/75; portrait T/B resize is clamped to 15/85.

On macOS iTerm2, the resize keys observed during validation are
`Shift-Left` and `Shift-Right` for the normal-mode `L`/`R` split.

The session list auto-refreshes through one selector-owned poll-backed path.
Visible rows are quietly refreshed once per second with `list_sessions()`,
which updates recency text, activity-descending ordering, and structural
session state in the same snapshot. The selector no longer starts a
separate `watch_host_events()` poller for its TUI list. Direct tmux
control-mode host notifications remain future work.

Each session row includes the display name, an attached-client marker, and a
right-aligned recency column. The attached marker is `*` when tmux reports one
or more clients attached to the session. Rows are sorted by
`activity_observed_at_local` (operator-side wall clock at last observed
`session.activity` advance) descending so the most recently active session
appears
first. The recency column is formatted as `  32h / 14.2d`. The left value
("active") is observer-relative — time since mmux last saw `session.activity`
advance — so it is immune to operator-vs-host clock skew. The right value
("age") is `local_now − session.created` under the NTP-synced clock
assumption — see §Recency math. The recency block is
right-aligned with a small right margin. Durations use `now`, `m`, `h`, or
`d`; days keep at most one decimal digit.
Window-level tmux alert flags such as `!`, `#`, and `~` are not shown in v1.

The top status bar uses the same blue background as the bottom status bar. It
shows `<hostname> | <ip address>` in bold at the left and the current time
right-justified. The Sessions pane title uses `Sessions [n]`, where `n` is the
current session count. The bottom blue status bar shows compact key hints and
status text only. Its direction hints are `↑/↓ sel` for selection and
`pane` for pane focus, with the shortcut letter underlined. It orders command
hints as `help`, `pane`, `monitor`, `enter/attach`, `new`, `kill`, `rename`,
`tags`, `quit`, `layout`, then mode-specific resize; the command
shortcut letters `h`/`p`/`m`/`a`/`n`/`k`/`r`/`t`/`q`/`l` are underlined in the
TUI. It does not repeat the host/time, show focus/layout mode, or prefix the
hints with a `keys` label.

Modal keys:

| Key | Behavior |
|-----|----------|
| Left / Right | Choose Cancel or Ok in New Session, Kill Session, and Rename Session modals. No-op in Help and Session Tags. |
| Tab / Shift-Tab | Cycle Session Tags edit-row focus between Key and Value. |
| Up / Down | Move focus row-to-row in Session Tags; Up from bottom controls returns to the last tag row when present. |
| `u` | In Session Tags, copy the focused row into the bottom Key/Value fields and focus Value. |
| `c` | In Session Tags, toggle the focused row as the sort key with `✓`. |
| `x` | In Session Tags, delete the focused row through the tmux tag API. |
| Enter | Close modal. Applies Ok when selected in New Session, Kill Session, or Rename Session; submits the Session Tags edit row when Key or Value is focused; closes when Cancel is focused. |
| Esc | Cancel action modals, close Session Tags, or close Help. |

Modal content is padded inside the border. New Session and Rename Session render
their text fields with their own borders. Session Tags lists `@mmux/<key>`
values sorted by stripped key in key/value/sort-marker columns. The list shows
up to five rows at a time, scrolls with row focus, and uses the same selected-row
styling as the session list. The key column is sized to the longest key plus
four characters, the marker column displays `✓` when toggled on by `c`, and the
value column takes the remaining width. The bottom Key/Value edit row is
separated from the list, has no marker/add column, and submits with Enter. Tag writes still
require non-empty values. Help
renders the built-in motlie logo, build date, last 8 characters of the build git
SHA, key functions, and a single Ok button. All modal content areas are
separated from the button bar by a horizontal line.

## ForceCommand

Recommended local deployment after installing the built binary to
`/usr/local/bin/mmux`:

```text
ForceCommand /usr/local/bin/mmux
```

Portrait-mode deployment:

```text
ForceCommand /usr/local/bin/mmux --portrait
```

Landscape-mode deployment:

```text
ForceCommand /usr/local/bin/mmux --landscape
```

By default, `SSH_ORIGINAL_COMMAND` is rejected with a clear stderr message.
Operators can configure an external bypass by allowing
`MOTLIE_MMUX_BYPASS=1` for selected users or groups. The value must be exactly
`1`; unset, empty, or any other value is treated as no bypass. When both
`SSH_ORIGINAL_COMMAND` and the exact bypass are present, the binary delegates to
the original command instead of launching the selector.

Env-gated SSH/ForceCommand integration coverage is tracked in
[issue #232](https://github.com/chungers/motlie/issues/232).

## Exit Semantics

| Condition | Exit behavior |
|-----------|---------------|
| `q` or `Ctrl-C` in TUI | Exit without attach. |
| Default attach child exits 0 | Re-enter selector after refresh. |
| Default attach child exits non-zero and selected session still exists | Re-enter selector after refresh. |
| Default attach child exits non-zero and selected session is gone | Exit with child status. |
| `--script` selection | Print name to stdout, exit 0. |
| `--script` cancel | Empty stdout, non-zero exit. |
| Startup argument error | Non-zero exit and stderr message. |

## Output Streams

- stdout is reserved for `--script` selected-session output.
- stderr carries TUI rendering, status, diagnostics, and errors.
- default attach mode inherits stdio for the attach child after terminal
  cleanup.
