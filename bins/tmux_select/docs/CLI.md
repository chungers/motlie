# tmux_select CLI

## Status

Implemented CLI contract for the initial `tmux_select` binary in
`bins/tmux_select/main.rs`.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
tmux_select [OPTIONS] [SSH_URI]
```

Examples:

```bash
tmux_select
tmux_select --portrait
tmux_select -p
tmux_select --landscape
tmux_select -l
tmux_select --script
tmux_select ssh://user@host
tmux_select 'ssh://user@host?identity-file=/home/user/.ssh/id_ed25519'
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
tmux_select
```

Default mode opens the TUI against the local host. Pressing Enter or `a`
attaches the user's current PTY to the highlighted tmux session. After detach,
the selector re-enters the TUI if the attach child exited successfully or the
selected session still exists. If the child exits non-zero and the selected
session is gone, the selector exits with the child status. `q` / `Ctrl-C` exits
without attach.

For remote targets:

```bash
tmux_select ssh://user@host
```

The TUI runs locally, but all tmux operations target the SSH host. Attach opens
an interactive SSH PTY and runs remote tmux attach against the selected session.

## Script Mode

```bash
tmux attach -t "$(tmux_select --script)"
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
tmux_select --portrait
tmux_select -p
tmux_select --portrait --script
tmux_select --portrait ssh://user@host
```

Portrait mode is optimized for 32 row by roughly 65 column terminals. It uses a
vertical split:

- `T`: session list, default focus
- `B`: detail pane
- one-row status bar

The initial T/B ratio is 30:70, giving the detail pane more vertical space by
default. `Ctrl-Up` / `Ctrl-Down` can resize the split after startup.

MOTD and the motlie placeholder are omitted in portrait mode. Use
`--landscape` / `-l` to force the normal `L`/`R` layout even when the PTY is
auto-detected as portrait.

Without `--portrait` / `-p` or `--landscape` / `-l`, startup calls
`crossterm::terminal::size()` on the connecting PTY and chooses portrait layout
when the character-cell aspect ratio is `columns / rows <= 4.0`. This treats
66x30, 80x24, 100x30, 160x40, and square-ish terminals as portrait layout.
Terminals wider than the 4.0 threshold use landscape layout. If the size cannot
be read, startup defaults to landscape layout.

## TUI Keymap

Normal mode main-view keys:

| Key | `Lb` focused | `R` focused |
|-----|--------------|-------------|
| Up / Down | Move highlighted session | Scroll detail one line |
| PgUp / PgDn | Page session list | Page detail buffer |
| Home / End | First / last session | Top / bottom detail; End resumes monitor tail |
| Right | Focus detail pane | No-op |
| Left | No-op | Focus session list |
| `Esc` | No-op outside modal | Focus session list |
| `Ctrl-Left` / `Ctrl-Right`, `Shift-Left` / `Shift-Right`, Alt Left / Right, or terminal word-left/word-right fallback | Resize L/R split (normal mode only) | Resize L/R split (normal mode only) |
| `m` | Monitor highlighted session | Monitor highlighted session |
| `n` | Open New Session modal | Open New Session modal |
| `k` | Open Kill Session modal | Open Kill Session modal |
| `h` | Open Help modal | Open Help modal |
| Enter / `a` | Attach highlighted session | Attach highlighted session |
| `q` / `Ctrl-C` | Exit without attach | Exit without attach |

Portrait mode maps `T` to `Lb` and `B` to `R`. It uses
`Ctrl-Up` / `Ctrl-Down` to resize `T` / `B`; Alt/Shift modified arrows are
accepted as compatibility fallbacks.

On macOS iTerm2, the resize keys observed during validation are
`Shift-Left` and `Shift-Right` for the normal-mode `L`/`R` split.

The session list auto-refreshes through `HostHandle::watch_host_events()`,
which is currently a one-second polling loop over `list_sessions()` with
stable-id snapshot diffing. It is not currently driven by direct tmux
control-mode host notifications.

The Sessions pane title includes the target host label. The blue status bar
shows current time and compact key hints only; it does not repeat the host,
focus, or layout mode.

Modal keys:

| Key | Behavior |
|-----|----------|
| Left / Right | Choose Cancel or Ok in New Session and Kill Session modals. No-op in Help. |
| Enter | Close modal. Applies Ok when selected in New Session or Kill Session. |
| Esc | Cancel New Session / Kill Session, or close Help. |

The Help modal shows the built-in motlie logo, key functions below the logo,
the build git SHA, and a single Ok button.

## ForceCommand

Recommended local deployment after installing the built binary to
`/usr/local/bin/tmux_select`:

```text
ForceCommand /usr/local/bin/tmux_select
```

Portrait-mode deployment:

```text
ForceCommand /usr/local/bin/tmux_select --portrait
```

Landscape-mode deployment:

```text
ForceCommand /usr/local/bin/tmux_select --landscape
```

By default, `SSH_ORIGINAL_COMMAND` is rejected with a clear stderr message.
Operators can configure an external bypass by allowing
`MOTLIE_TMUX_SELECT_BYPASS=1` for selected users or groups. When both
`SSH_ORIGINAL_COMMAND` and the bypass are present, the binary delegates to the
original command instead of launching the selector.

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
