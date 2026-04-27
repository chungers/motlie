# tmux_select CLI

## Status

Implemented CLI contract for the initial `tmux_select` binary in
`bins/tmux_select/main.rs`.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
tmux_select --dashboard
tmux_select --print-session
tmux_select ssh://user@host
tmux_select 'ssh://user@host?identity-file=/home/user/.ssh/id_ed25519'
```

## Arguments and Options

| Form | Behavior |
|------|----------|
| no `ssh-uri` | Operate on the local host. |
| `[ssh-uri]` | Operate on the remote SSH target for MOTD, session list, sampling, monitor, create, kill, and attach. |
| `--portrait` | Force portrait layout. Body is split into `T` session list and `B` detail pane. MOTD is omitted. Without this flag, layout is auto-detected from the current PTY dimensions. |
| `--print-session` | Select a session, print exactly `<name>\n` to stdout, and exit 0. Cancel exits non-zero with empty stdout. |
| `--dashboard` | Re-enter the selector after a clean attach child exit, or after a non-zero detach when the selected session still exists. Other non-zero child exits exit with that status. |
| `--print-session --dashboard` | Invalid. Startup error. |

The v1 target form is positional only. There is no `--target` flag in v1.

## Default Mode

```bash
tmux_select
```

Default mode opens the TUI against the local host. Pressing Enter or `a`
attaches the user's current PTY to the highlighted tmux session. The selector
restores terminal state and exits with the attach child status.

For remote targets:

```bash
tmux_select ssh://user@host
```

The TUI runs locally, but all tmux operations target the SSH host. Attach opens
an interactive SSH PTY and runs remote tmux attach against the selected session.

## Print-Session Mode

```bash
tmux attach -t "$(tmux_select --print-session)"
```

`--print-session` is for shell composition. On selection:

- stdout: selected session name plus one newline
- stderr: TUI, status, and diagnostics
- exit status: 0

On cancel:

- stdout: empty
- stderr: diagnostics if any
- exit status: non-zero

This mode is not appropriate for SSH `ForceCommand` deployments because the
user has no shell to consume stdout.

## Dashboard Mode

```bash
tmux_select --dashboard
```

`--dashboard` re-enters the selector after clean attach child exit. It also
re-enters after a non-zero attach child exit when the selected session still
exists, which covers tmux/SSH detach paths that report non-zero even though the
session survived. Re-entry is bounded:

1. attach child exits with success, or the selected session still exists after
   a non-zero child exit
2. session list refresh succeeds
3. user explicitly selects a session again

Non-zero child exit with no selected session remaining, or refresh failure,
exits instead of looping.

## Portrait Mode

```bash
tmux_select --portrait
tmux_select --portrait --dashboard
tmux_select --portrait ssh://user@host
```

Portrait mode is optimized for 32 row by roughly 65 column terminals. It uses a
vertical split:

- `T`: session list, default focus
- `B`: detail pane
- one-row status bar

MOTD and the motlie placeholder are omitted in portrait mode.

Without `--portrait`, startup calls `crossterm::terminal::size()` on the
connecting PTY and chooses portrait layout when the character-cell aspect ratio
is narrow (`columns / rows < 2.2`). This treats typical 80x24 and 100x30
terminals as normal landscape layout, while narrow or square-ish terminals use
portrait layout. If the size cannot be read, startup defaults to normal layout.

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

Modal keys:

| Key | Behavior |
|-----|----------|
| Left / Right | Choose Cancel or Ok |
| Enter | Close modal and apply Ok if selected |
| Esc | Cancel |

## ForceCommand

Recommended local deployment after installing the built binary to
`/usr/local/bin/tmux_select`:

```text
ForceCommand /usr/local/bin/tmux_select
```

Dashboard deployment:

```text
ForceCommand /usr/local/bin/tmux_select --dashboard
```

Portrait-mode dashboard deployment:

```text
ForceCommand /usr/local/bin/tmux_select --portrait --dashboard
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
| Default attach child exits | Exit with child status. |
| `--dashboard` attach child exits 0 | Re-enter selector after refresh. |
| `--dashboard` attach child exits non-zero and selected session still exists | Re-enter selector after refresh. |
| `--dashboard` attach child exits non-zero and selected session is gone | Exit with child status. |
| `--print-session` selection | Print name to stdout, exit 0. |
| `--print-session` cancel | Empty stdout, non-zero exit. |
| Startup argument error | Non-zero exit and stderr message. |

## Output Streams

- stdout is reserved for `--print-session` selected-session output.
- stderr carries TUI rendering, status, diagnostics, and errors.
- default attach and dashboard modes inherit stdio for the attach child after
  terminal cleanup.
