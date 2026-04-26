# tmux_select CLI

## Status

Draft CLI contract for the planned `tmux_select` binary. No implementation is
included in this PR. After implementation, this file must be updated to reflect
the exact shipped behavior.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-26 | @gpt55-dgx | Initial CLI contract for issue #226 and PR #227: modes, arguments, keymap, stdout/stderr behavior, ForceCommand usage, and exit semantics. |

## Synopsis

```text
tmux_select [OPTIONS] [ssh-uri]
```

Examples:

```bash
tmux_select
tmux_select -s
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
| `-s` | Use compact short mode. Body is split into `T` session list and `B` detail pane. MOTD is omitted. |
| `--print-session` | Select a session, print exactly `<name>\n` to stdout, and exit 0. Cancel exits non-zero with empty stdout. |
| `--dashboard` | After successful attach child exit, re-enter the selector. Non-zero child exit exits with that status. |
| `--print-session --dashboard` | Invalid. Startup error. |

The v1 target form is positional only. There is no `--target` flag in v1.

## Default Mode

```bash
tmux_select
```

Default mode opens the TUI against the local host. Pressing Enter or `g`
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

`--dashboard` re-enters the selector after clean attach child exit. Re-entry is
bounded:

1. attach child exits with success
2. session list refresh succeeds
3. user explicitly selects a session again

Non-zero child exit or refresh failure exits instead of looping.

## Short Mode

```bash
tmux_select -s
tmux_select -s --dashboard
tmux_select -s ssh://user@host
```

Short mode is optimized for 32 row by roughly 65 column terminals. It uses a
vertical split:

- `T`: session list, default focus
- `B`: detail pane
- one-row status bar

MOTD and the motlie placeholder are omitted in short mode.

## TUI Keymap

Normal mode main-view keys:

| Key | `Lb` focused | `R` focused |
|-----|--------------|-------------|
| Up / Down | Move highlighted session | Scroll detail one line |
| PgUp / PgDn | Page session list | Page detail buffer |
| Home / End | First / last session | Top / bottom detail; End resumes monitor tail |
| `v` | Focus detail pane | No-op |
| `l` | No-op | Focus session list |
| `Esc` | No-op outside modal | Focus session list |
| `Ctrl-Left` / `Ctrl-Right` | Resize L/R split | Resize L/R split |
| Left / Right | Reserved no-op | Reserved no-op |
| `m` | Monitor highlighted session | Monitor highlighted session |
| `n` | Open New Session modal | Open New Session modal |
| `k` | Open Kill Session modal | Open Kill Session modal |
| Enter / `g` | Attach highlighted session | Attach highlighted session |
| `Ctrl-C` | Exit without attach | Exit without attach |

Short mode maps `T` to `Lb` and `B` to `R`. It uses
`Ctrl-Up` / `Ctrl-Down` to resize `T` / `B`.

Modal keys:

| Key | Behavior |
|-----|----------|
| Left / Right | Choose Cancel or Ok |
| Enter | Close modal and apply Ok if selected |
| Esc | Cancel |

## ForceCommand

Recommended local deployment:

```text
ForceCommand /usr/local/bin/tmux_select
```

Dashboard deployment:

```text
ForceCommand /usr/local/bin/tmux_select --dashboard
```

Short-mode dashboard deployment:

```text
ForceCommand /usr/local/bin/tmux_select -s --dashboard
```

By default, `SSH_ORIGINAL_COMMAND` is rejected with a clear stderr message.
Operators can configure an external bypass by allowing
`MOTLIE_TMUX_SELECT_BYPASS=1` for selected users or groups. When both
`SSH_ORIGINAL_COMMAND` and the bypass are present, the binary delegates to the
original command instead of launching the selector.

## Exit Semantics

| Condition | Exit behavior |
|-----------|---------------|
| `Ctrl-C` in TUI | Exit without attach. |
| Default attach child exits | Exit with child status. |
| `--dashboard` attach child exits 0 | Re-enter selector after refresh. |
| `--dashboard` attach child exits non-zero | Exit with child status. |
| `--print-session` selection | Print name to stdout, exit 0. |
| `--print-session` cancel | Empty stdout, non-zero exit. |
| Startup argument error | Non-zero exit and stderr message. |

## Output Streams

- stdout is reserved for `--print-session` selected-session output.
- stderr carries TUI rendering, status, diagnostics, and errors.
- default attach and dashboard modes inherit stdio for the attach child after
  terminal cleanup.
