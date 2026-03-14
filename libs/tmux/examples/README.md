# motlie-tmux Examples

Runnable programs demonstrating the `motlie-tmux` API. Each example connects
via an `ssh://` URI and exercises a specific part of the library.

## Prerequisites

- **Working directory**: all commands assume you are in the workspace root (`motlie/`)
- **tmux** installed and on PATH
- For remote hosts: `ssh-agent` running with keys loaded (`ssh-add`)

## Building

Build all examples once, then invoke the binaries directly — avoids
`cargo run` recompilation checks on each invocation.

```sh
cargo build -p motlie-tmux --examples

# Binaries are at target/debug/examples/<name>
./target/debug/examples/uri_connect ssh://localhost
./target/debug/examples/list_sessions ssh://localhost
./target/debug/examples/session_lifecycle ssh://localhost
./target/debug/examples/target_navigate ssh://localhost dev
./target/debug/examples/send_and_capture ssh://localhost
./target/debug/examples/exec_command ssh://localhost "uname -a"
./target/debug/examples/target_spec ssh://localhost "dev:0.0"
./target/debug/examples/repl ssh://localhost
./target/debug/examples/stream_pane ssh://localhost my_session --lines 50
```

## Examples

All examples accept an SSH URI as the first argument. Default: `ssh://localhost`.

### uri_connect — Minimal connection

Parse a URI, connect, verify by listing sessions.

```sh
cargo run -p motlie-tmux --example uri_connect -- ssh://localhost
cargo run -p motlie-tmux --example uri_connect -- ssh://deploy@prod:2222
```

Expected output:
```
Parsing URI: ssh://localhost
  host=localhost, user=, port=22
Connecting...
Connected. 2 active session(s).
```

### list_sessions — Session discovery

Print a table of all active sessions.

```sh
cargo run -p motlie-tmux --example list_sessions -- ssh://localhost
```

Expected output:
```
NAME                 ID       WINDOWS  ATTACHED
--------------------------------------------------
dev                  $0       3        yes
build                $1       1        no
```

### session_lifecycle — Create, rename, kill

Full lifecycle demonstrating that `rename()` returns a new Target handle.

```sh
cargo run -p motlie-tmux --example session_lifecycle -- ssh://localhost
```

Expected output:
```
Creating session 'motlie_example_lifecycle'...
  Created: target=motlie_example_lifecycle, level=Session
  Confirmed in session list.
Renaming to 'motlie_example_renamed'...
  Renamed: target=motlie_example_renamed, session_name=motlie_example_renamed
  Confirmed rename in session list.
Killing session...
  Confirmed session is gone.
Done.
```

### target_navigate — Hierarchy navigation

Walk the session → window → pane tree, printing metadata at each level.

```sh
# Use an existing session:
cargo run -p motlie-tmux --example target_navigate -- ssh://localhost dev

# Or let it create a temporary session with 2 windows:
cargo run -p motlie-tmux --example target_navigate -- ssh://localhost
```

Expected output (temporary session with 2 windows):
```
Created temporary session 'motlie_example_nav' with 2 windows for demo.
Session: motlie_example_nav (level=Session)
  id=$5, windows=2, attached=false

  Windows (2):
    motlie_example_nav:0 (level=Window)
      name='win0', index=0, active=false, panes=1
      Panes (1):
        motlie_example_nav:0.0 (level=Pane)
          pane_id=%10, index=0
    motlie_example_nav:1 (level=Window)
      name='win1', index=1, active=true, panes=1
      Panes (1):
        motlie_example_nav:1.0 (level=Pane)
          pane_id=%11, index=0

Cleaning up temporary session...
```

### send_and_capture — Input and capture

Send text + Enter to a pane, then capture the visible content.

```sh
cargo run -p motlie-tmux --example send_and_capture -- ssh://localhost
```

Expected output:
```
Sending command...
--- Captured pane content ---
$ echo HELLO_FROM_MOTLIE
HELLO_FROM_MOTLIE
$
--- End ---
Output verified.
Done.
```

### exec_command — Structured execution

Run a shell command inside a tmux pane and get structured output.

```sh
cargo run -p motlie-tmux --example exec_command -- ssh://localhost
cargo run -p motlie-tmux --example exec_command -- ssh://localhost "uname -a"
```

Expected output:
```
Executing: echo hello_from_exec
Exit code: 0
Success:   true
Stdout:
hello_from_exec
```

### target_spec — TargetSpec resolution

Parse and resolve a tmux target string against a live server.

```sh
cargo run -p motlie-tmux --example target_spec -- ssh://localhost "dev"
cargo run -p motlie-tmux --example target_spec -- ssh://localhost "dev:0"
cargo run -p motlie-tmux --example target_spec -- ssh://localhost "dev:0.0"
```

Expected output:
```
Parsed TargetSpec: dev:0.0
  session=dev, window=Some("0"), pane=Some(0)

Resolved target: dev:0.0
  level: Pane
  Pane: pane_id=%0, address=dev:0.0
```

### repl — Interactive session manager

Interactive REPL for managing tmux sessions over SSH. Connects to a host,
then accepts commands in a loop. Exercises session lifecycle, target
resolution, text input, and scrollback capture in a single interactive program.

```sh
cargo run -p motlie-tmux --example repl -- ssh://localhost
./target/debug/examples/repl ssh://localhost
```

#### Commands

| Command | Description | API Used |
|---------|-------------|----------|
| `create <name> [--size WxH] [--history N]` | Create a session with optional size and history | `host.create_session()`, `CreateSessionOptions` |
| `kill <target>` | Kill a session, window, or pane | `target.kill()` |
| `targets` | List all sessions with target spec strings | `host.list_sessions()`, `target.children()` |
| `send <target> <text...>` | Send text + Enter to a target | `target.send_text()`, `target.send_keys()` |
| `capture <target> <n>` | Print last N scrollback lines | `target.sample_text(LastLines(n))` |
| `quit` | Disconnect and exit | — |

`create` only creates sessions — the library API (`host.create_session()`) operates
at session level. Windows and panes are assumed to be created out-of-band (e.g. via
`tmux new-window`, `tmux split-window`, or scripted setup). Optional flags:
- `--size WxH` — set initial window dimensions (e.g. `--size 200x50`)
- `--history N` — set scrollback history limit (e.g. `--history 50000`)

All other commands accept a target string at any granularity: `session`,
`session:window`, or `session:window.pane`. The target resolves to the
corresponding level and the command operates there. For example:
- `kill dev` kills the entire session
- `kill dev:0` kills window 0
- `kill dev:0.1` kills pane 1 of window 0
- `send dev:0.1 ls` sends to a specific pane
- `capture dev 10` captures the active pane of the session

#### Expected output

```
Connected to ssh://localhost
repl> targets
  dev                  (Session, 3 windows)
    dev:0              (Window, 'editor', 2 panes)
      dev:0.0          (Pane, %0)
      dev:0.1          (Pane, %1)
    dev:1              (Window, 'shell', 1 pane)
      dev:1.0          (Pane, %2)
    dev:2              (Window, 'logs', 1 pane)
      dev:2.0          (Pane, %3)
repl> create test_session
Created: test_session
repl> create automation --size 200x50 --history 50000
Created: automation (200x50) history=50000
repl> send test_session echo hello from repl
Sent to test_session
repl> capture test_session 5
$ echo hello from repl
hello from repl
$
repl> kill test_session
Killed: test_session
repl> quit
Disconnected.
```

#### Future

The `create` command currently only creates sessions. A full `create <target>`
that builds the entire hierarchy from a target string (e.g. `create myapp:build.1`
would create session `myapp`, window `build`, and split pane `.1`) would require
first-class `new_window()` and `split_pane()` methods on `Target`. The library
does not expose these today — the workaround is `target.exec("tmux new-window ...")`
which shells out rather than using a direct API. Adding `Target::new_window()` and
`Target::split_pane()` to the library would make hierarchical create viable as a
proper inverse of `kill`.

### stream_pane — Continuous pane streaming

Demonstrates the distinct capture and streaming techniques in the library.
Use `--mode` to select a strategy. Ctrl-C exits cleanly in all modes.
Run with `-h` for detailed help on all modes and options.

| Mode | API Used | Behavior |
|------|----------|----------|
| `tail` (default) | `sample_text(LastLines(n))` + `overlap_deduplicate()` | Like `tail -f` — prints only new scrollback lines |
| `visible` | `capture()` | Polls visible pane; reprints on change. Best for TUI programs |
| `until` | `sample_text(Until { pattern, max_lines })` | Scans back to regex match, shows everything since (e.g. last prompt) |
| `fidelity` | `capture_with_options(detect_reflow: true)` | Polls with geometry snapshots; shows content + fidelity status |

```sh
# Default: tail mode (incremental scrollback with overlap dedup)
./target/debug/examples/stream_pane ssh://localhost my_session

# Watch visible pane content (good for TUI programs like htop, vim)
./target/debug/examples/stream_pane ssh://localhost my_session --mode visible

# Tail with custom line count and poll interval
./target/debug/examples/stream_pane ssh://localhost my_session --mode tail --lines 100 --interval 500

# Show everything since last shell prompt (scan backwards until pattern)
./target/debug/examples/stream_pane ssh://localhost my_session --mode until --pattern '^\$ '

# Fidelity mode — try resizing the target terminal to see degradation
./target/debug/examples/stream_pane ssh://localhost my_session --mode fidelity

# Stream a specific pane
./target/debug/examples/stream_pane ssh://localhost "my_session:0.1" --lines 30
```

Expected output (`--mode tail`):
```
Streaming my_session [mode=tail, lines=50, interval=200ms]. Ctrl-C to stop.
$ echo hello
hello
$ make test
running 42 tests...
test result: ok. 42 passed; 0 failed
^C
Stopped.
```

Expected output (`--mode fidelity`, after resizing the target terminal):
```
$ echo hello
hello
$

 DEGRADED: ClientResize, PaneResize
```
