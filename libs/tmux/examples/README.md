# motlie-tmux Examples

Runnable programs demonstrating the `motlie-tmux` API. Each example connects
via an `ssh://` URI and exercises a specific part of the library.

## Prerequisites

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

# Or let it create a temporary session:
cargo run -p motlie-tmux --example target_navigate -- ssh://localhost
```

Expected output (existing session with 2 windows):
```
Session: dev (level=Session)
  id=$0, windows=2, attached=yes

  Windows (2):
    dev:0 (level=Window)
      name='editor', index=0, active=true, panes=2
      Panes (2):
        dev:0.0 (level=Pane)
          pane_id=%0, index=0
        dev:0.1 (level=Pane)
          pane_id=%1, index=1
    dev:1 (level=Window)
      name='shell', index=1, active=false, panes=1
      Panes (1):
        dev:1.0 (level=Pane)
          pane_id=%2, index=0
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
