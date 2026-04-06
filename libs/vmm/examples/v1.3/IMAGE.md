# v1.3 Guest Image Notes

`v1.3` uses one generic guest image plus per-launch overlay mutation.

The authoritative end-to-end design/status document for this example is
[README.md](./README.md). This file records the image/runtime constraints that
turned out to be easy to regress while bringing up the SSH proxy flow.

## Current Contract

- `artifacts/base/Image` and `artifacts/base/rootfs.squashfs` are shared across
  guests
- guest identity, CA trust material, cloud-init seed data, hostnames, and
  writable root state are launch-time inputs
- the guest image bakes:
  - `sshd` with `TrustedUserCAKeys /etc/ssh/ca/user_ca.pub`
  - `AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u`
  - `motlie-agent-state.service`
  - `motlie-vmm-vsock-ssh.service`
  - the generic `alice` / `bob` accounts and validation tooling

## Non-Regression Notes

### 1. Overlay ownership and mode matter to sshd

The launch overlay created by [launch-ch.sh](./launch-ch.sh) must preserve
OpenSSH's strict ownership and permission checks.

Required behavior:

- create the overlay seed under `umask 022`
- ensure intermediate directories are `755`, not `775`
- ensure `/etc/ssh/auth_principals` is `root:root` and `755`
- ensure `/etc/ssh/auth_principals/<user>` files are `644`
- ensure `/etc/ssh/ca/user_ca.pub` is `644`
- build the runtime overlay with `fakeroot mkfs.ext4 -d ...` so the injected SSH
  CA and principals files are seen as `root:root` in the guest

Why this matters:

- guest sshd checks ownership and mode on every directory component up to `/`
- group-writable paths like `775` can cause cert/principals auth to fail even
  when the file contents are correct
- non-root ownership on the CA or principals files is rejected by sshd

### 2. Agent-state setup is boot-time, not login-time

`motlie-agent-state-setup` must keep:

- `~/.codex -> /agent-state/codex`
- `~/.claude -> /agent-state/claude`
- `~/.config/claude-code -> /agent-state/claude-code`

and must also create:

- `~/.config` as `0755`
- `/agent-state/codex`, `/agent-state/codex/sqlite`, `/agent-state/claude`,
  `/agent-state/claude-code` as `0700`

Why this matters:

- if these paths are created in the disk-backed home instead of symlinked into
  `/agent-state`, tool auth/state silently lands in the wrong layer
- if ownership is wrong, Codex/Claude can read but fail to refresh/update state

### 3. The proxy's ephemeral user cert must include `permit-pty`

The SSH proxy in [`ca.rs`](../../src/ca.rs) signs an ephemeral user
certificate per guest connection. That cert must include OpenSSH user-cert
extensions, especially:

- `permit-pty`

The current implementation also includes:

- `permit-user-rc`
- `permit-port-forwarding`
- `permit-agent-forwarding`
- `permit-X11-forwarding`

Why this matters:

- without `permit-pty`, guest sshd accepts the login/exec path but rejects PTY
  allocation
- the observed symptom was:
  - plain `ssh -p 2222 alice@localhost` hung after MOTD
  - `/bin/cat -v` showed raw `^M` / literal `^D`
  - proxy logs showed guest-side `Failure` followed by `Success`
- once `permit-pty` was added, proxied PTY allocation succeeded and the normal
  interactive shell path worked again

### 4. PTY debugging should compare direct guest SSH with proxied SSH

When debugging PTY behavior, compare:

```bash
ssh -tt alice@192.168.249.2 /bin/cat -v
ssh -tt -p 2222 alice@localhost /bin/cat -v
```

or use the local probe:

```bash
cargo run -p motlie-vmm --example russh_pty_probe -- 192.168.249.2:22 '/bin/cat -v' true
cargo run -p motlie-vmm --example russh_pty_probe -- 127.0.0.1:2222 '/bin/cat -v' true
```

Why this matters:

- it separates guest-image/sshd problems from proxy problems quickly
- in this bring-up, direct guest SSH was correct while proxied SSH was broken,
  which isolated the bug to the proxy cert/session path

## Known Remaining Caveat

The current proxy PTY path is working, but the localhost proxy still treats
`request_pty(true)` / `request_shell(true)` / `exec(true)` as if the future
itself confirms remote success. In `russh`, those calls enqueue the request; the
actual remote `Success` / `Failure` arrives later as channel messages.

That does not block `v1.3` bring-up now that the guest really accepts PTYs, but
future refactors should be careful not to assume those method calls are a
round-trip acknowledgement.
