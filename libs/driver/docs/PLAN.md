# Driver Plan

## Current State

The first feasibility milestone is largely complete.

Implemented:
- `libs/driver` crate
- `CommandEngine<C, S>`
- `CommandSet<C>` with `CompletionContext`, `help()`, and async `execute()`
- `clap` integration for parse/help/completion analysis
- generic retained history buffer
- driver-owned asciicast recorder
- tmux adapter in `driver::commands::tmux`
- shared tmux REPL/TUI frontends in `driver::tmux_frontend`
- top-level `bins/tmux/driver` package
- host-backed tmux smoke-test artifacts under `bins/tmux/driver/validation`

Not implemented:
- VMM/VNET/VFS adapters on `main`
- imported/remote resource management
- generic non-tmux TUI abstraction
- frame-accurate alternate-screen recording

## Completed Phases

### 1. Crate and workspace wiring

- [x] Rename `libs/repl` to `libs/driver`
- [x] Add `motlie-driver` to the workspace
- [x] Add top-level `bins/tmux/driver`

### 2. Generic driver core

- [x] Implement `CommandEngine<C, S>`
- [x] Implement `CommandSet<C>` with sync completion snapshots
- [x] Implement built-in `help` / `quit` / `exit`
- [x] Implement static + dynamic completion merge
- [x] Add unit coverage for `clap.rs` and `engine.rs`

### 3. tmux vertical slice

- [x] Implement `driver::commands::tmux`
- [x] Add retained mirror history
- [x] Add `mirror history` / `mirror clear`
- [x] Add attached plain-REPL follow behavior
- [x] Add `monitor start` / `monitor stop`
- [x] Surface polling errors in stream state instead of silently swallowing them

### 4. Shared frontends

- [x] Move the real tmux REPL/TUI loops into library code
- [x] Remove duplicated bin/example frontend implementations
- [x] Keep top-level binary and examples as thin assembly layers

### 5. Validation

- [x] Build/test `motlie-driver`
- [x] Build selective tmux-driver feature combinations
- [x] Capture host-backed smoke-test artifacts
- [x] Add driver-owned asciicast recording



## Proposed Next Phase: Semantic Resolution And Namespaced Contexts

Goal:
- make name resolution a first-class driver concern instead of an ad hoc per-adapter string hack

Why this is next:
- tmux multi-host mode needs `connect <ssh> as <alias>` and `alias/<target>` semantics
- future VMM commands will likely need namespace-aware guest references as the surface grows
- bolting namespace parsing into each adapter `execute()` path would duplicate logic and drift over time

Planned core tasks:
- [ ] extend the `CommandSet<C>` contract with an optional sync `resolve_command()` stage ([Planned Semantic Resolution Stage](DESIGN.md#planned-semantic-resolution-stage), [Proposed trait shape](DESIGN.md#proposed-trait-shape), [Sync resolution rule](DESIGN.md#sync-resolution-rule))
- [ ] keep identity/no-op resolution as the default so current namespace-less adapters still work with only one-line boilerplate ([Compositional requirement](DESIGN.md#compositional-requirement))
- [ ] add a small generic naming module under `libs/driver` for qualified-name parsing and concrete generic resolution errors ([Planned Naming / Resolution Support In `libs/driver`](DESIGN.md#planned-naming--resolution-support-in-libsdriver))
- [ ] document the context-side resolution helper pattern for adapters and app-level command sets ([Shared namespace, adapter-relative resolution](DESIGN.md#shared-namespace-adapter-relative-resolution))
- [ ] update completion guidance so scoped-name completion follows the same rules as scoped-name resolution ([Completion Impact](DESIGN.md#completion-impact))

Planned tmux proving slice:
- [ ] add an opt-in `--multi-host` mode to `bins/tmux/driver` while keeping single-host mode as the default ([Verification Slice: tmux Multi-host Namespaced Mode](DESIGN.md#verification-slice-tmux-multi-host-namespaced-mode))
- [ ] add app-level commands as an outer command family:
  - [ ] `connect <ssh-uri> as <alias>`
  - [ ] `disconnect <alias>`
  - [ ] `use <alias>`
  - [ ] `connections`
  - references: [Planned composed command-family shape](DESIGN.md#planned-composed-command-family-shape), [Lifecycle rule for `disconnect`](DESIGN.md#lifecycle-rule-for-disconnect)
- [ ] support namespaced tmux entities like `alias/<target>` and preserve bare-name compatibility once `current` is set ([Verification Slice: tmux Multi-host Namespaced Mode](DESIGN.md#verification-slice-tmux-multi-host-namespaced-mode))
- [ ] support current-connection fallback when the user omits an alias ([Verification Slice: tmux Multi-host Namespaced Mode](DESIGN.md#verification-slice-tmux-multi-host-namespaced-mode))
- [ ] add dynamic completion for connection aliases and scoped tmux targets ([Completion Impact](DESIGN.md#completion-impact))

Acceptance:
- a command set that does not need semantic resolution can still behave exactly like the current design
- adapters that do need semantic resolution can express it without re-implementing parsing hacks
- the tmux driver can demonstrate one engine session managing multiple SSH hosts by alias
- the design is reusable by future VMM guest/namespace-aware command sets

## Remaining Work

### Near-term

- [ ] Run another live smoke test after the `monitor start` subcommand change and update the
      validation artifacts if behavior/output changed materially
- [ ] Decide whether tmux TUI recording should grow beyond command-flow events into richer
      alternate-screen capture
- [ ] Decide whether owned tmux sessions should ever be auto-cleaned on driver exit

### Future adapters

- [ ] Add a real VMM adapter once the VMM crate lands on `main`
- [ ] Add VNET/VFS adapters only when their lifecycle shape is truthful on `main`
- [ ] Add higher-level composed commands once more than one adapter exists

### Future lifecycle work

- [ ] Introduce attach/import semantics when a real backing implementation exists
- [ ] Add remote admin/proxy lifecycle support only after the local owned slice is stable

## Done Criteria For This PR

This PR is in good shape when:

1. the library and top-level tmux driver build across the supported feature matrix
2. the tmux adapter has unit coverage for completion/help/basic command behavior
3. duplicated tmux frontend code is removed
4. the docs describe the implemented system, not the speculative earlier design
