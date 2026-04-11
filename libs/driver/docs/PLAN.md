# Driver Implementation Plan

## Change Log

| Date | Change |
|------|--------|
| 2026-04-10 | Rewrite the plan from `libs/repl` to `libs/driver`. The plan now treats REPL as one frontend layer, command adapters as feature-gated modules, and the first implementation slice as `Owned + InProcess` only. |

This plan derives from [`DESIGN.md`](./DESIGN.md), [`API.md`](./API.md), and
[`LIFECYCLE.md`](./LIFECYCLE.md).

## Phase 0: Lock Surfaces

### 0.1 Crate naming and boundaries

- [x] Rename `libs/repl` to `libs/driver`.
- [x] Keep `repl` as a frontend submodule, not the umbrella crate.
- [x] Put subsystem adapters under `driver::commands::*`.

### 0.2 Feature topology

- [x] Add root feature gates for:
  - `driver`
  - `driver-repl`
  - `driver-tui`
  - `driver-term-vt100`
  - `driver-term-shadow`
  - `driver-commands-vmm`
  - `driver-commands-vnet`
  - `driver-commands-vfs`
  - `driver-commands-tmux`
- [ ] Decide whether future workflow/composed-command modules need their own feature flags.

### 0.3 First vertical slice

- [x] Lock the first implementation slice to `Owned + InProcess`.
- [x] Keep attach/import/remote management documented but deferred.

## Phase 1: Crate Scaffold

### 1.1 Workspace wiring

- [x] Add `libs/driver` to the workspace.
- [x] Add `motlie-driver` as an optional root dependency.

### 1.2 Module scaffold

- [x] Create the initial modules:
  - `engine.rs`
  - `completion.rs`
  - `clap.rs`
  - `repl.rs`
  - `tui.rs`
  - `term.rs`
  - `commands/{vmm,vnet,vfs,tmux}.rs`
- [x] Re-export the generic surface from `src/lib.rs`.
- [x] Keep future-facing adapter modules such as `commands::vmm` as placeholders only until
  their corresponding resource crates exist on `main`.

### 1.3 Verification

- [ ] Run:
  ```bash
  cargo check -p motlie-driver
  cargo check --bin motlie --features driver
  cargo check --bin motlie --features driver,driver-repl
  ```

## Phase 2: Generic Driver Core

### 2.1 Engine

- [ ] Flesh out `CommandEngine<C, S>` as the owner of mutable runtime context.
- [ ] Make `run_line()` and `run_argv()` the stable frontend-neutral execution entrypoints.
- [ ] Add explicit engine close semantics for owned resources.

### 2.2 `clap` integration

- [ ] Move parse/composition helpers into `clap.rs`.
- [ ] Ensure `root_command()` can build a static command tree from the typed aggregate command enum.
- [ ] Keep the default path typed through `CommandSet<C>` rather than dynamic handler registries.

### 2.3 Completion

- [ ] Implement static completion from `clap`.
- [ ] Implement dynamic completion from typed runtime state.
- [ ] Merge static and dynamic candidates with predictable ordering and de-duplication.

### 2.4 Verification

- [ ] Add focused unit tests for:
  - line tokenization
  - parse failure
  - successful execution
  - dynamic completion
  - explicit engine close for owned child resources

## Phase 3: REPL Frontend

### 3.1 `driver::repl`

- [ ] Wire `ReplFrontend` to `reedline`.
- [ ] Map editor completion requests into `CommandEngine::complete`.
- [ ] Add prompt/history/error rendering.
- [ ] Keep all subsystem lifecycle behavior out of the REPL layer.

### 3.2 Verification

- [ ] Run:
  ```bash
  cargo check -p motlie-driver --features repl
  cargo test -p motlie-driver
  ```

## Phase 4: Command Adapters

### 4.1 tmux

- [ ] Replace the ad hoc tmux example parsing with a `driver::commands::tmux` command family.
- [ ] Add typed tmux session/target registries where needed.
- [ ] Prove dynamic completion for target/session names.

### 4.2 VMM

- [ ] Add `driver::commands::vmm` around the public VMM lifecycle APIs.
- [ ] Model `VmHandle` as the durable root resource.
- [ ] Model PTYs as `Ephemeral` child resources.

### 4.3 VNET

- [ ] Add `driver::commands::vnet` for standalone VNET management if exposed directly.
- [ ] Keep VNET subordinate to VMM where that is the real lifecycle owner.

### 4.4 VFS

- [ ] Add `driver::commands::vfs` only around a truthful wrapper-level lifecycle.
- [ ] Avoid pretending raw `FsServer` is a strong top-level managed resource when it is not.

## Phase 5: Composed Commands

- [ ] Add a first composed command family that spans multiple resources.
- [ ] Make rollback and partial-failure cleanup explicit.
- [ ] Persist only driver-owned registry semantics, not domain logic, in this layer.

Good first examples:

- `workspace up`
- `workspace down`
- `guest attach`
- `session up`

## Phase 6: Additional Frontends

### 6.1 `driver::tui`

- [ ] Add a `ratatui + crossterm` frontend that consumes `CommandEngine`.
- [ ] Reuse the same typed command and completion surfaces where practical.

### 6.2 `driver::term`

- [ ] Expand terminal-buffer support behind feature flags such as `term-vt100` and `term-shadow`.
- [ ] Keep terminal emulation state separate from both driver execution and UI loops.

## Phase 7: Attach / Import / Remote Management

- [ ] Introduce import/rehydrate adapters only after the owned local slice is solid.
- [ ] Start with tmux, where rediscovery is already strong.
- [ ] Add VMM attach/import only when the raw crate exposes a stable enough discovery surface.
- [ ] Add remote admin proxies as a separate locality axis, not as a rewrite of the core engine.

## Done Criteria

The driver design is implemented enough for the first milestone when all of the following are
true:

1. `motlie-driver` builds as a standalone crate.
2. `ReplFrontend` is a thin `reedline` layer over `CommandEngine`.
3. At least one real command adapter module is wired through the generic driver surface.
4. Commands can create named resources and reuse them later through typed registries.
5. Dynamic completion is powered by live runtime state.
6. Engine close cleans up owned child resources explicitly.
