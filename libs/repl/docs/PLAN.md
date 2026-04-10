# Unified REPL Engine Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-10 | @codex-repl | Add feature-gating and Cargo-composition work so subsystem crates can be selectively compiled and registered in the REPL command surface. |
| 2026-04-10 | @codex-repl | Align the PLAN with the command-engine-first design: explicit engine-local resource management, frontend-neutral execution, attach/detach semantics, and a new [`LIFECYCLE.md`](./LIFECYCLE.md) resource inventory for tmux plus future VMM/VNET/VFS adopters. |
| 2026-04-10 | @codex-repl | Initial PLAN for `libs/repl`, derived from [`DESIGN.md`](./DESIGN.md). Uses greenfield product direction, phased delivery, explicit design references per task, and concrete verification guidance. |

Derived from [`DESIGN.md`](./DESIGN.md). This PLAN translates the approved design into
phased implementation work for a new `libs/repl` crate, tmux adoption, and optional `motlie`
binary integration. The new [`LIFECYCLE.md`](./LIFECYCLE.md) records the reviewed resource
inventory and management/cleanup expectations for current and future adopters. All tasks
below reference the design sections they implement.

---

## Phase 0: Lock Product and API Decisions

Resolve the open design questions before implementation spreads them across the codebase.

### 0.1 Product entrypoint and command shape

- [ ] Choose the first binary entrypoint as `motlie repl` or `motlie tmux repl`, then record
  the decision in `DESIGN.md` and keep `PLAN.md` aligned.
  Ref: [`Open Questions`](./DESIGN.md#open-questions), [`CLI Example`](./DESIGN.md#cli-example), [`Chosen Solution`](./DESIGN.md#chosen-solution)

### 0.2 Built-in command namespace policy

- [ ] Decide whether `help` and `quit` are reserved built-ins or whether subsystem command
  names may shadow them under explicit namespacing rules.
  Ref: [`Chosen Solution`](./DESIGN.md#chosen-solution), [`Open Questions`](./DESIGN.md#open-questions)

### 0.3 Typed registration API direction

- [ ] Choose whether `libs/repl` exposes clap-native typed registration
  (`CommandFactory` + `FromArgMatches`) directly or wraps them in a Motlie-specific trait.
  Record the chosen API in `DESIGN.md` with one short code snippet.
  Ref: [`Command Registration Model`](./DESIGN.md#command-registration-model), [`Open Questions`](./DESIGN.md#open-questions), [`API Ergonomics`](./DESIGN.md#api-ergonomics)

### 0.4 Session-state management and cleanup policy

- [ ] Lock the engine-side resource model for named resources:
  `Owned`, `Imported`, and `Ephemeral`, plus whether locality is tracked explicitly as
  `InProcess` vs `RemoteProxy`. Document how explicit engine close interacts with subsystem
  shutdown APIs.
  Ref: [`Session State Model`](./DESIGN.md#session-state-model), [`Lifecycle and Rehydration Model`](./DESIGN.md#lifecycle-and-rehydration-model), [`LIFECYCLE.md`](./LIFECYCLE.md)

- [ ] Decide the first attach/rehydrate scope to support explicitly:
  tmux-only attach, or a more generic resource-import surface.
  Ref: [`Open Questions`](./DESIGN.md#open-questions), [`LIFECYCLE.md`](./LIFECYCLE.md#attach-rehydrate-and-detach)

### 0.5 Phase-0 verification

- [ ] Verify the decision set is reflected consistently in both docs before code starts.
  Run:
  ```bash
  rg -n "motlie repl|motlie tmux repl|help|quit|CommandFactory|FromArgMatches|Owned|Imported|Ephemeral|RemoteProxy|rehydrate|repl-vmm|repl-tmux|optional = true|required-features" libs/repl/docs
  ```
  Ref: [`Summary`](./DESIGN.md#summary), [`Chosen Solution`](./DESIGN.md#chosen-solution)

### 0.6 Feature-flag topology

- [ ] Lock the first feature-flag topology for REPL composition in the root package:
  `repl`, per-subsystem flags such as `repl-tmux` and `repl-vmm`, and whether an aggregate
  `repl-all` feature exists.
  Ref: [`Feature Gating and Cargo Composition`](./DESIGN.md#feature-gating-and-cargo-composition), [`Open Questions`](./DESIGN.md#open-questions)

- [ ] Decide whether any subsystem needs finer-grained bundle features at the REPL layer, or
  whether the first slice should stay at one flag per subsystem crate.
  Ref: [`Feature Gating and Cargo Composition`](./DESIGN.md#feature-gating-and-cargo-composition), [`Open Questions`](./DESIGN.md#open-questions)

---

## Phase 1: Workspace and Crate Scaffolding

Create the `libs/repl` crate and baseline module structure without yet migrating any
consumer.

### 1.1 Workspace wiring

- [ ] Add `libs/repl` to the workspace members list in the root `Cargo.toml`.
  Ref: [`Crate Layout`](./DESIGN.md#crate-layout), [`Dependency Inventory`](./DESIGN.md#dependency-inventory)

- [ ] Add the initial `libs/repl/Cargo.toml` with `reedline`, `clap`, `shlex` or equivalent,
  `thiserror`, and `tokio` as needed by the final API shape.
  Ref: [`Dependency Inventory`](./DESIGN.md#dependency-inventory), [`Chosen Solution`](./DESIGN.md#chosen-solution)

- [ ] Add optional root-package dependencies and feature wiring for `motlie-repl` and the
  first subsystem adopters, using `dep:`-style feature expansion where applicable.
  Ref: [`Feature Gating and Cargo Composition`](./DESIGN.md#feature-gating-and-cargo-composition), [`Cargo and Binary Walkthrough`](./DESIGN.md#cargo-and-binary-walkthrough)

### 1.2 Source skeleton

- [ ] Create `src/lib.rs` and the initial modules named in the design:
  `repl.rs`, `command.rs`, `completion.rs`, `tokenize.rs`, and `error.rs`.
  Ref: [`Crate Layout`](./DESIGN.md#crate-layout)

- [ ] Define the public re-exports and crate-level docs for the first intended surface:
  `CommandEngine`, `InteractiveShell`, command registration helpers, completion provider
  traits, and error types.
  Ref: [`Core Types`](./DESIGN.md#core-types), [`API Ergonomics`](./DESIGN.md#api-ergonomics)

### 1.3 Baseline build validation

- [ ] Make `cargo check -p motlie-repl` pass with an empty but compilable crate surface.
  Ref: [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements), [`Crate Layout`](./DESIGN.md#crate-layout)

### 1.4 Phase-1 verification

- [ ] Verify the new crate builds in isolation and does not disturb existing workspace crates.
  Run:
  ```bash
  cargo check -p motlie-repl
  cargo check -p motlie-tmux
  cargo check --bin motlie --features repl
  cargo check --bin motlie --features repl,repl-tmux
  ```
  Ref: [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements), [`Dependency Inventory`](./DESIGN.md#dependency-inventory)

---

## Phase 2: Core Command Engine, Registry, and Shell

Implement the engine that owns context, named resource state, command registration,
tokenization, parsing, async dispatch, and the interactive shell frontend.

### 2.1 Error model

- [ ] Define typed library errors in `error.rs` covering tokenization failures, clap parsing
  errors, handler failures, and editor/runtime failures.
  Ref: [`Functional Requirements`](./DESIGN.md#functional-requirements), [`Dependency Inventory`](./DESIGN.md#dependency-inventory)

### 2.2 Tokenization

- [ ] Implement shell-aware line splitting in `tokenize.rs` and document the accepted quoting
  semantics for interactive commands.
  Ref: [`Data Flow`](./DESIGN.md#data-flow), [`Dependency Inventory`](./DESIGN.md#dependency-inventory)

- [ ] Add unit tests for empty input, quoted strings, escaped quotes, trailing spaces, and
  malformed quoting.
  Ref: [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements), [`Command Execution`](./DESIGN.md#command-execution)

### 2.3 Command registry and state management

- [ ] Implement `RegisteredCommand<C>` and the command storage model in `command.rs`.
  Ref: [`Core Types`](./DESIGN.md#core-types), [`Command Registration Model`](./DESIGN.md#command-registration-model)

- [ ] Implement `CommandEngine<C>` as the owner of mutable session context and named resource
  state, including per-resource management metadata (`Owned`, `Imported`, `Ephemeral`) and
  optional locality metadata (`InProcess`, `RemoteProxy`).
  Ref: [`Session State Model`](./DESIGN.md#session-state-model), [`Boundaries and Responsibilities`](./DESIGN.md#boundaries-and-responsibilities), [`LIFECYCLE.md`](./LIFECYCLE.md)

- [ ] Define the adapter layer for managed resources in the command-engine crate rather than
  requiring subsystem crates to implement engine traits directly.
  Ref: [`Managed Resource Adapters`](./DESIGN.md#managed-resource-adapters), [`LIFECYCLE.md`](./LIFECYCLE.md#design-implications)

- [ ] Implement raw clap command registration with async handler dispatch against `&mut C`.
  Ref: [`Chosen Solution`](./DESIGN.md#chosen-solution), [`Command Registration Model`](./DESIGN.md#command-registration-model)

- [ ] Implement the typed registration path chosen in Phase 0 and prove it can parse a
  derive-based clap command into a handler call.
  Ref: [`Command Registration Model`](./DESIGN.md#command-registration-model), [`API Ergonomics`](./DESIGN.md#api-ergonomics)

- [ ] Ensure the registry assembly API works cleanly with `#[cfg(feature = "...")]`-gated
  subsystem registration modules so disabled crates contribute no commands at all.
  Ref: [`Build-Time Command Surface`](./DESIGN.md#build-time-command-surface), [`Feature Gating and Cargo Composition`](./DESIGN.md#feature-gating-and-cargo-composition)

### 2.4 Engine execution and shell frontend

- [ ] Implement frontend-neutral execution helpers such as `run_line()` and/or `run_argv()`
  on the command engine, so the same registry can be reused by REPL, TUI, and future
  message/socket frontends.
  Ref: [`Chosen Solution`](./DESIGN.md#chosen-solution), [`Data Flow`](./DESIGN.md#data-flow)

- [ ] Implement explicit engine close/shutdown behavior for `Owned` resources, explicit detach
  behavior for `Imported` resources, and unconditional child cleanup for `Ephemeral`
  resources, with the exact first-slice semantics documented near the code.
  Ref: [`Lifecycle and Rehydration Model`](./DESIGN.md#lifecycle-and-rehydration-model), [`LIFECYCLE.md`](./LIFECYCLE.md)

- [ ] Implement `InteractiveShell<C>::new`, prompt/name configuration, built-in command
  registration, and the `reedline` event loop in `repl.rs`.
  Ref: [`Chosen Solution`](./DESIGN.md#chosen-solution), [`Core Types`](./DESIGN.md#core-types)

- [ ] Implement successful command execution, structured error display, and non-fatal loop
  continuation after handler or parse errors.
  Ref: [`Functional Requirements`](./DESIGN.md#functional-requirements), [`Command Execution`](./DESIGN.md#command-execution)

- [ ] Add unit tests or focused integration tests covering:
  `help`, `quit`, successful handler execution, clap parse failure, handler error paths, and
  the first close/detach semantics for owned, imported, and ephemeral records.
  Ref: [`Functional Requirements`](./DESIGN.md#functional-requirements), [`Chosen Solution`](./DESIGN.md#chosen-solution)

### 2.5 Phase-2 verification

- [ ] Verify the crate compiles and its direct tests pass.
  Run:
  ```bash
  cargo test -p motlie-repl
  ```
  Ref: [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements), [`Summary`](./DESIGN.md#summary)

---

## Phase 3: Static and Dynamic Completion

Implement completion as the main differentiator over off-the-shelf wrappers.

### 3.1 Static completion

- [ ] Implement clap-derived completion for command names, subcommands, flags, and static
  value choices.
  Ref: [`Completion Model`](./DESIGN.md#completion-model), [`Tab Completion`](./DESIGN.md#tab-completion)

- [ ] Add tests proving completion suggestions are correct at root, subcommand, and flag
  positions.
  Ref: [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements), [`Completion Model`](./DESIGN.md#completion-model)

### 3.2 Dynamic completion provider API

- [ ] Implement the completion-provider registration surface and associate providers with a
  command name plus argument id.
  Ref: [`Completion Model`](./DESIGN.md#completion-model), [`Chosen Solution`](./DESIGN.md#chosen-solution)

- [ ] Ensure provider evaluation uses read-oriented context access or snapshot-backed shared
  state rather than `&mut C`.
  Ref: [`Completion Model`](./DESIGN.md#completion-model), [`Risks and Mitigations`](./DESIGN.md#risks-and-mitigations)

- [ ] Ensure the first provider API can surface names from the engine's own session registry,
  not just ad hoc subsystem caches.
  Ref: [`Completion Model`](./DESIGN.md#completion-model), [`Session State Model`](./DESIGN.md#session-state-model)

### 3.3 Completion merge logic

- [ ] Merge static and dynamic suggestions with de-duplication and predictable ordering.
  Ref: [`Tab Completion`](./DESIGN.md#tab-completion), [`Chosen Solution`](./DESIGN.md#chosen-solution)

- [ ] Add tests for:
  dynamic-only suggestions, static-plus-dynamic overlap, empty providers, and prefix
  filtering.
  Ref: [`Completion Model`](./DESIGN.md#completion-model), [`Tab Completion`](./DESIGN.md#tab-completion)

### 3.4 Example-driven completion harness

- [ ] Add a minimal test-only fixture context with mutable names or resources so dynamic
  completion can be validated without tmux networking.
  Ref: [`Session State Model`](./DESIGN.md#session-state-model), [`Completion Model`](./DESIGN.md#completion-model)

### 3.5 Phase-3 verification

- [ ] Verify completion tests pass and the completion code remains isolated to `motlie-repl`.
  Run:
  ```bash
  cargo test -p motlie-repl completion
  cargo test -p motlie-repl
  ```
  Ref: [`Summary`](./DESIGN.md#summary), [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements)

---

## Phase 4: Tmux Command Extraction and REPL Adoption

Turn the existing tmux example REPL into the first real consumer of the new engine.

### 4.1 Extract tmux command registration

- [ ] Create a reusable tmux command-registration module in `libs/tmux` or a clearly-scoped
  tmux command-support module chosen during implementation.
  Ref: [`Tmux REPL`](./DESIGN.md#tmux-repl), [`Chosen Solution`](./DESIGN.md#chosen-solution)

- [ ] Move the current tmux command definitions and handler logic out of
  `libs/tmux/examples/repl/main.rs` into reusable registration functions.
  Ref: [`Tmux REPL`](./DESIGN.md#tmux-repl), [`Data Flow`](./DESIGN.md#data-flow)

- [ ] Define tmux-owned REPL context state explicitly instead of relying on binary-local
  variables hidden inside the example loop.
  Ref: [`Session State Model`](./DESIGN.md#session-state-model), [`Chosen Solution`](./DESIGN.md#chosen-solution)

### 4.2 Preserve the tmux surface

- [ ] Keep the existing tmux command names and semantics for:
  `create`, `new-window`, `split-pane`, `kill`, `targets`, `send`, `keys`, `capture`,
  `monitor`, `tui on`, `upload`, and `download`.
  Ref: [`Tmux REPL`](./DESIGN.md#tmux-repl), [`Goals`](./DESIGN.md#goals)

- [ ] Preserve the current TUI handoff behavior so `tui on` still delegates to the existing
  tmux TUI path rather than reimplementing TUI inside `libs/repl`.
  Ref: [`Non-Goals`](./DESIGN.md#non-goals), [`Tmux REPL`](./DESIGN.md#tmux-repl)

### 4.3 Add tmux dynamic completion

- [ ] Add dynamic completion for live tmux session names and target specs where the command
  surface benefits from it first.
  Ref: [`Goals`](./DESIGN.md#goals), [`Completion Model`](./DESIGN.md#completion-model)

- [ ] Add tests or fixture-backed checks proving tmux completion behavior without requiring a
  live tmux server for unit coverage.
  Ref: [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements), [`Risks and Mitigations`](./DESIGN.md#risks-and-mitigations)

### 4.4 Thin example bootstrap

- [ ] Rewrite `libs/tmux/examples/repl/main.rs` as a thin bootstrap that creates context,
  registers tmux commands, and launches `motlie_repl::InteractiveShell`.
  Ref: [`Chosen Solution`](./DESIGN.md#chosen-solution), [`Library Example`](./DESIGN.md#library-example)

### 4.5 Phase-4 verification

- [ ] Verify tmux still builds and the example remains runnable.
  Run:
  ```bash
  cargo check -p motlie-tmux --examples
  cargo test -p motlie-tmux
  cargo run -p motlie-tmux --example repl -- ssh://localhost
  ```
  Manual checks:
  1. Start the REPL.
  2. Run `help`.
  3. Run `targets`.
  4. Create a session and send a command.
  5. Confirm tab completion offers tmux names or targets.
  6. Confirm `tui on` still works.
  Ref: [`Tmux REPL`](./DESIGN.md#tmux-repl), [`CLI Example`](./DESIGN.md#cli-example)

---

## Phase 5: Root `motlie` Binary Integration

Add an explicit interactive entrypoint to the main binary once the engine and tmux consumer
are proven.

### 5.1 Cargo and feature wiring

- [ ] Add an optional `motlie-tmux` dependency and root feature flag wiring needed for REPL
  composition.
  Ref: [`Repo Reality Check`](./DESIGN.md#repo-reality-check), [`Feature Gating and Cargo Composition`](./DESIGN.md#feature-gating-and-cargo-composition)

- [ ] Add `motlie-repl` as a dependency of the root package.
  Ref: [`Crate Layout`](./DESIGN.md#crate-layout), [`Dependency Inventory`](./DESIGN.md#dependency-inventory)

- [ ] Add the first aggregate REPL features in the root package, for example `repl`,
  `repl-tmux`, and optionally `repl-all`, and document them in `--help` or package docs as
  appropriate.
  Ref: [`Feature Gating and Cargo Composition`](./DESIGN.md#feature-gating-and-cargo-composition), [`Cargo and Binary Walkthrough`](./DESIGN.md#cargo-and-binary-walkthrough)

### 5.2 Binary entrypoint

- [ ] Add the explicit REPL entrypoint selected in Phase 0 to `bins/motlie/src/main.rs`.
  Ref: [`CLI Example`](./DESIGN.md#cli-example), [`Open Questions`](./DESIGN.md#open-questions)

- [ ] Build the root application context and register only feature-enabled subsystem commands.
  Ref: [`Session State Model`](./DESIGN.md#session-state-model), [`Chosen Solution`](./DESIGN.md#chosen-solution)

- [ ] Ensure the binary handles missing subsystem features by omission, not runtime error:
  disabled command families should simply not exist in clap parsing or completion.
  Ref: [`Build-Time Command Surface`](./DESIGN.md#build-time-command-surface), [`Feature Gating and Cargo Composition`](./DESIGN.md#feature-gating-and-cargo-composition)

### 5.3 Reuse existing clap-derived commands where practical

- [ ] Prototype typed registration for one existing derive-based command family (`db` or
  `fulltext`) and document whether the ergonomics are acceptable for wider use.
  Ref: [`Goals`](./DESIGN.md#goals), [`Command Registration Model`](./DESIGN.md#command-registration-model)

- [ ] If typed reuse is acceptable, wire the chosen command family into the REPL. If not,
  record the constraint in `DESIGN.md` and limit initial `motlie` REPL integration to tmux.
  Ref: [`Risks and Mitigations`](./DESIGN.md#risks-and-mitigations), [`Open Questions`](./DESIGN.md#open-questions)

### 5.4 Phase-5 verification

- [ ] Verify the root binary builds with and without the REPL-related features.
  Run:
  ```bash
  cargo check --bin motlie
  cargo check --bin motlie --features tmux
  cargo run --bin motlie --features tmux -- repl
  ```
  Manual checks:
  1. Existing `motlie info`, `motlie db ...`, and `motlie fulltext ...` commands still work.
  2. The explicit REPL entrypoint starts cleanly.
  3. Feature-disabled builds do not expose tmux-only integration.
  Ref: [`Goals`](./DESIGN.md#goals), [`CLI Example`](./DESIGN.md#cli-example)

---

## Phase 6: Documentation, Examples, and Final Validation

Finish the delivery with accurate docs and comprehensive validation.

### 6.1 Docs alignment

- [ ] Update `libs/repl/docs/DESIGN.md` surgically wherever implementation decisions refine
  the API or command shape.
  Ref: [`Summary`](./DESIGN.md#summary), [`Open Questions`](./DESIGN.md#open-questions)

- [ ] Keep `libs/repl/docs/LIFECYCLE.md` aligned with the actual resource types and cleanup
  semantics used by tmux plus any newly adopted subsystem crates.
  Ref: [`LIFECYCLE.md`](./LIFECYCLE.md)

- [ ] Add `libs/repl/docs/API.md` if the implemented public API or user-facing REPL contract
  has enough surface area to justify a dedicated reference doc.
  Ref: [`API Ergonomics`](./DESIGN.md#api-ergonomics), [`Functional Requirements`](./DESIGN.md#functional-requirements)

- [ ] Update tmux example docs and any root-level help text to reflect the new REPL engine.
  Ref: [`Tmux REPL`](./DESIGN.md#tmux-repl), [`CLI Example`](./DESIGN.md#cli-example)

### 6.2 End-to-end validation

- [ ] Run the full validation set required for a ready commit in scope:
  ```bash
  cargo test -p motlie-repl
  cargo test -p motlie-tmux
  cargo test --bin motlie
  cargo check -p motlie-tmux --examples
  cargo check --bin motlie --features tmux
  ```
  Ref: [`Non-Functional Requirements`](./DESIGN.md#non-functional-requirements), [`Summary`](./DESIGN.md#summary)

- [ ] Perform one manual tmux REPL session and one root `motlie` REPL session, capturing any
  mismatches as inline doc updates in `DESIGN.md` or `PLAN.md`.
  Ref: [`CLI Example`](./DESIGN.md#cli-example), [`Summary`](./DESIGN.md#summary)

### 6.3 Ready-to-ship checklist

- [ ] Confirm all completed PLAN tasks are checked, docs are updated, and any remaining
  unchecked items are explicitly deferred with rationale nearby.
  Ref: [`Summary`](./DESIGN.md#summary), [`Chosen Solution`](./DESIGN.md#chosen-solution)
