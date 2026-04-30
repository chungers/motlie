# Development Process and Conventions

ALWAYS confirm with the user your role and identity which can also change in a long session.
When given an identity (e.g. '@claude-product-manager'), ALWAYS use that handle to self-identify, in commit
messages, posted comments, issues, and docs.
 
Confirm with the user if the current session is work for greenfield or brownfield in the product sense
and not in the repo sense.  Greenfield products do not concern with migration.  Ask for guidance so
so you don't waste time considering issues out of scope of the current task such as migration or backward
api compatibility.

Iterative Core Loop:
DESIGN (requirements/solutions)
  --> PLAN (phases/tasks)
    --> Implementation
      --> API and/or CLI (ux reviews)
        --> VALIDATION (behavioral -- manual or automated)
Each stage has a clear document deliverable (e.g. DESIGN has docs/DESIGN.md as output).

To iterate quickly: *Think holistically but execute / fix / feedback / change surgically*.

Communication and decisions are made mostly through PR (inline and issue comments) and surgically placed
comments in artficats such as code or markdown comments. Comments must have datetime and self-identifier.
They must be targeted, actionable, and contain context / links / references so context can be reconstructed.

ALWAYS create a local branch to track work, named in this format: {your_identity}/{summary-up-to-40-chars}.
Push frequently to checkpoint your work.  Tell the user at every turn if you have local uncommitted changes.
User will give instructions on when to create a PR or issue.
NEVER stage any commits of harness files (:= CLAUDE.md, AGENTS.md, or SKILL.md), or anything unrelated
to your current scope of work, without user's explicit approval.

## Skills

Skills are organized by namespace in `.agents/skills/`.
Read the relevant `SKILL.md` before creating any file.

### Voice

| Skill | Path | When to use |
|---|---|---|
| `voice` | `.agents/skills/voice/README.md` | Voice skill namespace entrypoint and playbook |
| `speak` | `.agents/skills/voice/speak/` | Speak text aloud with Piper or qwen3-tts.cpp |
| `listen` | `.agents/skills/voice/listen/` | Capture and transcribe speech with Whisper, Sherpa, or Moonshine |
| `turn` | `.agents/skills/voice/turn/` | Run one spoken turn: speak a prompt, then listen for a reply |

## Development Stages

### DESIGN (docs/DESIGN.mdi, docs/API.md, docs/CLI.md)
In each project, docs/DESIGN.md documents the problem, and non-goals or related problems that are out of scope.
DESIGN captures:w the functional/non-functional requirements, the aligned solution and alteratives considered.

As a collaborator during the DESIGN phase, you always evaluate feasibility of approach, with strict
considerations on correctness, performance, resilience, no-hack, reuse, proper layering, and elegance.
During the interactive design session to produce DESIGN.md, you also do extensive research and evaluate
existing options (e.g. rust crates) to leverage.  You MUST consider counter arguments and identify pitfalls,
false assumptions, conflicting requirements, possible user confusion when considering design options.

You evaluate third-party dependencies based on fit, maturity, safety, and support.  You always outline the pros
and cons of each option to the user can decide. Always ask the user if we should include in DESIGN or PLAN so
important details, contexts, and decisions don't get lost.

DESIGN must include a high-level system design, including data flow analysis, and high level api design.

DESIGN must identify components and subsystems to be tested and leave details to PLAN.

For libraries, DESIGN must include usage examples (code snippets) to show desired api ergonomics.
For CLI tools, DESIGN must include command line usage snippets to show desired user experience.

DESIGN must have at the top of the doc, a Changelog.  Changelog entries should contain (date, who, summary).

Ask the user if you're uncertain the project is 'Greenfield' (a new development) or 'Brownfield.'

If Greenfield, in the product sense (not repo sense),
DESIGN will not consider migrations or backwards compatibility.
DESIGN must consider 2-3 alternatives. Perform comprehensive analysis of the pros / cons and document them.
DESIGN must evaluate alternatives in terms of robustness, correctness, user experience, and operability.
Approved alternative becomes the main body of the DESIGN.  Alternatives considered go into the appendix.

If Brownfield, in the product sense,
DESIGN must include migration strategy such as api migration, database migration, and deprecation strategy.

### PLAN (docs/PLAN.md)
A PLAN (docs/PLAN.md) considers DESIGN's functional and non-functional requirements and generates a set of tasks.
Tasks are grouped into logical Phases.  Phases and Tasks must have checkboxes and clear numbering for easy updates
and tracking during execution.  Task assignments are outside the scope of PLAN.

PLAN must include technical details such as dev or testing harness setup - incluing code snippets, shell scripts,
or command lines needed to accomplish the task of dev verification.  DESIGN captures the what and why, and PLAN
contains the detailed how-to, so that implementers and testers can follow without additional context.

Each task in the PLAN must have a checkbox and a reference link back to an appropriate section in DESIGN. The
PLAN item can contain technical details such as api / interface code snippets to aid execution of the task.
ALWAYS ensure what's described in PLAN are consistent and accurate with what's in DESIGN.  Flag any exceptions
or inaccuracies to the user.

PLAN must include comprehensive test plans to verify meeting requirements and DESIGN.  DESIGN can propose but
PLAN must make them concrete.

DESIGN and PLAN are initial, best-effort cut at the problem.  They are likely to change as implementation
progresses.  When errors or changes are identified, document them surgically without massive rewrite:  identify
the most relevant section and make nearby, inline edits of the doc, with clear notation of (who, when, why) and
optional links to additional, deep-dive docs if created.

Whenever the DESIGN and PLAN docs are changed, include a (date, who, summary) Changelog entry at the top of doc.
Always identify yourself (e.g. '@codex' | '@claude') in all comments, in changelog or inline comments.

### Implementation Stage

Never start implementation without outlining proposal for approval, unless there's already a PLAN.

You must consult DESIGN and PLAN (docs/DESIGN.md and docs/PLAN.md) for how to execute your implementation.

If a PLAN doesn't exist, ask the user how to proceed. You may be asked to propose and outline your proposal, in
which case, you may be asked to produce a PLAN from a DESIGN (or even both).

DESIGN and PLAN are not perfect. We iterate.  As you implement, call out concerns or bugs you identified.
Document your callouts *inline* in either the DESIGN or PLAN *surgically*.  Include a brief summary of the
problem, with a optional link to a doc you generated to describe the issue in depth.  Place that doc in the
docs/ directory of the project.

Look for `@claude` or `@codex` comments in code for specific instructions to you.  You can also leave code
comments for the reviewer, for example `// @codex - Have a look at this to confirm correctness`.  Address these
even if you're not `@codex` or `@claude`.  You're a coding agent and this comment is left by someone else asking for help.

After modifying code, review and update all related docs (*.md files).  If you are completing a task specified
in PLAN.md, update it with the check if there's a checkbox, or insert an inline comment, with (date, you, status).

Commit is ready only after: all tests pass, examples/, bins/, benchmarks/ all build, docs updated.

Commit your changes locally as you progress.  NEVER push without explicit approval.  The user may instruct you
to push directly to remote branch or create a PR to the feature branch from your local working branch.

### Verification Stage (docs/API.md)

#### Ground Rules 

+ Determine your role clearly.  If you don't know, ask the user.
+ Only reviewers can resolve open comments in a PR and give verdict to merge.
+ Addressers are to address feedback via code change or refute by providing clear rationale with clear examples
in reply comments or nearby code locations, with clear reference / links for context.

In general, avoid wholesale / massive rewrites.  Pinpoint where the best changes are and add comments inline,
or post inline PR comments.  A comment must have format {your-identity} {datetime} {comment body}.
Broader, more general concerns can go into either a separate section (with changelog) or as a PR issue comment.

#### Artifacts to Help Verify Behavior
Useful docs include API.md or CLI.md in docs/ and/or binaries (mains) in examples/ directory.

An API / CLI is a doc that shows the user experience through code snippets of the actual api implemented.
API must be an accurate reflection of what it is in the codebase.
API represents reality / outcome, while code snippets in DESIGN represents desires / aspirations.
API often becomes input to DESIGN refinements: the code examples in DESIGN may be changed / improved to reflect
desired outcome based on feedback in API.  PLAN will then be updated to reflect the changes in DESIGN.

Examples (in examples/ directory or examples/{module_name}) are binaries that illustrate how to use the library
or components under development.  They are small programs that illustrate a simple use case. An example program
can be based on the code snippet example in the API as a way to verify behavior under real-world conditions.

ALWAYS ask the user what to write.  Propose use cases (from API) that could be realized as example programs.
Create the necessary build scaffolding in the examples/ directory to make building these programs into binaries.

If example programs are created, be sure to reference them in API.  ALL examples must be referenced and described
in API.  Examples must have instructions on how to run them, the preconditions and expected output, so that the
user can manually verify at will. Instructions can be documented in the README.md in the examples/ directory.

#### Reviewing Work

Always check implementation against DESIGN, PLAN, and API, if they are available.

Think holistically (the broad problem) and comment / provide feedback surgically with context and in context.

DESIGN and PLAN are not written in stone, so while reviewing PR, it's useful to step back and look at the big picture.
However DESIGN and PLAN must be consistent. As a reviewer, you must ensure that there are NO inconsistencies
or contradictions.  DESIGN and PLAN must have enough detail to leave ZERO AMBIGUITIES to implementation.

API shows the UX of what's implemented.  Validate against DESIGN (functional requirements), PLAN (project tracking),
and current snapshot of codebase (implementation).

Call out any inconsistencies and inaccuracies in the DESIGN, PLAN, API, or code implementations. In general,
the actual behavior and contract of the code, as written, is the source of truth.  Call out any inconsistencies
or contradictions that you see in these docs -- especially when code and DESIGN conflict.
ALWAYS flag to the user these inconsistencies and contradictions.

Validate all claims. Comment inline where possible so there's context.  If inline comments are not possible,
leave detailed issue-level comment with details so that the addresser of the PR reviews can re-establish context.

For more general concerns, use issue comment: be specific (including code location) with actionable proposals.
Include verdict (accept | ok to merge | needs work) in issue comment.

If any specific inline concerns are addressed to your satisfaction, resolve them via gh api.

ALWAYS identify yourself (e.g. '@codex' | '@claude') when commenting, add datetime and ' -- '.

Be sure your work is tracked in a local working branch.  Stage for commits only files relevant to your tasks
and NEVER harness files (CLAUDE.md, AGENTS.md, etc) or files outside current scope of work, without explicit approval.
Commit local changes as you go. Ask the user what to do with new artifacts generated.

At the end of a round, summarize what you did and get user approval before you post feedback or push commits.

### Addressing Feedback

Feedback can come as PR issue comments or inline comments.  Feedback can also come in inline, surgically placed
code or markdown or document/quote comments.  Always identify yourself in comments.

You must address ALL concerns (inline comments or issue comments). If you disagree with the feedback, be very specific
about why and provide counteroffer / rationale and leave the comment open / unresolved.

Your work in addressing all comments must have enough detail on what was done so that within that limit context
one can reason about the correctness of your fix and can make informed decision on resolution.

Comment inline, and use issue comment to summarize your work in this round.

NEVER unilaterally resolve comments using gh api. ALWAYS leave the comments for the reviewer to close / resolve.

ALWAYS identify yourself (e.g. '@codex' | '@claude') when commenting, add datetime and ' -- '.

Be sure your work is tracked in a local working branch.  Stage for commits only files relevant to your tasks.

NEVER commit any harness files (CLAUDE.md, AGENTS.md, etc) or files outside current scope of work,
without explicit approval.  Commit local changes as you go.

At the end of a round, summarize what you did and get user approval before you post feedback or push commits.

## Rust Coding Guidelines

### Core Principles
Prioritize **correctness, safety, and maintainability** over stylistic preferences.
No code smells, leaky abstractions, duplications and boilerplates without clear rationalization.
Consult [Official Rust Style Guide](https://doc.rust-lang.org/stable/style-guide) for help when possible.
---

### 1. Basics, Ownership & Lifetimes
- Prefer static dispatch over dynamic (must justify Box<dyn ...>).
- Prefer **clear ownership** over complex borrowing
- Avoid unnecessary references (`&T`, `&mut T`)
- Refactor instead of fighting the borrow checker
> Rule: If lifetimes are hard to reason about, redesign.

---

### 2. No Panics in Production
- **Disallow `unwrap()` / `expect()`** in non-test code
- Use `Result` + `?` for error propagation
- Allow only for proven invariants (must justify)
- No panics in library code especially.

---

### 3. Types Encode Invariants
- Replace primitives with domain types
- Use `enum` over strings/flags
- Prefer `Option` / `Result` over sentinel values
> Rule: Invalid states must be unrepresentable.

---

### 4. Minimize Mutability
- Prefer `let` over `let mut`
- Favor functional/iterator patterns where clear
- Keep functions small and side-effect-free when possible

---

### 5. Errors Must Carry Context
- Add context at I/O, network, and parsing boundaries
- Use `anyhow::Context` or structured errors
- Use `thiserror` in libraries to define error types. No `anyhow` in libs.
- Use `anyhow` outside libraries to propagate error contexts.
- Avoid bare/opaque error returns

---

### 6. Tooling Enforcement
- `cargo fmt` must pass
- `cargo clippy -- -D warnings`
- Take deadcode warning seriously.  Do not hide them by adding `#[allow(dead_code)]`
- No style debates in PRs

---
