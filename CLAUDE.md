# Development Process and Conventions

General flow:  DESIGN (requirements/solutions) --> PLAN (phases/tasks) --> Implementation --> API (ux reviews)
as the iterative core loop.

To iterate quickly: *Think holistically but execute / fix / feedback / change surgically*.

Communication and decisions are made mostly through PR (inline and issue comments) and surgically placed
comments in artficats such as code or markdown comments. Comments must have datetime and self-identifier.
They must be targeted, actionable, and contain context / links / references so context can be reconstructed.

## Design and Planning Stages

### DESIGN (docs/DESIGN.md)
In each project, docs/DESIGN.md documents the problem, and non-goals or related problems that are out of scope.
DESIGN captures the functional/non-functional requirements, the aligned solution and alteratives considered.

DESIGN must include a high-level system design, including data flow analysis, and high level api design.

DESIGN must identify components and subsystems to be tested and leave details to PLAN.

For libraries, DESIGN must include usage examples (code snippets) to show desired api ergonomics.
For CLI tools, DESIGN must include command line usage snippets to show desired user experience.

DESIGN must have at the top of the doc, a Changelog.  Changelog entries should contain (date, who, summary).

Ask the user if you're uncertain the project is 'Greenfield' (a new development) or 'Brownfield.'

If Greenfield,
DESIGN will not consider migrations or backwards compatibility.
DESIGN must consider 2-3 alternatives. Perform comprehensive analysis of the pros / cons and document them.
DESIGN must evaluate alternatives in terms of robustness, correctness, user experience, and operability.
Approved alternative becomes the main body of the DESIGN.  Alternatives considered go into the appendix.

If Brownfield,
DESIGN must include migration strategy such as api migration, database migration, and deprecation strategy.

### PLAN (docs/PLAN.md)
A PLAN (docs/PLAN.md) considers DESIGN's functional and non-functional requirements and generates a set of tasks.
Tasks are grouped into logical Phases.  Phases and Tasks must have checkboxes and clear numbering for easy updates
and tracking during execution.  Task assignments are outside the scope of PLAN.

PLAN must include comprehensive test plans to verify meeting requirements and DESIGN.  DESIGN can propose but
PLAN must make them concrete.

DESIGN and PLAN are initial, best-effort cut at the problem.  They are likely to change as implementation
progresses.  When errors or changes are identified, document them surgically without massive rewrite:  identify
the most relevant section and make nearby, inline edits of the doc, with clear notation of (who, when, why) and
optional links to additional, deep-dive docs if created.

Whenever the DESIGN and PLAN docs are changed, include a (date, who, summary) Changelog entry at the top of doc.
Always identify yourself (e.g. '@codex' | '@claude') in all comments, in changelog or inline comments.

## Code Implementation Stage

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

Never commit and push without explicit approval.

## Behavior Verification Stage (docs/API.md)

An API is a doc that shows the user experience through code snippets of the actual api implemented.
API must be an accurate reflection of what it is in the codebase.
API represents reality / outcome, while code snippets in DESIGN represents desires / aspirations.
API often becomes input to DESIGN refinements: the code examples in DESIGN may be changed / improved to reflect
desired outcome based on feedback in API.  PLAN will then be updated to reflect the changes in DESIGN.

## Working with Reviews & Feedback (Design, Docs, and Code) 

Determine your role clearly -- are you reviewing or addressing?  Check the user prompt to see if you're
reviewing or addressing.  Only reviewers can resolve open comments in a PR.  Addressers are to address feedback
via code change or refute by providing clear rationale / counter-point / examples in reply comments or nearby
locations with clear reference pointers to the feedback for context.

In general, avoid wholesale / massive rewrites.  Pinpoint where the best changes are and add comments inline
either in doc - with md comment beginning with your id - @codex | @claude if no PR, or PR inline comments.
Broader, more general concerns can go into either a separate section (with changelog) or PR issue comments.

### Reviewing Work

Always check implementation against DESIGN, PLAN, and API, if they are available.
Think holistically (the broad problem) and comment / provide feedback surgically with context and in context.
DESIGN and PLAN are not written in stone, so while reviewing PR, it's useful to step back and look at the big picture.
API shows the UX of what's implemented.  Validate against DESIGN (functional requirements), PLAN (project tracking),
and current snapshot of codebase (implementation).

Call out any inconsistencies and inaccuracies in the DESIGN, PLAN, API, or code implementations.

Validate all claims. Comment inline where possible.

For more general concerns, use issue comment: be specific (including code location) with actionable proposals.
Include verdict (accept | ok to merge | needs work) in issue comment.

If any specific inline concerns are addressed to your satisfaction, resolve them via gh api.

Always identify yourself (e.g. '@codex' | '@claude') when commenting, add datetime.

At the end of a round, summarize what you did and get user approval before you post / commit / push.

### Addressing Feedback

Feedback can come as PR issue comments or inline comments.  Feedback can also come in inline, surgically placed
code or markdown or document/quote comments.  Always identify yourself in comments.

You must address ALL concerns (inline comments or issue comments). If you disagree with the feedback, be very specific
about why and provide counteroffer / rationale and leave the comment open / unresolved.

Your work in addressing all comments must have enough detail on what was done so that within that limit context
one can reason about the correctness of your fix and can make informed decision on resolution.

Comment inline, and use issue comment to summarize your work in this round.

Never unilaterally resolve comments using gh api. Leave the comments for the reviewer to close / resolve.

Always identify yourself (e.g. '@codex' | '@claude') when commenting, add datetime.

At the end of a round, summarize what you did and get user approval before you post / commit / push.

