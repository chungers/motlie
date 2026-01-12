# Rules You MUST Follow

ALWAYS Look through the codebase, if you see code comments with '@Claude', know that it's a request / instruction from human.  You will
  + Analyze my request in this comment and summarize your understanding of my requests.
  + Provide pros and cons so I can understand the impact.
  + Outline the changes you propose to make and seek my approval before proceed with the changes.
  + After the change, remove the specific @Claude comment.

NEVER start implementation without first outlining your proposal for APPROVAL.

NEVER commit and push code without explicit approval.

Whenever you modified code, you MUST review all docs (*.md files in the current and sub directories) to ensure all type and usage references in the documentation are updated accordingly.

Commit is ready to push ONLY after all tests (unittest + integration) passed, examples/, bins/, and benchmarks all build and run, and docs are up to date.
