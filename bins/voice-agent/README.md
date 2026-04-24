# Voice Agent Skill

This directory contains the typed Rust implementation behind the repo-local
voice agent skill.

The user-facing skill surface lives under:

- [`.agents/skills/voice/`](../../.agents/skills/voice/)

That namespaced skill tree contains:

- `voice/speak`
- `voice/listen`
- `voice/turn`

The typed implementation here provides the shared `voice-agent` binary that the
voice skill wrappers invoke.

Design and plan docs for this implementation live at:

- [Design](./docs/DESIGN.md)
- [Plan](./docs/PLAN.md)

The conversational playbook with example human prompts and example agent
responses lives at:

- [`.agents/skills/voice/README.md`](../../.agents/skills/voice/README.md)
