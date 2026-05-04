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
voice skill wrappers invoke. At runtime it resolves the namespaced skill root,
bootstraps curated model weights into `.agents/skills/voice/artifacts/hf-cache/`,
and talks to the typed Motlie TTS/ASR backends directly without shelling out to
repo example binaries. On this branch it consumes the new speech-contract split explicitly:
buffered TTS for Piper and qwen3-tts.cpp, batch ASR for Whisper, and streaming
ASR for Sherpa/Moonshine. The CLI still exposes stable curated selectors such as
`--backend whisper` or `--backend qwen3cpp`, but the runtime now derives the
actual execution path from the selected bundle's capability metadata instead of
assuming semantics from the selector name alone.

When the full `motlie` repo is present, one repo-based build should seed the three
voice subskill `bin/` directories and repopulate the packaged runtime sidecars in
the skill tree:

- `.agents/skills/voice/lib/<os>-<arch>/` for ONNX Runtime shared libraries
- `.agents/skills/voice/{speak,listen,turn}/bin/` for `voice-agent-*` and
  `libqwen3tts.so*`

The current packaging design prefers dynamic ONNX Runtime linking. If those
shared libraries are missing, the wrapper should tell the human exactly what to
install on the host instead of failing opaquely.

Design and plan docs for this implementation live at:

- [Design](./docs/DESIGN.md)
- [Plan](./docs/PLAN.md)

The conversational playbook with example human prompts and example agent
responses lives at:

- [`.agents/skills/voice/README.md`](../../.agents/skills/voice/README.md)
