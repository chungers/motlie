# Layer B: Echo-Robust Audio Boundary for Barge-In

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
| 2026-07-01 PDT | @codex-541 | Initial #586 design: keep barge-in boundary audio-first, general across processors and ASR engines, with AEC/VAD evidence produced in media and consumed by conversation policy only as typed state. |

## Problem

Layer A and #587 made the transcript path safer: active/recent-playback finals
that look like assistant echo are guarded before they can cancel playback or
reach the processor. That fixed the observed post-playback echo/fragment
dispatch bug, but it did not create an audio-derived boundary for the moment a
caller starts speaking over active assistant playback.

The current barge-in onset path is still conservative during active playback.
Media can see frame energy, but it cannot yet distinguish caller speech from
assistant echo using an audio reference. When echo is likely, cancellation is
deferred to partial/final ASR and text-domain echo suppression. That is better
than guessing from words alone, but it still makes interruption timing depend on
ASR engine behavior, partial stability, and transcript timing.

Issue #586 should move the primary interruption boundary toward high-fidelity
audio processing:

- identify caller onset during active playback from inbound audio plus the
  outbound TTS reference;
- expose that evidence as typed, timestamped media state;
- let conversation policy consume the media state without learning DSP details;
- keep ASR text as corroboration or fallback, not the primary signal.

## Requirements

- Generalize across processors. The design must work for Identity, turn-batched
  Identity, future app processors, and prompt handlers. It must not compare
  caller text to assistant text as a correctness mechanism.
- Generalize across ASR engines. Sherpa, Moonshine, and future runtime-switched ASR backends must see the
  same media-derived barge-in evidence.
- Prefer audio evidence over text tokens for active-playback caller onset.
- Preserve existing `current_compat` behavior unless a config opts into the new
  boundary.
- Fail closed when the audio signal is unavailable, stale, or ambiguous.
- Keep the DSP and echo analysis in the media layer. Conversation policy should
  consume typed evidence and make arbitration decisions.
- Emit enough metrics to compare false cancels, missed cancels, cancel latency,
  echo return, correlation, and downstream processor-visible turns.
- Keep PII out of committed examples and docs.

## Non-Goals

- Do not replace #587. The post-playback dispatch guard remains the downstream
  protection that prevents stale assistant echo from reaching processors.
- Do not tune domain vocabulary, hotwords, keywords, or Identity-specific text.
- Do not add a production AEC dependency until echo characterization shows that
  AEC is needed and the dependency is evaluated for maturity, licensing, and
  runtime cost.
- Do not change turn batching semantics.
- Do not make prompt handlers responsible for echo or barge-in detection.

## Current Data Flow

Current live-call media flow:

```text
Telnyx inbound media
  -> decode/resample
  -> SampleStats / AsrGate speech state
  -> ASR partials/finals
  -> endpoint/coalescing
  -> conversation policy
  -> processor
  -> TTS
  -> outbound Telnyx media
```

Current outbound reference flow:

```text
TTS frames
  -> outbound playback registry / echo signature
  -> Telnyx media send
  -> optional echo_characterization rolling reference
```

The gap is between `SampleStats / AsrGate` and `conversation policy`.
`AsrGate` can detect energy transitions, but it is echo-blind. The echo
characterization diagnostic can measure inbound/outbound correlation and echo
delay, but no policy consumes that signal yet.

## Proposed Architecture

Introduce an audio evidence boundary owned by media:

```text
Telnyx inbound media + outbound TTS reference
  -> EchoDiscriminator
  -> CallerOnsetEvidence
  -> conversation policy
  -> cancel / defer / wait-for-ASR fallback
```

### Media-Owned Echo Discriminator

The media layer owns a pure discriminator that receives:

- recent inbound audio frames and `SampleStats`;
- the rolling outbound TTS reference;
- active/recent playback state and playback IDs;
- current sample rate and codec metadata;
- configured thresholds for measurement or policy mode.

It produces typed evidence, not a conversation decision:

```rust
pub struct CallerOnsetEvidence {
    pub decision: AudioOnsetDecision,
    pub confidence: f32,
    pub playback_state: PlaybackEchoState,
    pub playback_id: Option<PlaybackId>,
    pub caller_active_since: Option<Instant>,
    pub evidence_at: Instant,
    pub evidence_age_ms: u64,
    pub window_ms: u32,
    pub sample_rate_hz: u32,
    pub inbound_rms_dbfs: f32,
    pub outbound_rms_dbfs: Option<f32>,
    pub estimated_delay_ms: Option<u32>,
    pub correlation_peak: Option<f32>,
    pub echo_return_db: Option<f32>,
    pub echo_margin_db: Option<f32>,
}

pub enum AudioOnsetDecision {
    TrustedCallerOnset,
    LikelyAssistantEcho,
    Ambiguous,
    Unavailable,
}
```

The exact Rust names can change during implementation, but the layer boundary
should remain: media computes audio evidence; policy consumes the evidence.

### Policy Consumption

Conversation policy may cancel active playback only when one of these is true:

- `TrustedCallerOnset` is fresh enough and config allows audio-anchored cancel;
- ASR fallback later supplies a valid Layer A partial/final under conservative
  missing-signal rules;
- `current_compat` is selected and legacy behavior applies.

Policy should not compute correlation, AEC residuals, token overlap, or echo
matching itself. It should treat `Ambiguous` and `Unavailable` as "do not cancel
yet" and either continue playback or wait for ASR fallback according to config.

## Alternatives

### Option A: Acoustic Echo Cancellation

Use outbound TTS frames as an AEC reference, align to the measured playback
delay, and produce an echo-reduced inbound stream or residual speech activity
estimate before the speech-onset gate.

Benefits:

- highest expected quality ceiling;
- can improve both barge-in cancellation and ASR accuracy during double-talk;
- reduces dependence on ASR partial timing and transcript echo suppression;
- gives cleaner evidence to downstream VAD and endpointing.

Risks:

- highest implementation complexity;
- dependency selection matters for licensing, maintenance, CPU cost, and sample
  rate support;
- wrong delay alignment or nonlinear handset echo can create artifacts;
- needs replay and live validation before it should drive cancellation.

Feasibility: medium. The existing outbound reference and
`echo_characterization` spans make the integration plausible, but dependency and
runtime evaluation are still required.

Expected impact: high if live measurements show strong residual echo or frequent
double-talk. This is the most likely path to robust production barge-in when
the echo-aware heuristic is not sufficient.

### Option B: Echo-Aware Onset Gate

Use the existing outbound reference and correlation metrics to classify inbound
energy as likely echo, likely caller speech, or ambiguous. Cancel only when the
inbound signal has enough margin above predicted echo return and correlation
does not look like pure assistant playback.

Benefits:

- smaller implementation;
- no new DSP dependency in the first pass;
- directly builds on the #586 measurement spans already emitted by media;
- can be unit-tested with synthetic echo and double-talk fixtures.

Risks:

- weaker than true AEC under high echo return or nonlinear echo;
- threshold calibration must be based on live samples, not one smoke-test call;
- cannot improve the ASR input stream itself unless paired with AEC later.

Feasibility: high for a first policy-consuming layer after measurement.

Expected impact: medium to high if Telnyx/handset echo return is stable enough
that caller speech produces a measurable margin above assistant echo.

### Option C: ASR/Text-Only Barge-In

Continue relying on partial/final ASR confidence, stability, and text-domain
echo matching during active playback.

Benefits:

- already implemented as Layer A fallback;
- simple and conservative.

Risks:

- ASR partials are unstable and vary by engine;
- cancellation latency waits for transcript emission;
- false positives and missed cancels can change with ASR runtime switching;
- it does not satisfy the high-fidelity audio boundary needed by #586.

Recommendation: keep this only as fallback and compatibility behavior.

## Recommendation

Use a two-stage implementation plan:

1. Measurement lock: keep `voice_quality.echo_characterization` enabled in #586
   live probes and aggregate correlation, delay, echo return, and false-onset
   evidence across multiple calls. Add analysis tooling if needed, but do not
   let policy consume the signal until the thresholds are justified.
2. First policy boundary: implement Option B if the measured echo profile has a
   stable delay and caller speech creates a consistent positive echo margin. If
   the margin is not stable, implement Option A AEC first and feed the residual
   speech estimate into the same `CallerOnsetEvidence` boundary.

This keeps the design generalized: the policy consumes a stable audio evidence
API regardless of whether the evidence came from a heuristic discriminator or a
full AEC backend.

## Quality Metrics

Each run that exercises #586 should record:

- number of active-playback speech-onset candidates;
- audio decision counts: trusted caller onset, likely assistant echo,
  ambiguous, unavailable;
- false cancel count and missed cancel count from human/run-record labeling;
- onset-to-clear-request latency;
- clear-request-to-playback-terminal latency;
- cancel-to-replacement-first-audio latency;
- `correlation_peak` p50/p95/max while playback-only echo is present;
- `estimated_delay_ms` p50/p95/range;
- `echo_return_db` p50/p95/range;
- `echo_margin_db` p50/p95/range during labeled caller barge-in;
- processor-visible turn count and stale echo/fragment turn count;
- outbound underrun count and max inter-frame gap.

The key acceptance test is not Identity-specific. A pass means audio evidence
separates "caller is interrupting active playback" from "assistant playback is
leaking back into ASR" without relying on script keywords, assistant/caller text
comparison, or one ASR engine's partial behavior.

## Config Surface

Existing measurement knobs stay under:

```toml
[voice_quality.echo_characterization]
enabled = true
window_ms = 240
max_delay_ms = 160
emit_interval_ms = 500
```

Future policy-consuming knobs should be introduced separately so measurement
and behavior changes remain auditable:

```toml
[voice_quality.audio_barge_in]
enabled = false
mode = "measure_only" # measure_only, echo_aware_onset, aec
max_evidence_age_ms = 120
min_onset_frames = 2
min_echo_margin_db = 6.0
max_echo_correlation = 0.65
uncertain_policy = "defer_to_layer_a"
```

If Option A is selected, add backend-specific knobs only after dependency
selection:

```toml
[voice_quality.audio_barge_in.aec]
backend = "webrtc" # example only; dependency not selected in this design
reference_delay_ms = "auto"
```

Default behavior must remain disabled or `measure_only` until live validation
shows the audio signal is safe to trust.

## Testing Plan

Automated tests:

- synthetic playback-only echo: high correlation should produce
  `LikelyAssistantEcho`, not cancel;
- synthetic caller-only speech with no playback: produce `TrustedCallerOnset`;
- synthetic double-talk: produce `TrustedCallerOnset` only when echo margin and
  confidence exceed configured thresholds;
- stale evidence: policy must fail closed and wait for Layer A fallback;
- config strictness: unknown audio-barge-in keys fail once the new surface
  lands;
- `current_compat`: behavior remains unchanged.

Replay tests:

- capture inbound frames and outbound reference frames from live probes;
- replay them through the discriminator with fixed thresholds;
- verify decisions against manually labeled playback-only, barge-in, and
  ambiguous windows.

Live tests:

- no-barge-in baseline: verify no false audio cancels during assistant playback;
- playback-only echo sample: allow assistant audio to play into the handset and
  verify likely-echo decisions;
- barge-in sample: caller interrupts during active playback and replacement
  speech remains one processor-visible turn;
- cross-ASR sample: run the same media/policy profile with at least two ASR
  runtimes when #380 runtime switching is available.

## Migration And Rollout

- Phase 0: docs and measurement protocol only.
- Phase 1: collect and summarize echo-characterization spans from real calls.
- Phase 2: implement the `CallerOnsetEvidence` boundary with
  `mode = "measure_only"` and no behavior change.
- Phase 3: enable `echo_aware_onset` or `aec` behind explicit config for live
  validation.
- Phase 4: after repeated live validation, consider promoting a non-compat
  barge-in profile. Do not change `current_compat`.

## Open Questions

- Are echo delay and return level stable enough across handset, speakerphone,
  carrier route, and codec conditions for Option B?
- Which AEC dependency best fits Motlie's licensing, CPU, sample-rate, and
  deployment constraints if Option A is required?
- Should the residual AEC stream feed ASR directly, or only the VAD/onset
  decision, in the first implementation?
- What labels should the run-record schema use for human-confirmed false cancel
  and missed cancel events?
