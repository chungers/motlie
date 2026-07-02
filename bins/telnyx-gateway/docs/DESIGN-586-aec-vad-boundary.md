# Layer B: Echo-Robust Audio Boundary for Barge-In

## Changelog

| Date | Who | Summary |
| --- | --- | --- |
| 2026-07-01 23:43 PDT | @codex-541 | Addressed PR #596 review: added Layer A/B arbitration precedence, per-call ERL/delay calibration, evidence state machine, DSP/transport invalidation, playback epoch matching, and non-Identity prompt-handler acceptance cases. |
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
- Generalize across ASR engines. Sherpa, Moonshine, and future runtime-switched
  ASR backends must see the same media-derived barge-in evidence.
- Prefer audio evidence over text tokens for active-playback caller onset.
- Preserve existing `current_compat` behavior unless a config opts into the new
  boundary.
- Fail closed when the audio signal is unavailable, stale, invalidated, or
  ambiguous.
- Learn echo return loss and echo delay online per call during playback-only
  windows. Global config bounds may sanity-check calibration, but must not encode
  one handset, speakerphone, Bluetooth, codec, or carrier acoustic path.
- Keep the DSP and echo analysis in the media layer. Conversation policy should
  consume typed evidence and make arbitration decisions.
- Emit enough metrics to compare false cancels, missed cancels, cancel latency,
  echo return, correlation, and downstream processor-visible turns.
- Prove generality with at least one non-Identity prompt-handler or turn-batched
  live case that has variable assistant utterance length, a tool-use-style pause,
  streaming/provisional/committed TTS chunks, cancel-and-replace churn, and
  turn-batch reset behavior.
- Keep PII out of committed examples and docs.

## Non-Goals

- Do not replace #587. The post-playback dispatch guard remains the downstream
  protection that prevents stale assistant echo from reaching processors.
- Do not tune domain vocabulary, hotwords, keywords, or Identity-specific text.
- Do not add a production AEC dependency until echo characterization shows that
  AEC is needed and the dependency is evaluated for maturity, licensing, and
  runtime cost.
- Do not change turn batching semantics in this design. The #586 acceptance
  matrix must exercise turn-batch reset behavior, but the reset contract remains
  owned by the conversation policy and processor pipeline designs.
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
- active/recent playback state, playback IDs, and playback epochs;
- current codec, input sample rate, evidence sample rate, and resampling path;
- packet sequence/timestamp continuity, loss/reorder/jitter markers, comfort
  noise/DTX markers, and PLC/synthetic-frame markers when available;
- configured bounds for measurement or policy mode.

It produces typed evidence, not a conversation decision:

```rust
pub struct CallerOnsetEvidence {
    pub decision: AudioOnsetDecision,
    pub confidence: f32,
    pub playback_state: PlaybackEchoState,
    pub playback_id: Option<PlaybackId>,
    pub playback_epoch: Option<u64>,
    pub caller_active_since: Option<Instant>,
    pub evidence_at: Instant,
    pub evidence_age_ms: u64,
    pub window_ms: u32,
    pub input_codec: MediaCodec,
    pub input_sample_rate_hz: u32,
    pub evidence_sample_rate_hz: u32,
    pub inbound_rms_dbfs: f32,
    pub outbound_rms_dbfs: Option<f32>,
    pub estimated_delay_ms: Option<u32>,
    pub correlation_peak: Option<f32>,
    pub echo_return_db: Option<f32>,
    pub echo_margin_db: Option<f32>,
    pub invalidation: Option<AudioEvidenceInvalidation>,
}

pub enum AudioOnsetDecision {
    TrustedCallerOnset,
    LikelyAssistantEcho,
    Ambiguous,
    Unavailable,
}
```

The exact Rust names can change during implementation, but the layer boundary
and semantics should remain: media computes audio evidence; policy consumes the
classification, freshness, playback identity, and epoch. Raw DSP fields such as
correlation, echo return, and echo margin are telemetry and debug evidence.
Policy must not re-threshold those raw fields directly; otherwise DSP judgment
leaks into the policy layer.

`confidence` is a media-owned telemetry score on a 0.0..1.0 scale for replay and
quality reports. It is not a separate policy threshold. Low confidence must be
reflected by the media classifier as `Ambiguous` or `Unavailable`.

### DSP Normalization And Invalidation Contract

All evidence operates on decoded, normalized PCM windows at a declared evidence
sample rate. PCMU 8 kHz and L16 16 kHz inputs must be decoded and, if needed,
resampled through one explicit path before correlation, VAD, AEC, or residual
energy analysis. If an AEC backend requires a different internal rate, both the
inbound stream and outbound reference must use the same aligned conversion and
report the conversion in quality spans.

A media evidence window must resolve to `Ambiguous` or `Unavailable`, never
`TrustedCallerOnset`, when any of these are true:

- packet loss, reordering, jitter buffering, comfort noise/DTX, or PLC/synthetic
  replacement exceeds configured validity bounds for the window;
- the outbound reference is shorter than the required analysis window;
- playback has no first outbound frame yet, has already reached terminal state,
  or has been superseded by a newer playback epoch;
- the calibrated echo delay is missing or outside the configured delay search
  bounds;
- the window contains low-correlation non-speech energy such as clicks, carrier
  fill, music-on-hold, or background bursts without sustained speech-like VAD
  evidence.

The diagnostic `echo_characterization.emit_interval_ms` is an observability
cadence, not a realtime policy cadence. Policy-consuming evidence must be
computed at media-frame cadence or a bounded small multiple of it.

### Online Echo Calibration

`echo_return_db` is the measured relationship between inbound and outbound RMS
at the best valid lag for a window. It is not sufficient for caller onset.

`echo_margin_db` must be defined against a per-call predicted echo model:

```text
predicted_echo_dbfs = outbound_rms_dbfs_at_calibrated_delay + erl_baseline_db
echo_margin_db = inbound_rms_dbfs - predicted_echo_dbfs
```

`erl_baseline_db` and the calibrated echo delay are learned online during
playback-only windows for the current call. Playback-only means outbound TTS is
active, the media VAD does not see sustained caller speech, ASR has not emitted
a caller candidate for the same window, and transport validity checks pass. The
first usable playback windows of each call should seed calibration, and later
playback-only windows should update it slowly enough to avoid learning caller
speech as echo.

Global config supplies bounds and sanity checks only: delay search range, ERL
minimum/maximum plausible bounds, calibration minimum duration, invalid-frame
ratio, and maximum evidence age. Aggregated Phase 1 measurements should choose
those bounds and detect outliers; they must not replace per-call calibration.

Double-talk rules:

- high correlation does not imply pure echo when caller speech is mixed with
  assistant playback;
- low correlation does not imply caller speech when nonlinear echo, codec
  distortion, or an out-of-range delay is present;
- `TrustedCallerOnset` requires sustained speech-like VAD/residual evidence plus
  a calibrated positive margin and valid transport;
- otherwise double-talk-like or weak-correlation active-playback windows are
  `Ambiguous` and fall back to Layer A.

### Evidence State Machine

`PlaybackEchoState` should distinguish at least these states:

- `Idle`: no active or recent playback reference exists.
- `ActivePlayback`: outbound frames for a playback ID/epoch are currently being
  sent or are buffered for immediate send.
- `RecentTail`: playback ended recently enough that acoustic echo may still
  return. The tail window is a media evidence knob and may be aligned with, but
  should not be silently coupled to, `echo_suppression.tail_window_ms`.
- `InterSegmentGap`: one assistant turn or prompt-handler response has a gap
  between playback segments, such as a tool-use pause or queued TTS segment. The
  prior playback reference remains relevant for tail echo, but no new active
  cancel should target the next segment until its playback ID/epoch is active.

Classifier transitions are also part of the contract:

- `TrustedCallerOnset` requires the configured number of consecutive valid
  speech-like windows for the same playback ID/epoch.
- `LikelyAssistantEcho` resets trusted-onset accumulation for that playback
  window but does not block legitimate barge-in for the whole utterance; later
  fresh evidence can become `TrustedCallerOnset`.
- `Ambiguous` pauses accumulation and defers to Layer A. It should not by itself
  reset all caller-active history unless a reset timeout expires.
- `Unavailable` means no audio cancel is allowed from that evidence. Layer A may
  still run according to the arbitration table below.
- Caller-active state decays after a configured silence/reset timeout, playback
  terminal, call end, or playback epoch change.

### Policy Consumption And Arbitration

Conversation policy may cancel active playback only when all base invariants are
true:

- config allows barge-in cancellation for the current mode;
- the evidence is fresh enough;
- the evidence playback ID and epoch still match the current active playback;
- no clear request, playback terminal event, replacement playback, or newer
  assistant generation has superseded that playback;
- the evidence decision and Layer A state allow cancel under the table below.

Playback ID/epoch matching is required to avoid stale evidence canceling a
replacement utterance or a later prompt-handler response. A clear request or
terminal event must either retire the epoch or mark later evidence for that epoch
as stale. Evidence can only cancel the playback it was measured against.

Layer B and Layer A arbitration for non-`current_compat` barge-in modes:

| Audio decision for active playback ID/epoch | Layer A partial/final state | Action |
| --- | --- | --- |
| `TrustedCallerOnset` | absent, pending, or passed | Cancel active playback immediately; later ASR text dispatch/coalescing follows the selected conversation policy. |
| `LikelyAssistantEcho` | absent or pending | Do not cancel; continue playback and keep collecting evidence. |
| `LikelyAssistantEcho` | passed | Audio vetoes the Layer A cancel only for this fresh evidence window and playback epoch; suppress/hold the transcript as echo-risk and wait for later audio or final ASR evidence after the veto expires. |
| `Ambiguous` | absent or pending | Do not cancel; continue playback or wait. |
| `Ambiguous` | passed | Use `uncertain_policy`. The default is `defer_to_layer_a`, which allows the existing conservative Layer A cancel path; stricter configs may continue playback. |
| `Unavailable` | absent or pending | Do not cancel from audio. |
| `Unavailable` | passed | Use Layer A fallback if the mode enables fallback; otherwise continue playback. |
| stale ID/epoch mismatch | any | Ignore audio evidence; do not cancel from it. Layer A may act only if it independently targets the current active playback. |

`current_compat` keeps legacy behavior and does not consume Layer B evidence.
`no_barge_in_bounded_pending` must reject behavior-changing audio barge-in modes
because cancellation is disabled by definition. `barge_in_cancel_only` and
`barge_in_coalesce_after_silence` may opt into Layer B once validated.

`uncertain_policy` values:

- `defer_to_layer_a`: default for non-compat barge-in modes; ambiguous or
  unavailable audio can still fall back to conservative Layer A transcript gates.
- `continue_playback`: fail safer for long agentic responses; ambiguous audio
  never cancels, even if Layer A passes, until a later trusted audio onset or
  playback terminal.

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

Use the existing outbound reference, online ERL/delay calibration, transport
validity, residual/speech-like VAD, and correlation metrics to classify inbound
energy as likely echo, likely caller speech, or ambiguous. Cancel only when the
inbound signal has enough calibrated margin above predicted echo return,
transport is valid, the playback epoch still matches, and the residual energy is
speech-like for the configured number of windows.

Benefits:

- smaller implementation;
- no new DSP dependency in the first pass;
- directly builds on the #586 measurement spans already emitted by media;
- can be unit-tested with synthetic echo, noise, invalid transport, and
  double-talk fixtures.

Risks:

- weaker than true AEC under high echo return or nonlinear echo;
- per-call calibration must avoid learning caller speech as echo;
- cannot improve the ASR input stream itself unless paired with AEC later.

Feasibility: high for a first policy-consuming layer after measurement.

Expected impact: medium to high if Telnyx/handset echo return is stable enough
that caller speech produces a measurable calibrated margin above assistant echo.

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

Use a staged implementation plan:

1. Measurement lock: keep `voice_quality.echo_characterization` enabled in #586
   live probes and aggregate correlation, delay, echo return, invalidation, and
   false-onset evidence across multiple calls. Add analysis tooling if needed,
   but do not let policy consume the signal until calibration and thresholds are
   justified.
2. Evidence boundary: implement `CallerOnsetEvidence` with playback ID/epoch,
   transport normalization, invalidation, and `mode = "measure_only"`; emit the
   same evidence fields without changing cancellation behavior.
3. First policy boundary: implement Option B if per-call calibration is stable
   and caller speech creates a consistent positive margin. If calibration is not
   stable, implement Option A AEC first and feed the residual speech estimate
   into the same `CallerOnsetEvidence` boundary.

This keeps the design generalized: the policy consumes a stable audio evidence
API regardless of whether the evidence came from a heuristic discriminator or a
full AEC backend.

## Quality Metrics

Each run that exercises #586 should record:

- number of active-playback speech-onset candidates;
- audio decision counts: trusted caller onset, likely assistant echo,
  ambiguous, unavailable, invalidated;
- false cancel count and missed cancel count from human/run-record labeling;
- onset-to-clear-request latency;
- clear-request-to-playback-terminal latency;
- cancel-to-replacement-first-audio latency;
- `correlation_peak` p50/p95/max while playback-only echo is present;
- `estimated_delay_ms` p50/p95/range;
- per-call ERL baseline p50/range and calibration update count;
- `echo_return_db` p50/p95/range;
- `echo_margin_db` p50/p95/range during labeled caller barge-in;
- invalidation counts by reason: loss, reorder, jitter, DTX/comfort noise, PLC,
  short reference, stale playback epoch, delay out of range;
- processor-visible turn count and stale echo/fragment turn count;
- outbound underrun count and max inter-frame gap.

Initial behavior targets before production use:

- playback-only false audio cancels: zero in targeted live validation runs and
  zero in deterministic playback-only fixtures;
- replay false-onset rate: at or below 0.1% of labeled playback-only windows
  before enabling behavior-changing audio mode by default;
- missed-onset rate: reported separately from false cancels, with an initial
  target of 10% or lower on labeled barge-in replay windows before promotion;
- onset-to-clear-request latency: p95 at or below 250 ms and hard max at or
  below 500 ms in live validation. The current sketch of two 20 ms evidence
  windows implies about 40 ms of media evidence, but clear-request latency also
  includes scheduling and policy dispatch.

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

Future behavior knobs should keep media DSP bounds separate from policy
arbitration. The values below are illustrative placeholders, not proposed
shipping defaults. Phase 1 measurement and per-call calibration should choose
actual bounds.

Media-owned bounds:

```toml
[voice_quality.audio_barge_in.media]
mode = "measure_only" # measure_only, echo_aware_onset, aec
max_evidence_age_ms = 120
trusted_onset_min_windows = 2
calibration_min_playback_only_ms = 400
delay_search_min_ms = 0
delay_search_max_ms = 240
erl_min_db = -60.0
erl_max_db = 0.0
min_echo_margin_db_floor = 3.0
min_echo_margin_db_ceiling = 18.0
max_invalid_frame_ratio = 0.05
```

Policy-owned arbitration:

```toml
[voice_quality.audio_barge_in.policy]
uncertain_policy = "defer_to_layer_a" # defer_to_layer_a, continue_playback
```

If Option A is selected, add backend-specific knobs only after dependency
selection:

```toml
[voice_quality.audio_barge_in.aec]
backend = "webrtc" # example only; dependency not selected in this design
reference_delay_ms = "auto"
```

Strict config validation should enforce the cross-product:

- `mode = "measure_only"` is accepted with any conversation policy mode because
  it is observational only.
- `mode = "echo_aware_onset"` or `mode = "aec"` requires
  `[voice_quality.barge_in] enabled = true` and a conversation policy mode that
  can cancel playback, such as `barge_in_cancel_only` or
  `barge_in_coalesce_after_silence`.
- Behavior-changing audio modes are rejected with `current_compat` and
  `no_barge_in_bounded_pending`; they must not become silently inert or silently
  alter compatibility behavior.

Default behavior must remain disabled or `measure_only` until live validation
shows the audio signal is safe to trust.

## Testing Plan

Automated tests:

- synthetic playback-only echo: high correlation should produce
  `LikelyAssistantEcho`, not cancel;
- synthetic caller-only speech with no playback: produce `TrustedCallerOnset`;
- synthetic double-talk: high-correlation mixed caller+echo is `Ambiguous` unless
  calibrated residual/VAD evidence is strong enough for `TrustedCallerOnset`;
- nonlinear echo and codec distortion: pure echo with weak correlation must not
  become `TrustedCallerOnset`;
- out-of-range delay: pure echo beyond `delay_search_max_ms` becomes
  `Ambiguous` or `Unavailable`, not caller onset;
- PLC/noise/click/music-on-hold/background burst: low-correlation non-speech
  energy must not cancel playback;
- comfort noise/DTX, loss, jitter, and reorder above bounds invalidate evidence;
- short outbound reference: playback shorter than the analysis window produces
  `Unavailable` and falls back to Layer A;
- stale evidence: policy must fail closed and wait for Layer A fallback;
- playback ID/epoch mismatch: stale evidence cannot cancel replacement playback
  or a later prompt-handler utterance;
- config strictness: unknown audio-barge-in keys fail once the new surface
  lands, and invalid mode cross-products fail parse/validation;
- `current_compat`: behavior remains unchanged.

Replay tests:

- capture inbound frames and outbound reference frames from live probes;
- replay them through the discriminator with fixed thresholds and per-call
  calibration enabled;
- verify decisions against manually labeled playback-only, barge-in,
  inter-segment gap, short-reference, invalid-transport, and ambiguous windows.

Live tests:

- no-barge-in baseline: verify no false audio cancels during assistant playback;
- playback-only echo sample: allow assistant audio to play into the handset and
  verify likely-echo decisions;
- barge-in sample: caller interrupts during active playback and replacement
  speech remains one processor-visible turn;
- prompt-handler generality sample: use an actual prompt handler, or
  `turn_batched_identity` only if the prompt handler is not yet available, with
  variable assistant utterance length, streaming/provisional and committed TTS
  chunks, cancel-and-replace churn, turn-batch reset, early and late barge-in,
  and onset during an inter-segment/tool-use pause;
- cross-ASR sample: run the same media/policy profile with at least two ASR
  runtimes when #380 runtime switching is available.

Known limitation: very short assistant utterances such as "Sure." or "One
moment." can be shorter than the correlation/calibration window. That should
produce `Unavailable` and fall back to Layer A rather than loosening thresholds.
This is fail-safe, but it may add ASR-latency fallback for short agentic replies.

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
  carrier route, and codec conditions for Option B after per-call calibration?
- Which AEC dependency best fits Motlie's licensing, CPU, sample-rate, and
  deployment constraints if Option A is required?
- Should the residual AEC stream feed ASR directly, or only the VAD/onset
  decision, in the first implementation?
- What labels should the run-record schema use for human-confirmed false cancel
  and missed cancel events?
