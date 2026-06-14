# Eval Matrix Run

- snapshot: `curated-v2-smoke`
- profile: `local-cpu-x86_64`
- launched cells: `2`
- pre-run records: `33`

## Run context (@kroko-impl 2026-06-13 21:24 PDT) — issue #518

- **First-ever kroko-2025 measurement.** Build SHA `47b919b56749860cb66f48d63bc1858bfda578f7` (= identity.git_sha in every record). Run pin per mission = `e8f27b6e`; this branch commit is `e8f27b6e`-equivalent evals-crate code (tip `30315d02` differs only by a results-data commit) + the new kroko cell/runner-arm.
- host amd2 (`amd-ryzen7-2`), AMD Ryzen 7 7730U, x86_64, profile `local-cpu-x86_64`; HF_TOKEN_PRESENT=false; release matrix children; warmup_iterations=1, iterations=3.
- outcome mix: 2 passed (both sherpa cells, cached) / 33 blocked (19 hf_token_missing, 14 artifact_missing — non-sherpa audio + non-audio bundles not prefetched on this host) / 0 failed.

### kroko-2025 vs 2023-zipformer (amd2 x86 CPU, release, warm n=3)
| metric | sherpa 2023 (06-26) | kroko-2025 (08-06) | delta |
|---|---:|---:|---|
| mean_ttfp_first_partial_ms | 111.7 [110,114,111] | **99.3 [97,104,97]** | kroko ~11% faster first-partial |
| mean_transcription_latency_ms | 517.7 | **308.3** | kroko ~1.7x faster whole-utterance |
| real_time_factor | 0.104 | **0.062** | kroko ~1.7x lower RTF |
| transcript_chars | 86 | 62 | kroko shorter (see truncation note) |
| word_error_rate | null | null | scenario carries no reference_transcript |

### Transcript text (jarvis-ref-16k.wav, captured via temporary non-committed instrumentation)
- **2023:** `IT APPEARS THAT THE CONTINUED USE OF THE EYE MAN'S SUIT IS ACCELERATING YOUR CONDITION` (86 chars, complete sentence, uppercase)
- **kroko-2025:** `It appears that the continued use of the IMAN suit is accelera` (62 chars, mixed-case, **truncated mid-word at "accelera"**)

### FINDING — kroko final transcript truncates under the streaming eval driver
kroko-2025's final transcript is deterministically truncated mid-word (62 vs 86 chars) under the eval's chunk-feed-then-`finish()` streaming path. The cell still PASSES (min_transcript_chars=1) and ttfp/latency are valid (first-partial timing is unaffected), but the tail words never appear in the committed transcript. Most likely cause: kroko's tail is emitted as a non-final partial that the eval's `final_segments()` assembly drops at `finish()` (kroko endpointing/lookahead differs from the 2023 model under the identical 200ms-chunk cadence) — i.e. a streaming-finalization interaction, not necessarily a kroko accuracy defect. The 2023 model finalizes the full sentence under the same driver. Recommend a follow-up to verify whether trailing-silence padding or a finalized-tail flush recovers the kroko tail before treating this as a quality signal. WER is null because the curated scenario has no reference_transcript (adding one is a separate, shared-scenario decision).
