# TTS To ASR Pipeline Validation

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-20 | @codex-tts: Added the first end-to-end 2x3 shell-pipeline validation matrix for the shipped TTS and ASR examples, covering 60 runs with per-case input, output, elapsed time, and normalized word error rate. | All |

## Scope

This report validates the shipped example-layer shell contract after
`tts_qwen3_onnx` removal. The matrix covers the two supported TTS examples and
the three shipped ASR examples:

- `tts_piper -> asr_whisper`
- `tts_piper -> asr_sherpa_onnx`
- `tts_piper -> asr_moonshine`
- `tts_qwen3_tts_cpp -> asr_whisper`
- `tts_qwen3_tts_cpp -> asr_sherpa_onnx`
- `tts_qwen3_tts_cpp -> asr_moonshine`

`tts_qwen3_onnx` is intentionally excluded because the ONNX adapter path is
already documented as non-functional for real speech output.

## Method

- Build: release examples from `libs/models` with the two shipped TTS bundles
  and three shipped ASR bundles compiled in.
- Execution model: stdin text into the TTS example, stdout WAV directly piped
  into the ASR example, final transcript captured from stdout.
- Prompt set: 10 prompts ranging from short declarative sentences to longer
  technical-documentation-style utterances.
- WER metric: token-level Levenshtein distance divided by reference token count
  after lowercasing and stripping non-alphanumeric separators, while preserving
  apostrophes.
- Artifact roots used during evaluation: Piper HF cache under
  `/home/dchung/sessions/cdx-dgx-e2e/motlie/artifacts/models/hf-cache`,
  qwen3-tts.cpp GGUF root under `/tmp/qwen3-tts-models`, Whisper HF cache under
  `/home/dchung/sessions/cdx-tts/motlie/artifacts/models/hf-cache`, Sherpa HF
  cache under `/home/dchung/.cache/huggingface/hub`, and Moonshine HF cache
  under `/home/dchung/cld-mistral/motlie/artifacts/models/hf-cache`.

## Summary

| Pipeline | Avg WER | Avg Elapsed (s) | Worst WER | Non-zero Exit Runs |
|----------|---------|-----------------|-----------|--------------------|
| `tts_piper->asr_moonshine` | 0.260 | 12.1 | 1.000 | 2 |
| `tts_piper->asr_sherpa_onnx` | 0.272 | 2.5 | 0.500 | 0 |
| `tts_piper->asr_whisper` | 0.042 | 2.4 | 0.125 | 0 |
| `tts_qwen3_tts_cpp->asr_moonshine` | 0.050 | 30.0 | 0.159 | 0 |
| `tts_qwen3_tts_cpp->asr_sherpa_onnx` | 0.300 | 20.6 | 1.000 | 0 |
| `tts_qwen3_tts_cpp->asr_whisper` | 0.055 | 24.1 | 0.188 | 0 |

## Key Findings

- `tts_piper -> asr_whisper` is the strongest CPU-first shell pipeline in this
  matrix: lowest average WER (`0.042`) and low average elapsed time (`2.4 s`).
- `tts_qwen3_tts_cpp -> asr_moonshine` is the strongest qwen3-tts.cpp pairing
  by average WER (`0.050`), but it is materially slower (`30.0 s` average
  elapsed).
- `tts_qwen3_tts_cpp -> asr_whisper` also stays usable (`0.055` average WER)
  and is materially more stable than the Sherpa pairing.
- `asr_sherpa_onnx` is the weakest recognizer in this matrix for long-tail
  stability. It produced one catastrophic qwen3-tts.cpp failure (`QQQ...`) and
  consistently higher average WER on both TTS sources.
- `asr_moonshine` has mixed behavior: it is strong with qwen3-tts.cpp, but it
  exited non-zero and returned blank output on two Piper prompts.

## Per-Pipeline Results

### `tts_piper->asr_moonshine`

| Case | Words | WER | Elapsed (s) | Status | Input | Output |
|------|-------|-----|-------------|--------|-------|--------|
| 1 | 8 | 0.000 | 2.3 | ok | Please read this short sentence clearly and naturally. | Please read this short sentence clearly and naturally. |
| 2 | 10 | 0.100 | 3.3 | ok | Rust makes state management explicit and predictable for maintainable systems. | Press makes state management explicit and predictable for maintainable systems. |
| 3 | 15 | 0.000 | 3.7 | ok | The quick brown fox jumps over the lazy dog near the river bank at sunset. | The quick brown fox jumps over the lazy dog near the river bank at sunset. |
| 4 | 18 | 0.000 | 5.3 | ok | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. |
| 5 | 16 | 0.250 | 6.4 | ok | Shell pipelines should stay simple, predictable, and safe for stdin and stdout composition without temporary files. | Shell pipelines should stay simple, predictable, and safe for SD-DIN and SD-DAP composition without temporary files. |
| 6 | 24 | 1.000 | 8.0 | tts=0, asr=1 | This medium length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. |  |
| 7 | 35 | 0.086 | 12.5 | ok | When the transcription backend receives a clean wav stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. | When the transcription, Beckend receives a clean web stream from the text-to-speech example, a final transcript should stay close enough to the original request that word error rate remains manageable for manual review. |
| 8 | 34 | 1.000 | 12.9 | tts=0, asr=1 | For this longer sample we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact roots, and the difference between partial streaming updates and a final transcript line. |  |
| 9 | 44 | 0.114 | 25.9 | ok | The validation protocol should reveal where Piper remains the stronger CPU first backend, where qwen3-tts.cpp offers more natural output or cloning potential, and how each ASR backend responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. | The validation protocol should reveal where Piper remains the stronger CPU for Spakend, where Q13TTS about CPP offers more natural output or cloning potential, and how each ASR, akin, responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. |
| 10 | 63 | 0.048 | 40.9 | ok | Finally, this longest validation prompt verifies the full shell composition contract across two text to speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting wav stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. | Finally, this long-est validation prompt verifies the full shell composition contract across two text-to-speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary. Forward the resulting WAP stream through Unix standard output and recover a usable transcript on the far end without manual file staging or a custom transport adapter. |

### `tts_piper->asr_sherpa_onnx`

| Case | Words | WER | Elapsed (s) | Status | Input | Output |
|------|-------|-----|-------------|--------|-------|--------|
| 1 | 8 | 0.500 | 1.7 | ok | Please read this short sentence clearly and naturally. | THIS SHORT SENTENCE CLEARLY ANDQQQQQQQQ |
| 2 | 10 | 0.200 | 2.0 | ok | Rust makes state management explicit and predictable for maintainable systems. | DRESS MAKES STATE MANAGEMENT EXPLICIT AND PREDICTABLE FOR MAINTAINABLE SYS |
| 3 | 15 | 0.133 | 1.9 | ok | The quick brown fox jumps over the lazy dog near the river bank at sunset. | QUICK BROWN FOX JUMPS OVER THE LAZY DOG NEAR THE RIVER BANK AT SUQQQQQQQQ |
| 4 | 18 | 0.278 | 2.3 | ok | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. | TESTING SPEECH PIPE LINES THAT CONVERT TEXT TO ORDIO AND THEN RECOVER TEXT THROUGH AUTOMATIC SPEECH RECOGNITION |
| 5 | 16 | 0.438 | 2.3 | ok | Shell pipelines should stay simple, predictable, and safe for stdin and stdout composition without temporary files. | PIPE LINES SHOULD STAY SIMPLE PREDICTABLE AND SAFE FOR ESTY DIN AND ESTEE DOUBT COMPOSITION WITHOUT TEMPORARY FIQQQQQQQQ |
| 6 | 24 | 0.042 | 2.5 | ok | This medium length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. | THIS MEDIUM LENGTH PROMPT CHECKS WHETHER THE SYNTHETIZED VOICE PRESERVES PUNCTUATION RHYTHM AND WORD BOUNDARIES WELL ENOUGH FOR AUTOMATIC SPEECH RECOGNITION TO STAY USABLE |
| 7 | 35 | 0.257 | 2.6 | ok | When the transcription backend receives a clean wav stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. | A TRANSCRIPTION THAT CAN RECEIVES A CLEAN WEB STREAM FROM THE TEXT TO SPEECH EXAMPLE THE FINAL TRANSCRIPT SHOULD STAY CLOSE ENOUGH TO THE ORIGINAL REQUEST THAT WORD ARROWRATE REMAINS MANAGEABLE FOR MANUALQQQQQQQQ |
| 8 | 34 | 0.147 | 2.9 | ok | For this longer sample we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact roots, and the difference between partial streaming updates and a final transcript line. | FOR THIS LONGER SANT WILL WE WANT A SENTENCE THAT SOUNDS LIKE ORDINARY TECHNICAL DOCUMENTATION MENTIONING COMMAND LINE COMPOSITION REMOTE PLAYBACK ARTIFACT ROOTS AND THE DIFFERENCE BETWEEN PARTIAL STREAMING UP DATES AND A FINAL TRANSFRICT LINE |
| 9 | 44 | 0.500 | 3.2 | ok | The validation protocol should reveal where Piper remains the stronger CPU first backend, where qwen3-tts.cpp offers more natural output or cloning potential, and how each ASR backend responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. | VALIDATION PROTOCOL SHOULD REVEAL WHERE PIPER REMAINS THE STRONGER SEA PEA YOU FIRST BECKENED WHERE CUE WENT THREE T S DOPS P P OFFERS MORE NATURAL APUT OR CLONING POTENTIAL AND HOW HEAT A S R PEK AND RESPONSE TO DIFFERENT SPEAKING STYLES PASIVE AND SENTENCE LENGTHS UNDER THE SAME SHELL COMPOSITION CONTRACT |
| 10 | 63 | 0.222 | 3.8 | ok | Finally, this longest validation prompt verifies the full shell composition contract across two text to speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting wav stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. | FINALLY THIS LONGEST VALIDATION PROMPT VERIFIES THE FULL SHELL COMPOSITION CONTRACT ACROSS TWO TEXTS TO SPEECH BUCKHEMS AND THREE AUTOMATIC SPEECH RECOGNITION BECKHEM'S WHILE MEASURING WHETHER OR USAH CAN REASONABLY PIPE TEXT INTO A TETE BINERY FORWARD THE RESULTING WAX STREAM THROUGH EUNUCHS STANDARD OUTWARD AND RECOVER A USABLE TRANSFRICT ON THE FAR END WITHOUT MANUAL FILE STAGING OR A CUSTOMED TRANSPORT A DOCTOR |

### `tts_piper->asr_whisper`

| Case | Words | WER | Elapsed (s) | Status | Input | Output |
|------|-------|-----|-------------|--------|-------|--------|
| 1 | 8 | 0.000 | 2.2 | ok | Please read this short sentence clearly and naturally. | Please read this short sentence clearly and naturally. |
| 2 | 10 | 0.000 | 2.2 | ok | Rust makes state management explicit and predictable for maintainable systems. | Rust makes state management explicit and predictable for maintainable systems. |
| 3 | 15 | 0.067 | 2.2 | ok | The quick brown fox jumps over the lazy dog near the river bank at sunset. | The quick brown fox jumps over the lazy dog near the river banked at sunset. |
| 4 | 18 | 0.000 | 2.3 | ok | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. | We are testing speech pipelines that convert text to audio, and then recover text through automatic speech recognition. |
| 5 | 16 | 0.125 | 2.3 | ok | Shell pipelines should stay simple, predictable, and safe for stdin and stdout composition without temporary files. | Shell pipelines should stay simple, predictable, and safe for SDDEND and SDDOUD composition without temporary files. |
| 6 | 24 | 0.000 | 2.4 | ok | This medium length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. | This medium length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. |
| 7 | 35 | 0.029 | 2.4 | ok | When the transcription backend receives a clean wav stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. | When the transcription backend receives a clean, WAP stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. |
| 8 | 34 | 0.059 | 2.5 | ok | For this longer sample we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact roots, and the difference between partial streaming updates and a final transcript line. | For this longer sample, we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact routes, and the difference between partial streaming updates and a final transferbed line. |
| 9 | 44 | 0.114 | 2.7 | ok | The validation protocol should reveal where Piper remains the stronger CPU first backend, where qwen3-tts.cpp offers more natural output or cloning potential, and how each ASR backend responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. | The validation protocol should reveal where Piper remains the stronger CPU first bequed, where Q1 3 TTS dot CPP offers more natural output or cloning potential, and how each ASR bequed responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. |
| 10 | 63 | 0.032 | 3.0 | ok | Finally, this longest validation prompt verifies the full shell composition contract across two text to speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting wav stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. | Finally, this longest validation prompt verifies the full shell composition contract across two text-to-speech Bekens and three automatic speech recognition Bekens, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting WAV stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. |

### `tts_qwen3_tts_cpp->asr_moonshine`

| Case | Words | WER | Elapsed (s) | Status | Input | Output |
|------|-------|-----|-------------|--------|-------|--------|
| 1 | 8 | 0.000 | 9.6 | ok | Please read this short sentence clearly and naturally. | Please read this short sentence clearly and naturally. |
| 2 | 10 | 0.000 | 12.7 | ok | Rust makes state management explicit and predictable for maintainable systems. | Rust makes state management explicit and predictable for maintainable systems. |
| 3 | 15 | 0.133 | 12.1 | ok | The quick brown fox jumps over the lazy dog near the river bank at sunset. | The quick brown fox jumps over the lazy dog near the riverbank at sunset. |
| 4 | 18 | 0.000 | 16.1 | ok | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. |
| 5 | 16 | 0.125 | 19.6 | ok | Shell pipelines should stay simple, predictable, and safe for stdin and stdout composition without temporary files. | Shell pipelines should stay simple, predictable, and safe for stendin' and taushoid composition without temporary files. |
| 6 | 24 | 0.000 | 22.8 | ok | This medium length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. | This medium-length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. |
| 7 | 35 | 0.086 | 34.2 | ok | When the transcription backend receives a clean wav stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. | When the transcription back end receives a clean wave stream from the text-to-speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. |
| 8 | 34 | 0.000 | 37.0 | ok | For this longer sample we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact roots, and the difference between partial streaming updates and a final transcript line. | For this longer sample, we want a sentence that sounds like ordinary technical documentation mentioning command line composition, remote playback, artifact roots, and the difference between partial streaming updates and a final transcript line. |
| 9 | 44 | 0.159 | 59.3 | ok | The validation protocol should reveal where Piper remains the stronger CPU first backend, where qwen3-tts.cpp offers more natural output or cloning potential, and how each ASR backend responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. | The validation protocol should reveal where Piper remains the stronger CPU first back-end, where Cohen 3T's ECP offers more natural output or cloning potential, and how each ASR back-end responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. |
| 10 | 63 | 0.000 | 76.9 | ok | Finally, this longest validation prompt verifies the full shell composition contract across two text to speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting wav stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. | Finally, this longest validation prompt verifies the full shell composition contract across two text-to-speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting WAV stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. |

### `tts_qwen3_tts_cpp->asr_sherpa_onnx`

| Case | Words | WER | Elapsed (s) | Status | Input | Output |
|------|-------|-----|-------------|--------|-------|--------|
| 1 | 8 | 0.000 | 7.9 | ok | Please read this short sentence clearly and naturally. | PLEASE READ THIS SHORT SENTENCE CLEARLY AND NATURALLY |
| 2 | 10 | 0.200 | 11.9 | ok | Rust makes state management explicit and predictable for maintainable systems. | MUST MAKE STATE MANAGEMENT EXPLICIT AND PREDICTABLE FOR MAINTAINABLE SYSTEMS |
| 3 | 15 | 0.000 | 11.1 | ok | The quick brown fox jumps over the lazy dog near the river bank at sunset. | THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG NEAR THE RIVER BANK AT SUNSET |
| 4 | 18 | 0.111 | 14.1 | ok | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. | WE ARE TESTING SPEECH PIPE LINES THAT CONVERT TEXT TO AUDIO AND THEN RECOVER TEXT THROUGH AUTOMATIC SPEECH RECOGNITION |
| 5 | 16 | 0.312 | 17.0 | ok | Shell pipelines should stay simple, predictable, and safe for stdin and stdout composition without temporary files. | BLIND SHOULD STAY SIMPLE PREDICTABLE AND SAFE FOR DESTINE AND STOUT HOID COMPOSITION WITHOUT TEMPORARY FILES |
| 6 | 24 | 0.042 | 19.3 | ok | This medium length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. | THIS MEDIUM LENGTH PROMPT CHECKS WHETHER THE SYNTHIZED VOICE PRESERVES PUNCTUATION RHYTHM AND WORD BOUNDARIES WELL ENOUGH FOR AUTOMATIC SPEECH RECOGNITION TO STAY USABLE |
| 7 | 35 | 1.000 | 23.1 | ok | When the transcription backend receives a clean wav stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. | QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ |
| 8 | 34 | 0.147 | 32.2 | ok | For this longer sample we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact roots, and the difference between partial streaming updates and a final transcript line. | FOR THIS LONGER SAMPLE WE WANT A SENTENCE THAT SOUNDS LIKE ORDINARY TECHNICAL DOCUMENTATION MENTIONING COMMAND LINE COMPOSITION REMOTE PLAY BACK ARTIFACT ROOTS AND THE DIFFERENCE BETWEEN PARTIAL STREAMING UP DATES IN A FINAL TRANSCRIPT LINE |
| 9 | 44 | 0.455 | 31.0 | ok | The validation protocol should reveal where Piper remains the stronger CPU first backend, where qwen3-tts.cpp offers more natural output or cloning potential, and how each ASR backend responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. | LEDATION PROTOCOL SHOULD REVEAL WHERE PIPER REMAINS THE STRONGER SEE P E U FIRST BACK END WERE QUEN THREE TETEA P OFFERS MORE NATURAL OUTPUT OR CLOTHING POTENTIAL AND HOW EACH A S ARE BACK END RESPONDS TO DIFFERENT SPEAKING STYLES PACING AND SENTENCE LENGTHS UNDER THE SAME SHELL COMPOSITION CONTRAC |
| 10 | 63 | 0.730 | 38.5 | ok | Finally, this longest validation prompt verifies the full shell composition contract across two text to speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting wav stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. | FINALLY THIS LONGEST FALLIDATION PROMPT VERIFIES THE FULL SHELL COMPOSITION CONTRACT ACROSS TWO TEXTS TO SPEECH BECK ENDS AND THREE AUTOMATIC SPEECH RECOGNITIONQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ |

### `tts_qwen3_tts_cpp->asr_whisper`

| Case | Words | WER | Elapsed (s) | Status | Input | Output |
|------|-------|-----|-------------|--------|-------|--------|
| 1 | 8 | 0.000 | 8.6 | ok | Please read this short sentence clearly and naturally. | Please read this short sentence clearly and naturally. |
| 2 | 10 | 0.000 | 10.3 | ok | Rust makes state management explicit and predictable for maintainable systems. | Rust makes state management explicit and predictable for maintainable systems. |
| 3 | 15 | 0.133 | 11.5 | ok | The quick brown fox jumps over the lazy dog near the river bank at sunset. | The quick brown fox jumps over the lazy dog near the riverbank at sunset. |
| 4 | 18 | 0.000 | 20.1 | ok | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. | We are testing speech pipelines that convert text to audio and then recover text through automatic speech recognition. |
| 5 | 16 | 0.188 | 17.9 | ok | Shell pipelines should stay simple, predictable, and safe for stdin and stdout composition without temporary files. | Shell pipelines should stay simple, predictable, and safe for statinin and stop-toid composition without temporary files. |
| 6 | 24 | 0.000 | 29.1 | ok | This medium length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. | This medium-length prompt checks whether the synthesized voice preserves punctuation, rhythm, and word boundaries well enough for automatic speech recognition to stay usable. |
| 7 | 35 | 0.029 | 28.9 | ok | When the transcription backend receives a clean wav stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. | When the transcription backend receives a clean wave stream from the text to speech example, the final transcript should stay close enough to the original request that word error rate remains manageable for manual review. |
| 8 | 34 | 0.029 | 31.1 | ok | For this longer sample we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact roots, and the difference between partial streaming updates and a final transcript line. | For this longer sample, we want a sentence that sounds like ordinary technical documentation, mentioning command line composition, remote playback, artifact routes, and the difference between partial streaming updates and a final transcript line. |
| 9 | 44 | 0.159 | 42.5 | ok | The validation protocol should reveal where Piper remains the stronger CPU first backend, where qwen3-tts.cpp offers more natural output or cloning potential, and how each ASR backend responds to different speaking styles, pacing, and sentence lengths under the same shell composition contract. | The validation protocol should reveal where Piper remains the stronger CPU first back-end, where Quinn3TCCP offers more natural output or cloning potential, and how each ASR back-end responds to different speaking styles, pacing and sentence lengths under the same shell composition contract. |
| 10 | 63 | 0.016 | 41.4 | ok | Finally, this longest validation prompt verifies the full shell composition contract across two text to speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting wav stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. | Finally, this longest validation prompt verifies the full shell composition contract across two text-to-speech backends and three automatic speech recognition backends, while measuring whether a user can reasonably pipe text into a TTS binary, forward the resulting wave stream through Unix standard output, and recover a usable transcript on the far end without manual file staging or a custom transport adapter. |
