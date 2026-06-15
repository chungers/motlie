# Curated Artifact Provenance

Generated from the curated bundle registry. Do not edit by hand; regenerate with `evals artifacts provenance`.

Canonical cache root: `$HOME/artifacts/hf-cache`.

Snapshot hashes are resolved by `evals preflight` from the local Hugging Face cache and live HF metadata.

| Bundle | Capabilities | HF repo | License | Gating | Registry artifact rules |
| --- | --- | --- | --- | --- | --- |
| `qwen3_4b` | Chat<br>Completion<br>ToolUse | `Qwen/Qwen3-4B` | `apache-2.0` | `public` | config.json<br>tokenizer.json<br>tokenizer_config.json<br>generation_config.json<br>*.safetensors<br>*.safetensors.index.json |
| `gemma4_e2b` | Chat<br>Vision<br>ToolUse | `google/gemma-4-E2B-it` | `apache-2.0` | `public` | chat_template.jinja<br>config.json<br>generation_config.json<br>tokenizer.json<br>tokenizer_config.json<br>processor_config.json<br>*.safetensors |
| `gemma4_e4b` | Chat<br>Vision<br>ToolUse | `google/gemma-4-E4B-it` | `apache-2.0` | `public` | chat_template.jinja<br>config.json<br>generation_config.json<br>tokenizer.json<br>tokenizer_config.json<br>processor_config.json<br>*.safetensors |
| `qwen3_4b_gguf` | Chat<br>Completion<br>ToolUse | `Qwen/Qwen3-4B-GGUF` | `apache-2.0` | `public` | *-Q4_K_M.gguf<br>*-Q8_0.gguf |
| `qwen3_6_27b_gguf` | Chat<br>Completion | `unsloth/Qwen3.6-27B-GGUF` | `apache-2.0` | `public` | Qwen3.6-27B-Q4_K_M.gguf<br>Qwen3.6-27B-Q5_K_M.gguf<br>Qwen3.6-27B-Q8_0.gguf |
| `gemma4_e2b_gguf` | Chat<br>Completion<br>ToolUse | `unsloth/gemma-4-E2B-it-GGUF` | `apache-2.0` | `public` | *-Q4_K_M.gguf<br>*-Q8_0.gguf |
| `gemma4_e4b_gguf` | Chat<br>Completion<br>ToolUse | `unsloth/gemma-4-E4B-it-GGUF` | `apache-2.0` | `public` | *-Q8_0.gguf<br>*-Q4_K_M.gguf |
| `gemma4_12b_gguf` | Chat<br>Completion<br>ToolUse | `unsloth/gemma-4-12b-it-GGUF` | `apache-2.0` | `public` | gemma-4-12b-it-Q4_K_M.gguf<br>gemma-4-12b-it-Q8_0.gguf |
| `gemma4_12b_qat_gguf` | Chat<br>Completion<br>ToolUse | `google/gemma-4-12B-it-qat-q4_0-gguf` | `apache-2.0` | `public` | gemma-4-12b-it-qat-q4_0.gguf |
| `embeddinggemma_300m` | Embeddings | `google/embeddinggemma-300m` | `gemma` | `manual` | config.json<br>modules.json<br>tokenizer.json<br>tokenizer.model<br>tokenizer_config.json<br>special_tokens_map.json<br>1_Pooling/config.json<br>2_Dense/config.json<br>3_Dense/config.json<br>*.safetensors |
| `qwen3_embedding_06b` | Embeddings | `Qwen/Qwen3-Embedding-0.6B` | `apache-2.0` | `public` | config.json<br>modules.json<br>tokenizer.json<br>tokenizer_config.json<br>1_Pooling/config.json<br>*.safetensors |
| `whisper_base_en` | Transcription | `ggerganov/whisper.cpp` | `mit` | `public` | ggml-base.en.bin |
| `sherpa_onnx_streaming_zipformer_en` | Transcription | `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` | `apache-2.0` | `public` | encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx<br>decoder-epoch-99-avg-1-chunk-16-left-64.onnx<br>joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx<br>tokens.txt |
| `sherpa_onnx_streaming_zipformer_en_kroko_2025` | Transcription | `csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | `unknown` | `public` | encoder.onnx<br>decoder.onnx<br>joiner.onnx<br>tokens.txt |
| `moonshine_streaming_en` | Transcription | `UsefulSensors/moonshine-streaming` | `mit` | `public` | onnx/small/frontend.ort<br>onnx/small/encoder.ort<br>onnx/small/adapter.ort<br>onnx/small/cross_kv.ort<br>onnx/small/decoder_kv.ort<br>onnx/small/streaming_config.json<br>onnx/small/tokenizer.json |
| `piper_en_us_ljspeech_medium` | Speech[Buffered] | `rhasspy/piper-voices` | `mit` | `public` | en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx<br>en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json |
| `kokoro_82m` | Speech[Buffered,Streaming] | `onnx-community/Kokoro-82M-v1.0-ONNX` | `apache-2.0` | `public` | onnx/model_quantized.onnx<br>tokenizer.json<br>voices/af_bella.bin |
| `qwen3_tts_cpp_0_6b` | Speech[Buffered]<br>VoiceClone | `koboldcpp/tts` | `unknown` | `public` | qwen3-tts-0.6b-q8_0.gguf<br>qwen3-tts-0.6b-f16.gguf<br>qwen3-tts-tokenizer-f16.gguf |
