# Curated Artifact Provenance

Generated from the curated bundle registry. Do not edit by hand; regenerate with `evals artifacts provenance`.

Canonical cache root: `$HOME/artifacts/hf-cache`.

Snapshot hashes are resolved by `evals preflight` from the local Hugging Face cache and live HF metadata. Derived artifacts are reproducibly generated or copied by the registry sync path after downloads complete.

| Bundle | Capabilities | HF sources | License/Gating | Downloaded artifact rules | Derived/local artifacts |
| --- | --- | --- | --- | --- | --- |
| `qwen3_4b` | Chat<br>Completion<br>ToolUse | primary: `Qwen/Qwen3-4B` | primary: `apache-2.0`/`public` | primary:<br>config.json<br>tokenizer.json<br>tokenizer_config.json<br>generation_config.json<br>*.safetensors<br>*.safetensors.index.json | none |
| `gemma4_e2b` | Chat<br>Vision<br>ToolUse | primary: `google/gemma-4-E2B-it` | primary: `apache-2.0`/`public` | primary:<br>chat_template.jinja<br>config.json<br>generation_config.json<br>tokenizer.json<br>tokenizer_config.json<br>processor_config.json<br>*.safetensors | none |
| `gemma4_e4b` | Chat<br>Vision<br>ToolUse | primary: `google/gemma-4-E4B-it` | primary: `apache-2.0`/`public` | primary:<br>chat_template.jinja<br>config.json<br>generation_config.json<br>tokenizer.json<br>tokenizer_config.json<br>processor_config.json<br>*.safetensors | none |
| `qwen3_4b_gguf` | Chat<br>Completion<br>ToolUse | primary: `Qwen/Qwen3-4B-GGUF` | primary: `apache-2.0`/`public` | primary:<br>*-Q4_K_M.gguf<br>*-Q8_0.gguf | none |
| `qwen3_6_27b_gguf` | Chat<br>Completion | primary: `unsloth/Qwen3.6-27B-GGUF` | primary: `apache-2.0`/`public` | primary:<br>Qwen3.6-27B-Q4_K_M.gguf<br>Qwen3.6-27B-Q5_K_M.gguf<br>Qwen3.6-27B-Q8_0.gguf | none |
| `gemma4_e2b_gguf` | Chat<br>Completion<br>ToolUse | primary: `unsloth/gemma-4-E2B-it-GGUF` | primary: `apache-2.0`/`public` | primary:<br>*-Q4_K_M.gguf<br>*-Q8_0.gguf | none |
| `gemma4_e4b_gguf` | Chat<br>Completion<br>ToolUse | primary: `unsloth/gemma-4-E4B-it-GGUF` | primary: `apache-2.0`/`public` | primary:<br>*-Q8_0.gguf<br>*-Q4_K_M.gguf | none |
| `gemma4_12b_gguf` | Chat<br>Completion<br>ToolUse | primary: `unsloth/gemma-4-12b-it-GGUF` | primary: `apache-2.0`/`public` | primary:<br>gemma-4-12b-it-Q4_K_M.gguf<br>gemma-4-12b-it-Q8_0.gguf | none |
| `gemma4_12b_qat_gguf` | Chat<br>Completion<br>ToolUse | primary: `google/gemma-4-12B-it-qat-q4_0-gguf` | primary: `apache-2.0`/`public` | primary:<br>gemma-4-12b-it-qat-q4_0.gguf | none |
| `embeddinggemma_300m` | Embeddings | primary: `google/embeddinggemma-300m` | primary: `gemma`/`manual` | primary:<br>config.json<br>modules.json<br>tokenizer.json<br>tokenizer.model<br>tokenizer_config.json<br>special_tokens_map.json<br>1_Pooling/config.json<br>2_Dense/config.json<br>3_Dense/config.json<br>*.safetensors | none |
| `qwen3_embedding_06b` | Embeddings | primary: `Qwen/Qwen3-Embedding-0.6B` | primary: `apache-2.0`/`public` | primary:<br>config.json<br>modules.json<br>tokenizer.json<br>tokenizer_config.json<br>1_Pooling/config.json<br>*.safetensors | none |
| `whisper_base_en` | Transcription | primary: `ggerganov/whisper.cpp` | primary: `mit`/`public` | primary:<br>ggml-base.en.bin | none |
| `sherpa_onnx_streaming_zipformer_en` | Transcription | primary: `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` | primary: `apache-2.0`/`public` | primary:<br>encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx<br>decoder-epoch-99-avg-1-chunk-16-left-64.onnx<br>joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx<br>tokens.txt | none |
| `sherpa_onnx_streaming_zipformer_en_kroko_2025` | Transcription | primary: `csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | primary: `unknown`/`public` | primary:<br>encoder.onnx<br>decoder.onnx<br>joiner.onnx<br>tokens.txt | none |
| `moonshine_streaming_en` | Transcription | primary: `UsefulSensors/moonshine-streaming` | primary: `mit`/`public` | primary:<br>onnx/small/frontend.ort<br>onnx/small/encoder.ort<br>onnx/small/adapter.ort<br>onnx/small/cross_kv.ort<br>onnx/small/decoder_kv.ort<br>onnx/small/streaming_config.json<br>onnx/small/tokenizer.json | none |
| `piper_en_us_ljspeech_medium` | Speech[Buffered] | primary: `rhasspy/piper-voices` | primary: `mit`/`public` | primary:<br>en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx<br>en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json | none |
| `kokoro_82m` | Speech[Buffered,Streaming] | buffered: `onnx-community/Kokoro-82M-v1.0-ONNX`<br>streaming: `csukuangfj/kokoro-en-v0_19` | buffered: `apache-2.0`/`public`<br>streaming: `unknown`/`public` | buffered:<br>onnx/model_quantized.onnx<br>tokenizer.json<br>voices/af_bella.bin<br>streaming:<br>model.onnx<br>voices.bin<br>espeak-ng-data/intonations<br>espeak-ng-data/en_dict<br>espeak-ng-data/phondata<br>espeak-ng-data/phondata-manifest<br>espeak-ng-data/phonindex<br>espeak-ng-data/phontab<br>espeak-ng-data/lang/gmw/en<br>espeak-ng-data/lang/gmw/en-US | streaming:<br>model.onnx <- model.onnx (copy from downloaded source artifact)<br>voices.bin <- voices.bin (copy from downloaded source artifact)<br>espeak-ng-data/** <- espeak-ng-data/ (copy from downloaded source artifact)<br>tokens.txt <- tokenizer.json (generate from tokenizer.json model.vocab via kokoro_82m::tokens_txt_from_tokenizer_json (introduced by 91cc0f32)) |
| `qwen3_tts_cpp_0_6b` | Speech[Buffered]<br>VoiceClone | primary: `koboldcpp/tts` | primary: `unknown`/`public` | primary:<br>qwen3-tts-0.6b-q8_0.gguf<br>qwen3-tts-0.6b-f16.gguf<br>qwen3-tts-tokenizer-f16.gguf | none |
