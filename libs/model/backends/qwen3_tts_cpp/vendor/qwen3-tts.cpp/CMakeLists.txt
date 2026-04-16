cmake_minimum_required(VERSION 3.18)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

find_package(Threads REQUIRED)

add_subdirectory(ggml)

set(QWEN3_TTS_PUBLIC_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

set(QWEN3_TTS_GGML_LIBS
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

if(TARGET ggml-cuda)
    list(APPEND QWEN3_TTS_GGML_LIBS ggml-cuda)
endif()

if(TARGET ggml-metal)
    list(APPEND QWEN3_TTS_GGML_LIBS ggml-metal)
endif()

add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC ${QWEN3_TTS_PUBLIC_INCLUDES})
target_link_libraries(text_tokenizer PUBLIC ${QWEN3_TTS_GGML_LIBS})

set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC ${TTS_TRANSFORMER_SOURCES})
target_include_directories(tts_transformer PUBLIC ${QWEN3_TTS_PUBLIC_INCLUDES})
target_link_libraries(tts_transformer PUBLIC ${QWEN3_TTS_GGML_LIBS})

if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC "-framework Foundation" "-framework CoreML")
endif()

add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC ${QWEN3_TTS_PUBLIC_INCLUDES})
target_link_libraries(audio_tokenizer_encoder PUBLIC ${QWEN3_TTS_GGML_LIBS})

add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC ${QWEN3_TTS_PUBLIC_INCLUDES})
target_link_libraries(audio_tokenizer_decoder PUBLIC ${QWEN3_TTS_GGML_LIBS})

add_library(qwen3_tts STATIC src/qwen3_tts.cpp)
target_include_directories(qwen3_tts PUBLIC ${QWEN3_TTS_PUBLIC_INCLUDES})
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
)

add_library(qwen3tts STATIC src/qwen3tts_c_api.cpp)
target_include_directories(qwen3tts PUBLIC ${QWEN3_TTS_PUBLIC_INCLUDES})
target_link_libraries(qwen3tts PRIVATE qwen3_tts)

install(TARGETS
    qwen3tts
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    qwen3_tts
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES src/qwen3tts_c_api.h DESTINATION include)
