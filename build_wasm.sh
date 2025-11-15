#!/bin/bash

# WASM构建脚本
# 使用Emscripten编译C++代码到WebAssembly

# 检查emcc是否安装
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten (emcc) not found!"
    echo "Please install Emscripten first:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

echo "Building WebAssembly version..."

# 创建输出目录
mkdir -p web/dist

# 编译到WASM
emcc \
    -std=c++17 \
    -O3 \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createLLMEngine" \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
    -s ENVIRONMENT='web' \
    --bind \
    -I./include \
    src/matrix.cpp \
    src/layers.cpp \
    web/wasm/wasm_bindings.cpp \
    -o web/dist/llm_engine.js

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Output files:"
    echo "  - web/dist/llm_engine.js"
    echo "  - web/dist/llm_engine.wasm"
else
    echo "Build failed!"
    exit 1
fi
