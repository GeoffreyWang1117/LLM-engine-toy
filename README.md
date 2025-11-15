# LLM Engine Toy

一个从零开始构建的轻量级LLM推理引擎，用于学习和理解Transformer模型的底层实现原理。

## 项目目标

这是一个教育性质的toy项目，目标是：
- 深入理解Transformer架构的底层实现
- 掌握LLM推理的核心算法
- 从最基础的矩阵运算开始构建完整的推理引擎

## 当前功能（v0.1）

- 基础矩阵运算库
- Transformer核心组件：
  - Linear层（全连接层）
  - LayerNorm（层归一化）
  - Self-Attention机制
  - Feed-Forward Network

## 在线演示

🌐 [点击查看WebAssembly在线演示](https://geoffreywang1117.github.io/LLM-engine-toy/)

在浏览器中直接运行，无需安装任何依赖！

## 📚 文档

- [快速开始指南](docs/QUICKSTART.md) - 5分钟快速上手
- [代码详解](docs/CODE_GUIDE.md) - 深入理解每个组件
- [部署指南](docs/DEPLOYMENT.md) - GitHub Pages部署说明

## 项目结构

```
LLM-engine-toy/
├── include/          # 头文件
│   └── llm_engine/
├── src/              # C++源代码实现
├── examples/         # C++示例代码
├── docs/             # 文档
│   ├── QUICKSTART.md
│   ├── CODE_GUIDE.md
│   └── DEPLOYMENT.md
├── web/              # WebAssembly前端
│   ├── wasm/         # WASM绑定代码
│   ├── index.html    # Web界面
│   ├── style.css     # 样式
│   └── app.js        # JavaScript逻辑
├── Makefile          # 构建配置
└── build_wasm.sh     # WASM构建脚本
```

## 构建方法

### CPU版本（原生C++）

使用Makefile构建：

```bash
make
./build/simple_example
```

或使用CMake（如果已安装）：

```bash
mkdir build && cd build
cmake ..
make
./simple_example
```

### WebAssembly版本

需要先安装 [Emscripten](https://emscripten.org/):

```bash
# 安装Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# 返回项目目录并构建WASM
cd /path/to/LLM-engine-toy
./build_wasm.sh
```

构建完成后，使用HTTP服务器运行：

```bash
cd web
python3 -m http.server 8000
# 访问 http://localhost:8000
```

## 技术栈

### 核心
- C++17 - 核心实现语言
- 纯手工实现（不依赖PyTorch/TensorFlow等框架）

### 构建工具
- Make / CMake 3.10+
- Emscripten - C++到WebAssembly编译器

### Web技术
- WebAssembly - 浏览器端高性能计算
- HTML5 + CSS3 - 现代Web界面
- JavaScript - 交互逻辑

## 路线图

### v0.1 - 基础实现 ✅
- [x] 基础项目结构
- [x] 矩阵运算库（加法、乘法、转置）
- [x] Transformer核心组件
  - [x] Linear层
  - [x] LayerNorm层
  - [x] Self-Attention机制
  - [x] FeedForward Network
- [x] CPU版本示例程序
- [x] WebAssembly版本
- [x] 在线演示网站

### v0.2 - 功能扩展（计划中）
- [ ] 完整的Transformer Block
- [ ] 位置编码（Positional Encoding）
- [ ] 简单的文本生成示例
- [ ] 模型权重加载（支持常见格式）

### v0.3 - 性能优化（计划中）
- [ ] KV Cache优化
- [ ] 矩阵运算优化（SIMD）
- [ ] 多线程支持
- [ ] 量化支持（INT8）

## License

MIT
