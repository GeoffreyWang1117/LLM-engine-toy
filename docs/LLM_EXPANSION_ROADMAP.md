# LLM引擎扩展路线图

从BERT到现代大语言模型（GPT、LLaMA、Qwen等）

---

## 📊 当前状态分析

### ✅ 已完成的功能（Alpha 0.1）

**核心组件：**
- ✅ Matrix计算库（基础线性代数）
- ✅ Transformer基础层（Attention、FFN、LayerNorm）
- ✅ BERT Encoder架构（双向注意力）
- ✅ Embeddings（Token + Position + Type）
- ✅ SafeTensors权重加载
- ✅ WordPiece Tokenizer
- ✅ 端到端推理（text → BERT → output）

**支持的模型：**
- ✅ BERT-Base (110M参数, 12层)
- ✅ BERT-Large (340M参数, 24层)

### ⚠️ 当前架构的局限性

| 局限 | 描述 | 影响 |
|------|------|------|
| **Encoder-Only** | 仅支持双向编码器 | 无法做生成任务（GPT、LLaMA等） |
| **固定位置编码** | 使用绝对位置编码 | 无法支持RoPE、ALiBi等现代位置编码 |
| **缺少KV Cache** | 无自回归生成缓存 | 生成效率低 |
| **缺少采样策略** | 无Top-k/Top-p/Temperature | 无法控制生成质量 |
| **单一注意力机制** | 只有标准Multi-Head Attention | 无法支持GQA、MQA等优化 |
| **缺少Tokenizer通用性** | 只支持WordPiece | 无法支持BPE、SentencePiece等 |

---

## 🎯 扩展目标

### 短期目标（Beta 0.2 - GPT-2）
实现decoder-only架构，支持自回归文本生成

### 中期目标（v1.0 - LLaMA）
支持现代大模型（RoPE、GQA、SwiGLU等）

### 长期目标（v2.0+）
- 支持多模态（Vision Transformer）
- 支持MoE架构
- 量化推理（INT8/INT4）
- 分布式推理

---

## 🏗️ 架构对比分析

### BERT vs GPT vs LLaMA

| 特性 | BERT | GPT-2 | LLaMA/Qwen |
|------|------|-------|------------|
| **架构** | Encoder-Only | Decoder-Only | Decoder-Only |
| **注意力** | Bidirectional | Causal (Masked) | Causal + GQA |
| **位置编码** | Absolute | Absolute | RoPE |
| **前馈网络** | GELU FFN | GELU FFN | SwiGLU |
| **归一化** | LayerNorm (Post) | LayerNorm (Pre) | RMSNorm (Pre) |
| **激活函数** | GELU | GELU | SiLU/SwiGLU |
| **任务** | 理解（分类、NER） | 生成（续写） | 生成（对话） |
| **Tokenizer** | WordPiece | BPE | BPE/SentencePiece |

---

## 📋 分阶段实施计划

## 🔥 Beta 0.2 - GPT-2支持（基础生成能力）

**目标：** 实现decoder-only架构和自回归文本生成

### 阶段1: Decoder架构核心组件

#### 1.1 Causal Attention (因果注意力)
```cpp
// include/llm_engine/causal_attention.h
class CausalAttention {
    // 添加attention mask支持
    Matrix forward(const Matrix& input, const Matrix& attention_mask) const;

private:
    // 生成下三角mask（防止看到未来token）
    Matrix create_causal_mask(size_t seq_len) const;
};
```

**关键差异：**
- BERT: 可以看到所有token（双向）
- GPT: 只能看到之前的token（单向，causal mask）

#### 1.2 KV Cache（关键优化）
```cpp
// include/llm_engine/kv_cache.h
struct KVCache {
    Matrix key_cache;    // [batch, num_heads, seq_len, head_dim]
    Matrix value_cache;  // [batch, num_heads, seq_len, head_dim]

    // 增量更新（只计算新token）
    void update(const Matrix& new_keys, const Matrix& new_values);
};
```

**为什么需要：**
- 自回归生成需要逐token生成
- 每次只需计算新token的Q，复用之前的K、V
- 性能提升：O(n²) → O(n)

#### 1.3 GPT Decoder Block
```cpp
// include/llm_engine/gpt_decoder_block.h
class GPTDecoderBlock {
public:
    // Pre-LayerNorm架构
    Matrix forward(const Matrix& input, KVCache* cache = nullptr) const;

private:
    LayerNorm ln1_;
    CausalAttention attn_;
    LayerNorm ln2_;
    FeedForward ffn_;
};
```

**架构差异：**
```
BERT (Post-LN):
  x → Attention → Add → LN → FFN → Add → LN

GPT-2 (Pre-LN):
  x → LN → Attention → Add → LN → FFN → Add
```

### 阶段2: 文本生成引擎

#### 2.1 Sampling策略
```cpp
// include/llm_engine/sampling.h
class Sampler {
public:
    // Greedy: 选择概率最高的token
    int greedy_sample(const Matrix& logits) const;

    // Temperature: 控制随机性
    int temperature_sample(const Matrix& logits, float temperature) const;

    // Top-k: 从概率最高的k个中采样
    int top_k_sample(const Matrix& logits, int k) const;

    // Top-p (nucleus): 累积概率达到p时停止
    int top_p_sample(const Matrix& logits, float p) const;
};
```

#### 2.2 生成循环
```cpp
// include/llm_engine/generator.h
class TextGenerator {
public:
    std::vector<int> generate(
        const std::vector<int>& prompt_ids,
        int max_new_tokens,
        const SamplingConfig& config
    );

private:
    GPTModel model_;
    Sampler sampler_;
    std::vector<KVCache> kv_caches_;  // 每层一个cache
};
```

#### 2.3 BPE Tokenizer
```cpp
// include/llm_engine/bpe_tokenizer.h
class BPETokenizer {
public:
    // GPT-2使用Byte-level BPE
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

private:
    std::map<std::string, int> vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
};
```

### 阶段3: GPT-2模型实现

```cpp
// include/llm_engine/gpt_model.h
class GPTModel {
public:
    GPTModel(const GPTConfig& config);

    // 单次forward（用于训练）
    Matrix forward(const std::vector<int>& input_ids);

    // 生成模式（使用KV cache）
    Matrix forward_generate(int token_id, std::vector<KVCache>& caches);

private:
    Embedding token_embedding_;
    PositionalEncoding position_embedding_;
    std::vector<GPTDecoderBlock> layers_;
    LayerNorm ln_f_;  // Final layer norm
    Matrix lm_head_;  // 输出投影到vocab
};
```

**测试目标：**
- GPT-2 Small (117M): "Hello" → "Hello, world! How are you today?"
- GPT-2 Medium (345M)
- GPT-2 Large (774M)

---

## 🚀 v1.0 - LLaMA/Qwen支持（现代架构）

**目标：** 支持主流开源大模型

### 阶段1: 现代位置编码

#### 1.1 RoPE (Rotary Position Embedding)
```cpp
// include/llm_engine/rope.h
class RotaryPositionEmbedding {
public:
    // 旋转位置编码
    Matrix apply_rope(const Matrix& qk, int position_offset = 0) const;

private:
    Matrix precompute_freqs(int max_seq_len, int dim) const;
    std::vector<float> inv_freq_;
};
```

**优势：**
- 相对位置编码（更好的长度外推）
- 不增加参数量
- LLaMA、Qwen、Mistral等都使用

### 阶段2: 现代注意力机制

#### 2.1 GQA (Grouped-Query Attention)
```cpp
// include/llm_engine/grouped_query_attention.h
class GroupedQueryAttention {
    // num_kv_heads < num_q_heads
    // 例如：32个Q头，8个KV头（4:1分组）
    Matrix forward(const Matrix& input, KVCache* cache) const;

private:
    int num_heads_;     // Q的头数（如32）
    int num_kv_heads_;  // KV的头数（如8）
    int group_size_;    // 分组大小（如4）
};
```

**内存节省：**
- MHA (Multi-Head): Q, K, V都是32头
- GQA: Q=32头, K=V=8头 → KV Cache减少75%
- MQA (Multi-Query): Q=32头, K=V=1头 → KV Cache减少96.875%

### 阶段3: 现代FFN和归一化

#### 3.1 SwiGLU FFN
```cpp
// include/llm_engine/swiglu.h
class SwiGLU {
public:
    Matrix forward(const Matrix& x) const;

private:
    // y = (W1(x) * SiLU(W2(x))) * W3
    Matrix gate_proj_;  // W1
    Matrix up_proj_;    // W2
    Matrix down_proj_;  // W3
};
```

#### 3.2 RMSNorm
```cpp
// include/llm_engine/rms_norm.h
class RMSNorm {
public:
    Matrix forward(const Matrix& x) const;

private:
    // RMS = sqrt(mean(x^2) + eps)
    // y = x / RMS * weight
    Matrix weight_;
    float eps_;
};
```

### 阶段4: LLaMA模型

```cpp
// include/llm_engine/llama_model.h
class LLaMAModel {
public:
    LLaMAModel(const LLaMAConfig& config);

    std::vector<int> generate(
        const std::vector<int>& prompt,
        int max_new_tokens
    );

private:
    Embedding token_embedding_;
    std::vector<LLaMADecoderBlock> layers_;
    RMSNorm norm_;
    Matrix lm_head_;

    // LLaMA特有组件
    RotaryPositionEmbedding rope_;
    std::vector<KVCache> kv_caches_;
};
```

**测试目标：**
- LLaMA-2-7B
- Qwen-2.5-7B
- Mistral-7B

---

## 🔧 支撑系统升级

### 1. Tokenizer抽象层
```cpp
// include/llm_engine/tokenizer.h
class Tokenizer {
public:
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& ids) = 0;
    virtual ~Tokenizer() = default;
};

class WordPieceTokenizer : public Tokenizer { /* BERT */ };
class BPETokenizer : public Tokenizer { /* GPT-2 */ };
class SentencePieceTokenizer : public Tokenizer { /* LLaMA */ };
```

### 2. 模型配置统一
```cpp
// include/llm_engine/model_config.h
struct ModelConfig {
    enum class ArchType { BERT, GPT2, LLAMA, QWEN };
    enum class AttentionType { MHA, GQA, MQA };
    enum class PositionType { ABSOLUTE, ROPE, ALIBI };

    ArchType arch_type;
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    AttentionType attention_type;
    PositionType position_type;
    // ...
};
```

### 3. 权重加载器增强
```cpp
// 支持更多格式
class WeightLoader {
public:
    // 自动检测模型类型
    static ModelConfig detect_model_type(const std::string& path);

    // 支持不同命名规范
    void load_to_gpt_model(GPTModel& model);
    void load_to_llama_model(LLaMAModel& model);
};
```

---

## 📦 优化方向（v1.5+）

### 1. 性能优化
- **量化推理**: INT8/INT4 量化
- **算子融合**: LayerNorm + Linear 融合
- **SIMD加速**: AVX2/AVX-512优化
- **GPU支持**: CUDA/Metal后端

### 2. 内存优化
- **KV Cache量化**: INT8 KV Cache
- **Flash Attention**: O(n) 内存注意力
- **分页注意力**: 动态KV Cache管理

### 3. 高级功能
- **流式生成**: 逐token输出
- **批处理推理**: 多请求并行
- **动态批处理**: vLLM风格调度

---

## 🗂️ 建议的目录结构

```
LLM-engine/
├── include/llm_engine/
│   ├── core/              # 核心组件
│   │   ├── matrix.h
│   │   ├── tensor.h       # NEW: 多维张量
│   │   └── allocator.h    # NEW: 内存管理
│   ├── layers/            # 层实现
│   │   ├── attention/
│   │   │   ├── multi_head_attention.h
│   │   │   ├── causal_attention.h      # NEW
│   │   │   ├── grouped_query_attention.h  # NEW
│   │   │   └── flash_attention.h       # NEW (v1.5)
│   │   ├── ffn/
│   │   │   ├── feed_forward.h
│   │   │   └── swiglu.h                # NEW
│   │   ├── norm/
│   │   │   ├── layer_norm.h
│   │   │   └── rms_norm.h              # NEW
│   │   └── embedding/
│   │       ├── token_embedding.h
│   │       ├── absolute_position.h
│   │       └── rope.h                  # NEW
│   ├── models/            # 模型实现
│   │   ├── bert/
│   │   │   ├── bert_model.h
│   │   │   └── bert_config.h
│   │   ├── gpt2/                       # NEW
│   │   │   ├── gpt2_model.h
│   │   │   └── gpt2_config.h
│   │   └── llama/                      # NEW
│   │       ├── llama_model.h
│   │       └── llama_config.h
│   ├── tokenizers/        # 分词器
│   │   ├── tokenizer.h              # 抽象基类
│   │   ├── wordpiece_tokenizer.h
│   │   ├── bpe_tokenizer.h          # NEW
│   │   └── sentencepiece_tokenizer.h  # NEW
│   ├── generation/        # 生成相关  # NEW
│   │   ├── sampler.h
│   │   ├── kv_cache.h
│   │   └── generator.h
│   └── utils/
│       ├── safetensors.h
│       └── weight_loader.h
├── src/                   # 对应实现
├── examples/
│   ├── bert/
│   ├── gpt2/              # NEW
│   └── llama/             # NEW
└── tests/                 # 单元测试
```

---

## 📈 实施时间线（建议）

### Phase 1: Beta 0.2 (2-3周)
- Week 1: Causal Attention + KV Cache
- Week 2: GPT Decoder Block + Sampler
- Week 3: BPE Tokenizer + GPT-2 端到端测试

### Phase 2: v1.0 (4-6周)
- Week 1-2: RoPE + GQA
- Week 3-4: SwiGLU + RMSNorm + LLaMA Block
- Week 5-6: 完整LLaMA模型 + 测试

### Phase 3: v1.5 (持续优化)
- 性能profiling
- 量化支持
- 高级优化

---

## 🎓 学习资源

### 必读论文
1. **Attention Is All You Need** (Transformer原理)
2. **BERT**: Pre-training of Deep Bidirectional Transformers
3. **GPT-2**: Language Models are Unsupervised Multitask Learners
4. **LLaMA**: Open and Efficient Foundation Language Models
5. **GQA**: Training Generalized Multi-Query Transformer
6. **RoFormer**: Enhanced Transformer with Rotary Position Embedding

### 推荐实现参考
- **llama.cpp**: 高性能C++实现
- **vLLM**: 高吞吐量推理
- **TinyLlama**: 简洁的训练+推理实现
- **nanoGPT**: Karpathy的教学实现

---

## ✅ 检查清单

### Beta 0.2 - GPT-2
- [ ] Causal Attention实现
- [ ] KV Cache实现
- [ ] BPE Tokenizer
- [ ] Sampling策略（greedy/top-k/top-p）
- [ ] GPT-2模型完整实现
- [ ] 文本生成demo
- [ ] GPT-2 Small/Medium测试通过

### v1.0 - LLaMA
- [ ] RoPE实现
- [ ] GQA实现
- [ ] SwiGLU实现
- [ ] RMSNorm实现
- [ ] LLaMA模型完整实现
- [ ] LLaMA-2-7B测试通过
- [ ] Qwen-2.5测试通过

---

## 🤔 关键决策点

### 1. 是否需要多后端支持？
- **纯CPU**: 先专注CPU优化，易于调试
- **CPU+GPU**: 后期添加CUDA后端（v2.0+）

### 2. 是否支持训练？
- **仅推理**: 专注推理性能（推荐）
- **推理+训练**: 需要autograd系统（复杂度高）

### 3. 编程语言选择？
- **纯C++**: 保持当前路线，性能最优
- **C++ + Python绑定**: 增加易用性（pybind11）

### 4. 依赖管理？
- **最小依赖**: 保持当前状态，仅STL（推荐）
- **引入库**: Eigen/OpenBLAS（线性代数加速）

---

## 💡 建议优先级

### 🔥 最高优先级（Beta 0.2）
1. **Causal Attention**: GPT的核心
2. **KV Cache**: 性能关键
3. **BPE Tokenizer**: GPT-2兼容性
4. **基础生成**: Greedy + Temperature采样

### ⭐ 高优先级（v1.0）
5. **RoPE**: 现代LLM标配
6. **GQA**: 内存效率关键
7. **SwiGLU + RMSNorm**: LLaMA兼容性

### 📌 中优先级（v1.5+）
8. Flash Attention
9. INT8量化
10. 批处理推理

---

## 🎯 成功指标

### Beta 0.2
- ✅ GPT-2 Small生成连贯文本
- ✅ KV Cache加速10倍以上
- ✅ 支持top-k/top-p采样

### v1.0
- ✅ LLaMA-2-7B成功加载和推理
- ✅ 生成质量与官方实现一致
- ✅ 支持至少2048 token上下文

### v2.0
- ✅ 达到llama.cpp 50%性能
- ✅ 支持INT4量化
- ✅ 支持流式生成

---

**下一步行动：**
1. 从Beta 0.2的Causal Attention开始
2. 创建`examples/gpt2/simple_generation.cpp`作为目标demo
3. 逐步实现GPT-2所需的最小组件集

**预计总时间：** 3-6个月达到v1.0（支持主流开源LLM）
