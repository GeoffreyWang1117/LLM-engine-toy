# LLM架构对比详解

深入理解BERT、GPT-2、LLaMA的技术差异

---

## 📐 整体架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                    BERT (Encoder-Only)                      │
├─────────────────────────────────────────────────────────────┤
│  Input: "The cat [MASK] on the mat"                        │
│    ↓                                                        │
│  Token + Position Embeddings (Absolute)                    │
│    ↓                                                        │
│  ┌───────────────────────────────────┐                     │
│  │  Encoder Block (x12)              │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ Multi-Head Attention        │  │ (Bidirectional)     │
│  │  │ All tokens can see all      │  │                     │
│  │  └─────────────────────────────┘  │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ Feed Forward (GELU)         │  │                     │
│  │  └─────────────────────────────┘  │                     │
│  │  (Post-LayerNorm)                 │                     │
│  └───────────────────────────────────┘                     │
│    ↓                                                        │
│  Output: [CLS] representation for classification           │
│  Task: Understanding (classification, NER, QA)             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   GPT-2 (Decoder-Only)                      │
├─────────────────────────────────────────────────────────────┤
│  Input: "The cat sat on the"                               │
│    ↓                                                        │
│  Token + Position Embeddings (Absolute)                    │
│    ↓                                                        │
│  ┌───────────────────────────────────┐                     │
│  │  Decoder Block (x12)              │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ Masked Self-Attention       │  │ (Causal)            │
│  │  │ Tokens only see past        │  │                     │
│  │  └─────────────────────────────┘  │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ Feed Forward (GELU)         │  │                     │
│  │  └─────────────────────────────┘  │                     │
│  │  (Pre-LayerNorm)                  │                     │
│  └───────────────────────────────────┘                     │
│    ↓                                                        │
│  Output: Logits for next token prediction                  │
│  Predict: "mat" (auto-regressive generation)               │
│  Task: Generation (text completion, dialog)                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              LLaMA/Qwen (Modern Decoder-Only)               │
├─────────────────────────────────────────────────────────────┤
│  Input: "The cat sat on the"                               │
│    ↓                                                        │
│  Token Embeddings (no position yet)                        │
│    ↓                                                        │
│  ┌───────────────────────────────────┐                     │
│  │  Decoder Block (x32)              │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ RMSNorm                     │  │ (Pre-Norm)          │
│  │  └─────────────────────────────┘  │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ Grouped-Query Attention     │  │ (Causal + RoPE)     │
│  │  │ + Rotary Position Encoding  │  │                     │
│  │  └─────────────────────────────┘  │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ RMSNorm                     │  │                     │
│  │  └─────────────────────────────┘  │                     │
│  │  ┌─────────────────────────────┐  │                     │
│  │  │ SwiGLU FFN                  │  │                     │
│  │  └─────────────────────────────┘  │                     │
│  └───────────────────────────────────┘                     │
│    ↓                                                        │
│  RMSNorm                                                    │
│    ↓                                                        │
│  Output: Next token logits                                 │
│  Task: Chat, instruction following, code generation        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 注意力机制详解

### 1. BERT: Bidirectional Attention (双向注意力)

**特点：** 每个token可以关注序列中的所有token

**Attention Matrix示例：**
```
Input: "The cat sat on the mat"
Tokens:  T   c   s   o   t   m

Attention权重矩阵 (6x6):
      T    c    s    o    t    m
  T [0.3][0.2][0.1][0.1][0.2][0.1]  ← "The"可以看全部
  c [0.1][0.4][0.2][0.1][0.1][0.1]  ← "cat"可以看全部
  s [0.1][0.2][0.4][0.2][0.1][0.0]  ← "sat"可以看全部
  o [0.1][0.1][0.2][0.3][0.2][0.1]
  t [0.2][0.1][0.1][0.2][0.3][0.1]
  m [0.1][0.1][0.1][0.1][0.2][0.4]
```

**用途：**
- 文本分类：理解整个句子的语义
- 命名实体识别：需要上下文理解实体
- 问答：需要同时看问题和上下文

**代码实现：**
```cpp
Matrix BertAttention::forward(const Matrix& input) {
    Matrix Q = input * Wq;
    Matrix K = input * Wk;
    Matrix V = input * Wv;

    // Attention scores
    Matrix scores = (Q * K.transpose()) / sqrt(head_dim);

    // ✅ 无mask，全部可见
    Matrix attn_weights = softmax(scores);

    return attn_weights * V;
}
```

---

### 2. GPT-2: Causal Attention (因果注意力)

**特点：** 每个token只能关注它自己和之前的token（不能看未来）

**Attention Matrix示例：**
```
Input: "The cat sat on the mat"
Tokens:  T   c   s   o   t   m

Causal Attention权重矩阵 (6x6):
      T    c    s    o    t    m
  T [1.0][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ]  ← "The"只能看自己
  c [0.4][0.6][ 0 ][ 0 ][ 0 ][ 0 ]  ← "cat"只能看T,c
  s [0.2][0.3][0.5][ 0 ][ 0 ][ 0 ]  ← "sat"只能看T,c,s
  o [0.1][0.2][0.3][0.4][ 0 ][ 0 ]
  t [0.1][0.1][0.2][0.3][0.3][ 0 ]
  m [0.1][0.1][0.1][0.2][0.2][0.3]  ← "mat"可以看全部历史
```

**为什么需要Causal？**
- 生成任务：逐token预测，不能"作弊"看未来
- 自回归：下一个token只能基于历史生成

**代码实现：**
```cpp
Matrix GPT2Attention::forward(const Matrix& input) {
    Matrix Q = input * Wq;
    Matrix K = input * Wk;
    Matrix V = input * Wv;

    Matrix scores = (Q * K.transpose()) / sqrt(head_dim);

    // 🔥 应用因果mask
    size_t seq_len = scores.rows();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            scores(i, j) = -1e10;  // 设为负无穷
        }
    }

    Matrix attn_weights = softmax(scores);  // softmax(-inf) = 0

    return attn_weights * V;
}
```

---

### 3. LLaMA: Grouped-Query Attention + RoPE

**特点：**
- Causal mask（同GPT-2）
- 使用GQA减少KV cache内存
- 使用RoPE编码相对位置

**GQA示例（32 Q heads, 8 KV heads）：**
```
Query Heads:  Q1  Q2  Q3  Q4 | Q5  Q6  Q7  Q8 | ... | Q29 Q30 Q31 Q32
              └───────┬──────┘ └───────┬──────┘       └────────┬──────┘
                      │                │                       │
Key/Value:           K1/V1           K2/V2        ...        K8/V8

每4个Q头共享1组KV头 → 内存减少75%
```

**RoPE位置编码：**
```
传统绝对位置编码：
  x = token_emb + pos_emb[position]
  位置信息是加上去的

RoPE（旋转位置编码）：
  在attention计算时旋转Q和K
  q_rotated = rotate(q, position)
  k_rotated = rotate(k, position)

好处：
  1. 相对位置：关注的是相对距离，不是绝对位置
  2. 外推性：可以处理比训练时更长的序列
```

---

## 🧩 归一化方式对比

### LayerNorm (BERT/GPT-2)

**公式：**
```
mean = sum(x) / n
var = sum((x - mean)^2) / n
output = (x - mean) / sqrt(var + eps) * gamma + beta
```

**Post-LN (BERT):**
```cpp
x1 = x + attention(x)
x2 = LayerNorm(x1)  // 在残差之后norm
x3 = x2 + ffn(x2)
output = LayerNorm(x3)
```

**Pre-LN (GPT-2):**
```cpp
x1 = LayerNorm(x)    // 先norm
x2 = x + attention(x1)
x3 = LayerNorm(x2)
output = x2 + ffn(x3)
```

### RMSNorm (LLaMA/Qwen)

**公式（更简单，更快）：**
```
rms = sqrt(sum(x^2) / n + eps)
output = x / rms * gamma  // 没有beta，没有减mean
```

**为什么RMSNorm更好？**
- 计算更快（不需要计算mean）
- 效果相当（经验上）
- 训练更稳定

---

## 🔧 前馈网络对比

### GELU FFN (BERT/GPT-2)

```cpp
class FeedForward {
    Matrix forward(const Matrix& x) {
        Matrix h = x * W1 + b1;     // [hidden, 4*hidden]
        h = gelu(h);                // GELU激活
        Matrix output = h * W2 + b2; // [4*hidden, hidden]
        return output;
    }
};

// GELU: x * Φ(x)，其中Φ是标准正态分布CDF
float gelu(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)));
}
```

### SwiGLU (LLaMA/Qwen)

```cpp
class SwiGLU {
    Matrix forward(const Matrix& x) {
        Matrix gate = x * W_gate;     // [hidden, ffn_dim]
        Matrix up = x * W_up;         // [hidden, ffn_dim]

        // 门控机制
        Matrix h = swish(gate) * up;  // element-wise乘法

        Matrix output = h * W_down;   // [ffn_dim, hidden]
        return output;
    }
};

// SiLU (Swish): x * sigmoid(x)
float swish(float x) {
    return x / (1 + exp(-x));
}
```

**SwiGLU优势：**
- 门控机制：自适应控制信息流
- 性能提升：在多个任务上优于GELU
- 现代标准：几乎所有新LLM都用

---

## 📊 计算复杂度对比

### Attention复杂度

| 机制 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| **BERT** | O(n² · d) | O(n²) | n=序列长度, d=维度 |
| **GPT-2** | O(n² · d) | O(n²) | 同BERT，但有causal mask |
| **GPT-2 + KV Cache** | O(n · d) | O(n · d · layers) | 生成时只计算新token |
| **GQA** | O(n² · d) | O(n · d · kv_heads) | KV cache减少 |
| **Flash Attention** | O(n² · d) | O(n) | 内存优化，速度更快 |

### 参数量对比

**BERT-Base:**
```
Embeddings:    30K vocab * 768 = 23M
12 Layers:     12 * (
                 Attention: 4 * (768*768) = 2.4M
                 FFN: 2 * (768*3072) = 4.7M
                 LayerNorm: ~1.5K
               ) = 85M
Total: ~110M
```

**GPT-2 Small:**
```
Embeddings:    50K vocab * 768 = 38M
12 Layers:     12 * 7M = 84M
Total: ~117M  (比BERT稍大，因为vocab更大)
```

**LLaMA-2-7B:**
```
Embeddings:    32K vocab * 4096 = 131M
32 Layers:     32 * (
                 GQA: ~67M
                 SwiGLU: ~134M (3个矩阵)
               ) = 6.4B
Total: ~7B
```

---

## 🎯 任务适配对比

### BERT典型任务

**1. 文本分类：**
```cpp
// 使用[CLS] token的表示
Matrix cls_output = bert_output.pooled_output;  // [1, 768]
Matrix logits = cls_output * classifier_weights; // [1, num_classes]
int predicted_class = argmax(softmax(logits));
```

**2. 命名实体识别（NER）：**
```cpp
// 使用每个token的表示
Matrix token_outputs = bert_output.sequence_output;  // [seq_len, 768]
Matrix tag_logits = token_outputs * tag_weights;     // [seq_len, num_tags]

// 每个位置预测一个标签（如B-PER, I-ORG等）
for (int i = 0; i < seq_len; ++i) {
    int tag = argmax(tag_logits.row(i));
    tags.push_back(tag);
}
```

### GPT-2典型任务

**1. 文本生成：**
```cpp
// 自回归生成
std::string prompt = "Once upon a time";
std::vector<int> tokens = tokenizer.encode(prompt);

for (int i = 0; i < max_new_tokens; ++i) {
    Matrix logits = gpt2.forward(tokens);
    int next_token = sample(logits[-1]);  // 取最后一个位置的logits
    tokens.push_back(next_token);
}

std::string generated = tokenizer.decode(tokens);
```

**2. Few-shot学习：**
```cpp
// 在prompt中提供示例
std::string prompt = R"(
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush giraffe => girafe peluche
cheese =>
)";

std::string output = gpt2.generate(prompt);
// 期望输出: "fromage"
```

### LLaMA典型任务

**1. 指令跟随：**
```cpp
std::string prompt = R"(
<s>[INST] Write a Python function to calculate factorial [/INST]
)";

std::string response = llama.generate(prompt);
// 输出完整的Python代码
```

**2. 多轮对话：**
```cpp
std::string conversation = R"(
<s>[INST] What is the capital of France? [/INST]
The capital of France is Paris. </s>
[INST] What is its population? [/INST]
)";

std::string response = llama.generate(conversation);
// "Paris has a population of approximately 2.2 million..."
```

---

## 🔄 推理流程对比

### BERT推理（单次forward）

```
Input: "The cat sat on the mat"
       ↓
  [CLS] The cat sat on the mat [SEP]
       ↓
  Tokenize: [101, 1996, 4937, 2938, 2006, 1996, 13523, 102]
       ↓
  ┌──────────────────────────────────┐
  │ Embedding                        │
  │   Token Emb + Position Emb       │
  └──────────────────────────────────┘
       ↓
  ┌──────────────────────────────────┐
  │ 12 x Encoder Blocks              │
  │   (全部token并行处理)             │
  └──────────────────────────────────┘
       ↓
  Output: [8, 768] (每个token一个向量)
       ↓
  Pooled: [1, 768] ([CLS]向量)
       ↓
  Use for classification/NER/etc.
```

### GPT-2生成（循环forward）

```
Prompt: "The cat"
        ↓
Step 1: Forward全部prompt
  Input: [464, 3797]  (The cat)
  Output: [2, 50257] logits
  Sample: 3332 (sat)
  New tokens: [464, 3797, 3332]
        ↓
Step 2: Forward新token（使用KV cache）
  Input: [3332]  (只有新token！)
  Output: [1, 50257] logits
  Sample: 319 (on)
  New tokens: [464, 3797, 3332, 319]
        ↓
Step 3: Continue...
  Input: [319]
  Output: [1, 50257]
  Sample: 262 (the)
        ↓
... 重复直到生成EOS或达到max_tokens
        ↓
Final: "The cat sat on the mat"
```

---

## 🧮 内存使用对比

### 推理时内存占用（7B模型示例）

**模型权重（FP16）：**
```
7B params × 2 bytes = 14 GB
```

**激活值（batch=1, seq_len=2048）：**

**BERT (encoder):**
```
Embeddings: 2048 × 4096 × 2 = 16 MB
Per layer: 2048 × 4096 × 2 (attn) + 2048 × 4096 × 2 (ffn) = 32 MB
32 layers: 32 × 32 MB = 1 GB
Total activations: ~1 GB
```

**GPT-2 (decoder, 无KV cache):**
```
类似BERT: ~1 GB
```

**GPT-2 (decoder, 有KV cache):**
```
KV cache per layer:
  2 (K+V) × num_heads × seq_len × head_dim × 2 bytes
  = 2 × 32 × 2048 × 128 × 2 = 33 MB

32 layers: 32 × 33 MB = 1 GB (只存KV，不存全部激活)
Total: ~1 GB (内存换速度)
```

**LLaMA with GQA (KV cache):**
```
KV cache per layer:
  2 × num_kv_heads × seq_len × head_dim × 2
  = 2 × 8 × 2048 × 128 × 2 = 8 MB  (减少75%！)

32 layers: 32 × 8 MB = 256 MB
Total: ~256 MB
```

---

## 📝 总结表格

| 特性 | BERT | GPT-2 | LLaMA/Qwen |
|------|------|-------|------------|
| **架构类型** | Encoder-Only | Decoder-Only | Decoder-Only |
| **注意力** | Bidirectional | Causal | Causal + GQA |
| **位置编码** | Absolute (learned) | Absolute (learned) | RoPE |
| **归一化** | LayerNorm (Post) | LayerNorm (Pre) | RMSNorm (Pre) |
| **FFN** | GELU | GELU | SwiGLU |
| **主要任务** | 理解 | 生成 | 聊天/指令 |
| **推理模式** | 单次forward | 循环生成 | 循环生成 |
| **KV Cache** | ❌ 不需要 | ✅ 重要优化 | ✅ + GQA优化 |
| **参数量（典型）** | 110M-340M | 117M-1.5B | 7B-70B |
| **训练任务** | MLM + NSP | Next Token Prediction | Next Token + 指令微调 |
| **输出** | Token表示 | Token logits | Token logits |
| **典型应用** | 分类、NER、QA | 文本补全、对话 | ChatBot、助手 |

---

## 🎓 从BERT迁移到GPT的核心改动

**最小改动列表（Beta 0.2）：**

1. ✅ **添加Causal Mask**
   - 在attention scores上应用下三角mask
   - ~10行代码

2. ✅ **实现KV Cache**
   - 缓存历史的Key和Value
   - ~50行代码

3. ✅ **改为Pre-LayerNorm**
   - 调整norm和attention的顺序
   - ~5行代码

4. ✅ **添加生成循环**
   - while循环逐token生成
   - ~30行代码

5. ✅ **实现采样策略**
   - greedy/temperature/top-k/top-p
   - ~100行代码

**总计：约200行核心代码即可从BERT升级到GPT-2！**

---

下一步：开始实现Beta 0.2 👉 参见 `BETA_0.2_QUICKSTART.md`
