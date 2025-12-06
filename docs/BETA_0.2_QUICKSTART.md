# Beta 0.2 快速开始指南

从BERT到GPT-2：实现第一个文本生成模型

---

## 🎯 目标

实现一个最小可用的GPT-2推理引擎，能够：
1. 加载GPT-2 Small (117M)模型
2. 进行自回归文本生成
3. 支持基础采样策略（greedy, temperature）

---

## 📚 核心概念对比

### BERT vs GPT-2：关键差异

| 维度 | BERT | GPT-2 | 为什么不同？ |
|------|------|-------|-------------|
| **任务** | 理解（双向编码） | 生成（自回归） | BERT看全文，GPT逐字生成 |
| **注意力** | 全局可见 | 因果遮罩 | GPT不能"偷看"未来 |
| **推理** | 一次forward | 循环生成 | GPT需要逐token预测 |
| **输出** | 固定长度向量 | 下一个token概率 | 生成vs理解 |

### 注意力机制对比

**BERT (Bidirectional):**
```
Input:  [CLS]  The  cat  sat  [SEP]

Attention Matrix (全1，全部可见):
      CLS  The  cat  sat  SEP
CLS   [1]  [1]  [1]  [1]  [1]
The   [1]  [1]  [1]  [1]  [1]
cat   [1]  [1]  [1]  [1]  [1]
sat   [1]  [1]  [1]  [1]  [1]
SEP   [1]  [1]  [1]  [1]  [1]
```

**GPT-2 (Causal):**
```
Input:  The  cat  sat  on

Attention Mask (下三角，只看过去):
      The  cat  sat  on
The   [1]  [0]  [0]  [0]  ← 只能看到"The"
cat   [1]  [1]  [0]  [0]  ← 能看到"The cat"
sat   [1]  [1]  [1]  [0]  ← 能看到"The cat sat"
on    [1]  [1]  [1]  [1]  ← 能看到全部历史
```

---

## 🔨 实现步骤

### Step 1: Causal Attention（最小改动）

**现有代码：** `src/layers.cpp::SelfAttention::forward()`

**需要修改的部分：**
```cpp
// 当前BERT实现（简化版）
Matrix SelfAttention::forward(const Matrix& input) const {
    Matrix Q = input * Wq;
    Matrix K = input * Wk;
    Matrix V = input * Wv;

    // Attention scores
    Matrix scores = (Q * K.transpose()) / sqrt(d_k);

    // BERT: 没有mask，全部可见
    Matrix attn_weights = softmax(scores);

    return attn_weights * V;
}
```

**GPT-2需要的改动（添加causal mask）：**
```cpp
Matrix CausalAttention::forward(const Matrix& input) const {
    Matrix Q = input * Wq;
    Matrix K = input * Wk;
    Matrix V = input * Wv;

    Matrix scores = (Q * K.transpose()) / sqrt(d_k);

    // 🔥 NEW: 添加因果mask
    scores = apply_causal_mask(scores);  // 将上三角设为-inf

    Matrix attn_weights = softmax(scores);  // softmax(-inf) = 0

    return attn_weights * V;
}

Matrix CausalAttention::apply_causal_mask(const Matrix& scores) const {
    size_t seq_len = scores.rows();
    Matrix masked = scores;

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            masked(i, j) = -1e9;  // 负无穷（softmax后变0）
        }
    }
    return masked;
}
```

---

### Step 2: KV Cache（性能关键）

**问题：为什么需要KV Cache？**

生成"The cat sat on the mat"时的计算：

**不使用KV Cache（低效）：**
```
Step 1: "The" → 计算 Q1, K1, V1
Step 2: "The cat" → 重新计算 Q1, K1, V1, Q2, K2, V2  ❌ 重复计算
Step 3: "The cat sat" → 重新计算全部Q, K, V  ❌ 重复计算
...
```

**使用KV Cache（高效）：**
```
Step 1: "The" → 计算Q1, K1, V1，缓存K1, V1
Step 2: "cat" → 只计算Q2, K2, V2，复用K1, V1  ✅
Step 3: "sat" → 只计算Q3, K3, V3，复用K1, K2, V1, V2  ✅
...
```

**实现：**
```cpp
// include/llm_engine/kv_cache.h
struct KVCache {
    Matrix key_cache;    // 缓存所有历史的K
    Matrix value_cache;  // 缓存所有历史的V

    void append(const Matrix& new_keys, const Matrix& new_values) {
        // 将新的K, V追加到缓存
        key_cache = concat(key_cache, new_keys, /*axis=*/0);
        value_cache = concat(value_cache, new_values, /*axis=*/0);
    }
};

Matrix CausalAttention::forward_with_cache(
    const Matrix& input,
    KVCache& cache
) const {
    // 只计算新token的Q, K, V
    Matrix Q_new = input * Wq;  // [1, d_model] (单个新token)
    Matrix K_new = input * Wk;
    Matrix V_new = input * Wv;

    // 更新cache
    cache.append(K_new, V_new);

    // 用新Q和全部历史KV计算attention
    Matrix scores = Q_new * cache.key_cache.transpose();
    scores = scores / sqrt(d_k);

    // 因果mask（新token只能看历史，不能看未来）
    Matrix attn_weights = softmax(scores);

    return attn_weights * cache.value_cache;
}
```

---

### Step 3: GPT Decoder Block

**BERT Block (Post-LN):**
```cpp
Matrix BertBlock::forward(const Matrix& x) {
    // Attention
    Matrix attn_out = attention(x);
    Matrix x1 = x + attn_out;  // Residual
    x1 = layer_norm1(x1);      // Post-LN

    // FFN
    Matrix ffn_out = ffn(x1);
    Matrix x2 = x1 + ffn_out;
    x2 = layer_norm2(x2);

    return x2;
}
```

**GPT-2 Block (Pre-LN):**
```cpp
Matrix GPTBlock::forward(const Matrix& x, KVCache* cache) {
    // Attention (Pre-LN)
    Matrix x1 = layer_norm1(x);           // 🔥 先norm
    Matrix attn_out = attention(x1, cache);
    x = x + attn_out;                     // 残差连接

    // FFN (Pre-LN)
    Matrix x2 = layer_norm2(x);           // 🔥 先norm
    Matrix ffn_out = ffn(x2);
    x = x + ffn_out;

    return x;
}
```

**为什么Pre-LN更好？**
- 训练更稳定（梯度流更平滑）
- 现代大模型几乎都用Pre-LN

---

### Step 4: 文本生成循环

**核心生成算法（自回归）：**
```cpp
std::vector<int> generate(
    const std::vector<int>& prompt,  // 输入："The cat"
    int max_new_tokens               // 生成长度：10
) {
    std::vector<int> tokens = prompt;
    std::vector<KVCache> caches(num_layers);  // 每层一个cache

    for (int i = 0; i < max_new_tokens; ++i) {
        // 1. 取最后一个token
        int last_token = tokens.back();

        // 2. Forward（使用KV cache）
        Matrix logits = model.forward_single_token(last_token, caches);

        // 3. Sampling（从logits选择下一个token）
        int next_token = sampler.sample(logits);

        // 4. 添加到序列
        tokens.push_back(next_token);

        // 5. 检查结束符
        if (next_token == EOS_TOKEN) break;
    }

    return tokens;
}
```

**示例流程：**
```
Prompt: "The cat"
Tokens: [464, 3797]  (tokenizer.encode("The cat"))

Step 1:
  Input: [464, 3797]
  Output logits: [vocab_size] 概率分布
  Sample: 3332 (对应"sat")
  Tokens: [464, 3797, 3332]

Step 2:
  Input: 3332 (只需要新token！)
  Output logits: [vocab_size]
  Sample: 319 ("on")
  Tokens: [464, 3797, 3332, 319]

...
```

---

### Step 5: Sampling策略

**1. Greedy（贪婪）：**
```cpp
int greedy_sample(const Matrix& logits) {
    return argmax(logits);  // 选概率最高的
}
```
- ✅ 确定性，可复现
- ❌ 无创造性，容易重复

**2. Temperature（温度）：**
```cpp
int temperature_sample(const Matrix& logits, float temp) {
    Matrix scaled = logits / temp;  // temp < 1: 更确定, temp > 1: 更随机
    Matrix probs = softmax(scaled);
    return sample_from_distribution(probs);
}
```
- `temp = 0.1`: 几乎确定（接近greedy）
- `temp = 1.0`: 按原始概率
- `temp = 2.0`: 更随机，更有创造性

**3. Top-k：**
```cpp
int top_k_sample(const Matrix& logits, int k) {
    // 只从概率最高的k个token中采样
    auto top_k_indices = get_top_k(logits, k);
    Matrix top_k_logits = logits[top_k_indices];
    Matrix probs = softmax(top_k_logits);
    return top_k_indices[sample_from_distribution(probs)];
}
```

**4. Top-p (Nucleus)：**
```cpp
int top_p_sample(const Matrix& logits, float p) {
    // 选择累积概率达到p的最小token集
    Matrix probs = softmax(logits);
    auto sorted = sort_descending(probs);

    float cumsum = 0.0;
    std::vector<int> nucleus;
    for (int i = 0; i < probs.size(); ++i) {
        cumsum += sorted[i].prob;
        nucleus.push_back(sorted[i].index);
        if (cumsum >= p) break;
    }

    return sample_from(nucleus);
}
```

---

## 📁 建议的最小实现文件

### 新增头文件：
```
include/llm_engine/
├── causal_attention.h      # Causal masked attention
├── kv_cache.h              # KV缓存
├── gpt_block.h             # GPT decoder block
├── gpt_model.h             # 完整GPT模型
├── sampler.h               # 采样策略
└── bpe_tokenizer.h         # BPE分词器
```

### 核心实现优先级：

**Phase 1（1周）：基础组件**
1. `causal_attention.cpp` - 因果注意力
2. `kv_cache.cpp` - KV缓存结构
3. `gpt_block.cpp` - GPT解码块

**Phase 2（1周）：生成引擎**
4. `sampler.cpp` - 采样策略
5. `gpt_model.cpp` - 完整模型
6. `examples/gpt2/simple_generation.cpp` - demo

**Phase 3（1周）：Tokenizer**
7. `bpe_tokenizer.cpp` - BPE实现
8. 完整端到端测试

---

## 🧪 测试计划

### 测试1: Causal Mask
```cpp
// 验证mask正确性
Matrix scores(3, 3);
scores.fill(1.0);

Matrix masked = apply_causal_mask(scores);
// 期望：
// [[  1, -inf, -inf],
//  [  1,    1, -inf],
//  [  1,    1,    1]]
```

### 测试2: KV Cache
```cpp
// 验证缓存工作正常
KVCache cache;

Matrix K1(1, 64), V1(1, 64);  // 第1个token
cache.append(K1, V1);
assert(cache.key_cache.rows() == 1);

Matrix K2(1, 64), V2(1, 64);  // 第2个token
cache.append(K2, V2);
assert(cache.key_cache.rows() == 2);  // ✅ 累积
```

### 测试3: 生成质量
```cpp
// 输入固定prompt，验证输出合理
std::string prompt = "Once upon a time";
std::string generated = generate(prompt, max_tokens=20);

// 预期：连贯的英文句子
// "Once upon a time, there was a little girl who lived in..."
```

---

## 📊 性能预期

### GPT-2 Small (117M参数)

**不使用KV Cache：**
- 生成50 tokens: ~25秒
- 每token平均: ~500ms

**使用KV Cache：**
- 生成50 tokens: ~2.5秒
- 每token平均: ~50ms
- **加速：10倍** ✅

---

## 🎯 最小可行产品（MVP）

**目标Demo：**
```bash
$ ./build/gpt2_demo

请输入prompt: The quick brown fox
正在生成...

输出:
The quick brown fox jumps over the lazy dog. The dog is sleeping
under a tree. It is a beautiful sunny day.

生成统计:
  - 生成tokens: 25
  - 耗时: 1.2秒
  - 平均速度: 20.8 tokens/s
```

**核心功能：**
- ✅ 加载GPT-2 Small权重
- ✅ 编码输入文本（BPE）
- ✅ 自回归生成
- ✅ 解码输出文本
- ✅ 使用KV Cache加速

---

## 🚀 开始第一个任务

**推荐第一步：实现Causal Attention**

创建文件：`include/llm_engine/causal_attention.h`

```cpp
#ifndef LLM_ENGINE_CAUSAL_ATTENTION_H
#define LLM_ENGINE_CAUSAL_ATTENTION_H

#include "matrix.h"

namespace llm {

/**
 * @brief Causal Self-Attention (GPT风格)
 *
 * 只能看到当前位置及之前的token（因果mask）
 */
class CausalAttention {
public:
    CausalAttention(int hidden_size, int num_heads);

    /**
     * @brief 前向传播（训练模式，完整序列）
     */
    Matrix forward(const Matrix& input) const;

    /**
     * @brief 生成模式（使用KV cache）
     */
    Matrix forward_with_cache(
        const Matrix& input,      // [1, hidden_size] 单个新token
        KVCache& cache           // 历史KV
    ) const;

private:
    int hidden_size_;
    int num_heads_;
    int head_dim_;

    Matrix Wq_, Wk_, Wv_, Wo_;

    /**
     * @brief 应用因果mask（上三角设为-inf）
     */
    Matrix apply_causal_mask(const Matrix& scores) const;
};

} // namespace llm

#endif
```

---

**下一步：**
1. 实现`causal_attention.cpp`
2. 编写单元测试
3. 继续KV Cache实现

**预计Beta 0.2完成时间：2-3周**
