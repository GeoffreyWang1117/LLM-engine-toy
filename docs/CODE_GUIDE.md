# 代码详解指南

## 📖 目录

1. [项目架构](#项目架构)
2. [Matrix类详解](#matrix类详解)
3. [Transformer组件详解](#transformer组件详解)
4. [关键算法实现](#关键算法实现)
5. [运行教学示例](#运行教学示例)

---

## 项目架构

```
核心实现层次：
┌─────────────────────────────────────┐
│   Transformer Block (未来实现)      │
│   - 完整的编码器/解码器层           │
└─────────────────────────────────────┘
            ↑
┌─────────────────────────────────────┐
│   Transformer 组件层 (layers.h/cpp) │
│   - Linear (全连接层)               │
│   - LayerNorm (层归一化)            │
│   - SelfAttention (自注意力)        │
│   - FeedForward (前馈网络)          │
└─────────────────────────────────────┘
            ↑
┌─────────────────────────────────────┐
│   矩阵运算层 (matrix.h/cpp)         │
│   - 加法、减法、乘法                │
│   - 矩阵乘法 (matmul)               │
│   - 转置 (transpose)                │
└─────────────────────────────────────┘
```

---

## Matrix类详解

### 1. 数据存储

```cpp
class Matrix {
private:
    size_t rows_;              // 行数
    size_t cols_;              // 列数
    std::vector<float> data_;  // 一维数组存储数据 (行优先)
};
```

**存储方式：行优先**
```
矩阵:  [1 2 3]      内存: [1, 2, 3, 4, 5, 6]
       [4 5 6]
```

### 2. 核心功能

#### 2.1 矩阵创建
```cpp
// 方式1: 指定大小
Matrix A(2, 3);  // 2x3 零矩阵

// 方式2: 指定初始值
Matrix B(2, 3, 5.0f);  // 全部填充5.0

// 方式3: 从数组创建
Matrix C(2, 3, {1, 2, 3, 4, 5, 6});

// 方式4: 工厂方法
Matrix zeros = Matrix::zeros(2, 3);
Matrix ones = Matrix::ones(2, 3);
Matrix randn = Matrix::randn(2, 3);  // 正态分布随机数
```

#### 2.2 矩阵乘法实现 (最关键!)

```cpp
Matrix Matrix::matmul(const Matrix& other) const {
    // A[m x n] × B[n x p] = C[m x p]

    if (cols_ != other.rows_) {
        throw std::invalid_argument("维度不匹配");
    }

    Matrix result(rows_, other.cols_, 0.0f);

    // 三重循环计算
    for (size_t i = 0; i < rows_; ++i) {           // 遍历A的行
        for (size_t j = 0; j < other.cols_; ++j) { // 遍历B的列
            float sum = 0.0f;
            for (size_t k = 0; k < cols_; ++k) {   // 点积
                sum += at(i, k) * other.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }

    return result;
}
```

**计算示例：**
```
A = [1 2]    B = [5 6]    C = A × B
    [3 4]        [7 8]

C[0][0] = 1×5 + 2×7 = 5 + 14 = 19
C[0][1] = 1×6 + 2×8 = 6 + 16 = 22
C[1][0] = 3×5 + 4×7 = 15 + 28 = 43
C[1][1] = 3×6 + 4×8 = 18 + 32 = 50

结果: C = [19 22]
          [43 50]
```

#### 2.3 转置实现

```cpp
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);  // 交换行列

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.at(j, i) = at(i, j);  // 交换索引
        }
    }

    return result;
}
```

---

## Transformer组件详解

### 1. Linear层 (全连接层)

**公式：** `y = x·W^T + b`

```cpp
class Linear {
private:
    Matrix weight_;  // [out_features, in_features]
    Matrix bias_;    // [out_features, 1]
};
```

**前向传播实现：**
```cpp
Matrix Linear::forward(const Matrix& input) const {
    // input: [batch_size, in_features]
    // weight: [out_features, in_features]
    // output: [batch_size, out_features]

    // 1. 矩阵乘法: x·W^T
    Matrix output = input.matmul(weight_.transpose());

    // 2. 添加偏置
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += bias_(j, 0);
        }
    }

    return output;
}
```

**实际例子：**
```
输入: [1, 2, 3]  (3维特征)
权重: [[0.5, 1.0, 0.2],   (3 -> 2 维度变换)
       [-0.5, -1.0, 0.8]]
偏置: [0.1, -0.1]

输出: y = [1,2,3] × [[0.5, -0.5],   + [0.1, -0.1]
                      [1.0, -1.0],
                      [0.2,  0.8]]
       = [3.1, -0.1]
```

### 2. LayerNorm层 (层归一化)

**公式：** `y = (x - μ) / √(σ² + ε) × γ + β`

- μ: 均值
- σ²: 方差
- ε: 防止除零的小常数 (1e-5)
- γ, β: 可学习的缩放和偏移参数

```cpp
Matrix LayerNorm::forward(const Matrix& input) const {
    Matrix output(input.rows(), input.cols());

    for (size_t i = 0; i < input.rows(); ++i) {  // 对每个样本
        // 1. 计算均值
        float mean = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            mean += input(i, j);
        }
        mean /= input.cols();

        // 2. 计算方差
        float variance = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            float diff = input(i, j) - mean;
            variance += diff * diff;
        }
        variance /= input.cols();

        // 3. 归一化并应用缩放/偏移
        float std = sqrt(variance + eps_);
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = (input(i, j) - mean) / std;
            output(i, j) = output(i, j) * weight_(j, 0) + bias_(j, 0);
        }
    }

    return output;
}
```

**作用：**
- 稳定训练：消除不同batch/样本间的尺度差异
- 加速收敛：让优化更容易

### 3. Self-Attention (自注意力机制)

**这是Transformer的核心！**

**公式：** `Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V`

```cpp
class SelfAttention {
private:
    Linear q_proj_;    // Query投影
    Linear k_proj_;    // Key投影
    Linear v_proj_;    // Value投影
    Linear out_proj_;  // 输出投影
};
```

**完整流程：**

```
输入序列: X [seq_len, embed_dim]
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
   Q=X·W_q  K=X·W_k  V=X·W_v
    ↓         ↓         ↓
   [L,d]    [L,d]    [L,d]

    Q·K^T / √d
       ↓
   Scores [L,L]  ← 注意力分数矩阵
       ↓
   Softmax
       ↓
   Weights [L,L] ← 归一化的注意力权重
       ↓
   Weights·V
       ↓
   Output [L,d]  ← 加权求和的结果
       ↓
   Output·W_o
       ↓
   Final [L,d]
```

**Softmax实现：**
```cpp
Matrix SelfAttention::softmax(const Matrix& input) const {
    Matrix output(input.rows(), input.cols());

    for (size_t i = 0; i < input.rows(); ++i) {
        // 1. 找最大值（数值稳定性）
        float max_val = input(i, 0);
        for (size_t j = 1; j < input.cols(); ++j) {
            max_val = std::max(max_val, input(i, j));
        }

        // 2. 计算exp(x - max)并求和
        float sum = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = std::exp(input(i, j) - max_val);
            sum += output(i, j);
        }

        // 3. 归一化
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) /= sum;
        }
    }

    return output;
}
```

**注意力的直观理解：**
```
假设有3个单词的序列：["我", "爱", "你"]

Attention权重矩阵可能是：
        我    爱    你
我:   [0.6, 0.3, 0.1]  ← "我"主要关注自己，部分关注"爱"
爱:   [0.3, 0.4, 0.3]  ← "爱"均衡关注三个词
你:   [0.1, 0.3, 0.6]  ← "你"主要关注自己，部分关注"爱"

每一行的权重和为1，表示该位置如何分配注意力到所有位置
```

### 4. FeedForward Network (前馈网络)

**结构：** `x → Linear → GELU → Linear → y`

```cpp
class FeedForward {
private:
    Linear fc1_;  // [embed_dim, hidden_dim]
    Linear fc2_;  // [hidden_dim, embed_dim]
};
```

**GELU激活函数：**

公式：`GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))`

```cpp
Matrix FeedForward::gelu(const Matrix& input) const {
    Matrix output(input.rows(), input.cols());
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);

    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            float x = input(i, j);
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
            output(i, j) = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }

    return output;
}
```

**为什么用GELU而不是ReLU？**
- GELU是平滑的，处处可导
- 在负值区域有小的非零梯度
- GPT、BERT等现代模型的标准选择

---

## 关键算法实现

### 1. 矩阵乘法的计算复杂度

```
A[m × n] × B[n × p] = C[m × p]

时间复杂度: O(m × n × p)
空间复杂度: O(m × p)

示例：
- 输入序列: [512, 768]  (512个token，每个768维)
- Linear层权重: [768, 3072]
- 计算量: 512 × 768 × 3072 ≈ 12亿次乘法
```

### 2. Attention的计算复杂度

```
序列长度: L
特征维度: d

Q·K^T: O(L² × d)   ← 这是瓶颈！
Softmax: O(L²)
Weights·V: O(L² × d)

总计: O(L² × d)

这就是为什么长文本处理很慢的原因！
```

### 3. 完整Transformer Block的数据流

```
输入 x [batch, seq_len, embed_dim]
    ↓
┌─────────────────────────┐
│ LayerNorm               │
│    ↓                    │
│ Self-Attention          │
│    ↓                    │
│ 残差: x = x + attn_out  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ LayerNorm               │
│    ↓                    │
│ FeedForward             │
│    ↓                    │
│ 残差: x = x + ffn_out   │
└─────────────────────────┘
    ↓
输出 [batch, seq_len, embed_dim]
```

---

## 运行教学示例

### 1. Matrix类演示

```bash
make run-matrix
```

这会演示：
- 矩阵的各种创建方式
- 基本运算（加减乘、转置）
- 实际应用示例

### 2. Transformer组件演示

```bash
make run-layers
```

这会演示：
- Linear层的作用
- LayerNorm的效果
- Self-Attention的计算
- FeedForward的变换
- 完整Transformer Block的组合

### 3. 完整示例

```bash
make run
```

运行所有组件的综合测试。

---

## WebAssembly版本说明

### WASM绑定层

在 `web/wasm/wasm_bindings.cpp` 中，我们使用Emscripten的`embind`库将C++类暴露给JavaScript：

```cpp
EMSCRIPTEN_BINDINGS(llm_engine) {
    // 绑定Matrix类
    class_<MatrixWrapper>("Matrix")
        .constructor<size_t, size_t>()
        .function("matmul", &MatrixWrapper::matmul)
        .function("toArray", &MatrixWrapper::toArray)
        // ...

    // 绑定Layer类
    class_<LinearWrapper>("Linear")
        .constructor<size_t, size_t>()
        .function("forward", &LinearWrapper::forward);
}
```

### JavaScript调用示例

```javascript
// 创建矩阵
const A = Module.Matrix.fromArray([[1, 2], [3, 4]]);
const B = Module.Matrix.fromArray([[5, 6], [7, 8]]);

// 矩阵乘法
const C = A.matmul(B);

// 转换为JS数组
const result = C.toArray();
console.log(result);  // [[19, 22], [43, 50]]

// 创建Linear层
const linear = new Module.Linear(4, 3);
const input = Module.Matrix.randn(2, 4);
const output = linear.forward(input);
```

---

## 下一步学习建议

1. **深入理解Attention**
   - 阅读论文 "Attention Is All You Need"
   - 手工计算一个小例子的Attention

2. **性能优化**
   - 研究矩阵乘法优化（分块、SIMD）
   - 实现KV Cache（减少重复计算）

3. **扩展功能**
   - 实现完整的Transformer Block
   - 添加位置编码
   - 实现简单的文本生成

4. **模型加载**
   - 了解权重文件格式
   - 实现从PyTorch导出的权重加载

---

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始Transformer论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程
- [GPT系列论文](https://openai.com/research/) - OpenAI的语言模型
- [Emscripten文档](https://emscripten.org/docs/index.html) - WASM编译
