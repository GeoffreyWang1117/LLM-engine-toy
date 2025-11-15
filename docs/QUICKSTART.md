# 快速开始指南

## 🎯 5分钟快速体验

### 第一步：编译项目

```bash
cd /home/coder-gw/Projects/LLM-engine
make clean
make
```

### 第二步：运行教学示例

#### 示例1：Matrix类基础教学
```bash
make run-matrix
```

**你会看到：**
- 矩阵的8种创建和操作方式
- 矩阵乘法的详细计算过程
- Linear层的工作原理演示

#### 示例2：Transformer组件教学
```bash
make run-layers
```

**你会看到：**
- Linear层：如何做维度变换
- LayerNorm：如何归一化数据
- Self-Attention：序列中的信息如何交互
- FeedForward：如何增加非线性
- 完整Transformer Block：如何组合所有组件

#### 示例3：综合测试
```bash
make run
```

运行所有组件的快速测试。

---

## 📚 深入学习

### 阅读代码详解

```bash
cat CODE_GUIDE.md
```

这个文档包含：
- 每个组件的详细实现
- 关键算法的解释
- 数学公式和代码对照
- 性能分析

### 代码结构说明

```
核心代码（按学习顺序）：

1. include/llm_engine/matrix.h
   ├─ Matrix类接口定义
   └─ 基础矩阵运算

2. src/matrix.cpp
   ├─ 矩阵乘法实现 ⭐
   ├─ 转置实现
   └─ 工具函数

3. include/llm_engine/layers.h
   ├─ Linear层接口
   ├─ LayerNorm层接口
   ├─ SelfAttention层接口 ⭐⭐⭐
   └─ FeedForward层接口

4. src/layers.cpp
   ├─ Linear前向传播
   ├─ LayerNorm实现
   ├─ Attention机制 ⭐⭐⭐
   ├─ Softmax实现
   ├─ GELU激活函数
   └─ FeedForward实现
```

### 建议学习路径

**Day 1: 理解矩阵运算**
1. 运行 `make run-matrix`
2. 阅读 `src/matrix.cpp` 中的 `matmul` 函数
3. 手工计算一个 2×2 矩阵乘法验证理解

**Day 2: 理解Linear层**
1. 运行 `make run-layers`，关注Linear部分
2. 阅读 `src/layers.cpp` 中的 `Linear::forward`
3. 理解公式：y = x·W^T + b

**Day 3: 理解LayerNorm**
1. 关注tutorial中的LayerNorm演示
2. 理解归一化的作用
3. 观察不同样本归一化后的效果

**Day 4-5: 理解Self-Attention（最重要！）**
1. 仔细阅读CODE_GUIDE.md中的Attention部分
2. 理解Q、K、V的含义
3. 手工计算一个小规模的Attention
4. 理解为什么叫"注意力"

**Day 6: 理解完整流程**
1. 运行完整的Transformer Block示例
2. 理解残差连接的作用
3. 思考如何堆叠多层

---

## 🔧 自己动手实验

### 实验1：修改矩阵大小

编辑 `examples/tutorial_matrix.cpp`：

```cpp
// 原来：
Matrix G(2, 3, {1, 2, 3, 4, 5, 6});

// 改成：
Matrix G(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
```

重新编译运行：
```bash
make clean && make run-matrix
```

### 实验2：观察不同输入对Attention的影响

编辑 `examples/tutorial_layers.cpp`，在Self-Attention部分：

```cpp
// 尝试不同的输入
Matrix input3 = Matrix::zeros(3, 8);  // 全0输入
// 或
Matrix input3 = Matrix::ones(3, 8);   // 全1输入
```

### 实验3：改变网络维度

```cpp
// 尝试不同的维度
SelfAttention attn(16, 4);  // 16维，4个头
// 或
FeedForward ffn(8, 64);     // hidden扩大到8倍
```

---

## 💡 关键概念速查

### 矩阵乘法
```
[m×n] × [n×p] = [m×p]
      ↑
    必须相等！
```

### Linear层
```
y = x·W^T + b
输入维度 → 输出维度的映射
```

### LayerNorm
```
归一化 → 均值0，方差1
作用：稳定训练
```

### Self-Attention
```
Q (Query):  我在寻找什么？
K (Key):    我能提供什么？
V (Value):  我的内容是什么？

Attention = softmax(Q·K^T / √d) · V
```

### 残差连接
```
y = x + F(x)
作用：让梯度更好地流动，训练更深的网络
```

---

## 🚀 下一步计划

### 近期可以实现

1. **完整的Transformer Block类**
   ```cpp
   class TransformerBlock {
       LayerNorm ln1, ln2;
       SelfAttention attn;
       FeedForward ffn;
   };
   ```

2. **位置编码**
   ```cpp
   class PositionalEncoding {
       // sin/cos位置编码
   };
   ```

3. **简单的文本生成**
   ```cpp
   class SimpleGenerator {
       // 自回归生成
   };
   ```

### 中期目标

- 从PyTorch加载预训练权重
- 实现KV Cache优化
- 支持批处理推理

### 长期愿景

- 完整的GPT-2实现
- 性能优化（SIMD、多线程）
- 量化支持（INT8推理）

---

## 📖 推荐阅读

### 必读论文
1. **Attention Is All You Need** (2017)
   - Transformer原始论文
   - https://arxiv.org/abs/1706.03762

2. **Layer Normalization** (2016)
   - LayerNorm详解
   - https://arxiv.org/abs/1607.06450

### 可视化教程
1. **The Illustrated Transformer**
   - Jay Alammar的可视化教程
   - http://jalammar.github.io/illustrated-transformer/

2. **The Illustrated GPT-2**
   - GPT模型解析
   - http://jalammar.github.io/illustrated-gpt2/

### 代码实现参考
1. **nanoGPT** by Andrej Karpathy
   - 最小化的GPT实现
   - https://github.com/karpathy/nanoGPT

2. **llama.cpp**
   - C++实现的LLaMA推理
   - https://github.com/ggerganov/llama.cpp

---

## ❓ 常见问题

**Q: 为什么用行优先存储矩阵？**
A: C++的内存布局是行优先的，这样可以更好地利用CPU缓存。

**Q: Attention的时间复杂度为什么是O(L²)？**
A: 因为需要计算序列中每两个位置之间的关系，形成一个L×L的注意力矩阵。

**Q: 为什么需要残差连接？**
A: 残差连接让梯度可以直接流过，避免梯度消失，使得训练深层网络成为可能。

**Q: GELU和ReLU有什么区别？**
A: GELU是平滑的，处处可导，在现代Transformer中效果更好。

**Q: 多头注意力是怎么实现的？**
A: 将Q,K,V分成多个头，并行计算多个注意力，最后拼接。当前实现是简化版。

---

## 🎓 学习检查清单

- [ ] 理解矩阵乘法的计算过程
- [ ] 知道Linear层的作用
- [ ] 理解LayerNorm的归一化过程
- [ ] 掌握Self-Attention的Q、K、V概念
- [ ] 理解softmax的作用
- [ ] 知道残差连接的重要性
- [ ] 能够解释完整Transformer Block的数据流
- [ ] 理解为什么Attention的复杂度是O(L²)

---

## 💪 动手练习建议

1. **手工计算Attention**
   - 选一个3×3的小例子
   - 手工计算Q·K^T
   - 手工计算softmax
   - 手工计算加权求和

2. **可视化Attention权重**
   - 修改代码输出注意力矩阵
   - 观察不同输入的注意力分布

3. **实现Multi-Head Attention**
   - 改造当前的SelfAttention
   - 真正实现多头拆分和拼接

4. **性能测试**
   - 测试不同矩阵大小的运算时间
   - 分析性能瓶颈
   - 尝试优化矩阵乘法

Happy Learning! 🎉
