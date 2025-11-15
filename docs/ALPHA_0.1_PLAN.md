# Alpha 0.1 开发计划：BERT推理框架

## 🎯 项目目标

在v0.1基础上，开发一个完整的BERT推理框架，支持实际模型加载和推理。

## 💻 系统资源评估

✅ **硬件配置（优秀）**
- CPU: AMD Ryzen 9 5950X (16核32线程)
- 内存: 125GB (106GB可用)
- GPU: 2x NVIDIA RTX 3090 (24GB显存 × 2)

**结论：硬件资源完全满足BERT推理开发需求**

---

## 📋 Alpha 0.1 开发路线图

### 阶段1：完善基础架构（1-2天）

#### 1.1 完整的Transformer Block
```cpp
class TransformerBlock {
public:
    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ffn_dim);
    Matrix forward(const Matrix& input);

private:
    LayerNorm ln1_;
    SelfAttention attention_;
    LayerNorm ln2_;
    FeedForward ffn_;
};
```

**功能：**
- 组合现有组件
- 实现残差连接
- 支持批处理

#### 1.2 位置编码（Positional Encoding）
```cpp
class PositionalEncoding {
public:
    PositionalEncoding(size_t max_seq_len, size_t embed_dim);
    Matrix encode(const Matrix& input);

private:
    Matrix pos_encoding_;  // 预计算的位置编码
};
```

**实现：**
- 正弦/余弦位置编码
- 可学习的位置嵌入（BERT使用）

#### 1.3 Embedding层
```cpp
class Embedding {
public:
    Embedding(size_t vocab_size, size_t embed_dim);
    Matrix forward(const std::vector<int>& token_ids);

private:
    Matrix weight_;  // [vocab_size, embed_dim]
};
```

---

### 阶段2：BERT架构实现（2-3天）

#### 2.1 BERT Encoder
```cpp
class BertEncoder {
public:
    BertEncoder(size_t num_layers, size_t embed_dim,
                size_t num_heads, size_t ffn_dim);
    Matrix forward(const Matrix& input);

private:
    std::vector<TransformerBlock> layers_;
    LayerNorm final_ln_;
};
```

#### 2.2 完整BERT模型
```cpp
class BertModel {
public:
    BertModel(const BertConfig& config);

    // 主要推理接口
    Matrix forward(const std::vector<int>& input_ids,
                   const std::vector<int>& token_type_ids = {},
                   const std::vector<int>& attention_mask = {});

    // 获取特征
    Matrix get_pooled_output();  // [CLS] token
    Matrix get_sequence_output(); // 所有token

private:
    Embedding token_embedding_;
    Embedding position_embedding_;
    Embedding token_type_embedding_;
    BertEncoder encoder_;
    Linear pooler_;  // [CLS]的池化层
};
```

#### 2.3 配置管理
```cpp
struct BertConfig {
    size_t vocab_size = 30522;      // 词表大小
    size_t hidden_size = 768;        // 隐藏层维度
    size_t num_layers = 12;          // Transformer层数
    size_t num_heads = 12;           // 注意力头数
    size_t ffn_dim = 3072;           // FFN中间层维度
    size_t max_position = 512;       // 最大序列长度
    size_t type_vocab_size = 2;      // token type数量
    float layer_norm_eps = 1e-12;

    // 从JSON加载
    static BertConfig from_json(const std::string& config_path);
};
```

---

### 阶段3：权重加载（3-4天）

#### 3.1 支持的格式
- **优先级1**: PyTorch `.bin`文件（使用cnpy库）
- **优先级2**: ONNX格式
- **优先级3**: 自定义二进制格式

#### 3.2 权重加载器
```cpp
class WeightLoader {
public:
    // 从PyTorch checkpoint加载
    static bool load_pytorch_weights(
        BertModel& model,
        const std::string& checkpoint_path
    );

    // 从Hugging Face加载
    static bool load_from_pretrained(
        BertModel& model,
        const std::string& model_name  // e.g., "bert-base-uncased"
    );

private:
    static Matrix load_tensor(const std::string& key);
};
```

#### 3.3 依赖库
```cmake
# CMakeLists.txt 添加
find_package(cnpy REQUIRED)  # NumPy文件读取
# 或者手动实现简单的.bin读取器
```

---

### 阶段4：Tokenizer实现（2-3天）

#### 4.1 WordPiece Tokenizer
```cpp
class WordPieceTokenizer {
public:
    WordPieceTokenizer(const std::string& vocab_path);

    // 分词
    std::vector<std::string> tokenize(const std::string& text);

    // 转换为ID
    std::vector<int> encode(const std::string& text);

    // ID转回文本
    std::string decode(const std::vector<int>& ids);

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;

    std::string unk_token_ = "[UNK]";
    std::string cls_token_ = "[CLS]";
    std::string sep_token_ = "[SEP]";
    std::string pad_token_ = "[PAD]";
    std::string mask_token_ = "[MASK]";
};
```

#### 4.2 文本预处理
```cpp
class TextProcessor {
public:
    // 基础清理
    static std::string clean_text(const std::string& text);

    // 转小写
    static std::string lowercase(const std::string& text);

    // 分词前处理
    static std::vector<std::string> whitespace_tokenize(
        const std::string& text
    );
};
```

---

### 阶段5：推理接口和优化（2-3天）

#### 5.1 高层推理API
```cpp
class BertInference {
public:
    BertInference(const std::string& model_path,
                  const std::string& vocab_path);

    // 句子特征提取
    Matrix encode_sentence(const std::string& sentence);

    // 句子对分类（如文本蕴含）
    float predict_pair(const std::string& sent_a,
                       const std::string& sent_b);

    // 批量推理
    std::vector<Matrix> encode_batch(
        const std::vector<std::string>& sentences
    );

private:
    BertModel model_;
    WordPieceTokenizer tokenizer_;
};
```

#### 5.2 性能优化
- **KV Cache**: 减少重复计算
- **批处理优化**: 优化batch推理
- **矩阵乘法优化**:
  - 考虑使用OpenBLAS/MKL
  - 或手工SIMD优化
- **层融合**: LayerNorm + Add等

#### 5.3 GPU支持（可选，使用CUDA）
```cpp
#ifdef USE_CUDA
class CudaMatrix : public Matrix {
    // GPU加速的矩阵运算
};
#endif
```

---

## 🔧 技术选型

### 核心依赖
```
必需：
- C++17编译器
- 无外部深度学习框架依赖

推荐：
- cnpy (NumPy文件读取)
- nlohmann/json (配置文件)
- OpenBLAS/MKL (矩阵加速，可选)
```

### 文件格式支持
1. **模型配置**: JSON格式
   ```json
   {
     "vocab_size": 30522,
     "hidden_size": 768,
     "num_hidden_layers": 12,
     "num_attention_heads": 12,
     ...
   }
   ```

2. **权重文件**: PyTorch `.bin` 或自定义二进制
3. **词表文件**: `vocab.txt` (每行一个token)

---

## 📊 测试计划

### 单元测试
- [ ] TransformerBlock 正确性
- [ ] PositionalEncoding 输出验证
- [ ] Embedding层功能
- [ ] Tokenizer正确性

### 集成测试
- [ ] 加载真实BERT权重
- [ ] 与Hugging Face对比输出
- [ ] 性能基准测试

### 端到端测试
- [ ] 文本分类任务
- [ ] 语义相似度计算
- [ ] 批量推理

---

## 📈 性能目标

### CPU推理（单线程）
- BERT-base: ~50-100 ms/句 (seq_len=128)
- BERT-large: ~150-300 ms/句

### 优化后目标
- BERT-base: ~20-40 ms/句 (使用BLAS)
- 支持批处理: 提升5-10x吞吐量

### GPU推理（如实现）
- BERT-base: ~5-10 ms/句
- RTX 3090单卡支持大批量推理

---

## 🎯 里程碑

### Week 1: 基础架构完善
- [x] v0.1 基础组件（已完成）
- [ ] TransformerBlock
- [ ] PositionalEncoding
- [ ] Embedding层

### Week 2: BERT模型实现
- [ ] BertEncoder
- [ ] 完整BertModel
- [ ] 配置管理

### Week 3: 权重加载
- [ ] 权重文件解析
- [ ] 从PyTorch加载
- [ ] 验证正确性

### Week 4: Tokenizer和推理
- [ ] WordPiece tokenizer
- [ ] 高层API
- [ ] 示例应用

---

## 📝 示例应用

### 示例1: 句子编码
```cpp
#include "llm_engine/bert.h"

int main() {
    // 加载模型
    BertInference bert("bert-base-uncased.bin", "vocab.txt");

    // 编码句子
    std::string text = "Hello, this is a test sentence.";
    Matrix embedding = bert.encode_sentence(text);

    // 输出特征
    std::cout << "Sentence embedding shape: "
              << embedding.rows() << " x " << embedding.cols()
              << std::endl;

    return 0;
}
```

### 示例2: 语义相似度
```cpp
// 计算两个句子的相似度
float similarity = bert.compute_similarity(
    "The cat sits on the mat.",
    "A cat is sitting on a rug."
);

std::cout << "Similarity: " << similarity << std::endl;
```

### 示例3: 文本分类
```cpp
// 加载微调后的分类器
BertClassifier classifier("sentiment_model.bin");

std::string review = "This movie is amazing!";
int label = classifier.predict(review);  // 0: negative, 1: positive

std::cout << "Sentiment: " << (label ? "Positive" : "Negative")
          << std::endl;
```

---

## 🚀 下一步行动

### 立即开始
1. 切换到alpha分支
2. 创建alpha-0.1开发目录结构
3. 实现TransformerBlock

### 优先级排序
**P0 (必须)**:
- TransformerBlock
- PositionalEncoding
- Embedding
- BertModel基础结构

**P1 (重要)**:
- 权重加载
- Tokenizer
- 推理API

**P2 (优化)**:
- 性能优化
- GPU支持
- 更多示例

---

## 📚 参考资料

### 必读论文
1. **BERT: Pre-training of Deep Bidirectional Transformers**
   - https://arxiv.org/abs/1810.04805

2. **Attention Is All You Need** (已读)
   - https://arxiv.org/abs/1706.03762

### 实现参考
1. **Hugging Face Transformers**
   - https://github.com/huggingface/transformers
   - 主要参考其BERT实现

2. **BERT官方实现**
   - https://github.com/google-research/bert

3. **bert.cpp** (C++实现参考)
   - https://github.com/skeskinen/bert.cpp

### 工具和库
1. **cnpy** - NumPy文件读取
   - https://github.com/rogersce/cnpy

2. **nlohmann/json** - JSON解析
   - https://github.com/nlohmann/json

3. **OpenBLAS** - 矩阵加速
   - https://www.openblas.net/

---

## ⚠️ 风险和挑战

### 技术挑战
1. **权重加载复杂性**
   - 解决方案: 先支持简单格式，逐步扩展

2. **Tokenizer实现**
   - 解决方案: 参考Hugging Face实现，先做简化版

3. **性能优化**
   - 解决方案: 先保证正确性，再优化性能

### 时间估算
- **保守估算**: 3-4周
- **理想情况**: 2-3周
- **最小可用版本**: 1-2周

---

## ✅ 准备就绪

**硬件**: ✅ 优秀
**基础代码**: ✅ v0.1已完成
**文档**: ✅ 完善
**规划**: ✅ 清晰

**可以开始alpha-0.1开发！** 🚀
