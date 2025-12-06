# Alpha 0.1 阶段3开发计划：权重加载

## 目标
从Hugging Face加载预训练的BERT权重到我们的C++推理引擎中

## 技术路线选择

### 方案1：使用safetensors格式（推荐）
**优点：**
- 更安全（不执行任意代码）
- 更快（零拷贝加载）
- 格式简单（JSON header + 原始二进制数据）
- Hugging Face官方推荐

**缺点：**
- 需要实现safetensors解析器

### 方案2：使用PyTorch格式（.bin）
**优点：**
- 最常见的格式

**缺点：**
- 需要PyTorch库或复杂的pickle解析
- 安全性较低

### 方案3：导出为numpy格式
**优点：**
- 格式简单（.npy）

**缺点：**
- 需要额外转换步骤

**决定：使用safetensors格式**

## 阶段3详细任务

### 第1步：准备工作
- [ ] 创建`weights/`目录存放模型权重
- [ ] 从Hugging Face下载bert-base-uncased（safetensors格式）
- [ ] 研究safetensors文件格式
- [ ] 列出BERT所有权重的名称和形状

### 第2步：实现safetensors解析器
- [ ] 创建`include/llm_engine/safetensors.h`
- [ ] 实现JSON header解析
- [ ] 实现二进制数据读取
- [ ] 验证能正确读取tensor数据

### 第3步：权重映射
理解Hugging Face BERT权重命名 → 我们的C++类：

```
Hugging Face命名                     → 我们的结构
=====================================
embeddings.word_embeddings.weight    → BertModel::token_embeddings_
embeddings.position_embeddings.weight→ BertModel::position_embeddings_
embeddings.token_type_embeddings.weight→ BertModel::token_type_embeddings_
embeddings.LayerNorm.weight          → BertModel::embeddings_layer_norm_ (gamma)
embeddings.LayerNorm.bias            → BertModel::embeddings_layer_norm_ (beta)

encoder.layer.0.attention.self.query.weight → BertEncoder::layers_[0].attention.q_proj
encoder.layer.0.attention.self.query.bias
encoder.layer.0.attention.self.key.weight
encoder.layer.0.attention.self.key.bias
encoder.layer.0.attention.self.value.weight
encoder.layer.0.attention.self.value.bias
encoder.layer.0.attention.output.dense.weight → BertEncoder::layers_[0].attention.o_proj
encoder.layer.0.attention.output.dense.bias
encoder.layer.0.attention.output.LayerNorm.weight
encoder.layer.0.attention.output.LayerNorm.bias

encoder.layer.0.intermediate.dense.weight → BertEncoder::layers_[0].ffn.fc1
encoder.layer.0.intermediate.dense.bias
encoder.layer.0.output.dense.weight → BertEncoder::layers_[0].ffn.fc2
encoder.layer.0.output.dense.bias
encoder.layer.0.output.LayerNorm.weight
encoder.layer.0.output.LayerNorm.bias

... (重复11次，layer.0到layer.11)

pooler.dense.weight                  → BertModel::pooler_dense_
pooler.dense.bias
```

### 第4步：扩展现有类以支持权重加载

需要为以下类添加`load_weights()`方法：
- [ ] Matrix - 添加从原始数据创建的方法
- [ ] Linear - 添加加载weight和bias的方法
- [ ] LayerNorm - 添加加载gamma和beta的方法
- [ ] Embedding - 添加加载embedding矩阵的方法
- [ ] SelfAttention - 添加加载Q/K/V/O权重的方法
- [ ] FeedForward - 添加加载FC1/FC2权重的方法
- [ ] TransformerBlock - 协调加载所有子模块
- [ ] BertEncoder - 加载所有层
- [ ] BertModel - 主加载接口

### 第5步：实现WeightLoader类
创建`include/llm_engine/weight_loader.h`：
```cpp
class WeightLoader {
public:
    // 从safetensors文件加载
    explicit WeightLoader(const std::string& model_path);

    // 加载到BertModel
    void load_to_model(BertModel& model);

    // 获取单个tensor
    Matrix get_tensor(const std::string& name);

private:
    std::map<std::string, TensorInfo> tensors_;
    std::vector<uint8_t> data_;
};
```

### 第6步：验证正确性
- [ ] 使用PyTorch和我们的引擎对同一输入计算输出
- [ ] 对比输出差异（应该在1e-5以内）
- [ ] 创建验证脚本

### 第7步：文档和示例
- [ ] 创建权重加载教程
- [ ] 更新README
- [ ] 创建示例程序

## 文件结构

```
LLM-engine/
├── weights/                          # 新增：存放模型权重
│   ├── bert-base-uncased/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── README.md
│   └── download_weights.py          # 下载脚本
├── include/llm_engine/
│   ├── safetensors.h                # 新增：safetensors解析器
│   └── weight_loader.h              # 新增：权重加载器
├── src/
│   ├── safetensors.cpp
│   └── weight_loader.cpp
└── examples/
    └── tutorial_weight_loading.cpp   # 新增：权重加载演示
```

## 预计时间
- 第1-2步（准备+解析器）：1-2天
- 第3-4步（映射+扩展类）：1-2天
- 第5-6步（实现+验证）：1天
- 第7步（文档）：半天
- **总计：3-5天**

## 依赖库
考虑是否需要：
- JSON解析：可以使用nlohmann/json或手写简单解析器
- 文件I/O：标准库即可

## 下一步
先实现safetensors解析器的原型，验证能正确读取权重数据
