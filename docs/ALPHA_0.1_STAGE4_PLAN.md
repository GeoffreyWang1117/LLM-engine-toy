# Alpha 0.1 阶段4开发计划：WordPiece Tokenizer

## 目标
实现WordPiece Tokenizer，支持从原始文本到Token IDs的转换，实现端到端推理

## 当前状态 vs 目标状态

### 当前（阶段3完成后）
```cpp
// 必须手动指定Token IDs
std::vector<int> input_ids = {101, 7592, 2088, 102};  // [CLS] hello world [SEP]
auto output = model.forward(input_ids);
```

### 目标（阶段4完成后）
```cpp
// 直接输入文本
BertTokenizer tokenizer("vocab.txt");
std::string text = "Hello world!";
auto input_ids = tokenizer.encode(text);  // 自动转换
auto output = model.forward(input_ids);
```

## WordPiece算法原理

WordPiece是一种子词（subword）分词算法，被BERT使用。

### 核心思想
1. 词表包含：
   - 完整的单词（如"hello"）
   - 子词片段（如"##ing", "##ed"）
   - 特殊token（[CLS], [SEP], [PAD], [UNK], [MASK]）

2. 分词策略：
   - 优先匹配最长的词
   - 如果找不到完整单词，分解为子词
   - "##"前缀表示这是单词的后续部分

### 示例
```
文本: "playing"

分词过程：
1. 尝试匹配 "playing" → 不在词表
2. 尝试匹配 "play" → 找到！
3. 剩余 "ing"
4. 尝试匹配 "##ing" → 找到！

结果: ["play", "##ing"]
Token IDs: [2377, 2075]
```

### BERT的特殊token
- [CLS] (101): 句子开始标记
- [SEP] (102): 句子分隔符
- [PAD] (0): 填充标记
- [UNK] (100): 未知词
- [MASK] (103): 掩码（用于预训练）

## 阶段4详细任务

### 第1步：研究词表结构
- [ ] 查看vocab.txt格式
- [ ] 理解token到ID的映射
- [ ] 统计词表大小和特殊token位置

### 第2步：实现Vocab类
创建 `include/llm_engine/vocab.h`

```cpp
class Vocab {
public:
    explicit Vocab(const std::string& vocab_file);

    // Token <-> ID 转换
    int token_to_id(const std::string& token) const;
    std::string id_to_token(int id) const;

    // 检查
    bool has_token(const std::string& token) const;
    size_t size() const;

    // 特殊token
    int cls_token_id() const { return cls_id_; }
    int sep_token_id() const { return sep_id_; }
    int pad_token_id() const { return pad_id_; }
    int unk_token_id() const { return unk_id_; }
    int mask_token_id() const { return mask_id_; }

private:
    std::map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    int cls_id_;
    int sep_id_;
    int pad_id_;
    int unk_id_;
    int mask_id_;
};
```

### 第3步：实现BasicTokenizer（文本预处理）
处理：
- 小写转换（uncased模型）
- 标点符号分割
- 空格分割
- 中文字符处理

```cpp
class BasicTokenizer {
public:
    explicit BasicTokenizer(bool do_lower_case = true);

    std::vector<std::string> tokenize(const std::string& text);

private:
    bool do_lower_case_;

    std::string clean_text(const std::string& text);
    std::string lowercase(const std::string& text);
    std::vector<std::string> split_on_punctuation(const std::string& text);
};
```

### 第4步：实现WordPieceTokenizer
核心分词逻辑：

```cpp
class WordPieceTokenizer {
public:
    explicit WordPieceTokenizer(const Vocab& vocab,
                                const std::string& unk_token = "[UNK]",
                                size_t max_input_chars_per_word = 100);

    std::vector<std::string> tokenize(const std::string& text);

private:
    const Vocab& vocab_;
    std::string unk_token_;
    size_t max_input_chars_per_word_;

    std::vector<std::string> tokenize_word(const std::string& word);
};
```

### 第5步：实现BertTokenizer（完整接口）
整合所有组件：

```cpp
class BertTokenizer {
public:
    explicit BertTokenizer(const std::string& vocab_file,
                          bool do_lower_case = true);

    // 核心方法
    std::vector<int> encode(const std::string& text,
                           bool add_special_tokens = true);

    std::vector<int> encode_pair(const std::string& text_a,
                                const std::string& text_b,
                                bool add_special_tokens = true);

    std::string decode(const std::vector<int>& token_ids,
                      bool skip_special_tokens = true);

    // 辅助方法
    size_t vocab_size() const;

private:
    Vocab vocab_;
    BasicTokenizer basic_tokenizer_;
    WordPieceTokenizer wordpiece_tokenizer_;
};
```

### 第6步：创建端到端推理示例
`examples/end_to_end_inference.cpp`

展示完整流程：
1. 加载tokenizer
2. 输入原始文本
3. 分词和编码
4. 模型推理
5. 解码输出

## 实现细节

### WordPiece分词算法伪代码
```python
def tokenize_word(word):
    tokens = []
    start = 0

    while start < len(word):
        end = len(word)
        found = False

        # 从最长子串开始尝试
        while start < end:
            substr = word[start:end]

            # 非第一个子词需要加##前缀
            if start > 0:
                substr = "##" + substr

            if substr in vocab:
                tokens.append(substr)
                found = True
                break

            end -= 1

        if not found:
            return ["[UNK]"]

        start = end

    return tokens
```

### 文本预处理流程
```
原始文本: "Hello, World!"
    ↓ lowercase
"hello, world!"
    ↓ split_on_punctuation
["hello", ",", "world", "!"]
    ↓ wordpiece
["hello", ",", "world", "!"]
    ↓ add_special_tokens
["[CLS]", "hello", ",", "world", "!", "[SEP]"]
    ↓ convert_to_ids
[101, 7592, 1010, 2088, 999, 102]
```

## 测试计划

### 单元测试
1. Vocab加载和查询
2. BasicTokenizer标点符号处理
3. WordPieceTokenizer子词分割
4. 特殊token处理

### 集成测试
1. 简单句子："Hello world"
2. 包含标点："Hello, world!"
3. 未登录词："supercalifragilisticexpialidocious"
4. 句子对："Question? [SEP] Answer."

### 与Hugging Face对比
使用Python验证我们的输出与transformers库一致：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Hello, world!"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print(f"Tokens: {tokens}")
print(f"IDs: {ids}")
```

## 文件结构

```
LLM-engine/
├── include/llm_engine/
│   ├── vocab.h              # 新增：词表
│   ├── basic_tokenizer.h    # 新增：基础分词
│   ├── wordpiece_tokenizer.h # 新增：WordPiece
│   └── bert_tokenizer.h     # 新增：完整Tokenizer
├── src/
│   ├── vocab.cpp
│   ├── basic_tokenizer.cpp
│   ├── wordpiece_tokenizer.cpp
│   └── bert_tokenizer.cpp
└── examples/
    ├── test_tokenizer.cpp   # 单元测试
    └── end_to_end_inference.cpp # 端到端演示
```

## 预计时间
- 第1-2步（Vocab）：半天
- 第3步（BasicTokenizer）：半天
- 第4步（WordPieceTokenizer）：1天
- 第5步（BertTokenizer）：半天
- 第6步（测试和示例）：半天
- **总计：3天**

## 成功标准
1. ✅ 能够加载vocab.txt（30522个tokens）
2. ✅ 正确分词常见句子
3. ✅ 输出与Hugging Face transformers一致
4. ✅ 端到端推理："Hello world" → BERT输出
5. ✅ 处理边界情况（未登录词、标点、大小写）

## 下一步
完成阶段4后，我们将进入阶段5：
- 高层推理API
- 批处理支持
- 实际应用示例（文本分类、语义相似度等）
