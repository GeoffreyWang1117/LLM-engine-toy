#ifndef LLM_ENGINE_BERT_TOKENIZER_H
#define LLM_ENGINE_BERT_TOKENIZER_H

#include "vocab.h"
#include <string>
#include <vector>

namespace llm {

/**
 * @brief BertTokenizer - 完整的BERT分词器
 *
 * 实现WordPiece算法，将文本转换为token IDs
 *
 * 分词流程：
 * 1. 基础分词（BasicTokenizer）：
 *    - 小写转换（uncased模型）
 *    - 标点符号分割
 *    - 空格分割
 *
 * 2. WordPiece分词：
 *    - 贪心最长匹配
 *    - 子词分割（##前缀）
 *
 * 3. 转换为IDs：
 *    - 添加特殊token（[CLS], [SEP]）
 *    - 查询词表
 *
 * 示例：
 *   文本: "Hello, world!"
 *   ↓ lowercase
 *   "hello, world!"
 *   ↓ split
 *   ["hello", ",", "world", "!"]
 *   ↓ wordpiece（已在词表）
 *   ["hello", ",", "world", "!"]
 *   ↓ add special tokens
 *   ["[CLS]", "hello", ",", "world", "!", "[SEP]"]
 *   ↓ to ids
 *   [101, 7592, 1010, 2088, 999, 102]
 */
class BertTokenizer {
public:
    /**
     * @brief 构造函数
     *
     * @param vocab_file 词表文件路径
     * @param do_lower_case 是否转换为小写（uncased模型使用true）
     */
    explicit BertTokenizer(const std::string& vocab_file,
                          bool do_lower_case = true);

    /**
     * @brief 编码单个文本
     *
     * @param text 输入文本
     * @param add_special_tokens 是否添加[CLS]和[SEP]
     * @return Token IDs
     */
    std::vector<int> encode(const std::string& text,
                           bool add_special_tokens = true);

    /**
     * @brief 编码文本对（用于句子对任务）
     *
     * @param text_a 第一个文本
     * @param text_b 第二个文本
     * @param add_special_tokens 是否添加特殊token
     * @return Token IDs
     *
     * 格式: [CLS] text_a [SEP] text_b [SEP]
     */
    std::vector<int> encode_pair(const std::string& text_a,
                                const std::string& text_b,
                                bool add_special_tokens = true);

    /**
     * @brief 解码Token IDs为文本
     *
     * @param token_ids Token ID序列
     * @param skip_special_tokens 是否跳过特殊token
     * @return 解码后的文本
     */
    std::string decode(const std::vector<int>& token_ids,
                      bool skip_special_tokens = true);

    /**
     * @brief 分词（返回tokens，不转ID）
     *
     * @param text 输入文本
     * @return Tokens列表
     */
    std::vector<std::string> tokenize(const std::string& text);

    /**
     * @brief 获取词表大小
     */
    size_t vocab_size() const { return vocab_.size(); }

    /**
     * @brief 获取词表
     */
    const Vocab& vocab() const { return vocab_; }

private:
    Vocab vocab_;             // 词表
    bool do_lower_case_;      // 是否小写化
    size_t max_word_len_;     // 单词最大长度（用于WordPiece）

    /**
     * 基础分词：将文本分割为单词和标点
     */
    std::vector<std::string> basic_tokenize(const std::string& text);

    /**
     * WordPiece分词：将单词分割为子词
     */
    std::vector<std::string> wordpiece_tokenize(const std::vector<std::string>& tokens);

    /**
     * 对单个单词进行WordPiece分词
     */
    std::vector<std::string> wordpiece_tokenize_word(const std::string& word);

    /**
     * 文本清理和小写化
     */
    std::string clean_text(const std::string& text);

    /**
     * 判断是否是标点符号
     */
    bool is_punctuation(char c);

    /**
     * 判断是否是空白字符
     */
    bool is_whitespace(char c);
};

} // namespace llm

#endif // LLM_ENGINE_BERT_TOKENIZER_H
