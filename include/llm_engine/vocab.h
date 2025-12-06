#ifndef LLM_ENGINE_VOCAB_H
#define LLM_ENGINE_VOCAB_H

#include <string>
#include <vector>
#include <map>

namespace llm {

/**
 * @brief Vocab - BERT词表
 *
 * 负责管理token <-> ID的映射关系
 *
 * BERT词表特点：
 * 1. 30522个tokens（bert-base-uncased）
 * 2. 包含特殊token：[PAD], [CLS], [SEP], [UNK], [MASK]
 * 3. 包含完整单词和子词片段（##开头）
 * 4. 小写处理（uncased版本）
 *
 * 词表格式（vocab.txt）：
 *   每行一个token，行号就是token ID（从0开始）
 *
 * 示例：
 *   Line 0:    [PAD]      → ID 0
 *   Line 100:  [UNK]      → ID 100
 *   Line 101:  [CLS]      → ID 101
 *   Line 102:  [SEP]      → ID 102
 *   Line 2088: world      → ID 2088
 *   Line 2075: ##ing      → ID 2075
 */
class Vocab {
public:
    /**
     * @brief 构造函数：从文件加载词表
     *
     * @param vocab_file 词表文件路径（如 vocab.txt）
     *
     * 加载过程：
     * 1. 读取每一行作为token
     * 2. 行号（从0开始）作为token ID
     * 3. 构建双向映射：token→ID 和 ID→token
     * 4. 查找特殊token的ID
     */
    explicit Vocab(const std::string& vocab_file);

    /**
     * @brief Token转ID
     *
     * @param token Token字符串
     * @return Token ID，如果不存在返回unk_id_
     */
    int token_to_id(const std::string& token) const;

    /**
     * @brief ID转Token
     *
     * @param id Token ID
     * @return Token字符串
     * @throws std::out_of_range 如果ID超出范围
     */
    std::string id_to_token(int id) const;

    /**
     * @brief 检查token是否在词表中
     */
    bool has_token(const std::string& token) const;

    /**
     * @brief 获取词表大小
     */
    size_t size() const { return id_to_token_.size(); }

    /**
     * @brief 获取特殊token的ID
     */
    int pad_token_id() const { return pad_id_; }
    int unk_token_id() const { return unk_id_; }
    int cls_token_id() const { return cls_id_; }
    int sep_token_id() const { return sep_id_; }
    int mask_token_id() const { return mask_id_; }

    /**
     * @brief 获取特殊token的字符串
     */
    std::string pad_token() const { return "[PAD]"; }
    std::string unk_token() const { return "[UNK]"; }
    std::string cls_token() const { return "[CLS]"; }
    std::string sep_token() const { return "[SEP]"; }
    std::string mask_token() const { return "[MASK]"; }

private:
    /**
     * Token → ID 映射
     *
     * 用于encode：快速查找token对应的ID
     * 例如：token_to_id_["hello"] = 7592
     */
    std::map<std::string, int> token_to_id_;

    /**
     * ID → Token 映射
     *
     * 用于decode：快速查找ID对应的token
     * 例如：id_to_token_[7592] = "hello"
     */
    std::vector<std::string> id_to_token_;

    /**
     * 特殊token的ID
     *
     * BERT特殊token及其用途：
     * - [PAD] (0):   填充，用于对齐批次中的序列长度
     * - [UNK] (100): 未知词，不在词表中的词用这个代替
     * - [CLS] (101): 分类标记，句子开始，用于分类任务
     * - [SEP] (102): 分隔符，句子结束或句子之间的分隔
     * - [MASK] (103): 掩码，用于MLM预训练任务
     */
    int pad_id_;
    int unk_id_;
    int cls_id_;
    int sep_id_;
    int mask_id_;

    /**
     * @brief 查找特殊token的ID
     *
     * 在词表加载后调用，找到特殊token的位置
     */
    void find_special_tokens();
};

} // namespace llm

#endif // LLM_ENGINE_VOCAB_H
