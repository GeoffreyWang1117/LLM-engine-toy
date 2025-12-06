#include "llm_engine/vocab.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace llm {

/**
 * 构造函数：加载词表
 */
Vocab::Vocab(const std::string& vocab_file)
    : pad_id_(-1), unk_id_(-1), cls_id_(-1), sep_id_(-1), mask_id_(-1)
{
    std::cout << "Loading vocabulary from: " << vocab_file << std::endl;

    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_file);
    }

    /*
     * 加载词表
     *
     * 格式：每行一个token
     * 行号（从0开始）= token ID
     */
    std::string line;
    int id = 0;

    while (std::getline(file, line)) {
        // 移除行尾的空白字符（包括\r\n）
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                 line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }

        // 跳过空行
        if (line.empty()) {
            continue;
        }

        // 添加到映射
        token_to_id_[line] = id;
        id_to_token_.push_back(line);

        id++;
    }

    file.close();

    std::cout << "  Loaded " << id_to_token_.size() << " tokens" << std::endl;

    // 查找特殊token
    find_special_tokens();

    std::cout << "  Special tokens:" << std::endl;
    std::cout << "    [PAD]:  " << pad_id_ << std::endl;
    std::cout << "    [UNK]:  " << unk_id_ << std::endl;
    std::cout << "    [CLS]:  " << cls_id_ << std::endl;
    std::cout << "    [SEP]:  " << sep_id_ << std::endl;
    std::cout << "    [MASK]: " << mask_id_ << std::endl;

    std::cout << "✅ Vocabulary loaded successfully" << std::endl;
}

/**
 * 查找特殊token的ID
 */
void Vocab::find_special_tokens() {
    /*
     * 在BERT词表中查找特殊token
     *
     * 标准位置（bert-base-uncased）：
     * - [PAD]:  0
     * - [UNK]:  100
     * - [CLS]:  101
     * - [SEP]:  102
     * - [MASK]: 103
     *
     * 但我们通过查找来确保兼容性
     */

    auto find_token = [this](const std::string& token) -> int {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        return -1;
    };

    pad_id_ = find_token("[PAD]");
    unk_id_ = find_token("[UNK]");
    cls_id_ = find_token("[CLS]");
    sep_id_ = find_token("[SEP]");
    mask_id_ = find_token("[MASK]");

    // 验证必需的特殊token存在
    if (unk_id_ == -1) {
        throw std::runtime_error("Vocabulary missing required token: [UNK]");
    }
    if (cls_id_ == -1) {
        throw std::runtime_error("Vocabulary missing required token: [CLS]");
    }
    if (sep_id_ == -1) {
        throw std::runtime_error("Vocabulary missing required token: [SEP]");
    }
}

/**
 * Token转ID
 */
int Vocab::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }

    // 如果token不在词表中，返回[UNK]的ID
    return unk_id_;
}

/**
 * ID转Token
 */
std::string Vocab::id_to_token(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
        throw std::out_of_range(
            "Token ID " + std::to_string(id) +
            " out of range [0, " + std::to_string(id_to_token_.size()) + ")"
        );
    }

    return id_to_token_[id];
}

/**
 * 检查token是否存在
 */
bool Vocab::has_token(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

} // namespace llm
