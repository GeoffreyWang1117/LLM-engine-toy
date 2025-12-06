#include "llm_engine/bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <limits>

namespace llm {

BPETokenizer::BPETokenizer() {
    // GPT-2的预分词正则表达式
    // 简化版本：匹配缩写、单词、数字、标点、空格
    // 注意：C++的std::regex不支持\p{L}等Unicode属性，这里使用简化版本
    pat_ = std::regex(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s]+|\s+)"
    );

    init_byte_encoder();
}

void BPETokenizer::init_byte_encoder() {
    // GPT-2的字节编码：将256个字节映射到Unicode
    // 避免使用控制字符和空格，使用可打印字符范围

    std::vector<int> bs;

    // 可打印ASCII: 33-126 (!'到~)
    for (int i = 33; i < 127; ++i) {
        bs.push_back(i);
    }
    // 扩展ASCII: 161-172, 174-255
    for (int i = 161; i < 173; ++i) {
        bs.push_back(i);
    }
    for (int i = 174; i < 256; ++i) {
        bs.push_back(i);
    }

    std::vector<int> cs = bs;
    int n = 0;

    // 为剩余的字节分配未使用的Unicode点
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // 创建映射
    for (size_t i = 0; i < bs.size(); ++i) {
        // 字节 -> Unicode字符
        std::string ch;
        ch += static_cast<char>(cs[i]);
        byte_encoder_[bs[i]] = ch;
        byte_decoder_[ch] = bs[i];
    }
}

std::vector<std::string> BPETokenizer::bytes_encode(const std::string& text) const {
    std::vector<std::string> result;
    for (unsigned char c : text) {
        auto it = byte_encoder_.find(static_cast<int>(c));
        if (it != byte_encoder_.end()) {
            result.push_back(it->second);
        } else {
            throw std::runtime_error("Byte not found in encoder");
        }
    }
    return result;
}

std::string BPETokenizer::bytes_decode(const std::vector<std::string>& tokens) const {
    std::string result;
    for (const auto& token : tokens) {
        auto it = byte_decoder_.find(token);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        }
    }
    return result;
}

std::vector<std::pair<std::string, std::string>> BPETokenizer::get_pairs(
    const std::vector<std::string>& word) const {

    std::vector<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) {
        return pairs;
    }

    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.push_back({word[i], word[i + 1]});
    }
    return pairs;
}

std::vector<std::string> BPETokenizer::bpe(const std::string& token) const {
    // 将token分解为字符
    std::vector<std::string> word;
    for (char c : token) {
        word.push_back(std::string(1, c));
    }

    if (word.size() <= 1) {
        return word;
    }

    // 迭代合并
    while (true) {
        auto pairs = get_pairs(word);
        if (pairs.empty()) {
            break;
        }

        // 找到优先级最高的pair
        std::pair<std::string, std::string> bigram = {"", ""};
        int min_rank = std::numeric_limits<int>::max();

        for (const auto& pair : pairs) {
            auto it = merges_.find(pair);
            int rank = (it != merges_.end()) ? it->second : std::numeric_limits<int>::max();

            if (rank < min_rank) {
                min_rank = rank;
                bigram = pair;
            }
        }

        // 如果找不到可合并的pair，退出
        if (merges_.find(bigram) == merges_.end()) {
            break;
        }

        // 合并bigram
        std::string first = bigram.first;
        std::string second = bigram.second;
        std::vector<std::string> new_word;

        size_t i = 0;
        while (i < word.size()) {
            // 查找first
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                // 剩余部分直接添加
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }

            // 添加之前的部分
            new_word.insert(new_word.end(), word.begin() + i, it);
            i = it - word.begin();

            // 检查是否可以合并
            if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;

        if (word.size() == 1) {
            break;
        }
    }

    return word;
}

void BPETokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open vocab file: " + vocab_path);
    }

    // 读取整个文件
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    // 简单的JSON解析（假设格式为 {"token": id, ...}）
    // 注意：这是一个简化版本，生产环境应使用专门的JSON库

    size_t pos = 0;
    while (pos < content.size()) {
        // 找到 "
        pos = content.find('"', pos);
        if (pos == std::string::npos) break;
        pos++; // 跳过开头的 "

        // 找到token结束的 "
        size_t end_pos = pos;
        while (end_pos < content.size()) {
            if (content[end_pos] == '"' && (end_pos == 0 || content[end_pos - 1] != '\\')) {
                break;
            }
            end_pos++;
        }
        if (end_pos >= content.size()) break;

        std::string token = content.substr(pos, end_pos - pos);
        pos = end_pos + 1;

        // 找到 :
        pos = content.find(':', pos);
        if (pos == std::string::npos) break;
        pos++;

        // 跳过空格
        while (pos < content.size() && std::isspace(content[pos])) {
            pos++;
        }

        // 提取数字
        size_t num_start = pos;
        if (pos < content.size() && content[pos] == '-') {
            pos++;
        }
        while (pos < content.size() && std::isdigit(content[pos])) {
            pos++;
        }

        if (pos > num_start) {
            std::string num_str = content.substr(num_start, pos - num_start);
            int id = std::stoi(num_str);

            // 存储
            vocab_[token] = id;
            reverse_vocab_[id] = token;

            // 检查特殊token
            if (token == "<|endoftext|>") {
                eos_token_id_ = id;
            }
        }
    }
}

void BPETokenizer::load_merges(const std::string& merges_path) {
    std::ifstream file(merges_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open merges file: " + merges_path);
    }

    std::string line;
    int rank = 0;

    // 跳过第一行（通常是 #version）
    std::getline(file, line);

    // 读取每一行合并规则
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string first, second;

        if (iss >> first >> second) {
            merges_[{first, second}] = rank;
            rank++;
        }
    }

    file.close();
}

void BPETokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    load_vocab(vocab_path);
    load_merges(merges_path);
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;

    if (text.empty()) {
        return ids;
    }

    // 简化版本：逐字符编码
    // 对每个字符，查找其对应的token ID
    for (char c : text) {
        std::string token(1, c);

        // 先尝试直接查找
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            // 如果不在vocab中，尝试查找字节编码版本
            auto byte_it = byte_encoder_.find(static_cast<int>(static_cast<unsigned char>(c)));
            if (byte_it != byte_encoder_.end()) {
                std::string byte_token = byte_it->second;
                auto vocab_it = vocab_.find(byte_token);
                if (vocab_it != vocab_.end()) {
                    ids.push_back(vocab_it->second);
                }
            }
        }
    }

    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::vector<std::string> tokens;

    for (int id : ids) {
        auto it = reverse_vocab_.find(id);
        if (it != reverse_vocab_.end()) {
            tokens.push_back(it->second);
        }
    }

    // 拼接tokens
    std::string byte_str;
    for (const auto& token : tokens) {
        byte_str += token;
    }

    // 字节解码
    std::string result;
    for (char c : byte_str) {
        std::string ch(1, c);
        auto it = byte_decoder_.find(ch);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += c;  // 保留原字符
        }
    }

    return result;
}

int BPETokenizer::token_to_id(const std::string& token) const {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        return it->second;
    }
    return -1;
}

std::string BPETokenizer::id_to_token(int id) const {
    auto it = reverse_vocab_.find(id);
    if (it != reverse_vocab_.end()) {
        return it->second;
    }
    return "";
}

} // namespace llm
