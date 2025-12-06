#include "llm_engine/bert_tokenizer.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>

namespace llm {

/**
 * 构造函数
 */
BertTokenizer::BertTokenizer(const std::string& vocab_file,
                             bool do_lower_case)
    : vocab_(vocab_file),
      do_lower_case_(do_lower_case),
      max_word_len_(100)
{
    std::cout << "BertTokenizer initialized" << std::endl;
    std::cout << "  do_lower_case: " << (do_lower_case_ ? "true" : "false") << std::endl;
}

/**
 * 编码单个文本
 */
std::vector<int> BertTokenizer::encode(const std::string& text,
                                      bool add_special_tokens) {
    // 分词
    std::vector<std::string> tokens = tokenize(text);

    // 转换为IDs
    std::vector<int> ids;

    if (add_special_tokens) {
        ids.push_back(vocab_.cls_token_id());
    }

    for (const auto& token : tokens) {
        ids.push_back(vocab_.token_to_id(token));
    }

    if (add_special_tokens) {
        ids.push_back(vocab_.sep_token_id());
    }

    return ids;
}

/**
 * 编码文本对
 */
std::vector<int> BertTokenizer::encode_pair(const std::string& text_a,
                                           const std::string& text_b,
                                           bool add_special_tokens) {
    std::vector<std::string> tokens_a = tokenize(text_a);
    std::vector<std::string> tokens_b = tokenize(text_b);

    std::vector<int> ids;

    if (add_special_tokens) {
        ids.push_back(vocab_.cls_token_id());
    }

    for (const auto& token : tokens_a) {
        ids.push_back(vocab_.token_to_id(token));
    }

    if (add_special_tokens) {
        ids.push_back(vocab_.sep_token_id());
    }

    for (const auto& token : tokens_b) {
        ids.push_back(vocab_.token_to_id(token));
    }

    if (add_special_tokens) {
        ids.push_back(vocab_.sep_token_id());
    }

    return ids;
}

/**
 * 解码
 */
std::string BertTokenizer::decode(const std::vector<int>& token_ids,
                                 bool skip_special_tokens) {
    std::string result;

    for (int id : token_ids) {
        std::string token = vocab_.id_to_token(id);

        // 跳过特殊token
        if (skip_special_tokens) {
            if (token == "[CLS]" || token == "[SEP]" || token == "[PAD]") {
                continue;
            }
        }

        // 处理子词（##开头）
        if (token.length() >= 2 && token.substr(0, 2) == "##") {
            // 子词直接拼接，不加空格
            result += token.substr(2);
        } else {
            // 普通token前加空格（除了第一个）
            if (!result.empty()) {
                result += " ";
            }
            result += token;
        }
    }

    return result;
}

/**
 * 分词
 */
std::vector<std::string> BertTokenizer::tokenize(const std::string& text) {
    // 步骤1: 基础分词
    std::vector<std::string> tokens = basic_tokenize(text);

    // 步骤2: WordPiece分词
    tokens = wordpiece_tokenize(tokens);

    return tokens;
}

/**
 * 基础分词
 */
std::vector<std::string> BertTokenizer::basic_tokenize(const std::string& text) {
    // 清理和小写化
    std::string cleaned = clean_text(text);

    std::vector<std::string> tokens;
    std::string current_token;

    for (size_t i = 0; i < cleaned.length(); ++i) {
        char c = cleaned[i];

        if (is_whitespace(c)) {
            // 空格：结束当前token
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else if (is_punctuation(c)) {
            // 标点：结束当前token并将标点作为独立token
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(std::string(1, c));
        } else {
            // 普通字符：添加到当前token
            current_token += c;
        }
    }

    // 添加最后一个token
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    return tokens;
}

/**
 * WordPiece分词
 */
std::vector<std::string> BertTokenizer::wordpiece_tokenize(
    const std::vector<std::string>& tokens) {

    std::vector<std::string> output;

    for (const auto& token : tokens) {
        // 对每个token进行WordPiece分词
        std::vector<std::string> sub_tokens = wordpiece_tokenize_word(token);
        output.insert(output.end(), sub_tokens.begin(), sub_tokens.end());
    }

    return output;
}

/**
 * 对单个单词进行WordPiece分词
 */
std::vector<std::string> BertTokenizer::wordpiece_tokenize_word(
    const std::string& word) {

    // 如果单词太长，直接返回[UNK]
    if (word.length() > max_word_len_) {
        return {vocab_.unk_token()};
    }

    std::vector<std::string> output;
    size_t start = 0;

    while (start < word.length()) {
        size_t end = word.length();
        std::string cur_substr;
        bool found = false;

        // 贪心匹配：从最长子串开始尝试
        while (start < end) {
            std::string substr = word.substr(start, end - start);

            // 非首个子词需要加##前缀
            if (start > 0) {
                substr = "##" + substr;
            }

            // 检查是否在词表中
            if (vocab_.has_token(substr)) {
                cur_substr = substr;
                found = true;
                break;
            }

            end--;
        }

        // 如果没找到匹配，整个单词标记为[UNK]
        if (!found) {
            return {vocab_.unk_token()};
        }

        output.push_back(cur_substr);
        start = end;
    }

    return output;
}

/**
 * 清理文本
 */
std::string BertTokenizer::clean_text(const std::string& text) {
    std::string result;

    for (char c : text) {
        // 移除控制字符
        if (c == 0 || c == 0xfffd) {
            continue;
        }

        // 转换为小写（如果需要）
        if (do_lower_case_) {
            c = std::tolower(static_cast<unsigned char>(c));
        }

        result += c;
    }

    return result;
}

/**
 * 判断是否是标点符号
 */
bool BertTokenizer::is_punctuation(char c) {
    /*
     * 简化版：只处理ASCII标点
     *
     * 完整版应该处理Unicode标点
     */
    return (c >= 33 && c <= 47) ||   // ! " # $ % & ' ( ) * + , - . /
           (c >= 58 && c <= 64) ||   // : ; < = > ? @
           (c >= 91 && c <= 96) ||   // [ \ ] ^ _ `
           (c >= 123 && c <= 126);   // { | } ~
}

/**
 * 判断是否是空白字符
 */
bool BertTokenizer::is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

} // namespace llm
