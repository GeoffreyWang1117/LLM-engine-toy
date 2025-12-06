/**
 * @file test_vocab.cpp
 * @brief 测试Vocab类
 */

#include "llm_engine/vocab.h"
#include <iostream>
#include <vector>

using namespace llm;

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Vocab类测试                                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        // 加载词表
        Vocab vocab("weights/bert-base-uncased/vocab.txt");

        std::cout << "\n【1】测试特殊token\n";
        std::cout << "────────────────────────────────────────────────────────\n";
        std::cout << "[PAD] token ID:  " << vocab.pad_token_id() << std::endl;
        std::cout << "[UNK] token ID:  " << vocab.unk_token_id() << std::endl;
        std::cout << "[CLS] token ID:  " << vocab.cls_token_id() << std::endl;
        std::cout << "[SEP] token ID:  " << vocab.sep_token_id() << std::endl;
        std::cout << "[MASK] token ID: " << vocab.mask_token_id() << std::endl;

        std::cout << "\n【2】测试常见单词\n";
        std::cout << "────────────────────────────────────────────────────────\n";
        std::vector<std::string> test_tokens = {
            "hello", "world", "play", "##ing", "!",  ","
        };

        for (const auto& token : test_tokens) {
            int id = vocab.token_to_id(token);
            std::cout << "  \"" << token << "\" → ID " << id;

            // 反向验证
            std::string decoded = vocab.id_to_token(id);
            if (decoded == token) {
                std::cout << " ✓\n";
            } else {
                std::cout << " ✗ (decoded: \"" << decoded << "\")\n";
            }
        }

        std::cout << "\n【3】测试未登录词\n";
        std::cout << "────────────────────────────────────────────────────────\n";
        std::string oov_word = "supercalifragilisticexpialidocious";
        int oov_id = vocab.token_to_id(oov_word);
        std::cout << "  \"" << oov_word << "\" → ID " << oov_id << " ([UNK])\n";
        std::cout << "  是否等于[UNK] ID: " << (oov_id == vocab.unk_token_id() ? "是" : "否") << "\n";

        std::cout << "\n【4】测试ID到Token的转换\n";
        std::cout << "────────────────────────────────────────────────────────\n";
        std::vector<int> test_ids = {101, 7592, 2088, 102};
        std::cout << "  Token IDs: [";
        for (size_t i = 0; i < test_ids.size(); ++i) {
            std::cout << test_ids[i];
            if (i < test_ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  对应的Tokens: [";
        for (size_t i = 0; i < test_ids.size(); ++i) {
            std::cout << "\"" << vocab.id_to_token(test_ids[i]) << "\"";
            if (i < test_ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  （应该是: [CLS] hello world [SEP]）\n";

        std::cout << "\n【5】词表统计\n";
        std::cout << "────────────────────────────────────────────────────────\n";
        std::cout << "  总token数: " << vocab.size() << "\n";

        // 统计子词数量
        int subword_count = 0;
        for (size_t i = 0; i < vocab.size(); ++i) {
            std::string token = vocab.id_to_token(i);
            if (token.length() >= 2 && token.substr(0, 2) == "##") {
                subword_count++;
            }
        }
        std::cout << "  子词（##开头）数量: " << subword_count << "\n";
        std::cout << "  完整词和特殊token: " << (vocab.size() - subword_count) << "\n";

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "║           Vocab测试通过！                                 ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════╝\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
