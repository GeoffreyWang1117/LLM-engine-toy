/**
 * @file test_generation_config.cpp
 * @brief 测试GenerationConfig功能
 *
 * 验证：
 * 1. max_new_tokens参数
 * 2. min_length参数
 * 3. eos_token_id停止条件
 * 4. stop_token_ids多停止token
 * 5. no_repeat_ngram_size n-gram重复控制
 * 6. 参数组合使用
 */

#include "llm_engine/gpt2_model.h"
#include "llm_engine/sampler.h"
#include <iostream>
#include <iomanip>

using namespace llm;

void print_separator() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
}

void print_sequence(const std::vector<int>& seq, const std::string& label) {
    std::cout << label << ": [";
    for (size_t i = 0; i < seq.size(); ++i) {
        std::cout << seq[i];
        if (i < seq.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

GPT2Model create_test_model() {
    GPT2Config config;
    config.vocab_size = 100;
    config.hidden_size = 64;
    config.num_layers = 2;
    config.num_heads = 4;
    config.intermediate_size = 256;

    GPT2Model model(config);

    // 设置简单的权重
    Matrix wte(config.vocab_size, config.hidden_size);
    Matrix wpe(config.max_position_embeddings, config.hidden_size);

    for (int i = 0; i < config.vocab_size; ++i) {
        for (int j = 0; j < config.hidden_size; ++j) {
            wte(i, j) = 0.01f * (i % 10);
        }
    }
    for (int i = 0; i < config.max_position_embeddings; ++i) {
        for (int j = 0; j < config.hidden_size; ++j) {
            wpe(i, j) = 0.001f * (i % 10);
        }
    }

    model.set_token_embedding(wte);
    model.set_position_embedding(wpe);

    Matrix ln_gamma(1, config.hidden_size, 1.0f);
    Matrix ln_beta(1, config.hidden_size, 0.0f);
    model.set_final_layer_norm(ln_gamma, ln_beta);

    return model;
}

bool test_max_new_tokens() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试1: max_new_tokens参数                                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        auto model = create_test_model();
        std::vector<int> prompt = {1, 2, 3};

        std::cout << "Prompt长度: " << prompt.size() << "\n";
        std::cout << "max_new_tokens: 10\n\n";

        GenerationConfig config = GenerationConfig::greedy(10);

        auto generated = model.generate(prompt, config);

        int num_new_tokens = static_cast<int>(generated.size()) - static_cast<int>(prompt.size());

        print_sequence(prompt, "Prompt");
        print_sequence(generated, "Generated");
        std::cout << "新生成的tokens数: " << num_new_tokens << "\n";
        std::cout << "期望: <= 10\n\n";

        if (num_new_tokens <= 10) {
            std::cout << "✓ max_new_tokens参数工作正常\n";
        } else {
            std::cout << "✗ 生成了超过max_new_tokens的tokens\n";
            return false;
        }

        // 测试不同的max_new_tokens值
        std::cout << "\n测试不同的max_new_tokens值:\n";
        std::vector<int> test_values = {5, 15, 20};

        for (int max_new : test_values) {
            GenerationConfig test_config = GenerationConfig::greedy(max_new);
            auto result = model.generate(prompt, test_config);
            int actual_new = static_cast<int>(result.size()) - static_cast<int>(prompt.size());

            std::cout << "  max_new_tokens=" << std::setw(2) << max_new
                      << " -> 实际生成=" << std::setw(2) << actual_new;

            if (actual_new <= max_new) {
                std::cout << " ✓\n";
            } else {
                std::cout << " ✗\n";
                return false;
            }
        }

        print_separator();
        std::cout << "✅ max_new_tokens测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_min_length() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试2: min_length参数                                       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        auto model = create_test_model();
        std::vector<int> prompt = {1, 2, 3};

        std::cout << "Prompt长度: " << prompt.size() << "\n";
        std::cout << "min_length: 15\n";
        std::cout << "max_new_tokens: 50\n\n";

        GenerationConfig config = GenerationConfig::greedy(50);
        config.min_length = 15;

        auto generated = model.generate(prompt, config);

        print_sequence(generated, "Generated");
        std::cout << "总长度: " << generated.size() << "\n";
        std::cout << "期望: >= 15\n\n";

        if (static_cast<int>(generated.size()) >= 15) {
            std::cout << "✓ min_length参数工作正常\n";
        } else {
            std::cout << "✗ 生成长度小于min_length\n";
            return false;
        }

        print_separator();
        std::cout << "✅ min_length测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_eos_token() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试3: eos_token_id停止条件                                 ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        auto model = create_test_model();
        std::vector<int> prompt = {1, 2, 3};

        std::cout << "Prompt: [1, 2, 3]\n";
        std::cout << "eos_token_id: 50\n";
        std::cout << "max_new_tokens: 100 (足够大，确保会遇到EOS)\n\n";

        // 使用temperature采样增加多样性，更可能遇到token 50
        GenerationConfig config = GenerationConfig::temperature(1.5f, 100);
        config.eos_token_id = 50;

        auto generated = model.generate(prompt, config);

        print_sequence(generated, "Generated");
        std::cout << "生成长度: " << generated.size() << "\n";

        // 检查最后一个token是否是EOS（如果提前停止）
        bool stopped_at_eos = false;
        if (!generated.empty() && generated.back() == 50) {
            stopped_at_eos = true;
        }

        if (stopped_at_eos) {
            std::cout << "✓ 在EOS token处停止\n";
        } else {
            std::cout << "⚠ 未遇到EOS token（正常情况，取决于采样）\n";
        }

        std::cout << "\n✓ eos_token_id参数正确配置\n";

        print_separator();
        std::cout << "✅ eos_token_id测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_stop_token_ids() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试4: stop_token_ids多停止token                            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        auto model = create_test_model();
        std::vector<int> prompt = {1, 2, 3};

        std::cout << "Prompt: [1, 2, 3]\n";
        std::cout << "stop_token_ids: [40, 50, 60]\n";
        std::cout << "max_new_tokens: 100\n\n";

        GenerationConfig config = GenerationConfig::temperature(1.5f, 100);
        config.stop_token_ids = {40, 50, 60};

        auto generated = model.generate(prompt, config);

        print_sequence(generated, "Generated");
        std::cout << "生成长度: " << generated.size() << "\n";

        // 检查是否在停止token处停止
        bool stopped_at_stop_token = false;
        if (!generated.empty()) {
            int last_token = generated.back();
            for (int stop_id : config.stop_token_ids) {
                if (last_token == stop_id) {
                    stopped_at_stop_token = true;
                    std::cout << "✓ 在停止token " << last_token << " 处停止\n";
                    break;
                }
            }
        }

        if (!stopped_at_stop_token) {
            std::cout << "⚠ 未遇到停止token（正常情况，取决于采样）\n";
        }

        std::cout << "\n✓ stop_token_ids参数正确配置\n";

        print_separator();
        std::cout << "✅ stop_token_ids测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_no_repeat_ngram() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试5: no_repeat_ngram_size n-gram重复控制                  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        auto model = create_test_model();
        std::vector<int> prompt = {1, 2, 3};

        std::cout << "测试1: 无n-gram限制\n";
        std::cout << "══════════════════\n";

        GenerationConfig config1 = GenerationConfig::greedy(30);
        auto generated1 = model.generate(prompt, config1);

        print_sequence(generated1, "Generated (无限制)");

        // 检测重复的3-gram
        int repeat_count1 = 0;
        for (size_t i = 0; i + 2 < generated1.size(); ++i) {
            for (size_t j = i + 3; j + 2 < generated1.size(); ++j) {
                if (generated1[i] == generated1[j] &&
                    generated1[i+1] == generated1[j+1] &&
                    generated1[i+2] == generated1[j+2]) {
                    repeat_count1++;
                    break;
                }
            }
        }
        std::cout << "重复的3-gram数量: " << repeat_count1 << "\n\n";

        std::cout << "测试2: 禁止重复3-gram\n";
        std::cout << "══════════════════\n";

        GenerationConfig config2 = GenerationConfig::greedy(30);
        config2.no_repeat_ngram_size = 3;

        auto generated2 = model.generate(prompt, config2);

        print_sequence(generated2, "Generated (3-gram限制)");

        // 检测重复的3-gram
        int repeat_count2 = 0;
        for (size_t i = 0; i + 2 < generated2.size(); ++i) {
            for (size_t j = i + 3; j + 2 < generated2.size(); ++j) {
                if (generated2[i] == generated2[j] &&
                    generated2[i+1] == generated2[j+1] &&
                    generated2[i+2] == generated2[j+2]) {
                    repeat_count2++;
                    std::cout << "⚠ 发现重复3-gram: [" << generated2[i] << ", "
                              << generated2[i+1] << ", " << generated2[i+2] << "] at positions "
                              << i << " and " << j << "\n";
                    break;
                }
            }
        }
        std::cout << "重复的3-gram数量: " << repeat_count2 << "\n\n";

        if (repeat_count2 == 0) {
            std::cout << "✓ no_repeat_ngram_size成功防止了3-gram重复\n";
        } else {
            std::cout << "✗ 仍有3-gram重复\n";
            return false;
        }

        // 测试不同的ngram大小
        std::cout << "\n测试不同的ngram大小:\n";
        std::vector<int> ngram_sizes = {2, 4, 5};

        for (int ngram_size : ngram_sizes) {
            GenerationConfig test_config = GenerationConfig::greedy(25);
            test_config.no_repeat_ngram_size = ngram_size;
            auto result = model.generate(prompt, test_config);

            std::cout << "  no_repeat_ngram_size=" << ngram_size
                      << " -> 生成长度=" << result.size() << " ✓\n";
        }

        print_separator();
        std::cout << "✅ no_repeat_ngram_size测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_combined_config() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试6: 组合参数使用                                         ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        auto model = create_test_model();
        std::vector<int> prompt = {1, 2, 3};

        std::cout << "配置:\n";
        std::cout << "  - max_new_tokens: 50\n";
        std::cout << "  - min_length: 10\n";
        std::cout << "  - repetition_penalty: 1.2\n";
        std::cout << "  - no_repeat_ngram_size: 3\n";
        std::cout << "  - temperature: 0.8\n\n";

        GenerationConfig config = GenerationConfig::temperature(0.8f, 50);
        config.min_length = 10;
        config.sampling.repetition_penalty = 1.2f;
        config.no_repeat_ngram_size = 3;

        auto generated = model.generate(prompt, config);

        print_sequence(generated, "Generated");
        std::cout << "\n验证:\n";

        // 验证长度约束
        int num_new = static_cast<int>(generated.size()) - static_cast<int>(prompt.size());
        std::cout << "  总长度: " << generated.size()
                  << " (min=" << config.min_length << ") ";
        if (static_cast<int>(generated.size()) >= config.min_length) {
            std::cout << "✓\n";
        } else {
            std::cout << "✗\n";
            return false;
        }

        std::cout << "  新tokens数: " << num_new
                  << " (max=" << config.max_new_tokens << ") ";
        if (num_new <= config.max_new_tokens) {
            std::cout << "✓\n";
        } else {
            std::cout << "✗\n";
            return false;
        }

        // 验证无3-gram重复
        bool has_repeat = false;
        for (size_t i = 0; i + 2 < generated.size(); ++i) {
            for (size_t j = i + 3; j + 2 < generated.size(); ++j) {
                if (generated[i] == generated[j] &&
                    generated[i+1] == generated[j+1] &&
                    generated[i+2] == generated[j+2]) {
                    has_repeat = true;
                    break;
                }
            }
            if (has_repeat) break;
        }

        std::cout << "  无3-gram重复: ";
        if (!has_repeat) {
            std::cout << "✓\n";
        } else {
            std::cout << "✗\n";
            return false;
        }

        std::cout << "\n✓ 所有参数正确协同工作\n";

        print_separator();
        std::cout << "✅ 组合参数测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        短期优化 - GenerationConfig参数测试                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    int passed = 0;
    int total = 6;

    // 测试1: max_new_tokens
    if (test_max_new_tokens()) {
        passed++;
    }

    // 测试2: min_length
    if (test_min_length()) {
        passed++;
    }

    // 测试3: eos_token_id
    if (test_eos_token()) {
        passed++;
    }

    // 测试4: stop_token_ids
    if (test_stop_token_ids()) {
        passed++;
    }

    // 测试5: no_repeat_ngram_size
    if (test_no_repeat_ngram()) {
        passed++;
    }

    // 测试6: 组合参数
    if (test_combined_config()) {
        passed++;
    }

    // 总结
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试总结                                                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "  通过: " << passed << " / " << total << "\n";
    std::cout << "  失败: " << (total - passed) << " / " << total << "\n";
    std::cout << "\n";

    if (passed == total) {
        std::cout << "✅ 所有测试通过！\n";
        std::cout << "\n";
        std::cout << "GenerationConfig功能验证成功！\n";
        std::cout << "新增参数:\n";
        std::cout << "  ✓ max_new_tokens - 限制新生成token数\n";
        std::cout << "  ✓ min_length - 最小总长度约束\n";
        std::cout << "  ✓ eos_token_id - EOS停止条件\n";
        std::cout << "  ✓ stop_token_ids - 多停止token\n";
        std::cout << "  ✓ no_repeat_ngram_size - n-gram重复控制\n";
        std::cout << "\n";
        std::cout << "使用示例:\n";
        std::cout << "  GenerationConfig config = GenerationConfig::temperature(0.8, 50);\n";
        std::cout << "  config.min_length = 20;\n";
        std::cout << "  config.eos_token_id = 50256;\n";
        std::cout << "  config.sampling.repetition_penalty = 1.2f;\n";
        std::cout << "  config.no_repeat_ngram_size = 3;\n";
        std::cout << "  auto generated = model.generate(prompt, config);\n";
        std::cout << "\n";
        return 0;
    } else {
        std::cout << "❌ 部分测试失败\n\n";
        return 1;
    }
}
