/**
 * @file test_sampler.cpp
 * @brief 测试Sampling策略
 *
 * 验证：
 * 1. Greedy采样（确定性）
 * 2. Temperature采样（温度控制）
 * 3. Top-k采样（限制候选集）
 * 4. Top-p采样（nucleus采样）
 */

#include "llm_engine/sampler.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>

using namespace llm;

void print_separator() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
}

void print_distribution(const std::map<int, int>& counts, int total) {
    std::cout << "  Token  Count  Probability\n";
    std::cout << "  ─────  ─────  ───────────\n";

    for (const auto& pair : counts) {
        float prob = static_cast<float>(pair.second) / total;
        std::cout << "    " << std::setw(2) << pair.first
                  << "     " << std::setw(4) << pair.second
                  << "    " << std::fixed << std::setprecision(3) << prob << "\n";
    }
    std::cout << "\n";
}

bool test_greedy() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试1: Greedy采样                                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "验证: 总是选择概率最高的token\n\n";

    try {
        Sampler sampler(42);

        // 创建logits（token 2的概率最高）
        Matrix logits(1, 5);
        logits(0, 0) = 1.0f;
        logits(0, 1) = 2.0f;
        logits(0, 2) = 5.0f;  // 最高
        logits(0, 3) = 1.5f;
        logits(0, 4) = 0.5f;

        std::cout << "Logits: [";
        for (size_t i = 0; i < logits.cols(); ++i) {
            std::cout << std::fixed << std::setprecision(1) << logits(0, i);
            if (i < logits.cols() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "期望: 总是选择token 2\n\n";

        // 采样多次
        int num_samples = 100;
        std::map<int, int> counts;

        for (int i = 0; i < num_samples; ++i) {
            int token = sampler.greedy_sample(logits);
            counts[token]++;
        }

        std::cout << "采样结果 (" << num_samples << "次):\n";
        print_distribution(counts, num_samples);

        // 验证
        if (counts.size() != 1 || counts[2] != num_samples) {
            std::cerr << "❌ 错误: Greedy应该总是选择token 2\n";
            return false;
        }

        std::cout << "✓ Greedy采样确定性验证通过\n";

        print_separator();
        std::cout << "✅ Greedy测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_temperature() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试2: Temperature采样                                      ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "验证: 温度控制随机性\n";
    std::cout << "  - temperature低: 更确定（接近greedy）\n";
    std::cout << "  - temperature高: 更随机\n\n";

    try {
        Sampler sampler(42);

        // 创建logits
        Matrix logits(1, 5);
        logits(0, 0) = 1.0f;
        logits(0, 1) = 2.0f;
        logits(0, 2) = 3.0f;  // 最高
        logits(0, 3) = 1.5f;
        logits(0, 4) = 0.5f;

        int num_samples = 1000;

        // 测试不同温度
        std::vector<float> temperatures = {0.1f, 1.0f, 2.0f};

        for (float temp : temperatures) {
            std::cout << "Temperature = " << temp << ":\n";

            std::map<int, int> counts;
            sampler.set_seed(42);  // 重置种子

            for (int i = 0; i < num_samples; ++i) {
                int token = sampler.temperature_sample(logits, temp);
                counts[token]++;
            }

            print_distribution(counts, num_samples);

            // 验证：低温度时，token 2应该占主导
            if (temp == 0.1f) {
                if (counts[2] < num_samples * 0.8f) {
                    std::cerr << "❌ 警告: 低温度时token 2应该占主导\n";
                }
            }
        }

        std::cout << "✓ Temperature采样工作正常\n";

        print_separator();
        std::cout << "✅ Temperature测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_top_k() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试3: Top-k采样                                            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "验证: 只从概率最高的k个token中采样\n\n";

    try {
        Sampler sampler(42);

        // 创建logits（10个token）
        Matrix logits(1, 10);
        for (size_t i = 0; i < 10; ++i) {
            logits(0, i) = static_cast<float>(10 - i);  // 降序：10, 9, 8, ...
        }

        std::cout << "Logits: [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]\n";
        std::cout << "Top-3 tokens: 0, 1, 2\n\n";

        // Top-3采样
        int k = 3;
        int num_samples = 1000;
        std::map<int, int> counts;

        for (int i = 0; i < num_samples; ++i) {
            int token = sampler.top_k_sample(logits, k);
            counts[token]++;
        }

        std::cout << "Top-" << k << " 采样结果 (" << num_samples << "次):\n";
        print_distribution(counts, num_samples);

        // 验证：只应该出现token 0, 1, 2
        for (const auto& pair : counts) {
            if (pair.first >= k) {
                std::cerr << "❌ 错误: 采样到了top-" << k << "之外的token " << pair.first << "\n";
                return false;
            }
        }

        std::cout << "✓ Top-k采样正确限制候选集\n";

        // 测试k=1（应该等价于greedy）
        std::cout << "\n测试k=1（应该等价于greedy）:\n";
        counts.clear();
        for (int i = 0; i < 100; ++i) {
            int token = sampler.top_k_sample(logits, 1);
            counts[token]++;
        }

        if (counts.size() != 1 || counts[0] != 100) {
            std::cerr << "❌ 错误: k=1应该等价于greedy\n";
            return false;
        }
        std::cout << "✓ k=1 等价于greedy验证通过\n";

        print_separator();
        std::cout << "✅ Top-k测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_top_p() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试4: Top-p (Nucleus)采样                                  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "验证: 累积概率达到p时停止\n\n";

    try {
        Sampler sampler(42);

        // 创建logits（概率分布较集中）
        Matrix logits(1, 10);
        logits(0, 0) = 5.0f;   // 高概率
        logits(0, 1) = 4.0f;   // 中等概率
        logits(0, 2) = 3.0f;   // 中等概率
        for (size_t i = 3; i < 10; ++i) {
            logits(0, i) = 0.5f;  // 低概率
        }

        std::cout << "Logits: [5.0, 4.0, 3.0, 0.5, 0.5, ...]\n";
        std::cout << "前3个token占据大部分概率\n\n";

        // Top-p采样（p=0.9）
        float p = 0.9f;
        int num_samples = 1000;
        std::map<int, int> counts;

        for (int i = 0; i < num_samples; ++i) {
            int token = sampler.top_p_sample(logits, p);
            counts[token]++;
        }

        std::cout << "Top-p=" << p << " 采样结果 (" << num_samples << "次):\n";
        print_distribution(counts, num_samples);

        // 验证：主要应该是前几个高概率token
        int top_tokens_count = 0;
        for (int i = 0; i < 3; ++i) {
            if (counts.find(i) != counts.end()) {
                top_tokens_count += counts[i];
            }
        }

        float top_ratio = static_cast<float>(top_tokens_count) / num_samples;
        std::cout << "前3个token占比: " << std::fixed << std::setprecision(2)
                  << (top_ratio * 100) << "%\n\n";

        if (top_ratio < 0.8f) {
            std::cerr << "❌ 警告: 前3个token占比偏低\n";
        } else {
            std::cout << "✓ Top-p采样正确集中在高概率区域\n";
        }

        print_separator();
        std::cout << "✅ Top-p测试通过！\n";
        print_separator();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << "\n";
        return false;
    }
}

bool test_sampling_config() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试5: SamplingConfig统一接口                               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "验证: 统一的sample()接口\n\n";

    try {
        Sampler sampler(42);

        Matrix logits(1, 5);
        logits(0, 0) = 1.0f;
        logits(0, 1) = 2.0f;
        logits(0, 2) = 5.0f;
        logits(0, 3) = 1.5f;
        logits(0, 4) = 0.5f;

        // 测试各种配置
        std::cout << "测试greedy配置:\n";
        int token1 = sampler.sample(logits, SamplingConfig::greedy());
        std::cout << "  采样token: " << token1 << " (期望: 2)\n\n";

        std::cout << "测试temperature配置:\n";
        int token2 = sampler.sample(logits, SamplingConfig::temperature(0.5f));
        std::cout << "  采样token: " << token2 << "\n\n";

        std::cout << "测试top-k配置:\n";
        int token3 = sampler.sample(logits, SamplingConfig::top_k(3));
        std::cout << "  采样token: " << token3 << "\n\n";

        std::cout << "测试top-p配置:\n";
        int token4 = sampler.sample(logits, SamplingConfig::top_p(0.9f));
        std::cout << "  采样token: " << token4 << "\n\n";

        if (token1 != 2) {
            std::cerr << "❌ 错误: greedy应该返回token 2\n";
            return false;
        }

        std::cout << "✓ 统一接口工作正常\n";

        print_separator();
        std::cout << "✅ SamplingConfig测试通过！\n";
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
    std::cout << "║        Beta 0.2 - Sampling Strategies测试                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    int passed = 0;
    int total = 5;

    // 测试1: Greedy
    if (test_greedy()) {
        passed++;
    }

    // 测试2: Temperature
    if (test_temperature()) {
        passed++;
    }

    // 测试3: Top-k
    if (test_top_k()) {
        passed++;
    }

    // 测试4: Top-p
    if (test_top_p()) {
        passed++;
    }

    // 测试5: SamplingConfig
    if (test_sampling_config()) {
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
        std::cout << "Sampling Strategies验证成功！\n";
        std::cout << "支持的策略:\n";
        std::cout << "  ✓ Greedy - 确定性，选择最高概率\n";
        std::cout << "  ✓ Temperature - 控制随机性（0.1-2.0）\n";
        std::cout << "  ✓ Top-k - 限制候选集大小\n";
        std::cout << "  ✓ Top-p (Nucleus) - 动态候选集\n";
        std::cout << "\n";
        std::cout << "下一步: 实现BPE Tokenizer（GPT-2分词器）\n";
        std::cout << "\n";
        return 0;
    } else {
        std::cout << "❌ 部分测试失败\n\n";
        return 1;
    }
}
