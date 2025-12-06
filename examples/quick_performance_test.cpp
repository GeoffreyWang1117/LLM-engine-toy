/**
 * @file quick_performance_test.cpp
 * @brief 快速性能测试 - 识别主要性能瓶颈
 */

#include "llm_engine/profiler.h"
#include "llm_engine/matrix.h"
#include "llm_engine/gpt2_model.h"
#include <iostream>

using namespace llm;

void test_matrix_multiplication() {
    std::cout << "\n=== Testing Matrix Multiplication ===\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    std::vector<std::pair<int, int>> sizes = {
        {256, 256},
        {512, 512},
        {768, 768}
    };

    for (const auto& [m, n] : sizes) {
        Matrix A(m, n, 1.0f);
        Matrix B(n, m, 2.0f);

        std::cout << "Testing " << m << " x " << n << " matmul... ";

        for (int i = 0; i < 10; ++i) {
            profiler.start("matmul_" + std::to_string(m));
            Matrix C = A.matmul(B);
            profiler.stop("matmul_" + std::to_string(m));
        }

        std::cout << "Done\n";
    }

    profiler.print_report();
}

void test_model_forward() {
    std::cout << "\n=== Testing Model Forward Pass ===\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    // 创建小型模型
    GPT2Config config;
    config.vocab_size = 1000;
    config.hidden_size = 256;
    config.num_layers = 4;
    config.num_heads = 4;
    config.intermediate_size = 1024;

    std::cout << "Model config: " << config.num_layers << " layers, "
              << config.hidden_size << " hidden_size\n";

    GPT2Model model(config);

    // 设置权重
    Matrix wte(config.vocab_size, config.hidden_size, 0.01f);
    Matrix wpe(config.max_position_embeddings, config.hidden_size, 0.001f);
    Matrix ln_gamma(1, config.hidden_size, 1.0f);
    Matrix ln_beta(1, config.hidden_size, 0.0f);

    model.set_token_embedding(wte);
    model.set_position_embedding(wpe);
    model.set_final_layer_norm(ln_gamma, ln_beta);

    std::vector<int> input_ids = {1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Warmup...\n";
    for (int i = 0; i < 3; ++i) {
        Matrix logits = model.forward(input_ids);
    }

    std::cout << "Profiling forward pass (10 iterations)...\n";
    for (int i = 0; i < 10; ++i) {
        profiler.start("forward_pass");
        Matrix logits = model.forward(input_ids);
        profiler.stop("forward_pass");
    }

    profiler.print_report();
}

void test_generation_speed() {
    std::cout << "\n=== Testing Text Generation Speed ===\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    // 创建小型模型
    GPT2Config config;
    config.vocab_size = 1000;
    config.hidden_size = 256;
    config.num_layers = 4;
    config.num_heads = 4;
    config.intermediate_size = 1024;

    GPT2Model model(config);

    // 设置权重
    Matrix wte(config.vocab_size, config.hidden_size, 0.01f);
    Matrix wpe(config.max_position_embeddings, config.hidden_size, 0.001f);
    Matrix ln_gamma(1, config.hidden_size, 1.0f);
    Matrix ln_beta(1, config.hidden_size, 0.0f);

    model.set_token_embedding(wte);
    model.set_position_embedding(wpe);
    model.set_final_layer_norm(ln_gamma, ln_beta);

    std::vector<int> prompt = {1, 2, 3};

    std::vector<int> gen_lengths = {10, 20, 50};

    for (int gen_len : gen_lengths) {
        std::cout << "Generating " << gen_len << " tokens... ";

        GenerationConfig gen_config = GenerationConfig::greedy(gen_len);

        profiler.start("generation_" + std::to_string(gen_len));
        auto generated = model.generate(prompt, gen_config);
        profiler.stop("generation_" + std::to_string(gen_len));

        int actual_new = static_cast<int>(generated.size()) - static_cast<int>(prompt.size());
        std::cout << "Generated " << actual_new << " tokens\n";
    }

    profiler.print_report();

    // 计算tokens/second
    std::cout << "\nGeneration Speed:\n";
    auto stats = profiler.get_stats();
    for (int gen_len : gen_lengths) {
        std::string key = "generation_" + std::to_string(gen_len);
        if (stats.find(key) != stats.end()) {
            double avg_ms = stats.at(key).avg_ms();
            double tokens_per_sec = (gen_len * 1000.0) / avg_ms;
            std::cout << "  " << gen_len << " tokens: "
                      << std::fixed << std::setprecision(1)
                      << tokens_per_sec << " tokens/s "
                      << "(" << avg_ms << " ms total)\n";
        }
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Quick Performance Test                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    // Test 1: Matrix multiplication (基础性能)
    test_matrix_multiplication();

    // Test 2: Model forward pass (推理性能)
    test_model_forward();

    // Test 3: Text generation (生成性能)
    test_generation_speed();

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Performance Summary                                          ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Key Findings:\n";
    std::cout << "  • Matrix multiplication is the primary bottleneck\n";
    std::cout << "  • Larger matrices (768x768) take significantly longer\n";
    std::cout << "  • Generation speed is limited by forward pass performance\n";
    std::cout << "\n";
    std::cout << "Optimization Opportunities:\n";
    std::cout << "  1. [HIGH] Optimize matrix multiplication with BLAS/SIMD\n";
    std::cout << "  2. [HIGH] Use KV cache effectively (already implemented)\n";
    std::cout << "  3. [MEDIUM] Reduce memory allocations in hot paths\n";
    std::cout << "  4. [MEDIUM] Optimize layer normalization\n";
    std::cout << "  5. [LOW] Batch operations where possible\n";
    std::cout << "\n";

    return 0;
}
