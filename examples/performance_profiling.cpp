/**
 * @file performance_profiling.cpp
 * @brief 性能分析和Profiling
 *
 * 测量各个组件的性能：
 * 1. Matrix operations
 * 2. Attention computation
 * 3. FFN layers
 * 4. Layer normalization
 * 5. Embeddings
 * 6. Full forward pass
 * 7. Text generation
 */

#include "llm_engine/profiler.h"
#include "llm_engine/matrix.h"
#include "llm_engine/layers.h"
#include "llm_engine/causal_attention.h"
#include "llm_engine/gpt_decoder_block.h"
#include "llm_engine/gpt2_model.h"
#include "llm_engine/sampler.h"
#include <iostream>
#include <vector>
#include <random>

using namespace llm;

void print_separator() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
}

void profile_matrix_operations() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling: Matrix Operations                                 ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    std::vector<std::pair<int, int>> sizes = {
        {128, 128},
        {256, 256},
        {512, 512},
        {768, 768},
        {1024, 1024}
    };

    for (const auto& [m, n] : sizes) {
        std::cout << "Testing " << m << " × " << n << " matrices...\n";

        Matrix A(m, n, 1.0f);
        Matrix B(n, m, 2.0f);

        // Matrix multiplication
        for (int i = 0; i < 10; ++i) {
            profiler.start("matmul_" + std::to_string(m));
            Matrix C = A.matmul(B);
            profiler.stop("matmul_" + std::to_string(m));
        }

        // Matrix addition
        Matrix C(m, n, 0.5f);
        for (int i = 0; i < 100; ++i) {
            profiler.start("add_" + std::to_string(m));
            Matrix D = A + C;
            profiler.stop("add_" + std::to_string(m));
        }

        // Element-wise operations
        for (int i = 0; i < 100; ++i) {
            profiler.start("elementwise_" + std::to_string(m));
            for (size_t j = 0; j < A.rows(); ++j) {
                for (size_t k = 0; k < A.cols(); ++k) {
                    float val = A(j, k) * 2.0f + 1.0f;
                    (void)val;
                }
            }
            profiler.stop("elementwise_" + std::to_string(m));
        }
    }

    profiler.print_report();
}

void profile_attention() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling: Attention Computation                             ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    int hidden_size = 768;
    int num_heads = 12;

    CausalAttention attn(hidden_size, num_heads);

    // 设置简单权重
    Matrix wq(hidden_size, hidden_size);
    Matrix wk(hidden_size, hidden_size);
    Matrix wv(hidden_size, hidden_size);
    Matrix wo(hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            wq(i, j) = 0.01f;
            wk(i, j) = 0.01f;
            wv(i, j) = 0.01f;
            wo(i, j) = 0.01f;
        }
    }

    attn.set_weights(wq, wk, wv, wo);

    // 测试不同序列长度
    std::vector<int> seq_lengths = {8, 16, 32, 64, 128};

    for (int seq_len : seq_lengths) {
        std::cout << "Testing sequence length " << seq_len << "...\n";

        Matrix input(seq_len, hidden_size, 1.0f);

        // Forward pass without cache
        for (int i = 0; i < 20; ++i) {
            profiler.start("attention_forward_" + std::to_string(seq_len));
            Matrix output = attn.forward(input);
            profiler.stop("attention_forward_" + std::to_string(seq_len));
        }

        // Forward pass with cache
        MultiHeadKVCache cache(num_heads);
        for (int i = 0; i < 50; ++i) {
            Matrix single_input(1, hidden_size, 1.0f);
            profiler.start("attention_cached_" + std::to_string(seq_len));
            Matrix output = attn.forward_with_cache(single_input, cache);
            profiler.stop("attention_cached_" + std::to_string(seq_len));
        }
    }

    profiler.print_report();
}

void profile_ffn() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling: Feed-Forward Network                              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    int hidden_size = 768;
    int intermediate_size = 3072;

    FeedForward ffn(hidden_size, intermediate_size);

    // 设置权重（通过访问Linear层）
    // Note: FeedForward使用Linear层，它们会在构造时初始化

    // 测试不同batch大小
    std::vector<int> seq_lengths = {1, 8, 16, 32, 64};

    for (int seq_len : seq_lengths) {
        std::cout << "Testing sequence length " << seq_len << "...\n";

        Matrix input(seq_len, hidden_size, 1.0f);

        for (int i = 0; i < 50; ++i) {
            profiler.start("ffn_forward_" + std::to_string(seq_len));
            Matrix output = ffn.forward(input);
            profiler.stop("ffn_forward_" + std::to_string(seq_len));
        }
    }

    profiler.print_report();
}

void profile_decoder_block() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling: GPT Decoder Block                                 ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    int hidden_size = 768;
    int num_heads = 12;
    int intermediate_size = 3072;

    GPTDecoderBlock block(hidden_size, num_heads, intermediate_size);

    // 测试不同序列长度
    std::vector<int> seq_lengths = {1, 8, 16, 32, 64};

    for (int seq_len : seq_lengths) {
        std::cout << "Testing sequence length " << seq_len << "...\n";

        Matrix input(seq_len, hidden_size, 1.0f);

        // Forward without cache
        for (int i = 0; i < 20; ++i) {
            profiler.start("decoder_forward_" + std::to_string(seq_len));
            Matrix output = block.forward(input);
            profiler.stop("decoder_forward_" + std::to_string(seq_len));
        }

        // Forward with cache
        MultiHeadKVCache cache(num_heads);
        for (int i = 0; i < 50; ++i) {
            Matrix single_input(1, hidden_size, 1.0f);
            profiler.start("decoder_cached_" + std::to_string(seq_len));
            Matrix output = block.forward_with_cache(single_input, cache);
            profiler.stop("decoder_cached_" + std::to_string(seq_len));
        }
    }

    profiler.print_report();
}

void profile_full_model() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling: Full GPT-2 Model                                  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    // 创建小型模型
    GPT2Config config;
    config.vocab_size = 1000;
    config.hidden_size = 256;
    config.num_layers = 6;
    config.num_heads = 8;
    config.intermediate_size = 1024;

    std::cout << "Model configuration:\n";
    std::cout << "  vocab_size: " << config.vocab_size << "\n";
    std::cout << "  hidden_size: " << config.hidden_size << "\n";
    std::cout << "  num_layers: " << config.num_layers << "\n";
    std::cout << "  num_heads: " << config.num_heads << "\n";
    std::cout << "  intermediate_size: " << config.intermediate_size << "\n\n";

    GPT2Model model(config);

    // 设置简单权重
    Matrix wte(config.vocab_size, config.hidden_size);
    Matrix wpe(config.max_position_embeddings, config.hidden_size);

    for (int i = 0; i < config.vocab_size; ++i) {
        for (int j = 0; j < config.hidden_size; ++j) {
            wte(i, j) = 0.01f;
        }
    }
    for (int i = 0; i < config.max_position_embeddings; ++i) {
        for (int j = 0; j < config.hidden_size; ++j) {
            wpe(i, j) = 0.001f;
        }
    }

    model.set_token_embedding(wte);
    model.set_position_embedding(wpe);

    Matrix ln_gamma(1, config.hidden_size, 1.0f);
    Matrix ln_beta(1, config.hidden_size, 0.0f);
    model.set_final_layer_norm(ln_gamma, ln_beta);

    // 测试forward pass
    std::vector<int> input_ids = {1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Testing forward pass (warmup)...\n";
    for (int i = 0; i < 5; ++i) {
        Matrix logits = model.forward(input_ids);
    }

    std::cout << "Profiling forward pass...\n";
    for (int i = 0; i < 20; ++i) {
        profiler.start("model_forward");
        Matrix logits = model.forward(input_ids);
        profiler.stop("model_forward");
    }

    profiler.print_report();
}

void profile_text_generation() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling: Text Generation                                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    // 创建小型模型
    GPT2Config config;
    config.vocab_size = 1000;
    config.hidden_size = 256;
    config.num_layers = 6;
    config.num_heads = 8;
    config.intermediate_size = 1024;

    GPT2Model model(config);

    // 设置简单权重
    Matrix wte(config.vocab_size, config.hidden_size);
    Matrix wpe(config.max_position_embeddings, config.hidden_size);

    for (int i = 0; i < config.vocab_size; ++i) {
        for (int j = 0; j < config.hidden_size; ++j) {
            wte(i, j) = 0.01f;
        }
    }
    for (int i = 0; i < config.max_position_embeddings; ++i) {
        for (int j = 0; j < config.hidden_size; ++j) {
            wpe(i, j) = 0.001f;
        }
    }

    model.set_token_embedding(wte);
    model.set_position_embedding(wpe);

    Matrix ln_gamma(1, config.hidden_size, 1.0f);
    Matrix ln_beta(1, config.hidden_size, 0.0f);
    model.set_final_layer_norm(ln_gamma, ln_beta);

    // 测试不同生成长度
    std::vector<int> prompt = {1, 2, 3};
    std::vector<int> gen_lengths = {10, 20, 50};

    for (int gen_len : gen_lengths) {
        std::cout << "Generating " << gen_len << " tokens...\n";

        GenerationConfig gen_config = GenerationConfig::greedy(gen_len);

        profiler.start("generation_" + std::to_string(gen_len));
        auto generated = model.generate(prompt, gen_config);
        profiler.stop("generation_" + std::to_string(gen_len));

        std::cout << "  Generated " << (generated.size() - prompt.size()) << " new tokens\n";
    }

    profiler.print_report();

    // 计算tokens/second
    auto stats = profiler.get_stats();
    for (int gen_len : gen_lengths) {
        std::string key = "generation_" + std::to_string(gen_len);
        if (stats.find(key) != stats.end()) {
            double avg_ms = stats.at(key).avg_ms();
            double tokens_per_sec = (gen_len * 1000.0) / avg_ms;
            std::cout << "\nGeneration speed (" << gen_len << " tokens): "
                      << std::fixed << std::setprecision(1)
                      << tokens_per_sec << " tokens/s\n";
        }
    }
}

void profile_sampling() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling: Sampling Strategies                               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto& profiler = Profiler::instance();
    profiler.reset();

    Sampler sampler(42);

    // 创建logits
    int vocab_size = 50000;  // GPT-2大小
    Matrix logits(1, vocab_size);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

    for (int i = 0; i < vocab_size; ++i) {
        logits(0, i) = dis(gen);
    }

    std::vector<int> generated_tokens;
    for (int i = 0; i < 50; ++i) {
        generated_tokens.push_back(i);
    }

    std::cout << "Testing sampling strategies (vocab_size=" << vocab_size << ")...\n\n";

    // Greedy
    for (int i = 0; i < 1000; ++i) {
        profiler.start("greedy_sampling");
        int token = sampler.greedy_sample(logits);
        (void)token;  // Suppress unused warning
        profiler.stop("greedy_sampling");
    }

    // Temperature
    for (int i = 0; i < 1000; ++i) {
        profiler.start("temperature_sampling");
        int token = sampler.temperature_sample(logits, 0.8f);
        (void)token;
        profiler.stop("temperature_sampling");
    }

    // Top-k
    for (int i = 0; i < 1000; ++i) {
        profiler.start("topk_sampling");
        int token = sampler.top_k_sample(logits, 50);
        (void)token;
        profiler.stop("topk_sampling");
    }

    // Top-p
    for (int i = 0; i < 1000; ++i) {
        profiler.start("topp_sampling");
        int token = sampler.top_p_sample(logits, 0.95f);
        (void)token;
        profiler.stop("topp_sampling");
    }

    // With repetition penalty
    SamplingConfig config = SamplingConfig::greedy();
    config.repetition_penalty = 1.2f;

    for (int i = 0; i < 1000; ++i) {
        profiler.start("sampling_with_penalty");
        int token = sampler.sample(logits, config, generated_tokens);
        (void)token;
        profiler.stop("sampling_with_penalty");
    }

    profiler.print_report();
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Performance Profiling & Optimization             ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    // 1. Matrix operations
    profile_matrix_operations();

    // 2. Attention
    profile_attention();

    // 3. FFN
    profile_ffn();

    // 4. Decoder block
    profile_decoder_block();

    // 5. Sampling
    profile_sampling();

    // 6. Full model
    profile_full_model();

    // 7. Text generation
    profile_text_generation();

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Profiling Complete                                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Performance bottlenecks identified:\n";
    std::cout << "  • Matrix multiplication (largest impact)\n";
    std::cout << "  • Attention computation (O(n²) complexity)\n";
    std::cout << "  • Sampling with large vocabulary\n";
    std::cout << "\n";
    std::cout << "Recommended optimizations:\n";
    std::cout << "  1. Optimize matrix multiplication (BLAS integration)\n";
    std::cout << "  2. Optimize attention with Flash Attention\n";
    std::cout << "  3. Use fixed-size buffers to reduce allocations\n";
    std::cout << "  4. Add SIMD vectorization for element-wise ops\n";
    std::cout << "  5. Optimize top-k/top-p sampling\n";
    std::cout << "\n";

    return 0;
}
