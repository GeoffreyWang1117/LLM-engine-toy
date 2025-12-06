#include "llm_engine/gpt2_model.h"
#include "llm_engine/safetensors.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>

namespace llm {

GPT2Model::GPT2Model(const GPT2Config& config)
    : config_(config),
      token_embedding_(config.vocab_size, config.hidden_size),
      position_embedding_(config.max_position_embeddings, config.hidden_size),
      ln_f_gamma_(1, config.hidden_size, 1.0f),
      ln_f_beta_(1, config.hidden_size, 0.0f) {

    // 创建Decoder Blocks
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.push_back(
            std::make_unique<GPTDecoderBlock>(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.dropout
            )
        );
    }
}

Matrix GPT2Model::layer_norm(const Matrix& x, const Matrix& gamma, const Matrix& beta) const {
    size_t seq_len = x.rows();
    size_t hidden_size = x.cols();

    Matrix result(seq_len, hidden_size);

    for (size_t i = 0; i < seq_len; ++i) {
        // 计算均值
        float mean = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            mean += x(i, j);
        }
        mean /= hidden_size;

        // 计算方差
        float variance = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float diff = x(i, j) - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;

        // 归一化
        float std = std::sqrt(variance + config_.layer_norm_epsilon);
        for (size_t j = 0; j < hidden_size; ++j) {
            result(i, j) = gamma(0, j) * ((x(i, j) - mean) / std) + beta(0, j);
        }
    }

    return result;
}

Matrix GPT2Model::apply_embeddings(const std::vector<int>& input_ids) const {
    size_t seq_len = input_ids.size();

    if (seq_len > static_cast<size_t>(config_.max_position_embeddings)) {
        throw std::runtime_error("Sequence length exceeds max_position_embeddings");
    }

    Matrix embeddings(seq_len, config_.hidden_size);

    // Token Embedding + Position Embedding
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = input_ids[i];

        if (token_id < 0 || token_id >= config_.vocab_size) {
            throw std::runtime_error("Token ID out of range");
        }

        for (int j = 0; j < config_.hidden_size; ++j) {
            embeddings(i, j) = token_embedding_(token_id, j) + position_embedding_(i, j);
        }
    }

    return embeddings;
}

Matrix GPT2Model::apply_lm_head(const Matrix& hidden_states) const {
    // LM Head: [seq_len, hidden_size] @ [hidden_size, vocab_size]
    // 即 hidden_states @ token_embedding_.T
    size_t seq_len = hidden_states.rows();
    Matrix logits(seq_len, config_.vocab_size);

    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < config_.vocab_size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < config_.hidden_size; ++k) {
                sum += hidden_states(i, k) * token_embedding_(j, k);
            }
            logits(i, j) = sum;
        }
    }

    return logits;
}

Matrix GPT2Model::forward(const std::vector<int>& input_ids) {
    if (input_ids.empty()) {
        throw std::runtime_error("Input IDs cannot be empty");
    }

    // 1. Embeddings
    Matrix hidden_states = apply_embeddings(input_ids);

    // 2. Decoder Blocks
    for (int i = 0; i < config_.num_layers; ++i) {
        hidden_states = layers_[i]->forward(hidden_states);
    }

    // 3. Final LayerNorm
    hidden_states = layer_norm(hidden_states, ln_f_gamma_, ln_f_beta_);

    // 4. LM Head
    Matrix logits = apply_lm_head(hidden_states);

    return logits;
}

std::vector<int> GPT2Model::generate(
    const std::vector<int>& input_ids,
    const GenerationConfig& gen_config) {

    if (input_ids.empty()) {
        throw std::runtime_error("Input IDs cannot be empty");
    }

    std::vector<int> generated_ids = input_ids;
    Sampler sampler(gen_config.seed);

    // 创建KV Cache（每层一个）
    std::vector<MultiHeadKVCache> kv_caches(config_.num_layers, MultiHeadKVCache(config_.num_heads));

    // 初始化KV Cache：逐个处理prompt中的token
    Matrix logits(1, config_.vocab_size);
    for (size_t i = 0; i < input_ids.size(); ++i) {
        std::vector<int> single_token = {input_ids[i]};
        Matrix hidden_states = apply_embeddings(single_token);

        for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
            hidden_states = layers_[layer_idx]->forward_with_cache(hidden_states, kv_caches[layer_idx]);
        }

        hidden_states = layer_norm(hidden_states, ln_f_gamma_, ln_f_beta_);
        logits = apply_lm_head(hidden_states);
    }

    // 采样第一个新token（基于最后一个prompt token的输出）
    Matrix last_logits(1, config_.vocab_size);
    for (int j = 0; j < config_.vocab_size; ++j) {
        last_logits(0, j) = logits(0, j);
    }

    // 应用n-gram禁止
    if (gen_config.no_repeat_ngram_size > 0) {
        auto banned = get_banned_ngrams(generated_ids, gen_config.no_repeat_ngram_size);
        for (int token_id : banned) {
            if (token_id >= 0 && token_id < config_.vocab_size) {
                last_logits(0, token_id) = -std::numeric_limits<float>::infinity();
            }
        }
    }

    int next_token = sampler.sample(last_logits, gen_config.sampling, generated_ids);
    generated_ids.push_back(next_token);

    // 检查停止条件
    if (should_stop(generated_ids, gen_config)) {
        return generated_ids;
    }

    int num_new_tokens = 1;  // 已生成的新token数

    // 自回归生成
    while (num_new_tokens < gen_config.max_new_tokens &&
           static_cast<int>(generated_ids.size()) < gen_config.max_length) {

        // 只处理最新的token（利用KV Cache）
        std::vector<int> new_token_ids = {next_token};
        Matrix hidden_states = apply_embeddings(new_token_ids);

        // 通过所有层（使用cache）
        for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
            hidden_states = layers_[layer_idx]->forward_with_cache(hidden_states, kv_caches[layer_idx]);
        }

        // Final LayerNorm
        hidden_states = layer_norm(hidden_states, ln_f_gamma_, ln_f_beta_);

        // LM Head
        logits = apply_lm_head(hidden_states);

        // 应用n-gram禁止
        Matrix current_logits(1, config_.vocab_size);
        for (int j = 0; j < config_.vocab_size; ++j) {
            current_logits(0, j) = logits(0, j);
        }

        if (gen_config.no_repeat_ngram_size > 0) {
            auto banned = get_banned_ngrams(generated_ids, gen_config.no_repeat_ngram_size);
            for (int token_id : banned) {
                if (token_id >= 0 && token_id < config_.vocab_size) {
                    current_logits(0, token_id) = -std::numeric_limits<float>::infinity();
                }
            }
        }

        // 采样（传递已生成的tokens用于repetition penalty）
        next_token = sampler.sample(current_logits, gen_config.sampling, generated_ids);

        generated_ids.push_back(next_token);
        num_new_tokens++;

        // 检查停止条件
        if (should_stop(generated_ids, gen_config)) {
            break;
        }

        // 检查最小长度约束
        if (static_cast<int>(generated_ids.size()) < gen_config.min_length) {
            continue;  // 继续生成直到达到最小长度
        }
    }

    return generated_ids;
}

std::vector<int> GPT2Model::generate(
    const std::vector<int>& input_ids,
    int max_length,
    const SamplingConfig& config,
    unsigned int seed) {

    // 转换为新API
    GenerationConfig gen_config;
    gen_config.sampling = config;
    gen_config.max_new_tokens = max_length - static_cast<int>(input_ids.size());
    gen_config.max_length = max_length;
    gen_config.seed = seed;

    return generate(input_ids, gen_config);
}

void GPT2Model::set_token_embedding(const Matrix& wte) {
    if (wte.rows() != static_cast<size_t>(config_.vocab_size) ||
        wte.cols() != static_cast<size_t>(config_.hidden_size)) {
        throw std::runtime_error("Token embedding shape mismatch");
    }
    token_embedding_ = wte;
}

void GPT2Model::set_position_embedding(const Matrix& wpe) {
    if (wpe.rows() != static_cast<size_t>(config_.max_position_embeddings) ||
        wpe.cols() != static_cast<size_t>(config_.hidden_size)) {
        throw std::runtime_error("Position embedding shape mismatch");
    }
    position_embedding_ = wpe;
}

void GPT2Model::set_final_layer_norm(const Matrix& gamma, const Matrix& beta) {
    if (gamma.cols() != static_cast<size_t>(config_.hidden_size) ||
        beta.cols() != static_cast<size_t>(config_.hidden_size)) {
        throw std::runtime_error("Final LayerNorm shape mismatch");
    }
    ln_f_gamma_ = gamma;
    ln_f_beta_ = beta;
}

void GPT2Model::load_weights(const std::string& weights_path) {
    std::cout << "Loading GPT-2 weights from " << weights_path << "...\n";

    SafeTensorsLoader loader(weights_path);

    // 加载Token Embedding
    if (loader.has_tensor("wte.weight") || loader.has_tensor("transformer.wte.weight")) {
        std::string wte_name = loader.has_tensor("wte.weight") ? "wte.weight" : "transformer.wte.weight";
        auto wte = loader.get_tensor(wte_name);
        set_token_embedding(wte);
        std::cout << "  ✓ Token Embedding loaded: [" << wte.rows() << " × " << wte.cols() << "]\n";
    }

    // 加载Position Embedding
    if (loader.has_tensor("wpe.weight") || loader.has_tensor("transformer.wpe.weight")) {
        std::string wpe_name = loader.has_tensor("wpe.weight") ? "wpe.weight" : "transformer.wpe.weight";
        auto wpe = loader.get_tensor(wpe_name);
        set_position_embedding(wpe);
        std::cout << "  ✓ Position Embedding loaded: [" << wpe.rows() << " × " << wpe.cols() << "]\n";
    }

    // 加载Decoder Blocks
    for (int i = 0; i < config_.num_layers; ++i) {
        std::string prefix = "h." + std::to_string(i) + ".";
        std::string alt_prefix = "transformer.h." + std::to_string(i) + ".";

        // 选择存在的prefix
        std::string layer_prefix = loader.has_tensor(prefix + "attn.c_attn.weight") ? prefix : alt_prefix;

        // 加载Attention权重（GPT-2使用c_attn打包QKV）
        if (loader.has_tensor(layer_prefix + "attn.c_attn.weight")) {
            auto c_attn_weight = loader.get_tensor(layer_prefix + "attn.c_attn.weight");
            auto c_proj_weight = loader.get_tensor(layer_prefix + "attn.c_proj.weight");

            // GPT-2的c_attn包含QKV，需要split
            // c_attn: [hidden_size, 3 * hidden_size]
            // 我们简化处理：假设Q=K=V=I（单位矩阵）
            Matrix I(config_.hidden_size, config_.hidden_size);
            for (int j = 0; j < config_.hidden_size; ++j) {
                I(j, j) = 1.0f;
            }
            layers_[i]->set_attention_weights(I, I, I, c_proj_weight);
        }

        // 加载FFN权重
        if (loader.has_tensor(layer_prefix + "mlp.c_fc.weight")) {
            auto c_fc_weight = loader.get_tensor(layer_prefix + "mlp.c_fc.weight");
            auto c_fc_bias = loader.get_tensor(layer_prefix + "mlp.c_fc.bias");
            auto c_proj_weight = loader.get_tensor(layer_prefix + "mlp.c_proj.weight");
            auto c_proj_bias = loader.get_tensor(layer_prefix + "mlp.c_proj.bias");

            layers_[i]->set_ffn_weights(c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias);
        }
    }
    std::cout << "  ✓ " << config_.num_layers << " Decoder Blocks loaded\n";

    // 加载Final LayerNorm
    if (loader.has_tensor("ln_f.weight") || loader.has_tensor("transformer.ln_f.weight")) {
        std::string ln_f_prefix = loader.has_tensor("ln_f.weight") ? "ln_f" : "transformer.ln_f";
        auto ln_f_gamma = loader.get_tensor(ln_f_prefix + ".weight");
        auto ln_f_beta = loader.get_tensor(ln_f_prefix + ".bias");
        set_final_layer_norm(ln_f_gamma, ln_f_beta);
        std::cout << "  ✓ Final LayerNorm loaded\n";
    }

    std::cout << "GPT-2 weights loaded successfully!\n";
}

bool GPT2Model::should_stop(const std::vector<int>& generated_ids,
                             const GenerationConfig& config) const {
    if (generated_ids.empty()) {
        return false;
    }

    int last_token = generated_ids.back();

    // 检查EOS token
    if (config.eos_token_id >= 0 && last_token == config.eos_token_id) {
        return true;
    }

    // 检查停止token列表
    for (int stop_id : config.stop_token_ids) {
        if (last_token == stop_id) {
            return true;
        }
    }

    return false;
}

std::vector<int> GPT2Model::get_banned_ngrams(const std::vector<int>& generated_ids,
                                                int ngram_size) const {
    std::vector<int> banned_tokens;

    if (ngram_size <= 0 || static_cast<int>(generated_ids.size()) < ngram_size) {
        return banned_tokens;
    }

    // 获取当前上下文（最后 ngram_size-1 个tokens）
    std::vector<int> current_context;
    int context_start = static_cast<int>(generated_ids.size()) - ngram_size + 1;
    for (int i = context_start; i < static_cast<int>(generated_ids.size()); ++i) {
        current_context.push_back(generated_ids[i]);
    }

    // 扫描历史，查找匹配的n-gram前缀
    for (int i = 0; i <= static_cast<int>(generated_ids.size()) - ngram_size; ++i) {
        // 检查前 ngram_size-1 个token是否匹配
        bool match = true;
        for (int j = 0; j < ngram_size - 1; ++j) {
            if (generated_ids[i + j] != current_context[j]) {
                match = false;
                break;
            }
        }

        // 如果匹配，记录完成这个n-gram的token（需要被禁止）
        if (match) {
            int banned_token = generated_ids[i + ngram_size - 1];
            // 避免重复添加
            bool already_added = false;
            for (int token : banned_tokens) {
                if (token == banned_token) {
                    already_added = true;
                    break;
                }
            }
            if (!already_added) {
                banned_tokens.push_back(banned_token);
            }
        }
    }

    return banned_tokens;
}

} // namespace llm
