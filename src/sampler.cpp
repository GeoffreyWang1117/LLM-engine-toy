#include "llm_engine/sampler.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace llm {

Sampler::Sampler(unsigned int seed) : rng_(seed) {}

void Sampler::set_seed(unsigned int seed) {
    rng_.seed(seed);
}

std::vector<float> Sampler::softmax(const Matrix& logits) {
    if (logits.rows() != 1) {
        throw std::invalid_argument("logits must be [1, vocab_size]");
    }

    size_t vocab_size = logits.cols();
    std::vector<float> probs(vocab_size);

    // 找到最大值（数值稳定性）
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits(0, i));
    }

    // 计算exp(x - max)的和
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits(0, i) - max_logit);
        sum += probs[i];
    }

    // 归一化
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    return probs;
}

int Sampler::sample_from_probs(const std::vector<float>& probs) {
    // 使用uniform分布采样
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);

    // 累积概率采样
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            return static_cast<int>(i);
        }
    }

    // 极端情况：返回最后一个
    return static_cast<int>(probs.size() - 1);
}

std::vector<int> Sampler::argsort_descending(const std::vector<float>& values) {
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
        [&values](int a, int b) {
            return values[a] > values[b];  // 降序
        }
    );

    return indices;
}

std::vector<int> Sampler::get_top_k_indices(const std::vector<float>& values, int k) {
    auto sorted_indices = argsort_descending(values);

    // 只保留top-k
    int actual_k = std::min(k, static_cast<int>(sorted_indices.size()));
    sorted_indices.resize(actual_k);

    return sorted_indices;
}

int Sampler::greedy_sample(const Matrix& logits) {
    if (logits.rows() != 1) {
        throw std::invalid_argument("logits must be [1, vocab_size]");
    }

    // 找到最大值的索引
    int max_idx = 0;
    float max_val = logits(0, 0);

    for (size_t i = 1; i < logits.cols(); ++i) {
        if (logits(0, i) > max_val) {
            max_val = logits(0, i);
            max_idx = static_cast<int>(i);
        }
    }

    return max_idx;
}

int Sampler::temperature_sample(const Matrix& logits, float temperature) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("temperature must be > 0");
    }

    if (logits.rows() != 1) {
        throw std::invalid_argument("logits must be [1, vocab_size]");
    }

    // 应用temperature：logits' = logits / temperature
    size_t vocab_size = logits.cols();
    Matrix scaled_logits(1, vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
        scaled_logits(0, i) = logits(0, i) / temperature;
    }

    // Softmax
    std::vector<float> probs = softmax(scaled_logits);

    // 采样
    return sample_from_probs(probs);
}

int Sampler::top_k_sample(const Matrix& logits, int k) {
    if (k <= 0) {
        throw std::invalid_argument("k must be > 0");
    }

    if (logits.rows() != 1) {
        throw std::invalid_argument("logits must be [1, vocab_size]");
    }

    // 特殊情况：k=1等价于greedy
    if (k == 1) {
        return greedy_sample(logits);
    }

    // 转换为vector
    size_t vocab_size = logits.cols();
    std::vector<float> logits_vec(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        logits_vec[i] = logits(0, i);
    }

    // 获取top-k索引
    auto top_k_indices = get_top_k_indices(logits_vec, k);

    // 创建top-k的logits
    Matrix top_k_logits(1, top_k_indices.size());
    for (size_t i = 0; i < top_k_indices.size(); ++i) {
        top_k_logits(0, i) = logits_vec[top_k_indices[i]];
    }

    // Softmax
    std::vector<float> probs = softmax(top_k_logits);

    // 采样
    int sampled_idx = sample_from_probs(probs);

    // 映射回原始索引
    return top_k_indices[sampled_idx];
}

int Sampler::top_p_sample(const Matrix& logits, float p) {
    if (p <= 0.0f || p > 1.0f) {
        throw std::invalid_argument("p must be in (0, 1]");
    }

    if (logits.rows() != 1) {
        throw std::invalid_argument("logits must be [1, vocab_size]");
    }

    // 计算概率
    std::vector<float> probs = softmax(logits);

    // 按概率降序排序
    auto sorted_indices = argsort_descending(probs);

    // 累积概率，找到nucleus
    std::vector<int> nucleus_indices;
    float cumsum = 0.0f;

    for (int idx : sorted_indices) {
        cumsum += probs[idx];
        nucleus_indices.push_back(idx);

        if (cumsum >= p) {
            break;
        }
    }

    // 至少保留一个token
    if (nucleus_indices.empty()) {
        nucleus_indices.push_back(sorted_indices[0]);
    }

    // 从nucleus中采样
    std::vector<float> nucleus_probs;
    for (int idx : nucleus_indices) {
        nucleus_probs.push_back(probs[idx]);
    }

    // 重新归一化
    float sum = 0.0f;
    for (float prob : nucleus_probs) {
        sum += prob;
    }
    for (float& prob : nucleus_probs) {
        prob /= sum;
    }

    // 采样
    int sampled_idx = sample_from_probs(nucleus_probs);

    // 映射回原始索引
    return nucleus_indices[sampled_idx];
}

Matrix Sampler::apply_repetition_penalty(const Matrix& logits,
                                          const std::vector<int>& generated_tokens,
                                          float penalty) const {
    if (penalty == 1.0f || generated_tokens.empty()) {
        return logits;  // 不应用penalty
    }

    Matrix penalized_logits = logits;

    // 对已生成的tokens应用penalty
    for (int token_id : generated_tokens) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < logits.cols()) {
            // 如果logit > 0，除以penalty（降低概率）
            // 如果logit < 0，乘以penalty（进一步降低概率）
            if (penalized_logits(0, token_id) > 0) {
                penalized_logits(0, token_id) /= penalty;
            } else {
                penalized_logits(0, token_id) *= penalty;
            }
        }
    }

    return penalized_logits;
}

int Sampler::sample(const Matrix& logits, const SamplingConfig& config,
                    const std::vector<int>& generated_tokens) {
    // 应用repetition penalty
    Matrix adjusted_logits = apply_repetition_penalty(
        logits, generated_tokens, config.repetition_penalty);

    // 根据策略采样
    switch (config.strategy) {
        case SamplingConfig::Strategy::GREEDY:
            return greedy_sample(adjusted_logits);

        case SamplingConfig::Strategy::TEMPERATURE:
            return temperature_sample(adjusted_logits, config.temp);

        case SamplingConfig::Strategy::TOP_K:
            return top_k_sample(adjusted_logits, config.k);

        case SamplingConfig::Strategy::TOP_P:
            return top_p_sample(adjusted_logits, config.p);

        default:
            throw std::runtime_error("Unknown sampling strategy");
    }
}

} // namespace llm
