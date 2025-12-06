#include "llm_engine/causal_attention.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace llm {

CausalAttention::CausalAttention(int hidden_size, int num_heads)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      head_dim_(hidden_size / num_heads),
      Wq_(hidden_size, hidden_size),
      Wk_(hidden_size, hidden_size),
      Wv_(hidden_size, hidden_size),
      Wo_(hidden_size, hidden_size)
{
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument(
            "hidden_size must be divisible by num_heads"
        );
    }

    std::cout << "CausalAttention initialized:" << std::endl;
    std::cout << "  hidden_size: " << hidden_size_ << std::endl;
    std::cout << "  num_heads: " << num_heads_ << std::endl;
    std::cout << "  head_dim: " << head_dim_ << std::endl;
}

void CausalAttention::set_weights(
    const Matrix& Wq,
    const Matrix& Wk,
    const Matrix& Wv,
    const Matrix& Wo
) {
    Wq_ = Wq;
    Wk_ = Wk;
    Wv_ = Wv;
    Wo_ = Wo;
}

Matrix CausalAttention::apply_causal_mask(const Matrix& scores) const {
    size_t seq_len = scores.rows();
    Matrix masked = scores;

    // 将上三角（不包括对角线）设为负无穷
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            masked(i, j) = -std::numeric_limits<float>::infinity();
        }
    }

    return masked;
}

Matrix CausalAttention::softmax(const Matrix& x) const {
    Matrix result(x.rows(), x.cols());

    for (size_t i = 0; i < x.rows(); ++i) {
        // 找到该行的最大值（数值稳定性）
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < x.cols(); ++j) {
            if (std::isfinite(x(i, j))) {  // 跳过-inf
                max_val = std::max(max_val, x(i, j));
            }
        }

        // 计算exp(x - max)的和
        float sum = 0.0f;
        for (size_t j = 0; j < x.cols(); ++j) {
            if (std::isfinite(x(i, j))) {
                result(i, j) = std::exp(x(i, j) - max_val);
                sum += result(i, j);
            } else {
                result(i, j) = 0.0f;  // exp(-inf) = 0
            }
        }

        // 归一化
        if (sum > 0) {
            for (size_t j = 0; j < x.cols(); ++j) {
                result(i, j) /= sum;
            }
        }
    }

    return result;
}

std::vector<Matrix> CausalAttention::split_heads(const Matrix& x) const {
    size_t seq_len = x.rows();
    std::vector<Matrix> heads;
    heads.reserve(num_heads_);

    for (int h = 0; h < num_heads_; ++h) {
        Matrix head(seq_len, head_dim_);
        for (size_t i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim_; ++j) {
                head(i, j) = x(i, h * head_dim_ + j);
            }
        }
        heads.push_back(head);
    }

    return heads;
}

Matrix CausalAttention::merge_heads(const std::vector<Matrix>& heads) const {
    if (heads.empty()) {
        throw std::invalid_argument("heads cannot be empty");
    }

    size_t seq_len = heads[0].rows();
    Matrix result(seq_len, hidden_size_);

    for (int h = 0; h < num_heads_; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim_; ++j) {
                result(i, h * head_dim_ + j) = heads[h](i, j);
            }
        }
    }

    return result;
}

Matrix CausalAttention::single_head_attention(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V,
    bool use_causal_mask
) const {
    // Q: [seq_len_q, head_dim]
    // K: [seq_len_k, head_dim]
    // V: [seq_len_v, head_dim]

    size_t seq_len_q = Q.rows();
    size_t seq_len_k = K.rows();

    // 计算attention scores: Q * K^T / sqrt(d_k)
    Matrix scores(seq_len_q, seq_len_k);
    for (size_t i = 0; i < seq_len_q; ++i) {
        for (size_t j = 0; j < seq_len_k; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim_; ++k) {
                sum += Q(i, k) * K(j, k);
            }
            scores(i, j) = sum / std::sqrt(static_cast<float>(head_dim_));
        }
    }

    // 应用causal mask（如果需要）
    if (use_causal_mask && seq_len_q == seq_len_k) {
        scores = apply_causal_mask(scores);
    }

    // Softmax
    Matrix attn_weights = softmax(scores);

    // 加权求和: attn_weights * V
    Matrix output(seq_len_q, head_dim_);
    for (size_t i = 0; i < seq_len_q; ++i) {
        for (int k = 0; k < head_dim_; ++k) {
            float sum = 0.0f;
            for (size_t j = 0; j < seq_len_k; ++j) {
                sum += attn_weights(i, j) * V(j, k);
            }
            output(i, k) = sum;
        }
    }

    return output;
}

Matrix CausalAttention::forward(const Matrix& input) const {
    // 计算Q, K, V
    Matrix Q = input.matmul(Wq_);
    Matrix K = input.matmul(Wk_);
    Matrix V = input.matmul(Wv_);

    // 分割为多头
    std::vector<Matrix> Q_heads = split_heads(Q);
    std::vector<Matrix> K_heads = split_heads(K);
    std::vector<Matrix> V_heads = split_heads(V);

    // 每个头独立计算attention
    std::vector<Matrix> output_heads;
    output_heads.reserve(num_heads_);
    for (int h = 0; h < num_heads_; ++h) {
        output_heads.push_back(single_head_attention(
            Q_heads[h],
            K_heads[h],
            V_heads[h],
            true  // 使用causal mask
        ));
    }

    // 合并多头
    Matrix merged = merge_heads(output_heads);

    // Output投影
    Matrix output = merged.matmul(Wo_);

    return output;
}

Matrix CausalAttention::forward_with_cache(
    const Matrix& input,
    MultiHeadKVCache& cache
) const {
    // input: [1, hidden_size] 单个新token

    if (input.rows() != 1) {
        throw std::invalid_argument(
            "forward_with_cache expects single token input [1, hidden_size]"
        );
    }

    // 计算新token的Q, K, V
    Matrix Q = input.matmul(Wq_);
    Matrix K = input.matmul(Wk_);
    Matrix V = input.matmul(Wv_);

    // 分割为多头
    std::vector<Matrix> Q_heads = split_heads(Q);
    std::vector<Matrix> K_heads = split_heads(K);
    std::vector<Matrix> V_heads = split_heads(V);

    // 每个头独立计算
    std::vector<Matrix> output_heads;
    output_heads.reserve(num_heads_);

    for (int h = 0; h < num_heads_; ++h) {
        // 将新的K, V追加到cache
        cache[h].append(K_heads[h], V_heads[h]);

        // 用新Q和全部历史KV计算attention
        // Q: [1, head_dim]
        // K_cache: [seq_len, head_dim]
        // V_cache: [seq_len, head_dim]

        size_t seq_len = cache[h].seq_len();

        // 计算scores: Q * K_cache^T
        Matrix scores(1, seq_len);
        for (size_t j = 0; j < seq_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim_; ++k) {
                sum += Q_heads[h](0, k) * cache[h].key_cache(j, k);
            }
            scores(0, j) = sum / std::sqrt(static_cast<float>(head_dim_));
        }

        // 不需要causal mask（新token自然只能看到历史）

        // Softmax
        Matrix attn_weights = softmax(scores);

        // 加权求和: attn_weights * V_cache
        Matrix head_output(1, head_dim_);
        for (int k = 0; k < head_dim_; ++k) {
            float sum = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                sum += attn_weights(0, j) * cache[h].value_cache(j, k);
            }
            head_output(0, k) = sum;
        }

        output_heads.push_back(head_output);
    }

    // 合并多头
    Matrix merged = merge_heads(output_heads);

    // Output投影
    Matrix output = merged.matmul(Wo_);

    return output;
}

} // namespace llm
