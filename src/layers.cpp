#include "llm_engine/layers.h"
#include <cmath>
#include <algorithm>

namespace llm {

// ============================================================================
// Linear Layer
// ============================================================================

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(out_features, in_features),
      bias_(use_bias ? out_features : 0, 1, 0.0f) {

    // 简单的随机初始化 (Xavier/Glorot初始化)
    float stddev = std::sqrt(2.0f / (in_features + out_features));
    weight_ = Matrix::randn(out_features, in_features, 0.0f, stddev);
}

Matrix Linear::forward(const Matrix& input) const {
    // input: [batch_size, in_features]
    // weight: [out_features, in_features]
    // output: [batch_size, out_features]

    // y = xW^T
    Matrix output = input.matmul(weight_.transpose());

    // 添加偏置
    if (use_bias_) {
        for (size_t i = 0; i < output.rows(); ++i) {
            for (size_t j = 0; j < output.cols(); ++j) {
                output(i, j) += bias_(j, 0);
            }
        }
    }

    return output;
}

// ============================================================================
// LayerNorm Layer
// ============================================================================

LayerNorm::LayerNorm(size_t normalized_shape, float eps)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(normalized_shape, 1, 1.0f),
      bias_(normalized_shape, 1, 0.0f) {
}

Matrix LayerNorm::forward(const Matrix& input) const {
    // input: [batch_size, normalized_shape]
    Matrix output(input.rows(), input.cols());

    for (size_t i = 0; i < input.rows(); ++i) {
        // 计算均值
        float mean = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            mean += input(i, j);
        }
        mean /= input.cols();

        // 计算方差
        float variance = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            float diff = input(i, j) - mean;
            variance += diff * diff;
        }
        variance /= input.cols();

        // 归一化
        float std = std::sqrt(variance + eps_);
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = (input(i, j) - mean) / std;
            output(i, j) = output(i, j) * weight_(j, 0) + bias_(j, 0);
        }
    }

    return output;
}

// ============================================================================
// SelfAttention Layer
// ============================================================================

SelfAttention::SelfAttention(size_t embed_dim, size_t num_heads)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      q_proj_(embed_dim, embed_dim),
      k_proj_(embed_dim, embed_dim),
      v_proj_(embed_dim, embed_dim),
      out_proj_(embed_dim, embed_dim) {

    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
}

Matrix SelfAttention::softmax(const Matrix& input) const {
    Matrix output(input.rows(), input.cols());

    for (size_t i = 0; i < input.rows(); ++i) {
        // 找到最大值（数值稳定性）
        float max_val = input(i, 0);
        for (size_t j = 1; j < input.cols(); ++j) {
            max_val = std::max(max_val, input(i, j));
        }

        // 计算exp和sum
        float sum = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = std::exp(input(i, j) - max_val);
            sum += output(i, j);
        }

        // 归一化
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) /= sum;
        }
    }

    return output;
}

Matrix SelfAttention::scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V) const {
    // Q, K, V: [seq_len, head_dim]
    // 计算 Q * K^T / sqrt(head_dim)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    Matrix scores = Q.matmul(K.transpose()) * scale;

    // Softmax
    Matrix attn_weights = softmax(scores);

    // 加权求和
    Matrix output = attn_weights.matmul(V);

    return output;
}

Matrix SelfAttention::forward(const Matrix& input) const {
    // input: [seq_len, embed_dim]
    size_t seq_len = input.rows();

    // 线性投影得到Q, K, V
    Matrix Q = q_proj_.forward(input);  // [seq_len, embed_dim]
    Matrix K = k_proj_.forward(input);  // [seq_len, embed_dim]
    Matrix V = v_proj_.forward(input);  // [seq_len, embed_dim]

    // 简化版本：不做多头拆分，直接计算attention
    // (完整实现需要将Q,K,V reshape成[num_heads, seq_len, head_dim])
    Matrix attn_output = scaled_dot_product_attention(Q, K, V);

    // 输出投影
    Matrix output = out_proj_.forward(attn_output);

    return output;
}

// ============================================================================
// FeedForward Layer
// ============================================================================

FeedForward::FeedForward(size_t embed_dim, size_t hidden_dim)
    : fc1_(embed_dim, hidden_dim),
      fc2_(hidden_dim, embed_dim) {
}

Matrix FeedForward::gelu(const Matrix& input) const {
    // GELU激活函数: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Matrix output(input.rows(), input.cols());

    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);

    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            float x = input(i, j);
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
            output(i, j) = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }

    return output;
}

Matrix FeedForward::forward(const Matrix& input) const {
    // input -> fc1 -> GELU -> fc2
    Matrix hidden = fc1_.forward(input);
    hidden = gelu(hidden);
    Matrix output = fc2_.forward(hidden);

    return output;
}

} // namespace llm
