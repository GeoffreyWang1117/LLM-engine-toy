#include "llm_engine/gpt_decoder_block.h"
#include <cmath>
#include <iostream>

namespace llm {

GPTDecoderBlock::GPTDecoderBlock(
    int hidden_size,
    int num_heads,
    int intermediate_size,
    float dropout
)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      intermediate_size_(intermediate_size),
      dropout_(dropout),
      eps_(1e-5f),
      ln1_gamma_(1, hidden_size, 1.0f),
      ln1_beta_(1, hidden_size, 0.0f),
      ln2_gamma_(1, hidden_size, 1.0f),
      ln2_beta_(1, hidden_size, 0.0f),
      ffn_w1_(hidden_size, intermediate_size),
      ffn_b1_(1, intermediate_size, 0.0f),
      ffn_w2_(intermediate_size, hidden_size),
      ffn_b2_(1, hidden_size, 0.0f)
{
    // 创建Causal Attention
    attention_ = std::make_unique<CausalAttention>(hidden_size, num_heads);

    std::cout << "GPTDecoderBlock initialized:" << std::endl;
    std::cout << "  hidden_size: " << hidden_size_ << std::endl;
    std::cout << "  num_heads: " << num_heads_ << std::endl;
    std::cout << "  intermediate_size: " << intermediate_size_ << std::endl;
    std::cout << "  dropout: " << dropout_ << std::endl;
}

void GPTDecoderBlock::set_ln1_weights(const Matrix& gamma, const Matrix& beta) {
    ln1_gamma_ = gamma;
    ln1_beta_ = beta;
}

void GPTDecoderBlock::set_ln2_weights(const Matrix& gamma, const Matrix& beta) {
    ln2_gamma_ = gamma;
    ln2_beta_ = beta;
}

void GPTDecoderBlock::set_attention_weights(
    const Matrix& Wq,
    const Matrix& Wk,
    const Matrix& Wv,
    const Matrix& Wo
) {
    attention_->set_weights(Wq, Wk, Wv, Wo);
}

void GPTDecoderBlock::set_ffn_weights(
    const Matrix& W1,
    const Matrix& b1,
    const Matrix& W2,
    const Matrix& b2
) {
    ffn_w1_ = W1;
    ffn_b1_ = b1;
    ffn_w2_ = W2;
    ffn_b2_ = b2;
}

float GPTDecoderBlock::gelu(float x) const {
    // GELU近似：0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

Matrix GPTDecoderBlock::gelu(const Matrix& x) const {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            result(i, j) = gelu(x(i, j));
        }
    }
    return result;
}

Matrix GPTDecoderBlock::layer_norm(
    const Matrix& x,
    const Matrix& gamma,
    const Matrix& beta
) const {
    size_t seq_len = x.rows();
    size_t hidden_size = x.cols();
    Matrix result(seq_len, hidden_size);

    // 对每一行（每个token）分别归一化
    for (size_t i = 0; i < seq_len; ++i) {
        // 计算mean
        float mean = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            mean += x(i, j);
        }
        mean /= hidden_size;

        // 计算variance
        float var = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float diff = x(i, j) - mean;
            var += diff * diff;
        }
        var /= hidden_size;

        // 归一化并应用gamma和beta
        float std_dev = std::sqrt(var + eps_);
        for (size_t j = 0; j < hidden_size; ++j) {
            float normalized = (x(i, j) - mean) / std_dev;
            result(i, j) = normalized * gamma(0, j) + beta(0, j);
        }
    }

    return result;
}

Matrix GPTDecoderBlock::feed_forward(const Matrix& x) const {
    // x: [seq_len, hidden_size]
    // W1: [hidden_size, intermediate_size]
    // b1: [1, intermediate_size]

    // 第一层: h = GELU(x * W1 + b1)
    Matrix h = x.matmul(ffn_w1_);  // [seq_len, intermediate_size]

    // 添加bias（广播）
    for (size_t i = 0; i < h.rows(); ++i) {
        for (size_t j = 0; j < h.cols(); ++j) {
            h(i, j) += ffn_b1_(0, j);
        }
    }

    // GELU激活
    h = gelu(h);

    // 第二层: y = h * W2 + b2
    Matrix y = h.matmul(ffn_w2_);  // [seq_len, hidden_size]

    // 添加bias
    for (size_t i = 0; i < y.rows(); ++i) {
        for (size_t j = 0; j < y.cols(); ++j) {
            y(i, j) += ffn_b2_(0, j);
        }
    }

    return y;
}

Matrix GPTDecoderBlock::forward(const Matrix& input) const {
    // Pre-LN架构：
    // x → LN → Attention → Add → LN → FFN → Add

    Matrix x = input;

    // 1. LayerNorm + Attention + Residual
    Matrix normalized1 = layer_norm(x, ln1_gamma_, ln1_beta_);
    Matrix attn_out = attention_->forward(normalized1);

    // Residual connection
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            x(i, j) += attn_out(i, j);
        }
    }

    // 2. LayerNorm + FFN + Residual
    Matrix normalized2 = layer_norm(x, ln2_gamma_, ln2_beta_);
    Matrix ffn_out = feed_forward(normalized2);

    // Residual connection
    Matrix output(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            output(i, j) = x(i, j) + ffn_out(i, j);
        }
    }

    return output;
}

Matrix GPTDecoderBlock::forward_with_cache(
    const Matrix& input,
    MultiHeadKVCache& cache
) const {
    // Pre-LN + KV Cache
    // input: [1, hidden_size]

    Matrix x = input;

    // 1. LayerNorm + Attention (with cache) + Residual
    Matrix normalized1 = layer_norm(x, ln1_gamma_, ln1_beta_);
    Matrix attn_out = attention_->forward_with_cache(normalized1, cache);

    // Residual
    for (size_t j = 0; j < x.cols(); ++j) {
        x(0, j) += attn_out(0, j);
    }

    // 2. LayerNorm + FFN + Residual
    Matrix normalized2 = layer_norm(x, ln2_gamma_, ln2_beta_);
    Matrix ffn_out = feed_forward(normalized2);

    // Residual
    Matrix output(1, x.cols());
    for (size_t j = 0; j < x.cols(); ++j) {
        output(0, j) = x(0, j) + ffn_out(0, j);
    }

    return output;
}

} // namespace llm
