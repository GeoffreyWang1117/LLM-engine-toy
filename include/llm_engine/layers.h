#ifndef LLM_ENGINE_LAYERS_H
#define LLM_ENGINE_LAYERS_H

#include "matrix.h"
#include <memory>

namespace llm {

// Linear层 (全连接层): y = xW^T + b
class Linear {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    // 前向传播
    Matrix forward(const Matrix& input) const;

    // 获取权重和偏置
    Matrix& weight() { return weight_; }
    Matrix& bias() { return bias_; }
    const Matrix& weight() const { return weight_; }
    const Matrix& bias() const { return bias_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    Matrix weight_;  // [out_features, in_features]
    Matrix bias_;    // [out_features]
};

// LayerNorm层 (层归一化)
class LayerNorm {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5f);

    // 前向传播
    Matrix forward(const Matrix& input) const;

    // 获取权重和偏置
    Matrix& weight() { return weight_; }
    Matrix& bias() { return bias_; }
    const Matrix& weight() const { return weight_; }
    const Matrix& bias() const { return bias_; }

private:
    size_t normalized_shape_;
    float eps_;
    Matrix weight_;  // [normalized_shape]
    Matrix bias_;    // [normalized_shape]
};

// Self-Attention层
class SelfAttention {
public:
    SelfAttention(size_t embed_dim, size_t num_heads);

    // 前向传播
    Matrix forward(const Matrix& input) const;

    // 获取Q、K、V投影层
    Linear& q_proj() { return q_proj_; }
    Linear& k_proj() { return k_proj_; }
    Linear& v_proj() { return v_proj_; }
    Linear& out_proj() { return out_proj_; }

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;

    Linear q_proj_;    // Query投影
    Linear k_proj_;    // Key投影
    Linear v_proj_;    // Value投影
    Linear out_proj_;  // 输出投影

    // 辅助函数
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V) const;
    Matrix softmax(const Matrix& input) const;
};

// Feed-Forward Network (FFN)
class FeedForward {
public:
    FeedForward(size_t embed_dim, size_t hidden_dim);

    // 前向传播
    Matrix forward(const Matrix& input) const;

    // 获取层
    Linear& fc1() { return fc1_; }
    Linear& fc2() { return fc2_; }

private:
    Linear fc1_;  // 第一个全连接层
    Linear fc2_;  // 第二个全连接层

    // 激活函数 (GELU)
    Matrix gelu(const Matrix& input) const;
};

} // namespace llm

#endif // LLM_ENGINE_LAYERS_H
