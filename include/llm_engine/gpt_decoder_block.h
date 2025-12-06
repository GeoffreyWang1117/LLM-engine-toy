#ifndef LLM_ENGINE_GPT_DECODER_BLOCK_H
#define LLM_ENGINE_GPT_DECODER_BLOCK_H

#include "matrix.h"
#include "causal_attention.h"
#include "kv_cache.h"
#include <memory>

namespace llm {

/**
 * @brief GPT Decoder Block (Pre-LayerNorm架构)
 *
 * GPT-2使用Pre-LN架构，与BERT的Post-LN不同：
 *
 * BERT (Post-LN):
 *   x → Attention → Add → LN → FFN → Add → LN
 *
 * GPT-2 (Pre-LN):
 *   x → LN → Attention → Add → LN → FFN → Add
 *
 * Pre-LN优势：
 * - 训练更稳定（梯度流更平滑）
 * - 不需要warmup
 * - 现代LLM都使用Pre-LN
 *
 * 完整流程：
 * ```
 * 输入: x [seq_len, hidden_size]
 *   ↓
 * LN1: normalized_x = LayerNorm(x)
 *   ↓
 * Attention: attn_out = CausalAttention(normalized_x, cache)
 *   ↓
 * Residual: x = x + attn_out
 *   ↓
 * LN2: normalized_x = LayerNorm(x)
 *   ↓
 * FFN: ffn_out = FFN(normalized_x)
 *   ↓
 * Residual: x = x + ffn_out
 *   ↓
 * 输出: x [seq_len, hidden_size]
 * ```
 */
class GPTDecoderBlock {
public:
    /**
     * @brief 构造函数
     *
     * @param hidden_size 隐藏层维度（如768）
     * @param num_heads 注意力头数（如12）
     * @param intermediate_size FFN中间层维度（如3072，通常是4*hidden_size）
     * @param dropout Dropout率（默认0.1，推理时不使用）
     */
    GPTDecoderBlock(
        int hidden_size,
        int num_heads,
        int intermediate_size,
        float dropout = 0.1f
    );

    /**
     * @brief 训练/推理模式：处理完整序列
     *
     * 用于：
     * - 训练
     * - 首次推理（处理prompt）
     *
     * @param input 输入 [seq_len, hidden_size]
     * @return 输出 [seq_len, hidden_size]
     */
    Matrix forward(const Matrix& input) const;

    /**
     * @brief 生成模式：使用KV Cache
     *
     * 用于：
     * - 自回归生成（逐token）
     *
     * @param input 新token输入 [1, hidden_size]
     * @param cache KV缓存
     * @return 输出 [1, hidden_size]
     */
    Matrix forward_with_cache(
        const Matrix& input,
        MultiHeadKVCache& cache
    ) const;

    /**
     * @brief 设置LayerNorm1权重
     */
    void set_ln1_weights(const Matrix& gamma, const Matrix& beta);

    /**
     * @brief 设置LayerNorm2权重
     */
    void set_ln2_weights(const Matrix& gamma, const Matrix& beta);

    /**
     * @brief 设置Attention权重
     */
    void set_attention_weights(
        const Matrix& Wq,
        const Matrix& Wk,
        const Matrix& Wv,
        const Matrix& Wo
    );

    /**
     * @brief 设置FFN权重
     */
    void set_ffn_weights(
        const Matrix& W1,
        const Matrix& b1,
        const Matrix& W2,
        const Matrix& b2
    );

    /**
     * @brief 获取隐藏层维度
     */
    int hidden_size() const { return hidden_size_; }

    /**
     * @brief 获取注意力头数
     */
    int num_heads() const { return num_heads_; }

    /**
     * @brief 获取FFN中间层维度
     */
    int intermediate_size() const { return intermediate_size_; }

private:
    int hidden_size_;
    int num_heads_;
    int intermediate_size_;
    float dropout_;
    float eps_;  // LayerNorm的epsilon

    // LayerNorm 1 (before attention)
    Matrix ln1_gamma_;  // [hidden_size]
    Matrix ln1_beta_;   // [hidden_size]

    // Causal Attention
    std::unique_ptr<CausalAttention> attention_;

    // LayerNorm 2 (before FFN)
    Matrix ln2_gamma_;  // [hidden_size]
    Matrix ln2_beta_;   // [hidden_size]

    // Feed Forward Network
    Matrix ffn_w1_;     // [hidden_size, intermediate_size]
    Matrix ffn_b1_;     // [intermediate_size]
    Matrix ffn_w2_;     // [intermediate_size, hidden_size]
    Matrix ffn_b2_;     // [hidden_size]

    /**
     * @brief LayerNorm
     *
     * @param x 输入 [seq_len, hidden_size]
     * @param gamma 缩放参数 [hidden_size]
     * @param beta 偏置参数 [hidden_size]
     * @return 归一化后的输出 [seq_len, hidden_size]
     *
     * 计算：
     * mean = sum(x) / n
     * var = sum((x - mean)^2) / n
     * y = (x - mean) / sqrt(var + eps) * gamma + beta
     */
    Matrix layer_norm(
        const Matrix& x,
        const Matrix& gamma,
        const Matrix& beta
    ) const;

    /**
     * @brief Feed Forward Network (GELU激活)
     *
     * @param x 输入 [seq_len, hidden_size]
     * @return 输出 [seq_len, hidden_size]
     *
     * 计算：
     * h = GELU(x * W1 + b1)
     * y = h * W2 + b2
     */
    Matrix feed_forward(const Matrix& x) const;

    /**
     * @brief GELU激活函数
     *
     * GELU(x) = x * Φ(x)
     * 其中Φ是标准正态分布的CDF
     *
     * 近似: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     */
    float gelu(float x) const;

    /**
     * @brief 向量化GELU
     */
    Matrix gelu(const Matrix& x) const;
};

} // namespace llm

#endif // LLM_ENGINE_GPT_DECODER_BLOCK_H
