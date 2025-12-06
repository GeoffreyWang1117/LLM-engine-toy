#ifndef LLM_ENGINE_CAUSAL_ATTENTION_H
#define LLM_ENGINE_CAUSAL_ATTENTION_H

#include "matrix.h"
#include "kv_cache.h"
#include <cmath>

namespace llm {

/**
 * @brief Causal Self-Attention (GPT风格)
 *
 * 与BERT的双向注意力不同，Causal Attention使用因果mask：
 * - 每个token只能看到它自己和之前的token
 * - 不能"偷看"未来的token
 *
 * 关键特性：
 * 1. Causal Mask: 上三角设为-inf，防止看到未来
 * 2. KV Cache支持: 生成时只计算新token的Q，复用历史K和V
 * 3. Pre-LayerNorm: 配合GPT-2风格的Pre-LN架构
 *
 * Attention Matrix示例：
 * ```
 * Input: "The cat sat on"
 *
 * BERT (Bidirectional) - 全部可见:
 *       The  cat  sat  on
 * The   [1]  [1]  [1]  [1]
 * cat   [1]  [1]  [1]  [1]
 * sat   [1]  [1]  [1]  [1]
 * on    [1]  [1]  [1]  [1]
 *
 * GPT (Causal) - 只看过去:
 *       The  cat  sat  on
 * The   [1]  [0]  [0]  [0]  ← 只能看"The"
 * cat   [1]  [1]  [0]  [0]  ← 能看"The cat"
 * sat   [1]  [1]  [1]  [0]  ← 能看"The cat sat"
 * on    [1]  [1]  [1]  [1]  ← 能看全部历史
 * ```
 *
 * 使用场景：
 * - GPT-2: 文本生成
 * - LLaMA: 对话生成
 * - 所有decoder-only架构
 */
class CausalAttention {
public:
    /**
     * @brief 构造函数
     *
     * @param hidden_size 隐藏层维度（如768）
     * @param num_heads 注意力头数（如12）
     */
    CausalAttention(int hidden_size, int num_heads);

    /**
     * @brief 训练/推理模式：处理完整序列
     *
     * 用于：
     * - 训练时的forward pass
     * - 首次推理（处理prompt）
     *
     * @param input 输入矩阵 [seq_len, hidden_size]
     * @return 输出矩阵 [seq_len, hidden_size]
     *
     * 计算流程：
     * 1. 计算Q, K, V = input * Wq, Wk, Wv
     * 2. 分割为多头
     * 3. 计算attention scores
     * 4. 应用causal mask（上三角设-inf）
     * 5. Softmax + 加权求和
     * 6. 合并多头
     */
    Matrix forward(const Matrix& input) const;

    /**
     * @brief 生成模式：使用KV Cache增量计算
     *
     * 用于：
     * - 自回归生成（逐token生成）
     *
     * @param input 新token的输入 [1, hidden_size]
     * @param cache KV缓存（存储历史K和V）
     * @return 输出 [1, hidden_size]
     *
     * 性能优化：
     * - 只计算新token的Q, K, V
     * - 复用cache中的历史K和V
     * - 时间复杂度: O(n²) → O(n)
     * - 加速比: ~10x
     *
     * 计算流程：
     * 1. 计算新token的Q, K, V
     * 2. 将K, V追加到cache
     * 3. 用新Q和全部历史KV计算attention
     * 4. 返回新token的输出
     */
    Matrix forward_with_cache(
        const Matrix& input,
        MultiHeadKVCache& cache
    ) const;

    /**
     * @brief 设置权重矩阵
     *
     * @param Wq Query权重 [hidden_size, hidden_size]
     * @param Wk Key权重 [hidden_size, hidden_size]
     * @param Wv Value权重 [hidden_size, hidden_size]
     * @param Wo Output权重 [hidden_size, hidden_size]
     */
    void set_weights(
        const Matrix& Wq,
        const Matrix& Wk,
        const Matrix& Wv,
        const Matrix& Wo
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
     * @brief 获取每个头的维度
     */
    int head_dim() const { return head_dim_; }

private:
    int hidden_size_;  // 隐藏层维度
    int num_heads_;    // 注意力头数
    int head_dim_;     // 每个头的维度 = hidden_size / num_heads

    // 权重矩阵
    Matrix Wq_;  // Query投影 [hidden_size, hidden_size]
    Matrix Wk_;  // Key投影
    Matrix Wv_;  // Value投影
    Matrix Wo_;  // Output投影

    /**
     * @brief 应用因果mask（上三角设为-inf）
     *
     * @param scores Attention scores [seq_len, seq_len]
     * @return Masked scores
     *
     * 示例：
     * ```
     * Input scores (3x3):
     * [[1.0, 1.0, 1.0],
     *  [1.0, 1.0, 1.0],
     *  [1.0, 1.0, 1.0]]
     *
     * After causal mask:
     * [[  1.0, -inf, -inf],
     *  [  1.0,  1.0, -inf],
     *  [  1.0,  1.0,  1.0]]
     * ```
     */
    Matrix apply_causal_mask(const Matrix& scores) const;

    /**
     * @brief Softmax函数（支持-inf）
     *
     * softmax(-inf) = 0
     */
    Matrix softmax(const Matrix& x) const;

    /**
     * @brief 分割为多头
     *
     * @param x [seq_len, hidden_size]
     * @return [num_heads, seq_len, head_dim]
     */
    std::vector<Matrix> split_heads(const Matrix& x) const;

    /**
     * @brief 合并多头
     *
     * @param heads [num_heads, seq_len, head_dim]
     * @return [seq_len, hidden_size]
     */
    Matrix merge_heads(const std::vector<Matrix>& heads) const;

    /**
     * @brief 单头attention计算
     *
     * @param Q [seq_len_q, head_dim]
     * @param K [seq_len_k, head_dim]
     * @param V [seq_len_v, head_dim]
     * @param use_causal_mask 是否使用因果mask
     * @return [seq_len_q, head_dim]
     */
    Matrix single_head_attention(
        const Matrix& Q,
        const Matrix& K,
        const Matrix& V,
        bool use_causal_mask = true
    ) const;
};

} // namespace llm

#endif // LLM_ENGINE_CAUSAL_ATTENTION_H
