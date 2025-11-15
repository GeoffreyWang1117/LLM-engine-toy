#ifndef LLM_ENGINE_TRANSFORMER_BLOCK_H
#define LLM_ENGINE_TRANSFORMER_BLOCK_H

#include "matrix.h"
#include "layers.h"

namespace llm {

/**
 * @brief TransformerBlock - 完整的Transformer编码器块
 *
 * 这是构成BERT/GPT等模型的基本单元。一个标准的Transformer Block包含：
 * 1. Multi-Head Self-Attention子层 + 残差连接 + LayerNorm
 * 2. Feed-Forward Network子层 + 残差连接 + LayerNorm
 *
 * 数据流：
 *   输入x
 *     ↓
 *   LayerNorm -> Self-Attention -> Add(x) [残差连接]
 *     ↓
 *   LayerNorm -> FeedForward -> Add(上一步输出) [残差连接]
 *     ↓
 *   输出
 *
 * 残差连接的作用：
 * - 让梯度能够直接流过，避免梯度消失
 * - 允许训练更深的网络（BERT有12-24层）
 * - 提供"跳过路径"，模型可以学习身份映射
 */
class TransformerBlock {
public:
    /**
     * @brief 构造函数
     *
     * @param embed_dim 嵌入维度（隐藏层大小），BERT-base使用768
     * @param num_heads 注意力头数，BERT-base使用12
     * @param ffn_dim Feed-Forward Network的中间层维度，通常是embed_dim的4倍
     *                BERT-base使用3072 (768 * 4)
     * @param dropout_rate Dropout比例（当前版本暂不实现dropout，预留参数）
     */
    TransformerBlock(size_t embed_dim,
                     size_t num_heads,
                     size_t ffn_dim,
                     float dropout_rate = 0.1f);

    /**
     * @brief 前向传播
     *
     * @param input 输入张量 [batch_size, seq_len, embed_dim]
     *              - batch_size: 批次大小
     *              - seq_len: 序列长度（token数量）
     *              - embed_dim: 每个token的特征维度
     *
     * @return 输出张量，形状与输入相同 [batch_size, seq_len, embed_dim]
     *
     * 注意：为了简化实现，当前版本将batch_size和seq_len合并为一个维度，
     *      实际输入形状为 [batch_size * seq_len, embed_dim]
     */
    Matrix forward(const Matrix& input);

    /**
     * @brief 前向传播（带注意力掩码）
     *
     * @param input 输入张量
     * @param attention_mask 注意力掩码 [seq_len, seq_len]
     *                       用于屏蔽某些位置（如padding tokens）
     *                       1.0表示保留，0.0表示屏蔽
     *
     * @return 输出张量
     *
     * 注意：当前简化版本暂不实现，预留接口
     */
    Matrix forward(const Matrix& input, const Matrix& attention_mask);

    // 访问内部组件（主要用于调试和权重加载）
    LayerNorm& ln1() { return ln1_; }
    SelfAttention& attention() { return attention_; }
    LayerNorm& ln2() { return ln2_; }
    FeedForward& ffn() { return ffn_; }

    const LayerNorm& ln1() const { return ln1_; }
    const SelfAttention& attention() const { return attention_; }
    const LayerNorm& ln2() const { return ln2_; }
    const FeedForward& ffn() const { return ffn_; }

private:
    // 第一个子层：Self-Attention
    LayerNorm ln1_;           // 注意力层前的LayerNorm
    SelfAttention attention_; // 多头自注意力

    // 第二个子层：Feed-Forward Network
    LayerNorm ln2_;           // FFN层前的LayerNorm
    FeedForward ffn_;         // 前馈神经网络

    // 超参数
    float dropout_rate_;      // Dropout比例（预留）

    /**
     * @brief 残差连接辅助函数
     *
     * 执行 output = input + residual
     * 确保维度匹配
     *
     * @param input 主路径的输出
     * @param residual 残差（原始输入）
     * @return 相加后的结果
     */
    Matrix add_residual(const Matrix& input, const Matrix& residual) const;
};

} // namespace llm

#endif // LLM_ENGINE_TRANSFORMER_BLOCK_H
