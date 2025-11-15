#include "llm_engine/transformer_block.h"
#include <stdexcept>

namespace llm {

// ============================================================================
// TransformerBlock 实现
// ============================================================================

/**
 * 构造函数实现
 *
 * 初始化Transformer Block的所有组件。
 * 这里使用成员初始化列表来构造各个子组件。
 */
TransformerBlock::TransformerBlock(size_t embed_dim,
                                   size_t num_heads,
                                   size_t ffn_dim,
                                   float dropout_rate)
    : ln1_(embed_dim),              // 第一个LayerNorm，归一化维度为embed_dim
      attention_(embed_dim, num_heads), // Self-Attention层
      ln2_(embed_dim),              // 第二个LayerNorm
      ffn_(embed_dim, ffn_dim),     // FeedForward层
      dropout_rate_(dropout_rate)   // Dropout比例（当前未使用）
{
    // 验证参数合法性
    if (embed_dim == 0) {
        throw std::invalid_argument("embed_dim must be greater than 0");
    }

    if (num_heads == 0) {
        throw std::invalid_argument("num_heads must be greater than 0");
    }

    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "embed_dim must be divisible by num_heads for multi-head attention"
        );
    }

    if (ffn_dim == 0) {
        throw std::invalid_argument("ffn_dim must be greater than 0");
    }
}

/**
 * 前向传播实现
 *
 * 这是Transformer Block的核心函数，实现了标准的两个子层结构：
 *
 * 子层1：Self-Attention
 *   1. LayerNorm(x)
 *   2. Self-Attention
 *   3. 残差连接: x = x + Attention(LayerNorm(x))
 *
 * 子层2：Feed-Forward
 *   1. LayerNorm(x)
 *   2. FeedForward
 *   3. 残差连接: x = x + FFN(LayerNorm(x))
 *
 * 注意：这里使用"Pre-LN"（Pre-LayerNorm）架构，即在子层之前应用LayerNorm。
 *      这与原始Transformer论文的"Post-LN"不同，但在实践中训练更稳定。
 *
 * @param input 输入矩阵 [seq_len, embed_dim] 或 [batch*seq_len, embed_dim]
 * @return 输出矩阵，形状与输入相同
 */
Matrix TransformerBlock::forward(const Matrix& input) {
    // -------------------------------------------------------------------------
    // 子层1：Multi-Head Self-Attention
    // -------------------------------------------------------------------------

    // 步骤1：保存输入，用于残差连接
    // 残差连接让梯度可以直接流过，避免梯度消失
    Matrix residual1 = input;

    // 步骤2：应用第一个LayerNorm
    // LayerNorm将每个样本的特征归一化为均值0、方差1
    // 这有助于稳定训练过程
    Matrix normed1 = ln1_.forward(input);

    // 步骤3：应用Self-Attention
    // Self-Attention让序列中的每个位置都能关注到其他所有位置
    // 这是Transformer捕获长距离依赖的关键机制
    Matrix attn_output = attention_.forward(normed1);

    // 步骤4：残差连接
    // output = input + Attention(LayerNorm(input))
    // 这让网络可以选择保留原始信息或使用新的表示
    Matrix output1 = add_residual(attn_output, residual1);

    // -------------------------------------------------------------------------
    // 子层2：Feed-Forward Network
    // -------------------------------------------------------------------------

    // 步骤5：保存子层1的输出，用于第二个残差连接
    Matrix residual2 = output1;

    // 步骤6：应用第二个LayerNorm
    Matrix normed2 = ln2_.forward(output1);

    // 步骤7：应用Feed-Forward Network
    // FFN包含两个线性变换和一个非线性激活（GELU）
    // 结构：Linear -> GELU -> Linear
    // 通常中间层维度是输入的4倍（如768 -> 3072 -> 768）
    // 这增加了模型的非线性表达能力
    Matrix ffn_output = ffn_.forward(normed2);

    // 步骤8：第二个残差连接
    // output = output1 + FFN(LayerNorm(output1))
    Matrix final_output = add_residual(ffn_output, residual2);

    // 返回最终输出
    // 经过两个子层的处理，输入的表示得到了增强：
    // - Self-Attention整合了序列信息
    // - FFN增加了非线性变换能力
    return final_output;
}

/**
 * 前向传播（带注意力掩码）
 *
 * 当前简化版本：直接调用无掩码版本
 * 未来可以在Self-Attention中实现真正的掩码功能
 *
 * 掩码的作用：
 * - 屏蔽padding位置（填充的无效token）
 * - 在decoder中实现因果掩码（防止看到未来信息）
 *
 * @param input 输入矩阵
 * @param attention_mask 注意力掩码（当前未使用）
 * @return 输出矩阵
 */
Matrix TransformerBlock::forward(const Matrix& input,
                                 const Matrix& attention_mask) {
    // TODO: 在SelfAttention中实现真正的掩码支持
    // 目前忽略attention_mask，直接调用基础版本
    (void)attention_mask; // 避免未使用参数警告
    return forward(input);
}

/**
 * 残差连接辅助函数
 *
 * 执行逐元素相加：output = input + residual
 *
 * 残差连接的重要性：
 * 1. 梯度流动：让梯度可以直接通过，避免梯度消失
 * 2. 信息保留：允许网络保留原始信息
 * 3. 深度训练：使得训练非常深的网络（如BERT的24层）成为可能
 * 4. 学习增量：网络只需学习"增量"而不是完整的变换
 *
 * @param input 主路径的输出
 * @param residual 残差路径（原始输入）
 * @return 相加后的结果
 */
Matrix TransformerBlock::add_residual(const Matrix& input,
                                      const Matrix& residual) const {
    // 验证维度匹配
    if (input.rows() != residual.rows() || input.cols() != residual.cols()) {
        throw std::runtime_error(
            "Dimension mismatch in residual connection: input (" +
            std::to_string(input.rows()) + "x" + std::to_string(input.cols()) +
            ") vs residual (" +
            std::to_string(residual.rows()) + "x" + std::to_string(residual.cols()) + ")"
        );
    }

    // 执行逐元素相加
    // 使用Matrix类重载的 + 操作符
    return input + residual;
}

} // namespace llm
