#include "llm_engine/embedding.h"
#include <stdexcept>

namespace llm {

// ============================================================================
// Embedding 实现
// ============================================================================

/**
 * 构造函数实现
 *
 * 初始化嵌入矩阵，使用随机值
 */
Embedding::Embedding(size_t vocab_size, size_t embed_dim)
    : vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      weight_(vocab_size, embed_dim)  // 分配权重矩阵空间
{
    // 参数验证
    if (vocab_size == 0) {
        throw std::invalid_argument("vocab_size must be greater than 0");
    }

    if (embed_dim == 0) {
        throw std::invalid_argument("embed_dim must be greater than 0");
    }

    // 初始化嵌入权重
    // 使用正态分布随机初始化，均值0，标准差0.02
    // 这是BERT和大多数预训练模型的标准做法
    weight_ = Matrix::randn(vocab_size, embed_dim, 0.0f, 0.02f);

    /*
     * 为什么使用0.02的标准差？
     *
     * 1. 避免梯度爆炸/消失：
     *    - 如果初始值太大，会导致前向传播时激活值过大
     *    - 如果初始值太小，会导致梯度消失
     *    - 0.02是经验值，在多数情况下效果良好
     *
     * 2. 与LayerNorm配合：
     *    - BERT在embedding后会立即应用LayerNorm
     *    - 小的初始值让LayerNorm能够有效归一化
     *
     * 3. 与预训练权重一致：
     *    - 从Hugging Face或其他来源加载的权重通常也是这样初始化的
     *    - 保持一致性有助于权重加载和对比
     */
}

/**
 * 前向传播：查表获取嵌入
 *
 * 这是Embedding层的核心操作，将token IDs转换为dense vectors
 *
 * @param token_ids 输入的token ID序列
 * @return 嵌入矩阵 [seq_len, embed_dim]
 */
Matrix Embedding::forward(const std::vector<int>& token_ids) {
    size_t seq_len = token_ids.size();

    // 边界情况：空序列
    if (seq_len == 0) {
        throw std::invalid_argument("token_ids cannot be empty");
    }

    // 创建输出矩阵
    Matrix output(seq_len, embed_dim_);

    // 对每个token ID，查找对应的嵌入向量
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];

        // 验证token ID有效性
        validate_token_id(token_id);

        // 从权重矩阵中提取对应行
        // 这就是"查表"操作：weight_[token_id, :]
        for (size_t j = 0; j < embed_dim_; ++j) {
            output(i, j) = weight_(token_id, j);
        }
    }

    /*
     * 实现说明：
     *
     * 1. 这里是简单的索引操作，非常高效：O(seq_len * embed_dim)
     *
     * 2. 在GPU实现中，这通常是gather操作
     *
     * 3. 对于批处理（batch），我们需要：
     *    - 输入: [[id1, id2, ...], [id3, id4, ...], ...]  (batch_size个序列)
     *    - 输出: [batch_size, max_seq_len, embed_dim]
     *    当前简化版本一次处理一个序列
     *
     * 4. 反向传播时（训练）：
     *    - 梯度只会更新被使用的token的嵌入
     *    - 未出现的token的嵌入保持不变
     *    - 这是embedding的一个重要特性：稀疏更新
     */

    return output;
}

/**
 * 获取单个token的嵌入向量
 *
 * 用于调试、可视化或特殊用途
 *
 * @param token_id Token ID
 * @return 嵌入向量 [1, embed_dim]
 */
Matrix Embedding::get_embedding(int token_id) const {
    validate_token_id(token_id);

    // 提取对应的行
    Matrix embedding(1, embed_dim_);
    for (size_t j = 0; j < embed_dim_; ++j) {
        embedding(0, j) = weight_(token_id, j);
    }

    return embedding;
}

/**
 * 验证token ID的有效性
 *
 * 确保ID在有效范围内：[0, vocab_size)
 *
 * @param token_id 要验证的ID
 * @throws std::out_of_range 如果ID无效
 */
void Embedding::validate_token_id(int token_id) const {
    // 检查负数
    if (token_id < 0) {
        throw std::out_of_range(
            "Token ID must be non-negative, got " + std::to_string(token_id)
        );
    }

    // 检查是否超出词表大小
    if (static_cast<size_t>(token_id) >= vocab_size_) {
        throw std::out_of_range(
            "Token ID " + std::to_string(token_id) +
            " is out of range (vocab_size=" + std::to_string(vocab_size_) + ")"
        );
    }

    /*
     * 特殊token说明：
     *
     * BERT使用的特殊token：
     * - [PAD]: padding token，ID=0
     * - [UNK]: unknown token，ID=100
     * - [CLS]: 句子开始，ID=101
     * - [SEP]: 句子分隔，ID=102
     * - [MASK]: 掩码token（MLM任务），ID=103
     *
     * 这些特殊token的ID在词表中有固定位置
     * 它们的嵌入向量也会通过训练学习到特殊的语义
     */
}

} // namespace llm
