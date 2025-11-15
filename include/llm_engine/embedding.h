#ifndef LLM_ENGINE_EMBEDDING_H
#define LLM_ENGINE_EMBEDDING_H

#include "matrix.h"
#include <vector>

namespace llm {

/**
 * @brief Embedding - 嵌入层
 *
 * 将离散的token ID转换为连续的向量表示。
 *
 * 什么是Embedding？
 * - 在NLP中，词汇是离散的符号（如"cat", "dog", "run"）
 * - 神经网络需要连续的数值输入
 * - Embedding将每个token映射到一个高维向量空间
 * - 例如：token_id=1024 -> [0.12, -0.34, 0.56, ..., 0.23] (768维)
 *
 * 工作原理：
 * - 本质上是一个查找表（lookup table）
 * - 每个token ID对应嵌入矩阵的一行
 * - 前向传播就是索引操作：embedding_matrix[token_id]
 *
 * BERT中的Embedding：
 * BERT使用三种嵌入的和：
 * 1. Token Embedding: 词本身的语义
 * 2. Position Embedding: 位置信息
 * 3. Token Type Embedding: 句子标记（用于区分句子A和句子B）
 *
 * 例如，输入 "[CLS] Hello world [SEP]" 会被转换为：
 * - Token IDs:      [101,  7592,  2088,  102]
 * - Position IDs:   [0,    1,     2,     3]
 * - Token Type IDs: [0,    0,     0,     0]
 *
 * 最终嵌入 = Token_Emb + Position_Emb + TokenType_Emb
 */
class Embedding {
public:
    /**
     * @brief 构造函数
     *
     * @param vocab_size 词表大小
     *                   - BERT-base: 30522 (包括特殊token如[CLS], [SEP]等)
     *                   - GPT-2: 50257
     * @param embed_dim 嵌入维度
     *                   - BERT-base: 768
     *                   - BERT-large: 1024
     */
    Embedding(size_t vocab_size, size_t embed_dim);

    /**
     * @brief 前向传播：将token IDs转换为嵌入向量
     *
     * @param token_ids Token ID序列，例如 [101, 2023, 2003, 102]
     *                  其中每个ID在[0, vocab_size)范围内
     * @return 嵌入矩阵 [seq_len, embed_dim]
     *         其中seq_len = token_ids.size()
     *
     * 示例：
     *   输入: [1, 5, 10]  (3个token)
     *   输出: [[0.1, 0.2, ..., 0.8],    # token 1的嵌入
     *          [0.3, 0.1, ..., 0.5],    # token 5的嵌入
     *          [0.2, 0.4, ..., 0.7]]    # token 10的嵌入
     *   形状: [3, embed_dim]
     */
    Matrix forward(const std::vector<int>& token_ids);

    /**
     * @brief 获取单个token的嵌入
     *
     * @param token_id Token ID
     * @return 嵌入向量 [1, embed_dim]
     */
    Matrix get_embedding(int token_id) const;

    /**
     * @brief 访问嵌入权重矩阵
     *
     * 用于：
     * - 权重加载（从预训练模型）
     * - 权重保存
     * - 调试和可视化
     *
     * @return 嵌入矩阵 [vocab_size, embed_dim]
     */
    Matrix& weight() { return weight_; }
    const Matrix& weight() const { return weight_; }

    /**
     * @brief 获取词表大小
     */
    size_t vocab_size() const { return vocab_size_; }

    /**
     * @brief 获取嵌入维度
     */
    size_t embed_dim() const { return embed_dim_; }

private:
    size_t vocab_size_;  // 词表大小
    size_t embed_dim_;   // 嵌入维度

    /**
     * 嵌入权重矩阵 [vocab_size, embed_dim]
     *
     * 存储方式：
     * - 每一行对应一个token的嵌入向量
     * - weight_[i] 就是 token_id=i 的嵌入
     *
     * 初始化：
     * - 随机初始化（正态分布，均值0，标准差0.02）
     * - 在实际使用中会加载预训练权重
     */
    Matrix weight_;

    /**
     * @brief 验证token ID的有效性
     *
     * @param token_id 要验证的token ID
     * @throws std::out_of_range 如果ID超出范围
     */
    void validate_token_id(int token_id) const;
};

} // namespace llm

#endif // LLM_ENGINE_EMBEDDING_H
