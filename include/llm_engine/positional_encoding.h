#ifndef LLM_ENGINE_POSITIONAL_ENCODING_H
#define LLM_ENGINE_POSITIONAL_ENCODING_H

#include "matrix.h"
#include <vector>

namespace llm {

/**
 * @brief PositionalEncoding - 位置编码
 *
 * 为什么需要位置编码？
 * - Transformer的Self-Attention是置换不变的（permutation invariant）
 * - 即打乱序列顺序，Self-Attention的输出不变
 * - 但在NLP任务中，词序是重要的信息
 * - 位置编码为每个位置添加独特的信息，让模型能够区分不同位置
 *
 * 两种主流位置编码方法：
 *
 * 1. **正弦位置编码** (Sinusoidal) - 原始Transformer使用
 *    公式：
 *      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 *    优点：
 *    - 固定的、不需要学习
 *    - 可以外推到更长的序列
 *    - 不同维度的频率不同，能捕获多尺度信息
 *
 * 2. **可学习位置编码** (Learned) - BERT使用
 *    - 为每个位置学习一个嵌入向量
 *    - 类似word embedding
 *
 *    优点：
 *    - 可以学习到任务相关的位置信息
 *    - 在训练数据充足时效果更好
 *
 *    缺点：
 *    - 无法处理超过最大长度的序列
 *
 * 本实现同时支持两种方式
 */
class PositionalEncoding {
public:
    /**
     * @brief 位置编码类型
     */
    enum class Type {
        SINUSOIDAL,  // 正弦/余弦位置编码（固定）
        LEARNED      // 可学习的位置编码
    };

    /**
     * @brief 构造函数
     *
     * @param max_seq_len 最大序列长度，BERT通常使用512
     * @param embed_dim 嵌入维度，BERT-base使用768
     * @param type 位置编码类型，默认使用可学习编码（BERT风格）
     */
    PositionalEncoding(size_t max_seq_len,
                      size_t embed_dim,
                      Type type = Type::LEARNED);

    /**
     * @brief 添加位置编码到输入
     *
     * @param input 输入张量 [seq_len, embed_dim]
     * @return 添加位置编码后的张量，形状相同
     *
     * 操作：output = input + positional_encoding[:seq_len, :]
     */
    Matrix forward(const Matrix& input);

    /**
     * @brief 获取指定位置的编码
     *
     * @param position 位置索引（0-based）
     * @return 该位置的编码向量 [1, embed_dim]
     */
    Matrix get_encoding(size_t position) const;

    /**
     * @brief 获取完整的位置编码矩阵
     *
     * @return 位置编码矩阵 [max_seq_len, embed_dim]
     */
    const Matrix& get_encodings() const { return encodings_; }

    /**
     * @brief 访问可学习的权重（用于训练和权重加载）
     */
    Matrix& weight() { return encodings_; }
    const Matrix& weight() const { return encodings_; }

private:
    size_t max_seq_len_;  // 最大序列长度
    size_t embed_dim_;    // 嵌入维度
    Type type_;           // 编码类型

    Matrix encodings_;    // 位置编码矩阵 [max_seq_len, embed_dim]

    /**
     * @brief 初始化正弦位置编码
     *
     * 使用原始Transformer论文中的正弦/余弦公式
     */
    void initialize_sinusoidal();

    /**
     * @brief 初始化可学习位置编码
     *
     * 随机初始化，后续可通过训练或加载权重更新
     */
    void initialize_learned();
};

} // namespace llm

#endif // LLM_ENGINE_POSITIONAL_ENCODING_H
