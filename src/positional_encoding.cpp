#include "llm_engine/positional_encoding.h"
#include <cmath>
#include <stdexcept>

namespace llm {

// ============================================================================
// PositionalEncoding 实现
// ============================================================================

/**
 * 构造函数实现
 *
 * 根据指定的类型初始化位置编码矩阵
 */
PositionalEncoding::PositionalEncoding(size_t max_seq_len,
                                       size_t embed_dim,
                                       Type type)
    : max_seq_len_(max_seq_len),
      embed_dim_(embed_dim),
      type_(type),
      encodings_(max_seq_len, embed_dim)  // 预分配矩阵空间
{
    // 参数验证
    if (max_seq_len == 0) {
        throw std::invalid_argument("max_seq_len must be greater than 0");
    }

    if (embed_dim == 0) {
        throw std::invalid_argument("embed_dim must be greater than 0");
    }

    // 正弦编码要求embed_dim是偶数（因为使用sin和cos配对）
    if (type == Type::SINUSOIDAL && embed_dim % 2 != 0) {
        throw std::invalid_argument(
            "embed_dim must be even for sinusoidal positional encoding"
        );
    }

    // 根据类型初始化位置编码
    if (type_ == Type::SINUSOIDAL) {
        initialize_sinusoidal();
    } else {
        initialize_learned();
    }
}

/**
 * 正弦位置编码初始化
 *
 * 使用正弦和余弦函数生成位置编码。这是原始Transformer论文的方法。
 *
 * 公式详解：
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * 其中：
 *   - pos: 位置索引（0, 1, 2, ...）
 *   - i: 维度索引（0, 1, 2, ..., d_model/2-1）
 *   - 2i: 偶数维度使用sin
 *   - 2i+1: 奇数维度使用cos
 *
 * 为什么这样设计？
 * 1. 不同维度使用不同频率的波：
 *    - 低维度（i小）：高频波，快速变化，捕获局部位置信息
 *    - 高维度（i大）：低频波，缓慢变化，捕获全局位置信息
 *
 * 2. sin和cos的配对：
 *    - 提供相位信息
 *    - 任意位置的编码都是独特的
 *    - 位置之间的相对距离可以通过点积计算
 *
 * 3. 10000作为基数：
 *    - 经验值，提供合适的波长范围
 *    - 让最长波长约为2π * 10000 ≈ 62832
 *    - 足以处理常见的序列长度
 */
void PositionalEncoding::initialize_sinusoidal() {
    const float base = 10000.0f;  // 基数，控制波长范围

    // 遍历每个位置
    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        // 遍历每个维度（步长为2，因为sin和cos成对）
        for (size_t i = 0; i < embed_dim_ / 2; ++i) {
            // 计算角频率：pos / base^(2i/d_model)
            // 这里使用exp和log来计算幂：base^x = exp(x * log(base))
            float angle = static_cast<float>(pos) /
                         std::pow(base, 2.0f * i / embed_dim_);

            // 偶数维度（2i）：使用sin
            encodings_(pos, 2 * i) = std::sin(angle);

            // 奇数维度（2i+1）：使用cos
            encodings_(pos, 2 * i + 1) = std::cos(angle);
        }
    }

    // 注意：正弦编码是固定的，不需要训练
    // 但它的一个优点是可以外推到训练时未见过的更长序列
}

/**
 * 可学习位置编码初始化
 *
 * BERT使用这种方法：为每个位置学习一个嵌入向量
 *
 * 工作原理：
 * 1. 为每个位置创建一个可学习的向量
 * 2. 这些向量在训练过程中更新（类似word embedding）
 * 3. 模型可以学习到任务相关的位置模式
 *
 * 优点：
 * - 灵活性强，可以学习到复杂的位置关系
 * - 在有充足训练数据时效果通常优于固定编码
 *
 * 缺点：
 * - 无法处理超过max_seq_len的序列
 * - 需要更多训练数据
 */
void PositionalEncoding::initialize_learned() {
    // 使用小的随机值初始化
    // 标准差设为0.02，这是BERT的常用初始化策略
    encodings_ = Matrix::randn(max_seq_len_, embed_dim_, 0.0f, 0.02f);

    // 在实际应用中，这些权重会：
    // 1. 在训练时通过反向传播更新
    // 2. 或者从预训练模型加载
}

/**
 * 前向传播：添加位置编码
 *
 * 将位置编码加到输入上。这让模型能够感知token的位置信息。
 *
 * @param input 输入矩阵 [seq_len, embed_dim]
 * @return 添加位置编码后的结果，形状相同
 */
Matrix PositionalEncoding::forward(const Matrix& input) {
    size_t seq_len = input.rows();
    size_t input_dim = input.cols();

    // 验证输入维度
    if (input_dim != embed_dim_) {
        throw std::runtime_error(
            "Input embedding dimension (" + std::to_string(input_dim) +
            ") does not match positional encoding dimension (" +
            std::to_string(embed_dim_) + ")"
        );
    }

    // 验证序列长度
    if (seq_len > max_seq_len_) {
        throw std::runtime_error(
            "Input sequence length (" + std::to_string(seq_len) +
            ") exceeds maximum sequence length (" +
            std::to_string(max_seq_len_) + ")"
        );
    }

    // 创建输出矩阵
    Matrix output = input;  // 复制输入

    // 添加位置编码
    // 对每个位置，将对应的位置编码向量加到输入向量上
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t dim = 0; dim < embed_dim_; ++dim) {
            // output[pos, dim] = input[pos, dim] + encodings[pos, dim]
            output(pos, dim) += encodings_(pos, dim);
        }
    }

    return output;
}

/**
 * 获取指定位置的编码向量
 *
 * 用于调试或可视化位置编码
 *
 * @param position 位置索引
 * @return 该位置的编码向量 [1, embed_dim]
 */
Matrix PositionalEncoding::get_encoding(size_t position) const {
    if (position >= max_seq_len_) {
        throw std::out_of_range(
            "Position " + std::to_string(position) +
            " is out of range (max: " + std::to_string(max_seq_len_) + ")"
        );
    }

    // 提取指定位置的编码行
    Matrix encoding(1, embed_dim_);
    for (size_t dim = 0; dim < embed_dim_; ++dim) {
        encoding(0, dim) = encodings_(position, dim);
    }

    return encoding;
}

} // namespace llm
