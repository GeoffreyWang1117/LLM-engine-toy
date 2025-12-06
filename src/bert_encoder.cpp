#include "llm_engine/bert_encoder.h"
#include <iostream>

namespace llm {

// ============================================================================
// BertEncoder 实现
// ============================================================================

/**
 * 构造函数实现
 *
 * 根据配置创建多个TransformerBlock
 */
BertEncoder::BertEncoder(const BertConfig& config)
    : config_(config)
{
    // 首先验证配置的有效性
    config_.validate();

    // 预留空间，避免vector多次重新分配
    layers_.reserve(config_.num_hidden_layers);

    /*
     * 创建TransformerBlock层
     *
     * 每一层都是独立的TransformerBlock实例
     * - 有自己的权重参数
     * - 结构相同但参数不同（通过训练学习）
     *
     * 为什么每层参数不同？
     * - 如果所有层共享参数，模型退化为1层的重复应用
     * - 独立参数允许每层学习不同的变换
     * - 实践证明，不同层确实学到了不同的语言学知识
     */
    for (size_t i = 0; i < config_.num_hidden_layers; ++i) {
        // 使用emplace_back在vector中直接构造对象，避免拷贝
        layers_.emplace_back(
            config_.hidden_size,           // embedding维度
            config_.num_attention_heads,   // 注意力头数
            config_.intermediate_size,     // FFN中间层维度
            config_.hidden_dropout_prob    // dropout概率（当前未实现）
        );

        /*
         * 初始化说明：
         *
         * TransformerBlock的构造函数会：
         * 1. 创建LayerNorm层（使用随机初始化，实际会加载预训练权重）
         * 2. 创建SelfAttention层（Q、K、V、O投影矩阵随机初始化）
         * 3. 创建FeedForward层（两个Linear层随机初始化）
         *
         * 在实际使用时，这些随机权重会被预训练权重替换
         */
    }

    /*
     * 参数统计（以BERT-base为例）：
     *
     * 每个TransformerBlock包含：
     * - LayerNorm1: 768 * 2 = 1.5K 参数 (gamma和beta)
     * - Attention:
     *   - Q投影: 768 * 768 = 590K
     *   - K投影: 768 * 768 = 590K
     *   - V投影: 768 * 768 = 590K
     *   - O投影: 768 * 768 = 590K
     *   小计: 2.36M
     * - LayerNorm2: 1.5K
     * - FeedForward:
     *   - FC1: 768 * 3072 = 2.36M
     *   - FC2: 3072 * 768 = 2.36M
     *   小计: 4.72M
     *
     * 每层总计: 约7.09M参数
     * 12层总计: 约85M参数
     */

    std::cout << "BertEncoder initialized with "
              << config_.num_hidden_layers << " layers" << std::endl;
}

/**
 * 前向传播实现
 *
 * 核心功能：将输入依次通过所有TransformerBlock
 */
Matrix BertEncoder::forward(const Matrix& hidden_states) {
    /*
     * 数据流追踪：
     *
     * 输入: hidden_states [seq_len, hidden_size]
     *   例如：[128, 768] (128个token，每个768维)
     *
     * 第1层:
     *   input  [128, 768]
     *     ↓ TransformerBlock
     *   output [128, 768]  # 形状不变，但表示更丰富
     *
     * 第2层:
     *   input  [128, 768]  # 使用上一层的输出
     *     ↓ TransformerBlock
     *   output [128, 768]
     *
     * ...依此类推
     *
     * 第12层:
     *   input  [128, 768]
     *     ↓ TransformerBlock
     *   output [128, 768]
     *
     * 最终输出: [128, 768]
     */

    // 从输入开始
    Matrix output = hidden_states;

    // 逐层处理
    for (size_t i = 0; i < layers_.size(); ++i) {
        // 将当前层的输出作为下一层的输入
        // 这体现了"深度学习"的本质：层层变换
        output = layers_[i].forward(output);

        /*
         * 可选：打印中间层输出（用于调试）
         * 取消注释下面的代码可以看到每层的输出统计信息
         */
        // std::cout << "Layer " << i << " output shape: ["
        //           << output.rows() << ", " << output.cols() << "]"
        //           << std::endl;
    }

    return output;

    /*
     * 性能考虑：
     *
     * 1. 时间复杂度：
     *    - 每层的Attention: O(L² * d)，L是序列长度，d是hidden_size
     *    - 每层的FFN: O(L * d * i)，i是intermediate_size
     *    - N层总计: O(N * L * (L*d + d*i))
     *    - 对于BERT-base (N=12, L=512, d=768, i=3072):
     *      约12 * 512 * (512*768 + 768*3072) ≈ 27亿次运算
     *
     * 2. 空间复杂度：
     *    - 每层需要存储中间结果
     *    - 梯度反向传播时需要保存所有层的激活值
     *    - 总体: O(N * L * d)
     *
     * 3. 优化方向：
     *    - 使用梯度检查点（gradient checkpointing）减少内存
     *    - 混合精度训练（FP16）加速计算
     *    - 稀疏注意力减少L²的复杂度
     */
}

/**
 * 前向传播（带注意力掩码）
 *
 * 当前版本的简化实现：忽略掩码，直接调用基础版本
 * 未来可以在每个TransformerBlock中实现真正的掩码支持
 */
Matrix BertEncoder::forward(const Matrix& hidden_states,
                            const Matrix& attention_mask) {
    // TODO: 实现真正的注意力掩码
    // 需要在SelfAttention层中支持掩码参数
    (void)attention_mask;  // 避免未使用参数警告

    return forward(hidden_states);

    /*
     * 注意力掩码的作用：
     *
     * 1. 屏蔽Padding：
     *    句子长度不同时，短句用[PAD]填充
     *    掩码让attention不关注padding位置
     *
     *    示例：
     *    文本: "Hello world [PAD] [PAD]"
     *    掩码: [1, 1, 0, 0]
     *
     * 2. 因果掩码（Causal Mask）：
     *    在decoder中，防止看到未来信息
     *    当前token只能attend到过去的token
     *
     * 3. 实现方式：
     *    在softmax前，将mask=0的位置设为-inf
     *    softmax后这些位置的权重变为0
     */
}

/**
 * 获取指定层的输出
 *
 * 用于特征提取和分析
 */
Matrix BertEncoder::get_layer_output(size_t layer_idx,
                                     const Matrix& hidden_states) {
    if (layer_idx >= layers_.size()) {
        throw std::out_of_range(
            "Layer index " + std::to_string(layer_idx) +
            " out of range (num_layers=" + std::to_string(layers_.size()) + ")"
        );
    }

    // 从输入开始，逐层处理直到目标层
    Matrix output = hidden_states;

    for (size_t i = 0; i <= layer_idx; ++i) {
        output = layers_[i].forward(output);
    }

    return output;

    /*
     * 使用场景：
     *
     * 1. 不同层适合不同任务：
     *    - 层0-3: 词性标注、句法分析
     *    - 层4-8: 命名实体识别
     *    - 层9-11: 文本分类、情感分析
     *
     * 2. 特征融合：
     *    concat([layer_3_out, layer_7_out, layer_11_out])
     *    使用多层特征可能比只用最后一层更好
     *
     * 3. 模型蒸馏：
     *    训练小模型去模仿大模型的中间层表示
     */
}

/**
 * 获取所有层的输出
 *
 * 返回每一层的隐藏状态
 */
std::vector<Matrix> BertEncoder::get_all_layer_outputs(
    const Matrix& hidden_states) {

    std::vector<Matrix> all_outputs;
    all_outputs.reserve(layers_.size() + 1);  // +1 for input

    // 先加入输入（第0层）
    all_outputs.push_back(hidden_states);

    Matrix output = hidden_states;

    // 逐层处理并保存每层输出
    for (size_t i = 0; i < layers_.size(); ++i) {
        output = layers_[i].forward(output);
        all_outputs.push_back(output);
    }

    /*
     * 返回的vector包含：
     * - all_outputs[0]: 输入（embedding层输出）
     * - all_outputs[1]: 第1层TransformerBlock输出
     * - all_outputs[2]: 第2层TransformerBlock输出
     * - ...
     * - all_outputs[N]: 第N层TransformerBlock输出
     *
     * 总共N+1个矩阵
     */

    return all_outputs;

    /*
     * 应用示例：
     *
     * 1. 层表示分析：
     *    观察不同层如何编码语言信息
     *    研究发现浅层捕获语法，深层捕获语义
     *
     * 2. ELMo风格的特征：
     *    加权组合所有层：output = Σ(w_i * layer_i_out)
     *    权重w可以针对特定任务学习
     *
     * 3. 可视化：
     *    使用t-SNE降维可视化不同层的词向量分布
     *    观察语义空间的演化
     */
}

} // namespace llm
