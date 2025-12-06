#ifndef LLM_ENGINE_BERT_ENCODER_H
#define LLM_ENGINE_BERT_ENCODER_H

#include "bert_config.h"
#include "transformer_block.h"
#include "layers.h"
#include <vector>
#include <memory>

namespace llm {

/**
 * @brief BertEncoder - BERT的编码器部分
 *
 * BertEncoder由多个TransformerBlock堆叠而成，是BERT的核心结构。
 *
 * 架构：
 *   输入 [seq_len, hidden_size]
 *     ↓
 *   TransformerBlock 1
 *     ↓
 *   TransformerBlock 2
 *     ↓
 *   ...
 *     ↓
 *   TransformerBlock N
 *     ↓
 *   输出 [seq_len, hidden_size]
 *
 * 每一层的作用：
 * - 第1层：捕获局部依赖和浅层特征
 * - 中间层：逐步构建更抽象的表示
 * - 最后层：高层语义特征
 *
 * 为什么需要多层？
 * 1. **层次化表示**：浅层捕获语法，深层捕获语义
 * 2. **表达能力**：更多层 = 更强的非线性变换能力
 * 3. **任务适应性**：不同任务可能需要不同深度的特征
 *
 * 实践经验：
 * - 12层对大多数NLP任务已经足够
 * - 24层在大规模预训练时能学到更好的通用表示
 * - 2-4层适合资源受限或特定简单任务
 */
class BertEncoder {
public:
    /**
     * @brief 构造函数
     *
     * @param config BERT配置，包含层数、维度等信息
     *
     * 根据配置创建指定数量的TransformerBlock
     */
    explicit BertEncoder(const BertConfig& config);

    /**
     * @brief 前向传播
     *
     * 将输入依次通过所有TransformerBlock
     *
     * @param hidden_states 输入的隐藏状态 [seq_len, hidden_size]
     *                      通常是embedding层的输出
     * @return 编码后的隐藏状态 [seq_len, hidden_size]
     *         可以直接用于下游任务
     *
     * 数据流：
     *   h_0 (输入) -> Block_1 -> h_1 -> Block_2 -> h_2 -> ... -> h_N (输出)
     *
     * 每个h_i的形状都是 [seq_len, hidden_size]
     */
    Matrix forward(const Matrix& hidden_states);

    /**
     * @brief 前向传播（带注意力掩码）
     *
     * @param hidden_states 输入隐藏状态
     * @param attention_mask 注意力掩码 [seq_len, seq_len]
     *                       1.0表示保留，0.0表示屏蔽
     *
     * 用途：
     * - 屏蔽padding tokens（填充的无效位置）
     * - 在某些任务中实现特殊的注意力模式
     *
     * 注意：当前版本暂未完全实现，预留接口
     */
    Matrix forward(const Matrix& hidden_states, const Matrix& attention_mask);

    /**
     * @brief 获取指定层的输出
     *
     * 用于：
     * - 特征提取（有些任务使用中间层特征更好）
     * - 分析不同层学到的表示
     * - 多层特征融合
     *
     * @param layer_idx 层索引（0-based）
     * @param hidden_states 输入
     * @return 该层的输出
     *
     * 实验发现：
     * - 浅层（0-3）：更适合POS tagging等语法任务
     * - 中层（4-8）：适合NER等实体识别
     * - 深层（9-11）：适合分类、文本蕴含等语义任务
     */
    Matrix get_layer_output(size_t layer_idx, const Matrix& hidden_states);

    /**
     * @brief 获取所有层的输出
     *
     * 返回每一层的隐藏状态，用于：
     * - 详细分析
     * - 特征融合
     * - 可视化
     *
     * @param hidden_states 输入
     * @return 包含每层输出的向量 [layer_0_out, layer_1_out, ..., layer_N_out]
     */
    std::vector<Matrix> get_all_layer_outputs(const Matrix& hidden_states);

    /**
     * @brief 获取层数
     */
    size_t num_layers() const { return layers_.size(); }

    /**
     * @brief 访问指定的TransformerBlock
     *
     * 用于权重加载和调试
     */
    TransformerBlock& layer(size_t idx) {
        if (idx >= layers_.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers_[idx];
    }

    const TransformerBlock& layer(size_t idx) const {
        if (idx >= layers_.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers_[idx];
    }

private:
    /**
     * TransformerBlock层的容器
     *
     * 使用std::vector存储多个TransformerBlock
     * - BERT-base: 12个TransformerBlock
     * - BERT-large: 24个TransformerBlock
     *
     * 内存布局：
     * - 每个Block包含：2个LayerNorm + 1个Attention + 1个FFN
     * - BERT-base每个Block约7M参数
     * - 12层共约84M参数（embedding和pooler占剩余30M）
     */
    std::vector<TransformerBlock> layers_;

    /**
     * 配置信息（保存副本，用于调试）
     */
    BertConfig config_;
};

} // namespace llm

#endif // LLM_ENGINE_BERT_ENCODER_H
