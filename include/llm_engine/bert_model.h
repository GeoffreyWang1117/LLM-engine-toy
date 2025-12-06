#ifndef LLM_ENGINE_BERT_MODEL_H
#define LLM_ENGINE_BERT_MODEL_H

#include "bert_config.h"
#include "bert_encoder.h"
#include "embedding.h"
#include "positional_encoding.h"
#include "layers.h"
#include "matrix.h"
#include <vector>
#include <memory>

namespace llm {

/**
 * @brief BertModel - 完整的BERT模型
 *
 * 这是BERT的完整实现，包含：
 * 1. Embeddings (Token + Position + Token Type)
 * 2. Encoder (多层TransformerBlock)
 * 3. Pooler (用于分类任务的[CLS]池化层)
 *
 * BERT完整架构：
 *
 *   Token IDs [seq_len]
 *       ↓
 *   ┌──────────── Embeddings ────────────┐
 *   │  Token Embedding                    │
 *   │  + Position Embedding               │
 *   │  + Token Type Embedding             │
 *   │  → LayerNorm → Dropout              │
 *   └────────────────────────────────────┘
 *       ↓ [seq_len, hidden_size]
 *   ┌─────────── Encoder ────────────────┐
 *   │  TransformerBlock × N               │
 *   └────────────────────────────────────┘
 *       ↓ [seq_len, hidden_size]
 *   ┌─────────── Outputs ────────────────┐
 *   │  Sequence Output: 所有token的表示   │
 *   │  Pooled Output: [CLS]的池化表示     │
 *   └────────────────────────────────────┘
 *
 * 使用场景：
 * - Sequence Output: 用于token级任务（NER、问答等）
 * - Pooled Output: 用于句子级任务（分类、文本蕴含等）
 */
class BertModel {
public:
    /**
     * @brief 构造函数
     *
     * @param config BERT配置
     *
     * 创建所有子模块：
     * - 3个Embedding层（token、position、token_type）
     * - 1个LayerNorm（embedding后）
     * - N个TransformerBlock（encoder）
     * - 1个Pooler层（可选，用于分类）
     */
    explicit BertModel(const BertConfig& config);

    /**
     * @brief 前向传播
     *
     * @param input_ids Token ID序列，例如 [101, 2023, 2003, 102]
     *                  - [CLS] ... [SEP]格式
     *                  - 长度不超过max_position_embeddings
     *
     * @param token_type_ids Token类型ID（可选）
     *                       - 单句任务：全0
     *                       - 句对任务：句子A为0，句子B为1
     *                       - 默认：全0
     *
     * @param attention_mask 注意力掩码（可选）
     *                       - 1表示真实token，0表示padding
     *                       - 默认：全1（无padding）
     *
     * @return 包含sequence_output和pooled_output的结构
     *
     * 示例：
     *   输入: [101, 2023, 2003, 102]  # [CLS] this is [SEP]
     *   sequence_output: [4, 768]      # 每个token的表示
     *   pooled_output: [1, 768]        # [CLS]的池化表示
     */
    struct BertOutput {
        Matrix sequence_output;  // [seq_len, hidden_size]
        Matrix pooled_output;    // [1, hidden_size]
    };

    BertOutput forward(const std::vector<int>& input_ids,
                      const std::vector<int>& token_type_ids = {},
                      const std::vector<int>& attention_mask = {});

    /**
     * @brief 仅编码（不使用pooler）
     *
     * 适合：
     * - Token级任务（不需要pooled output）
     * - 特征提取
     * - 节省计算
     *
     * @return sequence_output [seq_len, hidden_size]
     */
    Matrix encode(const std::vector<int>& input_ids,
                  const std::vector<int>& token_type_ids = {});

    /**
     * @brief 获取[CLS] token的表示
     *
     * 用于分类任务的快捷方法
     *
     * @param input_ids Token IDs
     * @return [CLS]的池化表示 [1, hidden_size]
     */
    Matrix get_pooled_output(const std::vector<int>& input_ids,
                            const std::vector<int>& token_type_ids = {});

    /**
     * @brief 获取配置
     */
    const BertConfig& config() const { return config_; }

    /**
     * @brief 访问子模块（用于权重加载）
     */
    Embedding& token_embeddings() { return token_embeddings_; }
    Embedding& position_embeddings() { return position_embeddings_; }
    Embedding& token_type_embeddings() { return token_type_embeddings_; }
    LayerNorm& embeddings_layer_norm() { return embeddings_layer_norm_; }
    BertEncoder& encoder() { return encoder_; }
    Linear& pooler() { return pooler_dense_; }

    // const版本
    const Embedding& token_embeddings() const { return token_embeddings_; }
    const Embedding& position_embeddings() const { return position_embeddings_; }
    const Embedding& token_type_embeddings() const { return token_type_embeddings_; }
    const LayerNorm& embeddings_layer_norm() const { return embeddings_layer_norm_; }
    const BertEncoder& encoder() const { return encoder_; }
    const Linear& pooler() const { return pooler_dense_; }

private:
    /**
     * 配置
     */
    BertConfig config_;

    // ========================================================================
    // Embedding层
    // ========================================================================

    /**
     * Token Embedding
     *
     * 将token ID映射到向量
     * - vocab_size × hidden_size
     * - 例如：30522 × 768 = 23M参数
     */
    Embedding token_embeddings_;

    /**
     * Position Embedding (可学习的位置编码)
     *
     * BERT使用可学习的位置嵌入（不是正弦编码）
     * - max_position_embeddings × hidden_size
     * - 例如：512 × 768 = 393K参数
     */
    Embedding position_embeddings_;

    /**
     * Token Type Embedding
     *
     * 区分不同句子
     * - type_vocab_size × hidden_size
     * - 例如：2 × 768 = 1.5K参数
     */
    Embedding token_type_embeddings_;

    /**
     * Embedding后的LayerNorm
     *
     * 对embedding之和进行归一化
     * - hidden_size × 2 参数 (gamma和beta)
     */
    LayerNorm embeddings_layer_norm_;

    // ========================================================================
    // Encoder
    // ========================================================================

    /**
     * BERT编码器
     *
     * 包含N个TransformerBlock
     * - BERT-base: 12层，约85M参数
     * - BERT-large: 24层，约270M参数
     */
    BertEncoder encoder_;

    // ========================================================================
    // Pooler
    // ========================================================================

    /**
     * Pooler Dense层
     *
     * 用于分类任务：
     * 1. 提取[CLS] token的表示
     * 2. 通过一个Linear层
     * 3. 应用tanh激活
     *
     * - hidden_size × hidden_size
     * - 例如：768 × 768 = 590K参数
     *
     * 作用：
     * - 将[CLS]的表示转换为更适合分类的表示
     * - [CLS]被设计用来聚合整个句子的信息
     */
    Linear pooler_dense_;

    /**
     * @brief 创建Embedding
     *
     * 将token、position、token_type三种embedding相加
     *
     * @param input_ids Token IDs
     * @param token_type_ids Token类型IDs
     * @return Embedding后的表示 [seq_len, hidden_size]
     */
    Matrix create_embeddings(const std::vector<int>& input_ids,
                            const std::vector<int>& token_type_ids);

    /**
     * @brief 池化[CLS] token
     *
     * 提取第一个token（[CLS]）的表示并通过pooler
     *
     * @param sequence_output Encoder的输出 [seq_len, hidden_size]
     * @return 池化后的表示 [1, hidden_size]
     */
    Matrix pool_cls_token(const Matrix& sequence_output);

    /**
     * @brief Tanh激活函数
     *
     * 用于pooler
     * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     */
    Matrix tanh(const Matrix& input);
};

} // namespace llm

#endif // LLM_ENGINE_BERT_MODEL_H
