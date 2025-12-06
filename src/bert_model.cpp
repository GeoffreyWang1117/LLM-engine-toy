#include "llm_engine/bert_model.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace llm {

// ============================================================================
// BertModel 实现
// ============================================================================

/**
 * 构造函数实现
 *
 * 初始化BERT的所有组件
 */
BertModel::BertModel(const BertConfig& config)
    : config_(config),
      // 初始化Token Embedding
      token_embeddings_(config.vocab_size, config.hidden_size),
      // 初始化Position Embedding（BERT使用可学习的位置编码）
      position_embeddings_(config.max_position_embeddings, config.hidden_size),
      // 初始化Token Type Embedding
      token_type_embeddings_(config.type_vocab_size, config.hidden_size),
      // 初始化Embedding LayerNorm
      embeddings_layer_norm_(config.hidden_size),
      // 初始化Encoder
      encoder_(config),
      // 初始化Pooler
      pooler_dense_(config.hidden_size, config.hidden_size)
{
    // 验证配置
    config.validate();

    /*
     * 参数统计（以BERT-base为例）：
     *
     * Embeddings:
     * - Token: 30522 × 768 = 23,440,896
     * - Position: 512 × 768 = 393,216
     * - Token Type: 2 × 768 = 1,536
     * - LayerNorm: 768 × 2 = 1,536
     * 小计: 约23.8M
     *
     * Encoder:
     * - 12层 × 7.09M/层 = 85.1M
     *
     * Pooler:
     * - Dense: 768 × 768 = 589,824
     *
     * 总计: 约110M参数
     *
     * 内存占用（FP32）:
     * - 110M × 4字节 = 440MB (仅权重)
     * - 加上激活值和梯度，训练时约2-3GB
     */

    std::cout << "BertModel initialized:" << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Hidden size: " << config.hidden_size << std::endl;
    std::cout << "  Num layers: " << config.num_hidden_layers << std::endl;
    std::cout << "  Num heads: " << config.num_attention_heads << std::endl;
}

/**
 * 创建Embedding
 *
 * BERT的embedding是三个embedding的和：
 * embeddings = token_emb + position_emb + token_type_emb
 * 然后经过LayerNorm
 */
Matrix BertModel::create_embeddings(
    const std::vector<int>& input_ids,
    const std::vector<int>& token_type_ids) {

    size_t seq_len = input_ids.size();

    /*
     * 步骤1: Token Embedding
     *
     * 将每个token ID转换为向量
     * 例如：[101, 2023, 2003, 102] -> [[emb_101], [emb_2023], [emb_2003], [emb_102]]
     */
    Matrix token_emb = token_embeddings_.forward(input_ids);
    // 形状: [seq_len, hidden_size]

    /*
     * 步骤2: Position Embedding
     *
     * BERT使用可学习的位置嵌入
     * 为序列中的每个位置添加位置信息
     *
     * position_ids = [0, 1, 2, ..., seq_len-1]
     */
    std::vector<int> position_ids(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        position_ids[i] = static_cast<int>(i);
    }
    Matrix position_emb = position_embeddings_.forward(position_ids);
    // 形状: [seq_len, hidden_size]

    /*
     * 步骤3: Token Type Embedding
     *
     * 用于区分句子A和句子B
     * - 单句任务：全为0
     * - 句对任务：句子A为0，句子B为1
     *
     * 示例：
     *   输入: [CLS] What is NLP? [SEP] Natural Language Processing [SEP]
     *   Type: [0]   0000000000000 [0]   111111111111111111111111111 [1]
     */
    std::vector<int> type_ids = token_type_ids;
    if (type_ids.empty()) {
        // 默认全为0（单句任务）
        type_ids.resize(seq_len, 0);
    } else if (type_ids.size() != seq_len) {
        throw std::invalid_argument(
            "token_type_ids length must match input_ids length"
        );
    }
    Matrix token_type_emb = token_type_embeddings_.forward(type_ids);
    // 形状: [seq_len, hidden_size]

    /*
     * 步骤4: 三个Embedding相加
     *
     * 为什么是相加而不是concatenate？
     * - 相加保持维度不变，节省参数
     * - 实践证明效果很好
     * - 三种信息可以在向量空间中线性叠加
     */
    Matrix embeddings(seq_len, config_.hidden_size);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < config_.hidden_size; ++j) {
            embeddings(i, j) = token_emb(i, j) +
                              position_emb(i, j) +
                              token_type_emb(i, j);
        }
    }

    /*
     * 步骤5: LayerNorm
     *
     * 对embedding进行归一化
     * - 稳定训练
     * - 让不同样本的embedding在相似的尺度上
     */
    Matrix normalized = embeddings_layer_norm_.forward(embeddings);

    /*
     * 步骤6: Dropout (当前版本未实现)
     *
     * 在训练时随机丢弃一些神经元，防止过拟合
     * 推理时不使用dropout
     */
    // TODO: 添加dropout支持
    // if (training) {
    //     normalized = dropout(normalized, config_.hidden_dropout_prob);
    // }

    return normalized;
}

/**
 * 池化[CLS] token
 *
 * 提取序列的第一个token（[CLS]）并通过pooler层
 */
Matrix BertModel::pool_cls_token(const Matrix& sequence_output) {
    /*
     * [CLS] token的特殊作用：
     *
     * BERT在预训练时使用Next Sentence Prediction (NSP)任务
     * 训练[CLS]来聚合整个句子的信息
     *
     * 因此[CLS]的表示可以用作：
     * - 整个句子的语义表示
     * - 句子对关系判断
     * - 文本分类
     */

    size_t hidden_size = sequence_output.cols();

    // 提取第一个token的表示
    Matrix cls_token(1, hidden_size);
    for (size_t j = 0; j < hidden_size; ++j) {
        cls_token(0, j) = sequence_output(0, j);
    }

    /*
     * 通过Dense层变换
     *
     * 为什么需要这个额外的变换？
     * - [CLS]的表示经过12层Transformer后已经很丰富
     * - 但不一定是最适合特定下游任务的表示
     * - 这个Dense层可以学习到任务相关的变换
     */
    Matrix pooled = pooler_dense_.forward(cls_token);

    /*
     * 应用Tanh激活
     *
     * 为什么用Tanh而不是ReLU？
     * - Tanh输出范围[-1, 1]，有界
     * - 对于分类任务，有界的激活通常效果更好
     * - BERT的设计选择，实践中效果不错
     */
    Matrix activated = tanh(pooled);

    return activated;
}

/**
 * Tanh激活函数实现
 */
Matrix BertModel::tanh(const Matrix& input) {
    Matrix output(input.rows(), input.cols());

    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            // 或等价地: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
            output(i, j) = std::tanh(input(i, j));
        }
    }

    return output;
}

/**
 * 完整的前向传播
 *
 * 这是BERT的主要接口
 */
BertModel::BertOutput BertModel::forward(
    const std::vector<int>& input_ids,
    const std::vector<int>& token_type_ids,
    const std::vector<int>& attention_mask) {

    /*
     * 完整的BERT前向传播流程：
     *
     * 1. Embeddings:
     *    Token + Position + Token Type -> LayerNorm
     *
     * 2. Encoder:
     *    12 × TransformerBlock
     *
     * 3. Outputs:
     *    - Sequence output: 所有token的表示
     *    - Pooled output: [CLS]的池化表示
     */

    // 步骤1: 创建Embeddings
    Matrix embeddings = create_embeddings(input_ids, token_type_ids);
    // 形状: [seq_len, hidden_size]

    // 步骤2: 通过Encoder
    // TODO: 实现attention mask支持
    (void)attention_mask;  // 避免未使用参数警告
    Matrix encoded = encoder_.forward(embeddings);
    // 形状: [seq_len, hidden_size]

    // 步骤3: 创建输出并返回
    return BertOutput{
        encoded,                    // sequence_output
        pool_cls_token(encoded)     // pooled_output
    };

    /*
     * 输出的使用方式：
     *
     * 1. 文本分类任务：
     *    使用pooled_output [1, 768]
     *    -> Linear(768, num_classes) -> Softmax
     *
     * 2. Token分类任务（NER）：
     *    使用sequence_output [seq_len, 768]
     *    -> Linear(768, num_tags) -> CRF
     *
     * 3. 问答任务：
     *    使用sequence_output
     *    -> 两个Linear层分别预测起始和结束位置
     *
     * 4. 句子对任务：
     *    使用pooled_output
     *    -> Linear -> 二分类/三分类
     */
}

/**
 * 仅编码（不使用pooler）
 *
 * 对于不需要句子级表示的任务，可以跳过pooler节省计算
 */
Matrix BertModel::encode(
    const std::vector<int>& input_ids,
    const std::vector<int>& token_type_ids) {

    // 创建embeddings
    Matrix embeddings = create_embeddings(input_ids, token_type_ids);

    // 通过encoder
    Matrix encoded = encoder_.forward(embeddings);

    return encoded;

    /*
     * 适合的任务：
     * - Token级任务（NER、POS tagging）
     * - 特征提取
     * - 当你只需要sequence_output时
     */
}

/**
 * 获取池化输出（快捷方法）
 *
 * 用于分类任务
 */
Matrix BertModel::get_pooled_output(
    const std::vector<int>& input_ids,
    const std::vector<int>& token_type_ids) {

    // 方式1: 调用完整forward
    BertOutput output = forward(input_ids, token_type_ids);
    return output.pooled_output;

    /*
     * 注意：
     * - 这个方法会计算完整的sequence_output
     * - 如果你需要两个输出，直接调用forward更高效
     * - 如果只需要pooled_output，这个方法更方便
     */
}

} // namespace llm
