#ifndef LLM_ENGINE_BERT_CONFIG_H
#define LLM_ENGINE_BERT_CONFIG_H

#include <string>
#include <stdexcept>
#include <iostream>

namespace llm {

/**
 * @brief BertConfig - BERT模型配置类
 *
 * 这个类存储BERT模型的所有超参数（hyperparameters）。
 *
 * 为什么需要配置类？
 * 1. **集中管理参数**：所有超参数都在一个地方，便于修改和实验
 * 2. **模型变体支持**：通过不同的配置创建BERT-base、BERT-large等变体
 * 3. **权重加载**：从预训练模型加载时，需要知道模型的结构参数
 * 4. **可复现性**：保存配置可以确保模型结构完全一致
 *
 * BERT的主要变体：
 * - BERT-base:  12层, 768维, 12头  (~110M参数)
 * - BERT-large: 24层, 1024维, 16头 (~340M参数)
 * - BERT-tiny:  2层,  128维, 2头   (用于快速实验)
 */
struct BertConfig {
    // ========================================================================
    // 核心架构参数
    // ========================================================================

    /**
     * 词表大小 (Vocabulary Size)
     *
     * - BERT原始论文使用的WordPiece词表大小为30522
     * - 包含所有常用词以及特殊token：[PAD], [UNK], [CLS], [SEP], [MASK]
     * - 更大的词表可以减少[UNK]（未知词）的出现，但会增加模型大小
     */
    size_t vocab_size = 30522;

    /**
     * 隐藏层维度 (Hidden Size)
     *
     * 这是模型的核心维度，贯穿整个网络：
     * - Embedding输出维度
     * - 每个TransformerBlock的输入/输出维度
     * - Self-Attention的Q、K、V维度总和
     *
     * 常见值：
     * - BERT-base: 768
     * - BERT-large: 1024
     * - 必须能被num_attention_heads整除
     */
    size_t hidden_size = 768;

    /**
     * Transformer层数 (Number of Hidden Layers)
     *
     * 即TransformerBlock堆叠的层数
     * - BERT-base: 12层
     * - BERT-large: 24层
     * - 更多层可以捕获更复杂的模式，但训练和推理更慢
     *
     * 经验：
     * - 12层对大多数任务已经足够
     * - 24层在大规模预训练时效果更好
     */
    size_t num_hidden_layers = 12;

    /**
     * 注意力头数 (Number of Attention Heads)
     *
     * Multi-Head Attention的头数
     * - BERT-base: 12头
     * - BERT-large: 16头
     *
     * 要求：
     * - hidden_size必须能被num_attention_heads整除
     * - head_dim = hidden_size / num_attention_heads
     * - BERT-base: 768 / 12 = 64维/头
     *
     * 作用：
     * - 多头允许模型关注不同的表示子空间
     * - 类似卷积网络的多个滤波器
     */
    size_t num_attention_heads = 12;

    /**
     * 前馈网络中间层维度 (Intermediate Size)
     *
     * FeedForward Network的中间层大小
     * - 通常是hidden_size的4倍
     * - BERT-base: 3072 (768 * 4)
     * - BERT-large: 4096 (1024 * 4)
     *
     * 为什么是4倍？
     * - 经验值，在表达能力和计算成本间取得平衡
     * - FFN: hidden -> 4*hidden -> hidden
     * - 中间层提供了强大的非线性变换能力
     */
    size_t intermediate_size = 3072;

    // ========================================================================
    // 序列和位置参数
    // ========================================================================

    /**
     * 最大序列长度 (Max Position Embeddings)
     *
     * 模型能处理的最大token数量
     * - BERT默认: 512
     * - 更长的序列需要更多的计算（O(n²)的attention）
     *
     * 限制：
     * - 训练时固定，推理时不能超过这个长度
     * - 如果需要处理更长文本，需要切分或使用特殊技术
     */
    size_t max_position_embeddings = 512;

    /**
     * Token类型词表大小 (Type Vocab Size)
     *
     * 用于区分不同的句子段
     * - BERT: 2 (句子A=0, 句子B=1)
     * - 单句任务只用0
     * - 句子对任务（如问答、文本蕴含）用0和1
     *
     * 示例：
     *   输入: [CLS] 问题 [SEP] 答案 [SEP]
     *   Type: [0]   0000  [0]   1111  [1]
     */
    size_t type_vocab_size = 2;

    // ========================================================================
    // 训练和正则化参数
    // ========================================================================

    /**
     * LayerNorm的epsilon值
     *
     * 防止除零错误的小常数
     * - 典型值: 1e-12 (BERT) 或 1e-5 (其他模型)
     * - 在计算标准差时加上这个值: std = sqrt(var + eps)
     */
    float layer_norm_eps = 1e-12f;

    /**
     * Dropout概率
     *
     * 训练时随机丢弃神经元的比例
     * - BERT: 0.1 (10%的神经元被丢弃)
     * - 推理时不使用dropout
     *
     * 注意：当前版本暂未实现dropout，预留参数
     */
    float hidden_dropout_prob = 0.1f;

    /**
     * Attention的Dropout概率
     *
     * 在attention权重上应用dropout
     * - BERT: 0.1
     * - 避免过度依赖某些特定的attention模式
     */
    float attention_probs_dropout_prob = 0.1f;

    // ========================================================================
    // 激活函数
    // ========================================================================

    /**
     * 激活函数类型
     *
     * FeedForward Network中使用的激活函数
     * - BERT: "gelu" (Gaussian Error Linear Unit)
     * - 其他选择: "relu", "swish"等
     *
     * GELU特点：
     * - 平滑，处处可导
     * - 在负值区域有小的梯度（不像ReLU完全为0）
     * - 现代Transformer的标准选择
     */
    std::string hidden_act = "gelu";

    // ========================================================================
    // 初始化参数
    // ========================================================================

    /**
     * 权重初始化的标准差
     *
     * 使用截断正态分布初始化权重
     * - BERT: 0.02
     * - 小的初始值有助于训练稳定性
     */
    float initializer_range = 0.02f;

    // ========================================================================
    // 预定义配置
    // ========================================================================

    /**
     * @brief 创建BERT-base配置
     *
     * 标准的BERT-base模型配置
     * - 12层, 768维, 12头
     * - 约110M参数
     * - 在大多数任务上有良好的性能和速度平衡
     */
    static BertConfig bert_base() {
        BertConfig config;
        config.vocab_size = 30522;
        config.hidden_size = 768;
        config.num_hidden_layers = 12;
        config.num_attention_heads = 12;
        config.intermediate_size = 3072;
        config.max_position_embeddings = 512;
        config.type_vocab_size = 2;
        return config;
    }

    /**
     * @brief 创建BERT-large配置
     *
     * 大型BERT模型配置
     * - 24层, 1024维, 16头
     * - 约340M参数
     * - 性能更好但速度更慢
     */
    static BertConfig bert_large() {
        BertConfig config;
        config.vocab_size = 30522;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        config.max_position_embeddings = 512;
        config.type_vocab_size = 2;
        return config;
    }

    /**
     * @brief 创建BERT-tiny配置（用于快速实验）
     *
     * 极小的BERT模型，适合：
     * - 快速原型验证
     * - 资源受限环境
     * - 单元测试
     */
    static BertConfig bert_tiny() {
        BertConfig config;
        config.vocab_size = 30522;
        config.hidden_size = 128;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 2;
        config.intermediate_size = 512;
        config.max_position_embeddings = 512;
        config.type_vocab_size = 2;
        return config;
    }

    /**
     * @brief 验证配置的有效性
     *
     * 检查参数间的约束关系
     * @throws std::invalid_argument 如果配置无效
     */
    void validate() const {
        // 检查hidden_size能否被num_attention_heads整除
        if (hidden_size % num_attention_heads != 0) {
            throw std::invalid_argument(
                "hidden_size (" + std::to_string(hidden_size) +
                ") must be divisible by num_attention_heads (" +
                std::to_string(num_attention_heads) + ")"
            );
        }

        // 检查基本参数大于0
        if (vocab_size == 0 || hidden_size == 0 ||
            num_hidden_layers == 0 || num_attention_heads == 0) {
            throw std::invalid_argument(
                "Core parameters must be greater than 0"
            );
        }

        // 检查dropout在有效范围
        if (hidden_dropout_prob < 0.0f || hidden_dropout_prob >= 1.0f) {
            throw std::invalid_argument(
                "Dropout probability must be in [0, 1)"
            );
        }
    }

    /**
     * @brief 打印配置信息
     *
     * 用于调试和日志记录
     */
    void print() const {
        std::cout << "BertConfig:\n"
                  << "  vocab_size: " << vocab_size << "\n"
                  << "  hidden_size: " << hidden_size << "\n"
                  << "  num_hidden_layers: " << num_hidden_layers << "\n"
                  << "  num_attention_heads: " << num_attention_heads << "\n"
                  << "  intermediate_size: " << intermediate_size << "\n"
                  << "  max_position_embeddings: " << max_position_embeddings << "\n"
                  << "  type_vocab_size: " << type_vocab_size << "\n"
                  << "  layer_norm_eps: " << layer_norm_eps << "\n"
                  << "  hidden_act: " << hidden_act << std::endl;
    }
};

} // namespace llm

#endif // LLM_ENGINE_BERT_CONFIG_H
