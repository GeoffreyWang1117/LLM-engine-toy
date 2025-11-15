/**
 * @file tutorial_alpha.cpp
 * @brief Alpha 0.1 新组件教学示例
 *
 * 本示例展示：
 * 1. TransformerBlock - 完整的Transformer编码器块
 * 2. PositionalEncoding - 位置编码（正弦和可学习两种）
 * 3. Embedding - 词嵌入层
 *
 * 这些是构建BERT模型的基础组件
 */

#include "llm_engine/matrix.h"
#include "llm_engine/layers.h"
#include "llm_engine/transformer_block.h"
#include "llm_engine/positional_encoding.h"
#include "llm_engine/embedding.h"
#include <iostream>
#include <iomanip>

using namespace llm;

// 辅助函数：打印分隔线
void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
}

// 辅助函数：打印矩阵（简化版，只显示部分元素）
void print_matrix_summary(const Matrix& mat, const std::string& name) {
    std::cout << name << " [" << mat.rows() << " x " << mat.cols() << "]" << std::endl;

    size_t show_rows = std::min(size_t(3), mat.rows());
    size_t show_cols = std::min(size_t(6), mat.cols());

    for (size_t i = 0; i < show_rows; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < show_cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                     << mat(i, j);
            if (j < show_cols - 1) std::cout << ", ";
        }
        if (show_cols < mat.cols()) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
    }

    if (show_rows < mat.rows()) {
        std::cout << "  ..." << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          Alpha 0.1 - 新组件教学示例                             ║\n";
    std::cout << "║          BERT基础架构组件详解                                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    // ============================================================================
    // 1. Embedding层演示
    // ============================================================================
    print_separator("【1】Embedding层 - 将Token ID转换为向量");

    std::cout << "Embedding是NLP模型的第一步：\n";
    std::cout << "- 输入：离散的token ID（整数）\n";
    std::cout << "- 输出：连续的向量表示（浮点数）\n";
    std::cout << "- 每个token对应embedding矩阵的一行\n" << std::endl;

    // 创建一个小型词表的Embedding
    // vocab_size=1000, embed_dim=8（实际BERT使用30522和768）
    size_t vocab_size = 1000;
    size_t embed_dim = 8;
    Embedding embedding(vocab_size, embed_dim);

    std::cout << "创建Embedding层：\n";
    std::cout << "- 词表大小: " << vocab_size << "\n";
    std::cout << "- 嵌入维度: " << embed_dim << "\n" << std::endl;

    // 示例：将一个句子的token IDs转换为嵌入
    // 假设： "[CLS] Hello world [SEP]" -> [101, 520, 128, 102]
    // 注意：这里使用小的ID以适应我们的toy词表大小(1000)
    std::vector<int> token_ids = {101, 520, 128, 102};

    std::cout << "输入Token IDs: [";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::cout << token_ids[i];
        if (i < token_ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n" << std::endl;

    Matrix token_embeddings = embedding.forward(token_ids);
    print_matrix_summary(token_embeddings, "Token Embeddings");

    std::cout << "说明：\n";
    std::cout << "- 每一行是一个token的嵌入向量\n";
    std::cout << "- 这些向量会在训练中学习语义信息\n";
    std::cout << "- 例如，\"cat\"和\"dog\"的嵌入会比较接近\n" << std::endl;

    // ============================================================================
    // 2. PositionalEncoding演示 - 正弦编码
    // ============================================================================
    print_separator("【2】位置编码 - 正弦/余弦方式（Transformer原始方法）");

    std::cout << "为什么需要位置编码？\n";
    std::cout << "- Self-Attention对位置不敏感\n";
    std::cout << "- 打乱单词顺序，attention结果不变\n";
    std::cout << "- 但词序在语言中很重要！\n";
    std::cout << "- 位置编码让模型知道每个token的位置\n" << std::endl;

    // 创建正弦位置编码
    size_t max_seq_len = 512;
    PositionalEncoding pos_enc_sin(
        max_seq_len,
        embed_dim,
        PositionalEncoding::Type::SINUSOIDAL
    );

    std::cout << "正弦位置编码特点：\n";
    std::cout << "- 使用sin和cos函数生成\n";
    std::cout << "- 不需要训练，是固定的\n";
    std::cout << "- 不同频率捕获不同尺度的位置信息\n";
    std::cout << "- 可以外推到更长的序列\n" << std::endl;

    // 查看前几个位置的编码
    std::cout << "前3个位置的正弦编码：\n";
    for (size_t pos = 0; pos < 3; ++pos) {
        Matrix pos_vec = pos_enc_sin.get_encoding(pos);
        std::cout << "  位置 " << pos << ": [";
        for (size_t j = 0; j < std::min(size_t(6), embed_dim); ++j) {
            std::cout << std::fixed << std::setprecision(4) << pos_vec(0, j);
            if (j < std::min(size_t(6), embed_dim) - 1) std::cout << ", ";
        }
        std::cout << ", ...]" << std::endl;
    }
    std::cout << std::endl;

    // 将位置编码添加到token embeddings
    Matrix embeddings_with_pos = pos_enc_sin.forward(token_embeddings);
    print_matrix_summary(embeddings_with_pos,
                        "Token Embeddings + Positional Encoding");

    std::cout << "说明：现在每个token不仅包含语义信息，还包含位置信息\n" << std::endl;

    // ============================================================================
    // 3. PositionalEncoding演示 - 可学习编码
    // ============================================================================
    print_separator("【3】位置编码 - 可学习方式（BERT使用的方法）");

    PositionalEncoding pos_enc_learned(
        max_seq_len,
        embed_dim,
        PositionalEncoding::Type::LEARNED
    );

    std::cout << "可学习位置编码特点：\n";
    std::cout << "- 为每个位置学习一个嵌入向量\n";
    std::cout << "- 类似token embedding\n";
    std::cout << "- 可以学习到任务相关的位置模式\n";
    std::cout << "- BERT、GPT等现代模型常用此方法\n" << std::endl;

    Matrix learned_pos_embeddings = pos_enc_learned.forward(token_embeddings);
    print_matrix_summary(learned_pos_embeddings,
                        "Token Embeddings + Learned Positional Encoding");

    // ============================================================================
    // 4. TransformerBlock演示
    // ============================================================================
    print_separator("【4】TransformerBlock - Transformer编码器的基本单元");

    std::cout << "TransformerBlock结构：\n";
    std::cout << "  输入 x\n";
    std::cout << "    ↓\n";
    std::cout << "  LayerNorm → Self-Attention → Add(x)  [残差连接]\n";
    std::cout << "    ↓\n";
    std::cout << "  LayerNorm → FeedForward → Add(上一步输出)  [残差连接]\n";
    std::cout << "    ↓\n";
    std::cout << "  输出\n" << std::endl;

    // 创建TransformerBlock
    size_t num_heads = 2;      // 注意力头数（实际BERT-base用12）
    size_t ffn_dim = 32;       // FFN中间层维度（实际BERT-base用3072）

    TransformerBlock transformer_block(embed_dim, num_heads, ffn_dim);

    std::cout << "TransformerBlock配置：\n";
    std::cout << "- 嵌入维度: " << embed_dim << "\n";
    std::cout << "- 注意力头数: " << num_heads << "\n";
    std::cout << "- FFN中间维度: " << ffn_dim << "\n" << std::endl;

    // 通过TransformerBlock处理
    Matrix block_output = transformer_block.forward(embeddings_with_pos);
    print_matrix_summary(block_output, "TransformerBlock输出");

    std::cout << "TransformerBlock的作用：\n";
    std::cout << "1. Self-Attention: 让每个token关注序列中的所有token\n";
    std::cout << "   - 捕获长距离依赖\n";
    std::cout << "   - 例如：\"The cat that chased the mouse was tired\"\n";
    std::cout << "     \"was\"需要关注到\"cat\"而非\"mouse\"\n\n";
    std::cout << "2. FeedForward: 增加非线性变换能力\n";
    std::cout << "   - 每个位置独立处理\n";
    std::cout << "   - 提升表达能力\n\n";
    std::cout << "3. 残差连接: 让网络可以训练得更深\n";
    std::cout << "   - BERT有12或24层TransformerBlock\n";
    std::cout << "   - 残差连接避免梯度消失\n" << std::endl;

    // ============================================================================
    // 5. 模拟多层Transformer
    // ============================================================================
    print_separator("【5】多层Transformer - 模拟BERT的结构");

    std::cout << "BERT模型结构：\n";
    std::cout << "  Token IDs\n";
    std::cout << "    ↓\n";
    std::cout << "  Token Embedding + Position Embedding + Token Type Embedding\n";
    std::cout << "    ↓\n";
    std::cout << "  TransformerBlock × N  (BERT-base: N=12, BERT-large: N=24)\n";
    std::cout << "    ↓\n";
    std::cout << "  输出表示\n" << std::endl;

    // 创建3层TransformerBlock（简化的BERT）
    std::vector<TransformerBlock> layers;
    size_t num_layers = 3;

    for (size_t i = 0; i < num_layers; ++i) {
        layers.emplace_back(embed_dim, num_heads, ffn_dim);
    }

    std::cout << "创建" << num_layers << "层Transformer编码器\n" << std::endl;

    // 逐层处理
    Matrix output = embeddings_with_pos;

    for (size_t i = 0; i < num_layers; ++i) {
        std::cout << "通过第" << (i + 1) << "层...\n";
        output = layers[i].forward(output);
    }

    std::cout << std::endl;
    print_matrix_summary(output, "最终输出（经过3层Transformer）");

    std::cout << "每一层的作用：\n";
    std::cout << "- 第1层: 捕获局部依赖和简单模式\n";
    std::cout << "- 第2层: 基于第1层构建更复杂的特征\n";
    std::cout << "- 第3层: 更抽象的语义表示\n";
    std::cout << "- 层数越深，表示越抽象\n" << std::endl;

    // ============================================================================
    // 6. 输出形状追踪
    // ============================================================================
    print_separator("【6】数据流和形状变化总结");

    std::cout << "完整的数据流（以seq_len=4, embed_dim=8为例）：\n\n";

    std::cout << "1. Token IDs:          [4]           # 4个token\n";
    std::cout << "                         ↓ Embedding\n";
    std::cout << "2. Token Embeddings:   [4, 8]        # 4个token，每个8维\n";
    std::cout << "                         ↓ + Positional Encoding\n";
    std::cout << "3. Input Embeddings:   [4, 8]        # 形状不变，但加上了位置信息\n";
    std::cout << "                         ↓ TransformerBlock\n";
    std::cout << "4. Layer 1 Output:     [4, 8]        # 形状不变\n";
    std::cout << "                         ↓ TransformerBlock\n";
    std::cout << "5. Layer 2 Output:     [4, 8]        # 形状不变\n";
    std::cout << "                         ↓ TransformerBlock\n";
    std::cout << "6. Layer 3 Output:     [4, 8]        # 形状不变\n";
    std::cout << "                         ↓\n";
    std::cout << "7. Final Output:       [4, 8]        # 最终表示\n\n";

    std::cout << "关键观察：\n";
    std::cout << "- Transformer保持形状不变：输入[seq_len, embed_dim]，输出相同形状\n";
    std::cout << "- 这让我们可以堆叠任意多层\n";
    std::cout << "- 但每一层都在逐步增强表示的质量\n" << std::endl;

    // ============================================================================
    // 总结
    // ============================================================================
    print_separator("【总结】Alpha 0.1 核心组件");

    std::cout << "我们实现了构建BERT的三大基础组件：\n\n";

    std::cout << "1. ✅ Embedding层\n";
    std::cout << "   - Token ID → Dense Vector\n";
    std::cout << "   - 将离散符号转换为连续表示\n\n";

    std::cout << "2. ✅ PositionalEncoding\n";
    std::cout << "   - 添加位置信息\n";
    std::cout << "   - 支持正弦编码（Transformer原始方法）\n";
    std::cout << "   - 支持可学习编码（BERT方法）\n\n";

    std::cout << "3. ✅ TransformerBlock\n";
    std::cout << "   - Self-Attention: 捕获长距离依赖\n";
    std::cout << "   - FeedForward: 增强非线性\n";
    std::cout << "   - 残差连接: 支持深层网络\n\n";

    std::cout << "下一步计划：\n";
    std::cout << "- 实现完整的BertModel类\n";
    std::cout << "- 添加权重加载功能\n";
    std::cout << "- 实现Tokenizer\n";
    std::cout << "- 支持实际的推理任务\n" << std::endl;

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Alpha 0.1 基础组件测试成功！                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    return 0;
}
