/**
 * @file tutorial_bert.cpp
 * @brief 完整BERT模型教学示例
 *
 * 本示例展示：
 * 1. BertConfig - 配置不同的BERT变体
 * 2. BertEncoder - 多层Transformer编码器
 * 3. BertModel - 完整的BERT模型
 * 4. 模拟真实的推理流程
 *
 * 这是Alpha 0.1的最终演示！
 */

#include "llm_engine/bert_config.h"
#include "llm_engine/bert_encoder.h"
#include "llm_engine/bert_model.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace llm;

// 辅助函数：打印分隔线
void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(75, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(75, '=') << "\n" << std::endl;
}

// 辅助函数：打印矩阵摘要
void print_matrix_summary(const Matrix& mat, const std::string& name) {
    std::cout << name << " shape: [" << mat.rows() << " × " << mat.cols() << "]" << std::endl;

    // 显示统计信息
    float sum = 0.0f;
    float min_val = mat(0, 0);
    float max_val = mat(0, 0);

    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            float val = mat(i, j);
            sum += val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }

    float mean = sum / (mat.rows() * mat.cols());

    std::cout << "  Statistics:" << std::endl;
    std::cout << "    Mean: " << std::fixed << std::setprecision(6) << mean << std::endl;
    std::cout << "    Min:  " << min_val << std::endl;
    std::cout << "    Max:  " << max_val << std::endl;

    // 显示第一行的前几个元素
    std::cout << "  First row preview: [";
    size_t show_cols = std::min(size_t(5), mat.cols());
    for (size_t j = 0; j < show_cols; ++j) {
        std::cout << std::setprecision(4) << mat(0, j);
        if (j < show_cols - 1) std::cout << ", ";
    }
    if (mat.cols() > show_cols) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Alpha 0.1 - 完整BERT模型演示                                ║\n";
    std::cout << "║           从Embedding到Encoder的完整流程                              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";

    // ========================================================================
    // 1. BertConfig - 配置管理
    // ========================================================================
    print_separator("【1】BertConfig - 配置不同的BERT变体");

    std::cout << "BERT有多种预定义配置：\n" << std::endl;

    // BERT-tiny (用于快速实验)
    std::cout << "1. BERT-tiny (超小模型，用于快速实验):" << std::endl;
    BertConfig tiny_config = BertConfig::bert_tiny();
    tiny_config.print();

    std::cout << "\n2. BERT-base (标准配置):" << std::endl;
    BertConfig base_config = BertConfig::bert_base();
    base_config.print();

    std::cout << "\n3. BERT-large (大型模型):" << std::endl;
    BertConfig large_config = BertConfig::bert_large();
    large_config.print();

    std::cout << "\n配置说明：" << std::endl;
    std::cout << "- BERT-tiny:  适合快速实验和单元测试" << std::endl;
    std::cout << "- BERT-base:  最常用，性能和速度的良好平衡" << std::endl;
    std::cout << "- BERT-large: 最强性能，但计算成本高" << std::endl;

    // ========================================================================
    // 2. 使用BERT-tiny进行演示
    // ========================================================================
    print_separator("【2】创建BERT-tiny模型");

    std::cout << "为了快速演示，我们使用BERT-tiny配置：" << std::endl;
    std::cout << "- 2层Transformer" << std::endl;
    std::cout << "- 128维隐藏层" << std::endl;
    std::cout << "- 2个注意力头" << std::endl;
    std::cout << "- 足够小可以快速运行，但保留了BERT的核心特性\n" << std::endl;

    // 创建模型
    std::cout << "正在创建BertModel..." << std::endl;
    BertModel model(tiny_config);
    std::cout << "✅ 模型创建成功！\n" << std::endl;

    // ========================================================================
    // 3. 准备输入数据
    // ========================================================================
    print_separator("【3】准备输入数据");

    /*
     * 模拟一个句子的Token IDs
     *
     * 在真实场景中，这些ID由Tokenizer生成
     * 例如："[CLS] Hello world [SEP]" -> [101, 7592, 2088, 102]
     *
     * 这里使用随机的合法ID来演示
     */
    std::vector<int> input_ids = {
        101,   // [CLS]
        2023,  // token 1
        2003,  // token 2
        1037,  // token 3
        3231,  // token 4
        102    // [SEP]
    };

    std::cout << "输入Token IDs: [";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        std::cout << input_ids[i];
        if (i < input_ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "序列长度: " << input_ids.size() << " tokens\n" << std::endl;

    std::cout << "说明：" << std::endl;
    std::cout << "- input_ids[0] = 101  →  [CLS] token (句子开始)" << std::endl;
    std::cout << "- input_ids[1-4]      →  实际的词token" << std::endl;
    std::cout << "- input_ids[5] = 102  →  [SEP] token (句子结束)" << std::endl;

    // ========================================================================
    // 4. BERT前向传播
    // ========================================================================
    print_separator("【4】BERT前向传播");

    std::cout << "执行前向传播..." << std::endl;
    std::cout << "数据流：" << std::endl;
    std::cout << "  Token IDs" << std::endl;
    std::cout << "      ↓" << std::endl;
    std::cout << "  Embeddings (Token + Position + Token Type)" << std::endl;
    std::cout << "      ↓" << std::endl;
    std::cout << "  LayerNorm" << std::endl;
    std::cout << "      ↓" << std::endl;
    std::cout << "  TransformerBlock × 2" << std::endl;
    std::cout << "      ↓" << std::endl;
    std::cout << "  Outputs (Sequence + Pooled)\n" << std::endl;

    // 执行forward
    BertModel::BertOutput output = model.forward(input_ids);

    std::cout << "✅ 前向传播完成！\n" << std::endl;

    // ========================================================================
    // 5. 分析输出
    // ========================================================================
    print_separator("【5】分析BERT输出");

    std::cout << "BERT产生两种输出：\n" << std::endl;

    std::cout << "1. Sequence Output (序列输出):" << std::endl;
    std::cout << "   - 形状: [seq_len, hidden_size]" << std::endl;
    std::cout << "   - 包含每个token的上下文化表示" << std::endl;
    std::cout << "   - 用于token级任务（NER、问答等）\n" << std::endl;
    print_matrix_summary(output.sequence_output, "Sequence Output");

    std::cout << "\n2. Pooled Output (池化输出):" << std::endl;
    std::cout << "   - 形状: [1, hidden_size]" << std::endl;
    std::cout << "   - [CLS] token的池化表示" << std::endl;
    std::cout << "   - 用于句子级任务（分类、文本蕴含等）\n" << std::endl;
    print_matrix_summary(output.pooled_output, "Pooled Output");

    // ========================================================================
    // 6. 不同配置的对比
    // ========================================================================
    print_separator("【6】不同BERT配置的特点对比");

    std::cout << "┌──────────────┬────────┬───────────┬───────┬──────────┬─────────┐" << std::endl;
    std::cout << "│ 配置         │ 层数   │ 隐藏维度  │ 注意力头│ 参数量   │ 用途    │" << std::endl;
    std::cout << "├──────────────┼────────┼───────────┼───────┼──────────┼─────────┤" << std::endl;
    std::cout << "│ BERT-tiny    │ 2      │ 128       │ 2     │ ~4M      │ 快速实验│" << std::endl;
    std::cout << "│ BERT-base    │ 12     │ 768       │ 12    │ ~110M    │ 标准应用│" << std::endl;
    std::cout << "│ BERT-large   │ 24     │ 1024      │ 16    │ ~340M    │ 高性能  │" << std::endl;
    std::cout << "└──────────────┴────────┴───────────┴───────┴──────────┴─────────┘" << std::endl;

    // ========================================================================
    // 7. 应用场景示例
    // ========================================================================
    print_separator("【7】BERT的典型应用场景");

    std::cout << "使用Sequence Output的任务：\n" << std::endl;

    std::cout << "1. 命名实体识别 (NER):" << std::endl;
    std::cout << "   输入: \"Apple is looking at buying U.K. startup\"" << std::endl;
    std::cout << "   对每个token分类: [ORG] [O] [O] [O] [O] [LOC] [O]" << std::endl;
    std::cout << "   实现: sequence_output → Linear(hidden_size, num_tags) → CRF\n" << std::endl;

    std::cout << "2. 问答系统:" << std::endl;
    std::cout << "   问题: \"Who wrote Pride and Prejudice?\"" << std::endl;
    std::cout << "   上下文: \"... Jane Austen wrote Pride and Prejudice ...\"" << std::endl;
    std::cout << "   预测: 答案的起始和结束位置\n" << std::endl;

    std::cout << "使用Pooled Output的任务：\n" << std::endl;

    std::cout << "3. 文本分类:" << std::endl;
    std::cout << "   输入: \"This movie is amazing!\"" << std::endl;
    std::cout << "   输出: Positive (positive/negative sentiment)" << std::endl;
    std::cout << "   实现: pooled_output → Linear(hidden_size, num_classes)\n" << std::endl;

    std::cout << "4. 文本蕴含 (Textual Entailment):" << std::endl;
    std::cout << "   前提: \"A man is riding a bike\"" << std::endl;
    std::cout << "   假设: \"A man is exercising\"" << std::endl;
    std::cout << "   判断: Entailment / Contradiction / Neutral" << std::endl;

    // ========================================================================
    // 8. 性能分析
    // ========================================================================
    print_separator("【8】性能和资源分析");

    std::cout << "BERT-base的计算需求（以512长度序列为例）：\n" << std::endl;

    std::cout << "内存占用：" << std::endl;
    std::cout << "- 模型权重 (FP32): ~440MB" << std::endl;
    std::cout << "- 推理时激活值: ~500MB" << std::endl;
    std::cout << "- 总计: ~1GB\n" << std::endl;

    std::cout << "计算量：" << std::endl;
    std::cout << "- 每层Attention: O(L² × d) ≈ 512² × 768 = 200M ops" << std::endl;
    std::cout << "- 每层FFN: O(L × d × i) ≈ 512 × 768 × 3072 = 1.2B ops" << std::endl;
    std::cout << "- 12层总计: ~17B ops\n" << std::endl;

    std::cout << "推理速度（单核CPU）：" << std::endl;
    std::cout << "- 序列长度128: ~50-100ms" << std::endl;
    std::cout << "- 序列长度512: ~200-400ms\n" << std::endl;

    std::cout << "GPU加速（RTX 3090）：" << std::endl;
    std::cout << "- 批量推理可达100-1000句/秒" << std::endl;
    std::cout << "- 相比CPU快10-50倍" << std::endl;

    // ========================================================================
    // 9. 下一步计划
    // ========================================================================
    print_separator("【9】Alpha 0.1 下一步开发计划");

    std::cout << "当前进度：阶段2完成 ✅\n" << std::endl;

    std::cout << "已实现：" << std::endl;
    std::cout << "✅ 完整的BERT模型架构" << std::endl;
    std::cout << "✅ 所有Transformer组件" << std::endl;
    std::cout << "✅ 配置管理系统\n" << std::endl;

    std::cout << "下一阶段（阶段3）：权重加载" << std::endl;
    std::cout << "⏳ 从PyTorch checkpoint加载权重" << std::endl;
    std::cout << "⏳ 支持Hugging Face格式" << std::endl;
    std::cout << "⏳ 权重映射和验证\n" << std::endl;

    std::cout << "阶段4：Tokenizer" << std::endl;
    std::cout << "⏳ WordPiece分词算法" << std::endl;
    std::cout << "⏳ 词表管理" << std::endl;
    std::cout << "⏳ 文本预处理\n" << std::endl;

    std::cout << "阶段5：推理优化和应用" << std::endl;
    std::cout << "⏳ 高层推理API" << std::endl;
    std::cout << "⏳ 批处理支持" << std::endl;
    std::cout << "⏳ 实际应用示例" << std::endl;

    // ========================================================================
    // 总结
    // ========================================================================
    print_separator("【总结】Alpha 0.1 阶段2成果");

    std::cout << "我们成功实现了完整的BERT架构：\n" << std::endl;

    std::cout << "核心组件：" << std::endl;
    std::cout << "1. ✅ BertConfig    - 灵活的配置管理" << std::endl;
    std::cout << "2. ✅ BertEncoder   - 多层Transformer编码器" << std::endl;
    std::cout << "3. ✅ BertModel     - 完整的BERT模型\n" << std::endl;

    std::cout << "代码质量：" << std::endl;
    std::cout << "- ~1350行新增代码" << std::endl;
    std::cout << "- 每个函数都有详细注释" << std::endl;
    std::cout << "- 解释了算法原理和设计决策\n" << std::endl;

    std::cout << "技术亮点：" << std::endl;
    std::cout << "- 支持多种BERT变体（tiny/base/large）" << std::endl;
    std::cout << "- 模块化设计，易于扩展" << std::endl;
    std::cout << "- 为权重加载预留了接口\n" << std::endl;

    std::cout << "学习价值：" << std::endl;
    std::cout << "- 深入理解BERT的每个细节" << std::endl;
    std::cout << "- 掌握Transformer的工程实践" << std::endl;
    std::cout << "- 从零构建工业级NLP模型" << std::endl;

    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Alpha 0.1 阶段2完成！                                     ║\n";
    std::cout << "║              完整的BERT推理框架已就绪                                  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n";

    return 0;
}
