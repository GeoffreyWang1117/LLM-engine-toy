/**
 * @file tutorial_weight_loading.cpp
 * @brief 演示如何加载预训练BERT权重
 *
 * 这是Alpha 0.1 阶段3的最终演示：
 * 展示如何从Hugging Face的safetensors文件加载预训练权重到我们的C++模型
 */

#include "llm_engine/bert_config.h"
#include "llm_engine/bert_model.h"
#include "llm_engine/weight_loader.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace llm;

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      Alpha 0.1 阶段3 - 加载预训练BERT权重                      ║\n";
    std::cout << "║      从随机权重到真实的预训练模型                              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        // ====================================================================
        // 第1步：创建BERT-base模型（使用随机权重）
        // ====================================================================
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "  【1】创建BERT-base模型\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "\n";

        BertConfig config = BertConfig::bert_base();
        std::cout << "使用配置：\n";
        std::cout << "  - 词表大小: " << config.vocab_size << "\n";
        std::cout << "  - 隐藏层大小: " << config.hidden_size << "\n";
        std::cout << "  - 层数: " << config.num_hidden_layers << "\n";
        std::cout << "  - 注意力头数: " << config.num_attention_heads << "\n";
        std::cout << "\n";

        std::cout << "正在创建模型...\n";
        BertModel model(config);
        std::cout << "✅ 模型创建完成（当前使用随机权重）\n";
        std::cout << "\n";

        // ====================================================================
        // 第2步：使用随机权重进行推理（参考）
        // ====================================================================
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "  【2】使用随机权重推理（作为对比）\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "\n";

        // 输入：[CLS] hello world [SEP]
        std::vector<int> input_ids = {101, 7592, 2088, 102};
        std::cout << "输入Token IDs: [";
        for (size_t i = 0; i < input_ids.size(); ++i) {
            std::cout << input_ids[i];
            if (i < input_ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "（对应文本: [CLS] hello world [SEP]）\n\n";

        std::cout << "使用随机权重推理...\n";
        auto random_output = model.forward(input_ids);

        std::cout << "随机权重输出统计：\n";
        std::cout << "  Sequence output shape: ["
                  << random_output.sequence_output.rows() << " × "
                  << random_output.sequence_output.cols() << "]\n";

        // 计算统计信息
        float random_mean = 0.0f;
        float random_min = random_output.sequence_output(0, 0);
        float random_max = random_output.sequence_output(0, 0);

        for (size_t i = 0; i < random_output.sequence_output.rows(); ++i) {
            for (size_t j = 0; j < random_output.sequence_output.cols(); ++j) {
                float val = random_output.sequence_output(i, j);
                random_mean += val;
                random_min = std::min(random_min, val);
                random_max = std::max(random_max, val);
            }
        }
        random_mean /= (random_output.sequence_output.rows() * random_output.sequence_output.cols());

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Mean: " << random_mean << "\n";
        std::cout << "  Min:  " << random_min << "\n";
        std::cout << "  Max:  " << random_max << "\n";
        std::cout << "\n";

        // ====================================================================
        // 第3步：加载预训练权重
        // ====================================================================
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "  【3】加载预训练权重\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "\n";

        std::string model_path = "weights/bert-base-uncased/model.safetensors";
        std::cout << "从文件加载: " << model_path << "\n\n";

        WeightLoader loader(model_path);
        loader.load_to_model(model);

        // ====================================================================
        // 第4步：使用预训练权重进行推理
        // ====================================================================
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "  【4】使用预训练权重推理\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "\n";

        std::cout << "再次推理（使用预训练权重）...\n";
        auto pretrained_output = model.forward(input_ids);

        std::cout << "预训练权重输出统计：\n";
        std::cout << "  Sequence output shape: ["
                  << pretrained_output.sequence_output.rows() << " × "
                  << pretrained_output.sequence_output.cols() << "]\n";

        // 计算统计信息
        float pretrained_mean = 0.0f;
        float pretrained_min = pretrained_output.sequence_output(0, 0);
        float pretrained_max = pretrained_output.sequence_output(0, 0);

        for (size_t i = 0; i < pretrained_output.sequence_output.rows(); ++i) {
            for (size_t j = 0; j < pretrained_output.sequence_output.cols(); ++j) {
                float val = pretrained_output.sequence_output(i, j);
                pretrained_mean += val;
                pretrained_min = std::min(pretrained_min, val);
                pretrained_max = std::max(pretrained_max, val);
            }
        }
        pretrained_mean /= (pretrained_output.sequence_output.rows() * pretrained_output.sequence_output.cols());

        std::cout << "  Mean: " << pretrained_mean << "\n";
        std::cout << "  Min:  " << pretrained_min << "\n";
        std::cout << "  Max:  " << pretrained_max << "\n";
        std::cout << "\n";

        // 显示[CLS] token的表示（用于分类任务）
        std::cout << "Pooled output（[CLS] token，用于分类）:\n";
        std::cout << "  Shape: ["
                  << pretrained_output.pooled_output.rows() << " × "
                  << pretrained_output.pooled_output.cols() << "]\n";
        std::cout << "  前10维: [";
        for (size_t i = 0; i < 10; ++i) {
            std::cout << std::setprecision(4) << pretrained_output.pooled_output(0, i);
            if (i < 9) std::cout << ", ";
        }
        std::cout << ", ...]\n";
        std::cout << "\n";

        // ====================================================================
        // 第5步：对比分析
        // ====================================================================
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "  【5】对比分析\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "\n";

        std::cout << "┌────────────────┬───────────────┬───────────────┐\n";
        std::cout << "│                │ 随机权重      │ 预训练权重    │\n";
        std::cout << "├────────────────┼───────────────┼───────────────┤\n";
        std::cout << "│ Mean           │ " << std::setw(13) << random_mean
                  << " │ " << std::setw(13) << pretrained_mean << " │\n";
        std::cout << "│ Min            │ " << std::setw(13) << random_min
                  << " │ " << std::setw(13) << pretrained_min << " │\n";
        std::cout << "│ Max            │ " << std::setw(13) << random_max
                  << " │ " << std::setw(13) << pretrained_max << " │\n";
        std::cout << "└────────────────┴───────────────┴───────────────┘\n";
        std::cout << "\n";

        std::cout << "观察：\n";
        std::cout << "  • 预训练权重的输出更加稳定（范围更小）\n";
        std::cout << "  • 预训练权重已经学习了有意义的语言表示\n";
        std::cout << "  • 可以直接用于下游任务（分类、NER等）\n";
        std::cout << "\n";

        // ====================================================================
        // 总结
        // ====================================================================
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Alpha 0.1 阶段3完成！                             ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        std::cout << "已实现功能：\n";
        std::cout << "  ✅ SafeTensors文件解析\n";
        std::cout << "  ✅ 权重映射和加载\n";
        std::cout << "  ✅ 完整的BERT模型推理（使用预训练权重）\n";
        std::cout << "\n";

        std::cout << "下一步（阶段4）：\n";
        std::cout << "  ⏳ WordPiece Tokenizer\n";
        std::cout << "  ⏳ 支持输入原始文本（\"Hello world\"）\n";
        std::cout << "  ⏳ 端到端推理流程\n";
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
