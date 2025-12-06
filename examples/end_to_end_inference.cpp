/**
 * @file end_to_end_inference.cpp
 * @brief 端到端BERT推理演示
 *
 * 展示完整流程：文本 → Tokenizer → BERT模型 → 输出
 *
 * Alpha 0.1 阶段4的最终演示
 */

#include "llm_engine/bert_tokenizer.h"
#include "llm_engine/bert_model.h"
#include "llm_engine/bert_config.h"
#include "llm_engine/weight_loader.h"
#include <iostream>
#include <iomanip>

using namespace llm;

void print_separator() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        Alpha 0.1 阶段4 - 端到端BERT推理                       ║\n";
    std::cout << "║        从原始文本到模型输出                                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        print_separator();
        std::cout << "  【1】初始化Tokenizer\n";
        print_separator();
        std::cout << "\n";

        BertTokenizer tokenizer("weights/bert-base-uncased/vocab.txt", true);
        std::cout << "✅ Tokenizer初始化完成\n\n";

        print_separator();
        std::cout << "  【2】测试分词\n";
        print_separator();
        std::cout << "\n";

        // 测试几个例子
        std::vector<std::string> test_texts = {
            "Hello world!",
            "I love natural language processing.",
            "BERT is amazing!"
        };

        for (const auto& text : test_texts) {
            std::cout << "文本: \"" << text << "\"\n";

            // 分词
            auto tokens = tokenizer.tokenize(text);
            std::cout << "  Tokens: [";
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << "\"" << tokens[i] << "\"";
                if (i < tokens.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";

            // 编码
            auto ids = tokenizer.encode(text);
            std::cout << "  IDs: [";
            for (size_t i = 0; i < ids.size(); ++i) {
                std::cout << ids[i];
                if (i < ids.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";

            // 解码
            auto decoded = tokenizer.decode(ids);
            std::cout << "  解码: \"" << decoded << "\"\n";
            std::cout << "\n";
        }

        print_separator();
        std::cout << "  【3】初始化BERT模型\n";
        print_separator();
        std::cout << "\n";

        BertConfig config = BertConfig::bert_base();
        BertModel model(config);
        std::cout << "✅ 模型创建完成\n\n";

        print_separator();
        std::cout << "  【4】加载预训练权重\n";
        print_separator();
        std::cout << "\n";

        WeightLoader loader("weights/bert-base-uncased/model.safetensors");
        loader.load_to_model(model);

        print_separator();
        std::cout << "  【5】端到端推理\n";
        print_separator();
        std::cout << "\n";

        std::string inference_text = "Hello, world!";
        std::cout << "输入文本: \"" << inference_text << "\"\n\n";

        // 步骤1: 分词
        std::cout << "步骤1: 分词...\n";
        auto tokens = tokenizer.tokenize(inference_text);
        std::cout << "  Tokens: [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << "\"" << tokens[i] << "\"";
            if (i < tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";

        // 步骤2: 编码
        std::cout << "步骤2: 转换为Token IDs...\n";
        auto input_ids = tokenizer.encode(inference_text);
        std::cout << "  IDs: [";
        for (size_t i = 0; i < input_ids.size(); ++i) {
            std::cout << input_ids[i];
            if (i < input_ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";

        // 步骤3: 模型推理
        std::cout << "步骤3: BERT模型推理...\n";
        auto output = model.forward(input_ids);
        std::cout << "✅ 推理完成\n\n";

        // 步骤4: 分析输出
        std::cout << "步骤4: 输出分析\n";
        std::cout << "  Sequence output: ["
                  << output.sequence_output.rows() << " × "
                  << output.sequence_output.cols() << "]\n";
        std::cout << "    (每个token的表示)\n";

        std::cout << "  Pooled output: ["
                  << output.pooled_output.rows() << " × "
                  << output.pooled_output.cols() << "]\n";
        std::cout << "    ([CLS] token，用于分类)\n\n";

        // 显示[CLS] token的表示
        std::cout << "  [CLS] token表示（前10维）: [";
        for (size_t i = 0; i < 10; ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << output.pooled_output(0, i);
            if (i < 9) std::cout << ", ";
        }
        std::cout << ", ...]\n\n";

        print_separator();
        std::cout << "  【6】应用示例\n";
        print_separator();
        std::cout << "\n";

        std::cout << "现在可以使用输出向量进行：\n";
        std::cout << "  1. 文本分类：\n";
        std::cout << "     pooled_output → Linear(768, num_classes) → Softmax\n";
        std::cout << "\n";
        std::cout << "  2. 命名实体识别（NER）：\n";
        std::cout << "     sequence_output → Linear(768, num_tags) → CRF\n";
        std::cout << "\n";
        std::cout << "  3. 问答系统：\n";
        std::cout << "     sequence_output → 预测起始/结束位置\n";
        std::cout << "\n";
        std::cout << "  4. 语义相似度：\n";
        std::cout << "     cosine_similarity(pooled_output_1, pooled_output_2)\n";
        std::cout << "\n";

        print_separator();
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Alpha 0.1 阶段4完成！                            ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        std::cout << "已实现功能：\n";
        std::cout << "  ✅ 词表加载（30522个tokens）\n";
        std::cout << "  ✅ WordPiece分词器\n";
        std::cout << "  ✅ 文本编码和解码\n";
        std::cout << "  ✅ 端到端推理（文本 → BERT输出）\n";
        std::cout << "\n";

        std::cout << "可以使用的接口：\n";
        std::cout << "  • tokenizer.encode(\"文本\") → Token IDs\n";
        std::cout << "  • model.forward(token_ids) → BERT输出\n";
        std::cout << "  • tokenizer.decode(token_ids) → 文本\n";
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
