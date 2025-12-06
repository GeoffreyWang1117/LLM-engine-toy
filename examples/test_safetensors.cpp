/**
 * @file test_safetensors.cpp
 * @brief 测试safetensors加载器
 *
 * 验证我们能否正确加载BERT权重
 */

#include "llm_engine/safetensors.h"
#include "llm_engine/matrix.h"
#include <iostream>
#include <iomanip>

using namespace llm;

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           SafeTensors加载器测试                               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        // 加载safetensors文件
        std::string model_path = "weights/bert-base-uncased/model.safetensors";

        std::cout << "【1】加载模型文件\n";
        std::cout << "────────────────────────────────────────────────────────────\n";
        SafeTensorsLoader loader(model_path);
        std::cout << "\n";

        // 列出所有tensor
        std::cout << "【2】列出部分tensor\n";
        std::cout << "────────────────────────────────────────────────────────────\n";
        auto tensor_names = loader.list_tensors();
        std::cout << "总共 " << tensor_names.size() << " 个tensors\n\n";

        // 只显示BERT基础模型的tensor（不包括MLM head）
        std::cout << "BERT基础模型tensor（示例）:\n";
        int count = 0;
        for (const auto& name : tensor_names) {
            if (name.find("bert.") == 0 && name.find("pooler") == std::string::npos) {
                const auto& info = loader.get_info(name);
                std::cout << "  " << name << "\n";
                std::cout << "    shape: [";
                for (size_t i = 0; i < info.shape.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << info.shape[i];
                }
                std::cout << "]\n";
                std::cout << "    elements: " << info.num_elements() << "\n";

                count++;
                if (count >= 5) break;  // 只显示前5个
            }
        }
        std::cout << "  ...\n\n";

        // 测试加载embedding权重
        std::cout << "【3】加载word_embeddings权重\n";
        std::cout << "────────────────────────────────────────────────────────────\n";
        Matrix word_emb = loader.get_tensor("bert.embeddings.word_embeddings.weight");
        std::cout << "✅ 成功加载!\n";
        std::cout << "  Shape: [" << word_emb.rows() << " × " << word_emb.cols() << "]\n";
        std::cout << "  (vocab_size × hidden_size)\n";

        // 显示前几个token的embedding
        std::cout << "\n前3个token的embedding (前10维):\n";
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "  Token " << i << ": [";
            for (size_t j = 0; j < 10; ++j) {
                std::cout << std::fixed << std::setprecision(4) << word_emb(i, j);
                if (j < 9) std::cout << ", ";
            }
            std::cout << ", ...]\n";
        }
        std::cout << "\n";

        // 测试加载其他权重
        std::cout << "【4】加载其他关键权重\n";
        std::cout << "────────────────────────────────────────────────────────────\n";

        struct TestCase {
            std::string name;
            std::string description;
        };

        std::vector<TestCase> test_cases = {
            {"bert.embeddings.position_embeddings.weight", "Position Embeddings"},
            {"bert.embeddings.LayerNorm.gamma", "Embedding LayerNorm gamma"},
            {"bert.embeddings.LayerNorm.beta", "Embedding LayerNorm beta"},
            {"bert.encoder.layer.0.attention.self.query.weight", "Layer 0 Query权重"},
            {"bert.encoder.layer.0.attention.self.query.bias", "Layer 0 Query bias"},
            {"bert.pooler.dense.weight", "Pooler Dense权重"},
        };

        for (const auto& test : test_cases) {
            if (loader.has_tensor(test.name)) {
                const auto& info = loader.get_info(test.name);
                std::cout << "✅ " << test.description << "\n";
                std::cout << "   名称: " << test.name << "\n";
                std::cout << "   形状: [";
                for (size_t i = 0; i < info.shape.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << info.shape[i];
                }
                std::cout << "]\n\n";
            }
        }

        // 统计信息
        std::cout << "【5】统计信息\n";
        std::cout << "────────────────────────────────────────────────────────────\n";

        size_t total_params = 0;
        size_t bert_params = 0;

        for (const auto& name : tensor_names) {
            const auto& info = loader.get_info(name);
            total_params += info.num_elements();

            if (name.find("bert.") == 0) {
                bert_params += info.num_elements();
            }
        }

        std::cout << "总参数数: " << total_params << " ("
                  << total_params / 1e6 << "M)\n";
        std::cout << "BERT模型参数: " << bert_params << " ("
                  << bert_params / 1e6 << "M)\n";
        std::cout << "其他参数(MLM head等): " << (total_params - bert_params) << " ("
                  << (total_params - bert_params) / 1e6 << "M)\n";

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           SafeTensors加载器测试通过！                         ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
