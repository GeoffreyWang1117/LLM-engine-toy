/**
 * @file test_bert_variants.cpp
 * @brief 测试不同的BERT模型变种
 *
 * 支持测试：
 * - bert-base-uncased (110M参数, 12层, 768维, 12头)
 * - bert-large-uncased (340M参数, 24层, 1024维, 16头)
 */

#include "llm_engine/bert_tokenizer.h"
#include "llm_engine/bert_model.h"
#include "llm_engine/bert_config.h"
#include "llm_engine/weight_loader.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

using namespace llm;

void print_separator() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
}

struct ModelInfo {
    std::string name;
    std::string weights_path;
    std::string vocab_path;
    BertConfig config;
    int num_params_millions;
};

ModelInfo get_bert_base() {
    ModelInfo info;
    info.name = "BERT-Base-Uncased";
    info.weights_path = "weights/bert-base-uncased/model.safetensors";
    info.vocab_path = "weights/bert-base-uncased/vocab.txt";
    info.config = BertConfig::bert_base();
    info.num_params_millions = 110;
    return info;
}

ModelInfo get_bert_large() {
    ModelInfo info;
    info.name = "BERT-Large-Uncased";
    info.weights_path = "weights/bert-large-uncased/model.safetensors";
    info.vocab_path = "weights/bert-large-uncased/vocab.txt";

    // BERT-Large配置
    info.config.vocab_size = 30522;
    info.config.hidden_size = 1024;
    info.config.num_hidden_layers = 24;
    info.config.num_attention_heads = 16;
    info.config.intermediate_size = 4096;
    info.config.max_position_embeddings = 512;
    info.config.type_vocab_size = 2;

    info.num_params_millions = 340;
    return info;
}

bool test_model(const ModelInfo& model_info) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试模型: " << std::setw(48) << std::left << model_info.name << " ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        print_separator();
        std::cout << "  【1】模型配置\n";
        print_separator();
        std::cout << "\n";
        std::cout << "  名称: " << model_info.name << "\n";
        std::cout << "  参数量: ~" << model_info.num_params_millions << "M\n";
        std::cout << "  层数: " << model_info.config.num_hidden_layers << "\n";
        std::cout << "  隐藏维度: " << model_info.config.hidden_size << "\n";
        std::cout << "  注意力头数: " << model_info.config.num_attention_heads << "\n";
        std::cout << "  前馈层维度: " << model_info.config.intermediate_size << "\n";
        std::cout << "  词表大小: " << model_info.config.vocab_size << "\n";
        std::cout << "\n";

        print_separator();
        std::cout << "  【2】加载Tokenizer\n";
        print_separator();
        std::cout << "\n";

        auto tokenizer_start = std::chrono::high_resolution_clock::now();
        BertTokenizer tokenizer(model_info.vocab_path, true);
        auto tokenizer_end = std::chrono::high_resolution_clock::now();
        auto tokenizer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            tokenizer_end - tokenizer_start).count();

        std::cout << "✅ Tokenizer加载完成 (" << tokenizer_ms << "ms)\n\n";

        print_separator();
        std::cout << "  【3】创建模型\n";
        print_separator();
        std::cout << "\n";

        auto model_start = std::chrono::high_resolution_clock::now();
        BertModel model(model_info.config);
        auto model_end = std::chrono::high_resolution_clock::now();
        auto model_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            model_end - model_start).count();

        std::cout << "✅ 模型创建完成 (" << model_ms << "ms)\n\n";

        print_separator();
        std::cout << "  【4】加载预训练权重\n";
        print_separator();
        std::cout << "\n";

        auto weights_start = std::chrono::high_resolution_clock::now();
        WeightLoader loader(model_info.weights_path);
        loader.load_to_model(model);
        auto weights_end = std::chrono::high_resolution_clock::now();
        auto weights_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            weights_end - weights_start).count();

        std::cout << "✅ 权重加载完成 (" << weights_ms << "ms)\n\n";

        print_separator();
        std::cout << "  【5】推理测试\n";
        print_separator();
        std::cout << "\n";

        std::vector<std::string> test_texts = {
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing with transformers."
        };

        for (const auto& text : test_texts) {
            std::cout << "输入: \"" << text << "\"\n";

            // 编码
            auto encode_start = std::chrono::high_resolution_clock::now();
            auto input_ids = tokenizer.encode(text);
            auto encode_end = std::chrono::high_resolution_clock::now();
            auto encode_us = std::chrono::duration_cast<std::chrono::microseconds>(
                encode_end - encode_start).count();

            std::cout << "  Token数量: " << input_ids.size() << " (编码耗时: " << encode_us << "μs)\n";

            // 推理
            auto infer_start = std::chrono::high_resolution_clock::now();
            auto output = model.forward(input_ids);
            auto infer_end = std::chrono::high_resolution_clock::now();
            auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                infer_end - infer_start).count();

            std::cout << "  输出形状: [" << output.sequence_output.rows() << " × "
                      << output.sequence_output.cols() << "]";
            std::cout << " (推理耗时: " << infer_ms << "ms)\n";

            // 显示[CLS]向量的前5维
            std::cout << "  [CLS]向量(前5维): [";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << std::fixed << std::setprecision(4) << output.pooled_output(0, i);
                if (i < 4) std::cout << ", ";
            }
            std::cout << ", ...]\n\n";
        }

        print_separator();
        std::cout << "\n";
        std::cout << "✅ " << model_info.name << " 测试通过！\n";
        std::cout << "\n";

        return true;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 错误: " << e.what() << "\n";
        std::cerr << "   可能原因: 权重文件不存在或格式不匹配\n\n";
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           BERT模型变种测试                                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    std::vector<ModelInfo> models_to_test;

    // 解析命令行参数
    if (argc > 1) {
        std::string model_name = argv[1];
        if (model_name == "base") {
            models_to_test.push_back(get_bert_base());
        } else if (model_name == "large") {
            models_to_test.push_back(get_bert_large());
        } else if (model_name == "all") {
            models_to_test.push_back(get_bert_base());
            models_to_test.push_back(get_bert_large());
        } else {
            std::cerr << "\n用法: " << argv[0] << " [base|large|all]\n";
            std::cerr << "  base  - 测试BERT-Base (110M参数)\n";
            std::cerr << "  large - 测试BERT-Large (340M参数)\n";
            std::cerr << "  all   - 测试所有变种\n";
            std::cerr << "  (默认: base)\n\n";
            return 1;
        }
    } else {
        // 默认测试bert-base
        models_to_test.push_back(get_bert_base());
    }

    int success_count = 0;
    int total_count = models_to_test.size();

    for (const auto& model : models_to_test) {
        if (test_model(model)) {
            success_count++;
        }
    }

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  测试总结                                                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "  总计: " << total_count << " 个模型\n";
    std::cout << "  成功: " << success_count << " 个\n";
    std::cout << "  失败: " << (total_count - success_count) << " 个\n";
    std::cout << "\n";

    if (success_count == total_count) {
        std::cout << "✅ 所有测试通过！\n\n";
        return 0;
    } else {
        std::cout << "⚠️  部分测试失败\n\n";
        return 1;
    }
}
