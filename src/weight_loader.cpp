#include "llm_engine/weight_loader.h"
#include "llm_engine/embedding.h"
#include "llm_engine/layers.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace llm {

// ============================================================================
// WeightLoader 实现
// ============================================================================

/**
 * 构造函数
 */
WeightLoader::WeightLoader(const std::string& safetensors_path)
    : loader_(safetensors_path)
{
    std::cout << "WeightLoader initialized with " << loader_.list_tensors().size()
              << " tensors" << std::endl;
}

/**
 * 加载权重到BertModel
 */
void WeightLoader::load_to_model(BertModel& model) {
    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  开始加载BERT权重到模型\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "\n";

    // 步骤1: 加载embeddings
    std::cout << "【1/3】加载Embeddings层...\n";
    load_embeddings(model);
    std::cout << "✅ Embeddings加载完成\n\n";

    // 步骤2: 加载encoder
    std::cout << "【2/3】加载Encoder层...\n";
    load_encoder(model);
    std::cout << "✅ Encoder加载完成\n\n";

    // 步骤3: 加载pooler
    std::cout << "【3/3】加载Pooler层...\n";
    load_pooler(model);
    std::cout << "✅ Pooler加载完成\n\n";

    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  BERT权重加载完成！\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
}

/**
 * 加载embeddings层权重
 */
void WeightLoader::load_embeddings(BertModel& model) {
    /*
     * Embeddings包含：
     * 1. Token Embeddings (word_embeddings)
     * 2. Position Embeddings (position_embeddings)
     * 3. Token Type Embeddings (token_type_embeddings)
     * 4. LayerNorm
     */

    // 加载token embeddings
    std::cout << "  - Token Embeddings... ";
    load_embedding(model.token_embeddings(),
                  "bert.embeddings.word_embeddings.weight");
    std::cout << "✓\n";

    // 加载position embeddings
    std::cout << "  - Position Embeddings... ";
    load_embedding(model.position_embeddings(),
                  "bert.embeddings.position_embeddings.weight");
    std::cout << "✓\n";

    // 加载token type embeddings
    std::cout << "  - Token Type Embeddings... ";
    load_embedding(model.token_type_embeddings(),
                  "bert.embeddings.token_type_embeddings.weight");
    std::cout << "✓\n";

    // 加载embeddings LayerNorm
    std::cout << "  - Embeddings LayerNorm... ";
    load_layer_norm(model.embeddings_layer_norm(),
                   "bert.embeddings.LayerNorm.gamma",
                   "bert.embeddings.LayerNorm.beta");
    std::cout << "✓\n";
}

/**
 * 加载encoder层权重
 */
void WeightLoader::load_encoder(BertModel& model) {
    /*
     * Encoder包含N个TransformerBlock
     * 每个block包含：
     * - SelfAttention (Q, K, V, O projections)
     * - LayerNorm1
     * - FeedForward (fc1, fc2)
     * - LayerNorm2
     */

    BertEncoder& encoder = model.encoder();
    size_t num_layers = model.config().num_hidden_layers;

    for (size_t i = 0; i < num_layers; ++i) {
        std::cout << "  - Layer " << i << "... ";

        std::ostringstream prefix;
        prefix << "bert.encoder.layer." << i;
        std::string layer_prefix = prefix.str();

        // 获取当前层的TransformerBlock
        TransformerBlock& block = encoder.layer(i);

        // 加载Attention的Q, K, V, O权重
        load_linear(block.attention().q_proj(),
                   layer_prefix + ".attention.self.query.weight",
                   layer_prefix + ".attention.self.query.bias");

        load_linear(block.attention().k_proj(),
                   layer_prefix + ".attention.self.key.weight",
                   layer_prefix + ".attention.self.key.bias");

        load_linear(block.attention().v_proj(),
                   layer_prefix + ".attention.self.value.weight",
                   layer_prefix + ".attention.self.value.bias");

        load_linear(block.attention().out_proj(),
                   layer_prefix + ".attention.output.dense.weight",
                   layer_prefix + ".attention.output.dense.bias");

        // 加载LayerNorm1
        load_layer_norm(block.ln1(),
                       layer_prefix + ".attention.output.LayerNorm.gamma",
                       layer_prefix + ".attention.output.LayerNorm.beta");

        // 加载FeedForward的fc1和fc2
        load_linear(block.ffn().fc1(),
                   layer_prefix + ".intermediate.dense.weight",
                   layer_prefix + ".intermediate.dense.bias");

        load_linear(block.ffn().fc2(),
                   layer_prefix + ".output.dense.weight",
                   layer_prefix + ".output.dense.bias");

        // 加载LayerNorm2
        load_layer_norm(block.ln2(),
                       layer_prefix + ".output.LayerNorm.gamma",
                       layer_prefix + ".output.LayerNorm.beta");

        std::cout << "✓\n";
    }
}

/**
 * 加载pooler层权重
 */
void WeightLoader::load_pooler(BertModel& model) {
    /*
     * Pooler是一个简单的Linear层
     * 用于将[CLS] token的表示转换为分类用的表示
     */

    std::cout << "  - Pooler Dense... ";
    load_linear(model.pooler(),
               "bert.pooler.dense.weight",
               "bert.pooler.dense.bias");
    std::cout << "✓\n";
}

/**
 * 加载Linear层权重
 */
void WeightLoader::load_linear(Linear& linear,
                               const std::string& weight_name,
                               const std::string& bias_name) {
    /*
     * PyTorch的Linear层：
     * - weight形状: [out_features, in_features]
     * - bias形状: [out_features]
     * - 计算: y = xW^T + b
     *
     * 我们的实现：
     * - weight形状: [out_features, in_features]（一致）
     * - 在forward中也是做xW^T
     * - 所以直接复制即可
     */

    // 加载weight
    Matrix weight = loader_.get_tensor(weight_name);

    // 验证形状
    std::vector<size_t> expected_shape = {linear.weight().rows(), linear.weight().cols()};
    std::vector<size_t> actual_shape = {weight.rows(), weight.cols()};
    verify_shape(expected_shape, actual_shape, weight_name);

    // 复制权重
    linear.weight() = weight;

    // 加载bias（如果有）
    if (!bias_name.empty() && loader_.has_tensor(bias_name)) {
        std::vector<float> bias_data = loader_.get_tensor_1d(bias_name);

        // 验证形状（bias是[out_features, 1]）
        if (bias_data.size() != linear.bias().rows()) {
            throw std::runtime_error(
                "Bias size mismatch for " + bias_name +
                ": expected " + std::to_string(linear.bias().rows()) +
                ", got " + std::to_string(bias_data.size())
            );
        }

        // 复制bias（bias是[out_features, 1]）
        for (size_t i = 0; i < bias_data.size(); ++i) {
            linear.bias()(i, 0) = bias_data[i];
        }
    }
}

/**
 * 加载LayerNorm权重
 */
void WeightLoader::load_layer_norm(LayerNorm& layer_norm,
                                   const std::string& gamma_name,
                                   const std::string& beta_name) {
    /*
     * LayerNorm包含：
     * - gamma (weight): 缩放参数
     * - beta (bias): 偏移参数
     */

    // 加载gamma
    std::vector<float> gamma_data = loader_.get_tensor_1d(gamma_name);

    // 验证形状（weight是[normalized_shape, 1]）
    if (gamma_data.size() != layer_norm.weight().rows()) {
        throw std::runtime_error(
            "Gamma size mismatch for " + gamma_name +
            ": expected " + std::to_string(layer_norm.weight().rows()) +
            ", got " + std::to_string(gamma_data.size())
        );
    }

    // 复制gamma（weight是[normalized_shape, 1]）
    for (size_t i = 0; i < gamma_data.size(); ++i) {
        layer_norm.weight()(i, 0) = gamma_data[i];
    }

    // 加载beta
    std::vector<float> beta_data = loader_.get_tensor_1d(beta_name);

    // 验证形状（bias是[normalized_shape, 1]）
    if (beta_data.size() != layer_norm.bias().rows()) {
        throw std::runtime_error(
            "Beta size mismatch for " + beta_name +
            ": expected " + std::to_string(layer_norm.bias().rows()) +
            ", got " + std::to_string(beta_data.size())
        );
    }

    // 复制beta（bias是[normalized_shape, 1]）
    for (size_t i = 0; i < beta_data.size(); ++i) {
        layer_norm.bias()(i, 0) = beta_data[i];
    }
}

/**
 * 加载Embedding权重
 */
void WeightLoader::load_embedding(Embedding& embedding,
                                  const std::string& weight_name) {
    /*
     * Embedding权重形状: [vocab_size, embedding_dim]
     * 直接对应embedding table
     */

    Matrix weight = loader_.get_tensor(weight_name);

    // 验证形状
    std::vector<size_t> expected_shape = {
        embedding.weight().rows(),
        embedding.weight().cols()
    };
    std::vector<size_t> actual_shape = {weight.rows(), weight.cols()};
    verify_shape(expected_shape, actual_shape, weight_name);

    // 复制权重
    embedding.weight() = weight;
}

/**
 * 验证形状
 */
void WeightLoader::verify_shape(const std::vector<size_t>& expected,
                                const std::vector<size_t>& actual,
                                const std::string& name) const {
    if (expected.size() != actual.size()) {
        throw std::runtime_error(
            "Shape dimension mismatch for " + name +
            ": expected " + std::to_string(expected.size()) + "D" +
            ", got " + std::to_string(actual.size()) + "D"
        );
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::ostringstream oss;
            oss << "Shape mismatch for " << name << " at dimension " << i
                << ": expected " << expected[i]
                << ", got " << actual[i];
            throw std::runtime_error(oss.str());
        }
    }
}

} // namespace llm
