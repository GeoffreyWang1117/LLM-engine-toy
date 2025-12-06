#ifndef LLM_ENGINE_WEIGHT_LOADER_H
#define LLM_ENGINE_WEIGHT_LOADER_H

#include "safetensors.h"
#include "bert_model.h"
#include <string>

namespace llm {

/**
 * @brief WeightLoader - 从safetensors加载权重到BertModel
 *
 * 负责将Hugging Face格式的BERT权重映射到我们的C++模型结构
 *
 * 权重映射关系：
 * Hugging Face                          → Our C++ Structure
 * ═══════════════════════════════════════════════════════════════
 * bert.embeddings.word_embeddings.weight
 *   → BertModel::token_embeddings_
 *
 * bert.embeddings.position_embeddings.weight
 *   → BertModel::position_embeddings_
 *
 * bert.embeddings.token_type_embeddings.weight
 *   → BertModel::token_type_embeddings_
 *
 * bert.embeddings.LayerNorm.gamma/beta
 *   → BertModel::embeddings_layer_norm_
 *
 * bert.encoder.layer.{i}.attention.self.query.weight/bias
 *   → BertEncoder::layers_[{i}].attention_.q_proj_
 *
 * bert.encoder.layer.{i}.attention.self.key.weight/bias
 *   → BertEncoder::layers_[{i}].attention_.k_proj_
 *
 * bert.encoder.layer.{i}.attention.self.value.weight/bias
 *   → BertEncoder::layers_[{i}].attention_.v_proj_
 *
 * bert.encoder.layer.{i}.attention.output.dense.weight/bias
 *   → BertEncoder::layers_[{i}].attention_.out_proj_
 *
 * bert.encoder.layer.{i}.attention.output.LayerNorm.gamma/beta
 *   → BertEncoder::layers_[{i}].ln1_
 *
 * bert.encoder.layer.{i}.intermediate.dense.weight/bias
 *   → BertEncoder::layers_[{i}].ffn_.fc1_
 *
 * bert.encoder.layer.{i}.output.dense.weight/bias
 *   → BertEncoder::layers_[{i}].ffn_.fc2_
 *
 * bert.encoder.layer.{i}.output.LayerNorm.gamma/beta
 *   → BertEncoder::layers_[{i}].ln2_
 *
 * bert.pooler.dense.weight/bias
 *   → BertModel::pooler_dense_
 *
 * 使用方式：
 *   BertModel model(config);
 *   WeightLoader loader("model.safetensors");
 *   loader.load_to_model(model);
 */
class WeightLoader {
public:
    /**
     * @brief 构造函数
     *
     * @param safetensors_path safetensors文件路径
     */
    explicit WeightLoader(const std::string& safetensors_path);

    /**
     * @brief 加载权重到BertModel
     *
     * @param model 要加载权重的模型
     *
     * 步骤：
     * 1. 加载embeddings层权重
     * 2. 加载encoder层权重（所有TransformerBlock）
     * 3. 加载pooler层权重
     * 4. 验证所有权重形状匹配
     */
    void load_to_model(BertModel& model);

    /**
     * @brief 获取SafeTensorsLoader
     */
    const SafeTensorsLoader& loader() const { return loader_; }

private:
    SafeTensorsLoader loader_;  // safetensors加载器

    /**
     * @brief 加载embeddings层权重
     */
    void load_embeddings(BertModel& model);

    /**
     * @brief 加载encoder层权重
     */
    void load_encoder(BertModel& model);

    /**
     * @brief 加载pooler层权重
     */
    void load_pooler(BertModel& model);

    /**
     * @brief 加载Linear层权重
     *
     * @param linear Linear层对象
     * @param weight_name 权重tensor名称
     * @param bias_name bias tensor名称（如果有）
     */
    void load_linear(Linear& linear,
                    const std::string& weight_name,
                    const std::string& bias_name);

    /**
     * @brief 加载LayerNorm权重
     *
     * @param layer_norm LayerNorm层对象
     * @param gamma_name gamma tensor名称
     * @param beta_name beta tensor名称
     */
    void load_layer_norm(LayerNorm& layer_norm,
                        const std::string& gamma_name,
                        const std::string& beta_name);

    /**
     * @brief 加载Embedding权重
     *
     * @param embedding Embedding层对象
     * @param weight_name 权重tensor名称
     */
    void load_embedding(Embedding& embedding,
                       const std::string& weight_name);

    /**
     * @brief 验证形状是否匹配
     *
     * @param expected 期望的形状
     * @param actual 实际的形状
     * @param name tensor名称（用于错误信息）
     */
    void verify_shape(const std::vector<size_t>& expected,
                     const std::vector<size_t>& actual,
                     const std::string& name) const;
};

} // namespace llm

#endif // LLM_ENGINE_WEIGHT_LOADER_H
