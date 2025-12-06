#ifndef LLM_ENGINE_GPT2_MODEL_H
#define LLM_ENGINE_GPT2_MODEL_H

#include "matrix.h"
#include "gpt_decoder_block.h"
#include "kv_cache.h"
#include "sampler.h"
#include <vector>
#include <memory>
#include <string>

namespace llm {

/**
 * @brief GPT-2模型配置
 */
struct GPT2Config {
    int vocab_size = 50257;              // 词汇表大小
    int hidden_size = 768;               // 隐藏层维度
    int num_layers = 12;                 // Decoder层数
    int num_heads = 12;                  // 注意力头数
    int intermediate_size = 3072;        // FFN中间层大小
    int max_position_embeddings = 1024;  // 最大位置编码
    float dropout = 0.1f;                // Dropout率
    float layer_norm_epsilon = 1e-5f;    // LayerNorm epsilon

    /**
     * @brief GPT-2模型变体配置
     */
    static GPT2Config gpt2_small() {
        GPT2Config config;
        config.vocab_size = 50257;
        config.hidden_size = 768;
        config.num_layers = 12;
        config.num_heads = 12;
        config.intermediate_size = 3072;
        return config;
    }

    static GPT2Config gpt2_medium() {
        GPT2Config config;
        config.vocab_size = 50257;
        config.hidden_size = 1024;
        config.num_layers = 24;
        config.num_heads = 16;
        config.intermediate_size = 4096;
        return config;
    }

    static GPT2Config gpt2_large() {
        GPT2Config config;
        config.vocab_size = 50257;
        config.hidden_size = 1280;
        config.num_layers = 36;
        config.num_heads = 20;
        config.intermediate_size = 5120;
        return config;
    }

    static GPT2Config gpt2_xl() {
        GPT2Config config;
        config.vocab_size = 50257;
        config.hidden_size = 1600;
        config.num_layers = 48;
        config.num_heads = 25;
        config.intermediate_size = 6400;
        return config;
    }
};

/**
 * @brief 文本生成配置
 */
struct GenerationConfig {
    // 采样策略配置
    SamplingConfig sampling;

    // 长度控制
    int max_new_tokens = 50;          // 最大生成token数（不包括prompt）
    int min_length = 0;               // 最小总长度（包括prompt）
    int max_length = 1024;            // 最大总长度（包括prompt）

    // 停止条件
    int eos_token_id = -1;            // EOS token ID，-1表示不使用
    std::vector<int> stop_token_ids;  // 多个停止token IDs

    // N-gram重复控制
    int no_repeat_ngram_size = 0;     // 防止重复的n-gram大小，0表示不使用

    // 随机种子
    unsigned int seed = 42;

    /**
     * @brief 创建默认greedy配置
     */
    static GenerationConfig greedy(int max_new_tokens = 50) {
        GenerationConfig config;
        config.sampling = SamplingConfig::greedy();
        config.max_new_tokens = max_new_tokens;
        return config;
    }

    /**
     * @brief 创建temperature配置
     */
    static GenerationConfig temperature(float temp, int max_new_tokens = 50) {
        GenerationConfig config;
        config.sampling = SamplingConfig::temperature(temp);
        config.max_new_tokens = max_new_tokens;
        return config;
    }

    /**
     * @brief 创建top-k配置
     */
    static GenerationConfig top_k(int k, int max_new_tokens = 50) {
        GenerationConfig config;
        config.sampling = SamplingConfig::top_k(k);
        config.max_new_tokens = max_new_tokens;
        return config;
    }

    /**
     * @brief 创建top-p配置
     */
    static GenerationConfig top_p(float p, int max_new_tokens = 50) {
        GenerationConfig config;
        config.sampling = SamplingConfig::top_p(p);
        config.max_new_tokens = max_new_tokens;
        return config;
    }
};

/**
 * @brief GPT-2模型
 *
 * 完整的GPT-2 Transformer Decoder架构。
 *
 * **架构：**
 * ```
 * Input IDs [batch, seq_len]
 *   ↓
 * Token Embedding + Position Embedding
 *   ↓
 * Decoder Block 1 (Causal Attention + FFN)
 *   ↓
 * Decoder Block 2
 *   ↓
 * ...
 *   ↓
 * Decoder Block N
 *   ↓
 * Final LayerNorm
 *   ↓
 * LM Head (Linear: hidden -> vocab_size)
 *   ↓
 * Logits [batch, seq_len, vocab_size]
 * ```
 *
 * **关键特性：**
 * 1. Causal (自回归) 注意力机制
 * 2. Pre-LayerNorm架构
 * 3. GELU激活函数
 * 4. KV Cache支持（高效生成）
 *
 * **使用示例：**
 * ```cpp
 * // 创建模型
 * GPT2Model model(GPT2Config::gpt2_small());
 *
 * // 加载权重
 * model.load_weights("gpt2.safetensors");
 *
 * // Forward推理
 * std::vector<int> input_ids = {15496, 995};  // "Hello world"
 * Matrix logits = model.forward(input_ids);
 *
 * // 文本生成（新API）
 * GenerationConfig gen_config = GenerationConfig::temperature(0.8, 50);
 * gen_config.sampling.repetition_penalty = 1.2f;
 * gen_config.eos_token_id = 50256;
 * std::vector<int> generated = model.generate(input_ids, gen_config);
 * ```
 */
class GPT2Model {
public:
    /**
     * @brief 构造函数
     *
     * @param config 模型配置
     */
    explicit GPT2Model(const GPT2Config& config);

    /**
     * @brief Forward推理
     *
     * @param input_ids Token IDs [seq_len]
     * @return Logits [seq_len, vocab_size]
     */
    Matrix forward(const std::vector<int>& input_ids);

    /**
     * @brief 文本生成（新API，推荐使用）
     *
     * @param input_ids 输入token IDs（prompt）
     * @param config 生成配置
     * @return 生成的完整序列（包括prompt）
     */
    std::vector<int> generate(
        const std::vector<int>& input_ids,
        const GenerationConfig& config
    );

    /**
     * @brief 文本生成（旧API，保持向后兼容）
     *
     * @param input_ids 输入token IDs（prompt）
     * @param max_length 最大生成长度
     * @param config 采样配置
     * @param seed 随机种子
     * @return 生成的完整序列（包括prompt）
     */
    std::vector<int> generate(
        const std::vector<int>& input_ids,
        int max_length,
        const SamplingConfig& config = SamplingConfig::greedy(),
        unsigned int seed = 42
    );

    /**
     * @brief 从SafeTensors文件加载权重
     *
     * @param weights_path SafeTensors文件路径
     */
    void load_weights(const std::string& weights_path);

    /**
     * @brief 获取配置
     */
    const GPT2Config& config() const { return config_; }

    /**
     * @brief 设置Token Embedding权重
     *
     * @param wte Token Embedding矩阵 [vocab_size, hidden_size]
     */
    void set_token_embedding(const Matrix& wte);

    /**
     * @brief 设置Position Embedding权重
     *
     * @param wpe Position Embedding矩阵 [max_position, hidden_size]
     */
    void set_position_embedding(const Matrix& wpe);

    /**
     * @brief 设置Final LayerNorm权重
     *
     * @param gamma Gamma参数 [hidden_size]
     * @param beta Beta参数 [hidden_size]
     */
    void set_final_layer_norm(const Matrix& gamma, const Matrix& beta);

private:
    GPT2Config config_;

    // Token Embedding: [vocab_size, hidden_size]
    Matrix token_embedding_;

    // Position Embedding: [max_position, hidden_size]
    Matrix position_embedding_;

    // Decoder Blocks
    std::vector<std::unique_ptr<GPTDecoderBlock>> layers_;

    // Final LayerNorm
    Matrix ln_f_gamma_;  // [hidden_size]
    Matrix ln_f_beta_;   // [hidden_size]

    // Note: LM Head通常与Token Embedding共享权重（tied weights）
    // 所以我们复用token_embedding_作为LM Head

    /**
     * @brief LayerNorm
     *
     * @param x 输入 [seq_len, hidden_size]
     * @param gamma Gamma参数
     * @param beta Beta参数
     * @return 归一化后的结果
     */
    Matrix layer_norm(const Matrix& x, const Matrix& gamma, const Matrix& beta) const;

    /**
     * @brief 应用Embedding
     *
     * @param input_ids Token IDs [seq_len]
     * @return Embedding [seq_len, hidden_size]
     */
    Matrix apply_embeddings(const std::vector<int>& input_ids) const;

    /**
     * @brief 应用LM Head
     *
     * @param hidden_states [seq_len, hidden_size]
     * @return Logits [seq_len, vocab_size]
     */
    Matrix apply_lm_head(const Matrix& hidden_states) const;

    /**
     * @brief 检查是否应该停止生成
     *
     * @param generated_ids 已生成的序列
     * @param config 生成配置
     * @return 是否应该停止
     */
    bool should_stop(const std::vector<int>& generated_ids,
                     const GenerationConfig& config) const;

    /**
     * @brief 获取被n-gram重复限制的token IDs
     *
     * @param generated_ids 已生成的序列
     * @param ngram_size n-gram大小
     * @return 被禁止的token IDs集合
     */
    std::vector<int> get_banned_ngrams(const std::vector<int>& generated_ids,
                                        int ngram_size) const;
};

} // namespace llm

#endif // LLM_ENGINE_GPT2_MODEL_H
