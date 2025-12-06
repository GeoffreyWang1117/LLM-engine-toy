#ifndef LLM_ENGINE_SAMPLER_H
#define LLM_ENGINE_SAMPLER_H

#include "matrix.h"
#include <vector>
#include <random>

namespace llm {

/**
 * @brief 采样配置
 */
struct SamplingConfig {
    enum class Strategy {
        GREEDY,      // 贪婪采样：选择概率最高的token
        TEMPERATURE, // 温度采样：调整概率分布的锐度
        TOP_K,       // Top-k采样：从概率最高的k个token中采样
        TOP_P        // Top-p (nucleus)采样：累积概率达到p
    };

    Strategy strategy = Strategy::GREEDY;
    float temp = 1.0f;              // 温度参数（0.1-2.0，默认1.0）
    int k = 50;                     // Top-k参数（默认50）
    float p = 0.9f;                 // Top-p参数（0.0-1.0，默认0.9）
    float repetition_penalty = 1.0f;// Repetition penalty（1.0-2.0，1.0=不惩罚）
    unsigned int seed = 42;         // 随机种子

    /**
     * @brief 创建greedy配置
     */
    static SamplingConfig greedy() {
        SamplingConfig config;
        config.strategy = Strategy::GREEDY;
        return config;
    }

    /**
     * @brief 创建temperature配置
     */
    static SamplingConfig temperature(float temperature) {
        SamplingConfig config;
        config.strategy = Strategy::TEMPERATURE;
        config.temp = temperature;
        return config;
    }

    /**
     * @brief 创建top-k配置
     */
    static SamplingConfig top_k(int top_k) {
        SamplingConfig config;
        config.strategy = Strategy::TOP_K;
        config.k = top_k;
        return config;
    }

    /**
     * @brief 创建top-p配置
     */
    static SamplingConfig top_p(float top_p) {
        SamplingConfig config;
        config.strategy = Strategy::TOP_P;
        config.p = top_p;
        return config;
    }
};

/**
 * @brief 文本生成采样器
 *
 * 从模型输出的logits中采样下一个token。
 *
 * 支持的采样策略：
 *
 * 1. Greedy（贪婪）：
 *    - 选择概率最高的token
 *    - 确定性，可复现
 *    - 缺点：容易陷入重复
 *
 * 2. Temperature（温度）：
 *    - logits' = logits / temperature
 *    - temperature < 1: 更确定（接近greedy）
 *    - temperature = 1: 按原始概率
 *    - temperature > 1: 更随机，更有创造性
 *
 * 3. Top-k：
 *    - 只从概率最高的k个token中采样
 *    - 过滤低概率选项
 *    - k=1等价于greedy
 *
 * 4. Top-p (Nucleus)：
 *    - 选择累积概率达到p的最小token集
 *    - 动态调整候选集大小
 *    - p=1.0使用所有token
 *
 * 使用示例：
 * ```cpp
 * Sampler sampler(42);  // 随机种子
 *
 * // Greedy
 * int token = sampler.sample(logits, SamplingConfig::greedy());
 *
 * // Temperature
 * int token = sampler.sample(logits, SamplingConfig::temperature(0.8));
 *
 * // Top-k
 * int token = sampler.sample(logits, SamplingConfig::top_k(40));
 *
 * // Top-p
 * int token = sampler.sample(logits, SamplingConfig::top_p(0.95));
 * ```
 */
class Sampler {
public:
    /**
     * @brief 构造函数
     *
     * @param seed 随机种子（用于可复现性）
     */
    explicit Sampler(unsigned int seed = 42);

    /**
     * @brief 采样下一个token
     *
     * @param logits 模型输出的logits [vocab_size]
     * @param config 采样配置
     * @param generated_tokens 已生成的tokens（用于repetition penalty）
     * @return 采样的token ID
     */
    int sample(const Matrix& logits, const SamplingConfig& config,
               const std::vector<int>& generated_tokens = {});

    /**
     * @brief Greedy采样（选择概率最高的）
     *
     * @param logits [vocab_size]
     * @return Token ID
     */
    int greedy_sample(const Matrix& logits);

    /**
     * @brief Temperature采样
     *
     * @param logits [vocab_size]
     * @param temperature 温度参数（0.1-2.0）
     * @return Token ID
     */
    int temperature_sample(const Matrix& logits, float temperature);

    /**
     * @brief Top-k采样
     *
     * @param logits [vocab_size]
     * @param k 保留的top-k个候选
     * @return Token ID
     */
    int top_k_sample(const Matrix& logits, int k);

    /**
     * @brief Top-p (Nucleus) 采样
     *
     * @param logits [vocab_size]
     * @param p 累积概率阈值（0.0-1.0）
     * @return Token ID
     */
    int top_p_sample(const Matrix& logits, float p);

    /**
     * @brief 设置随机种子
     */
    void set_seed(unsigned int seed);

private:
    std::mt19937 rng_;  // 随机数生成器

    /**
     * @brief Softmax函数（将logits转换为概率）
     *
     * @param logits [vocab_size]
     * @return 概率分布 [vocab_size]
     */
    std::vector<float> softmax(const Matrix& logits);

    /**
     * @brief 从概率分布中采样
     *
     * @param probs 概率分布
     * @return 采样的索引
     */
    int sample_from_probs(const std::vector<float>& probs);

    /**
     * @brief 获取top-k索引
     *
     * @param values 值数组
     * @param k 保留的top-k个
     * @return top-k索引（降序）
     */
    std::vector<int> get_top_k_indices(const std::vector<float>& values, int k);

    /**
     * @brief 对索引进行排序（按值降序）
     *
     * @param values 值数组
     * @return 排序后的索引
     */
    std::vector<int> argsort_descending(const std::vector<float>& values);

    /**
     * @brief 应用Repetition Penalty
     *
     * 对已生成过的tokens降低其logits，减少重复
     *
     * @param logits 原始logits
     * @param generated_tokens 已生成的token列表
     * @param penalty 惩罚系数（>1.0降低重复概率）
     * @return 应用penalty后的logits
     */
    Matrix apply_repetition_penalty(const Matrix& logits,
                                     const std::vector<int>& generated_tokens,
                                     float penalty) const;
};

} // namespace llm

#endif // LLM_ENGINE_SAMPLER_H
