#ifndef LLM_ENGINE_KV_CACHE_H
#define LLM_ENGINE_KV_CACHE_H

#include "matrix.h"
#include <vector>

namespace llm {

/**
 * @brief KV Cache for auto-regressive generation
 *
 * 在自回归生成时缓存历史的Key和Value，避免重复计算
 *
 * 用途：
 * - GPT-2/LLaMA等decoder模型生成时使用
 * - 每层维护一个KVCache
 * - 每次只需计算新token的Q，复用历史的K和V
 *
 * 性能提升：
 * - 不使用cache: O(n²) - 每次重新计算所有token的attention
 * - 使用cache: O(n) - 只计算新token与历史的attention
 * - 加速比: ~10x（对于长序列）
 *
 * 示例：
 * ```
 * KVCache cache;
 *
 * // 第1个token
 * Matrix K1(1, head_dim), V1(1, head_dim);
 * cache.append(K1, V1);  // cache: [1, head_dim]
 *
 * // 第2个token
 * Matrix K2(1, head_dim), V2(1, head_dim);
 * cache.append(K2, V2);  // cache: [2, head_dim]
 *
 * // 使用cache
 * Matrix Q_new(1, head_dim);  // 新token的Query
 * Matrix scores = Q_new * cache.key_cache.transpose();  // [1, 2]
 * ```
 */
struct KVCache {
    /**
     * @brief 默认构造函数
     */
    KVCache() : key_cache(0, 0), value_cache(0, 0) {}

    /**
     * @brief Key缓存
     *
     * Shape: [seq_len, head_dim]
     * - seq_len: 已生成的token数量（动态增长）
     * - head_dim: 每个头的维度
     */
    Matrix key_cache;

    /**
     * @brief Value缓存
     *
     * Shape: [seq_len, head_dim]
     */
    Matrix value_cache;

    /**
     * @brief 当前缓存的序列长度
     */
    size_t seq_len() const {
        return key_cache.rows();
    }

    /**
     * @brief 检查缓存是否为空
     */
    bool empty() const {
        return key_cache.rows() == 0;
    }

    /**
     * @brief 追加新的K和V到缓存
     *
     * @param new_keys 新token的Key [1, head_dim] 或 [num_new_tokens, head_dim]
     * @param new_values 新token的Value [1, head_dim] 或 [num_new_tokens, head_dim]
     */
    void append(const Matrix& new_keys, const Matrix& new_values);

    /**
     * @brief 清空缓存（开始新的生成时调用）
     */
    void clear() {
        key_cache = Matrix(0, 0);
        value_cache = Matrix(0, 0);
    }

    /**
     * @brief 获取指定范围的Key
     *
     * @param start 起始位置
     * @param end 结束位置（不包含）
     * @return 子矩阵 [end-start, head_dim]
     */
    Matrix get_keys(size_t start, size_t end) const;

    /**
     * @brief 获取指定范围的Value
     */
    Matrix get_values(size_t start, size_t end) const;
};

/**
 * @brief Multi-head KV Cache
 *
 * 为多头注意力维护缓存
 * 每个head一个独立的KVCache
 */
class MultiHeadKVCache {
public:
    MultiHeadKVCache(int num_heads) : caches_(num_heads) {}

    /**
     * @brief 获取指定head的cache
     */
    KVCache& operator[](size_t head_idx) {
        return caches_[head_idx];
    }

    const KVCache& operator[](size_t head_idx) const {
        return caches_[head_idx];
    }

    /**
     * @brief 清空所有head的cache
     */
    void clear() {
        for (auto& cache : caches_) {
            cache.clear();
        }
    }

    /**
     * @brief 获取head数量
     */
    size_t num_heads() const {
        return caches_.size();
    }

    /**
     * @brief 获取当前序列长度（假设所有head长度相同）
     */
    size_t seq_len() const {
        if (caches_.empty()) return 0;
        return caches_[0].seq_len();
    }

private:
    std::vector<KVCache> caches_;
};

} // namespace llm

#endif // LLM_ENGINE_KV_CACHE_H
