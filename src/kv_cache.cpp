#include "llm_engine/kv_cache.h"
#include <stdexcept>
#include <iostream>

namespace llm {

void KVCache::append(const Matrix& new_keys, const Matrix& new_values) {
    // 检查维度匹配
    if (new_keys.rows() != new_values.rows() ||
        new_keys.cols() != new_values.cols()) {
        throw std::runtime_error(
            "KVCache::append: new_keys and new_values must have same shape"
        );
    }

    // 第一次添加
    if (empty()) {
        key_cache = new_keys;
        value_cache = new_values;
        return;
    }

    // 检查维度兼容性
    if (new_keys.cols() != key_cache.cols()) {
        throw std::runtime_error(
            "KVCache::append: dimension mismatch. Expected " +
            std::to_string(key_cache.cols()) + " but got " +
            std::to_string(new_keys.cols())
        );
    }

    // 追加到现有缓存
    size_t old_seq_len = key_cache.rows();
    size_t new_tokens = new_keys.rows();
    size_t total_seq_len = old_seq_len + new_tokens;
    size_t head_dim = key_cache.cols();

    // 创建新的扩展矩阵
    Matrix new_key_cache(total_seq_len, head_dim);
    Matrix new_value_cache(total_seq_len, head_dim);

    // 复制旧数据
    for (size_t i = 0; i < old_seq_len; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            new_key_cache(i, j) = key_cache(i, j);
            new_value_cache(i, j) = value_cache(i, j);
        }
    }

    // 追加新数据
    for (size_t i = 0; i < new_tokens; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            new_key_cache(old_seq_len + i, j) = new_keys(i, j);
            new_value_cache(old_seq_len + i, j) = new_values(i, j);
        }
    }

    // 更新缓存
    key_cache = new_key_cache;
    value_cache = new_value_cache;
}

Matrix KVCache::get_keys(size_t start, size_t end) const {
    if (end > seq_len()) {
        throw std::out_of_range("KVCache::get_keys: end index out of range");
    }
    if (start >= end) {
        throw std::invalid_argument("KVCache::get_keys: start must be < end");
    }

    size_t len = end - start;
    size_t head_dim = key_cache.cols();
    Matrix result(len, head_dim);

    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            result(i, j) = key_cache(start + i, j);
        }
    }

    return result;
}

Matrix KVCache::get_values(size_t start, size_t end) const {
    if (end > seq_len()) {
        throw std::out_of_range("KVCache::get_values: end index out of range");
    }
    if (start >= end) {
        throw std::invalid_argument("KVCache::get_values: start must be < end");
    }

    size_t len = end - start;
    size_t head_dim = value_cache.cols();
    Matrix result(len, head_dim);

    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            result(i, j) = value_cache(start + i, j);
        }
    }

    return result;
}

} // namespace llm
