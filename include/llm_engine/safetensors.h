#ifndef LLM_ENGINE_SAFETENSORS_H
#define LLM_ENGINE_SAFETENSORS_H

#include "matrix.h"
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <cstdint>

namespace llm {

/**
 * @brief TensorInfo - Safetensors中tensor的元数据
 *
 * 每个tensor包含：
 * - dtype: 数据类型（F32, F16等）
 * - shape: 张量形状
 * - data_offsets: 数据在文件中的位置[start, end)
 */
struct TensorInfo {
    std::string dtype;            // 数据类型，如"F32"
    std::vector<size_t> shape;    // 张量形状，如[768, 768]
    size_t offset_begin;          // 数据起始位置
    size_t offset_end;            // 数据结束位置

    /**
     * @brief 计算tensor的元素数量
     */
    size_t num_elements() const {
        if (shape.empty()) return 0;
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * @brief 计算tensor的字节数
     */
    size_t num_bytes() const {
        return offset_end - offset_begin;
    }
};

/**
 * @brief SafeTensorsLoader - 加载safetensors格式的模型权重
 *
 * Safetensors格式：
 * ┌─────────────────┬──────────────────┬────────────────────┐
 * │ Header Size     │ Header (JSON)    │ Data (binary)      │
 * │ 8 bytes (u64)   │ N bytes          │ M bytes            │
 * └─────────────────┴──────────────────┴────────────────────┘
 *
 * Header是JSON格式，包含所有tensor的元数据：
 * {
 *   "tensor_name": {
 *     "dtype": "F32",
 *     "shape": [768, 768],
 *     "data_offsets": [0, 2359296]
 *   },
 *   "__metadata__": { ... }  // 可选的元数据
 * }
 *
 * 使用方式：
 *   SafeTensorsLoader loader("model.safetensors");
 *   Matrix weights = loader.get_tensor("embeddings.word_embeddings.weight");
 *
 * 优点：
 * - 安全：不执行任意代码
 * - 快速：零拷贝加载
 * - 简单：格式清晰易解析
 */
class SafeTensorsLoader {
public:
    /**
     * @brief 构造函数：加载safetensors文件
     *
     * @param filepath safetensors文件路径
     *
     * 步骤：
     * 1. 读取header长度（前8字节）
     * 2. 读取并解析JSON header
     * 3. 将整个文件映射到内存（或读取到buffer）
     */
    explicit SafeTensorsLoader(const std::string& filepath);

    /**
     * @brief 获取tensor数据并转换为Matrix
     *
     * @param name tensor名称（如"bert.embeddings.word_embeddings.weight"）
     * @return Matrix对象
     *
     * 注意：
     * - 当前仅支持F32类型
     * - 会进行字节序转换（如果需要）
     */
    Matrix get_tensor(const std::string& name) const;

    /**
     * @brief 获取tensor的1D数据（如bias）
     *
     * @param name tensor名称
     * @return vector<float>
     */
    std::vector<float> get_tensor_1d(const std::string& name) const;

    /**
     * @brief 检查tensor是否存在
     */
    bool has_tensor(const std::string& name) const;

    /**
     * @brief 获取所有tensor名称
     */
    std::vector<std::string> list_tensors() const;

    /**
     * @brief 获取tensor信息
     */
    const TensorInfo& get_info(const std::string& name) const;

    /**
     * @brief 获取文件路径
     */
    const std::string& filepath() const { return filepath_; }

private:
    std::string filepath_;                        // 文件路径
    std::map<std::string, TensorInfo> tensors_;   // tensor名称 -> 元数据
    std::vector<uint8_t> data_;                   // 原始二进制数据
    size_t data_offset_;                          // 数据在文件中的起始位置

    /**
     * @brief 解析JSON header
     *
     * 简化的JSON解析器，专门用于safetensors的header格式
     *
     * @param json_str JSON字符串
     */
    void parse_header(const std::string& json_str);

    /**
     * @brief 从二进制数据中提取tensor
     *
     * @param info tensor元数据
     * @return 原始float数据
     */
    std::vector<float> extract_tensor_data(const TensorInfo& info) const;

    /**
     * @brief 简化的JSON解析辅助函数
     */
    std::string parse_string(const std::string& json, size_t& pos) const;
    std::vector<size_t> parse_array(const std::string& json, size_t& pos) const;
    void skip_whitespace(const std::string& json, size_t& pos) const;
};

} // namespace llm

#endif // LLM_ENGINE_SAFETENSORS_H
