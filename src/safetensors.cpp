#include "llm_engine/safetensors.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace llm {

// ============================================================================
// SafeTensorsLoader 实现
// ============================================================================

/**
 * 构造函数：加载safetensors文件
 */
SafeTensorsLoader::SafeTensorsLoader(const std::string& filepath)
    : filepath_(filepath), data_offset_(0)
{
    std::cout << "Loading safetensors file: " << filepath << std::endl;

    // 打开文件
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    /*
     * 步骤1: 读取header长度
     *
     * Safetensors格式的前8字节是header的长度
     * - 类型：uint64_t
     * - 字节序：little-endian
     */
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), 8);

    if (!file) {
        throw std::runtime_error("Failed to read header size");
    }

    std::cout << "  Header size: " << header_size << " bytes" << std::endl;

    /*
     * 步骤2: 读取header（JSON格式）
     */
    std::vector<char> header_buffer(header_size);
    file.read(header_buffer.data(), header_size);

    if (!file) {
        throw std::runtime_error("Failed to read header");
    }

    std::string header_json(header_buffer.begin(), header_buffer.end());

    /*
     * 步骤3: 解析JSON header
     *
     * header包含所有tensor的元数据：
     * - dtype: 数据类型
     * - shape: 形状
     * - data_offsets: 数据位置
     */
    parse_header(header_json);

    std::cout << "  Found " << tensors_.size() << " tensors" << std::endl;

    /*
     * 步骤4: 读取所有二进制数据
     *
     * 为了简化实现，我们将所有数据读入内存
     * 对于大模型，可以考虑mmap或按需加载
     */
    data_offset_ = 8 + header_size;

    // 获取文件总大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t data_size = file_size - data_offset_;

    std::cout << "  Data size: " << data_size / (1024.0 * 1024.0)
              << " MB" << std::endl;

    // 读取数据
    file.seekg(data_offset_);
    data_.resize(data_size);
    file.read(reinterpret_cast<char*>(data_.data()), data_size);

    if (!file) {
        throw std::runtime_error("Failed to read tensor data");
    }

    file.close();

    std::cout << "✅ Safetensors file loaded successfully" << std::endl;
}

/**
 * 解析JSON header
 *
 * 简化版：使用字符串查找提取信息
 */
void SafeTensorsLoader::parse_header(const std::string& json_str) {
    /*
     * 使用更简单的方法：逐个查找tensor定义
     *
     * 格式："tensor_name":{"dtype":"F32","shape":[...],"data_offsets":[...]}
     */

    size_t pos = 0;
    while (true) {
        // 查找下一个tensor名称（以引号开始）
        pos = json_str.find('"', pos);
        if (pos == std::string::npos) break;

        pos++;  // 跳过引号

        // 提取tensor名称
        size_t name_end = json_str.find('"', pos);
        if (name_end == std::string::npos) break;

        std::string tensor_name = json_str.substr(pos, name_end - pos);
        pos = name_end + 1;

        // 跳过__metadata__
        if (tensor_name == "__metadata__") {
            // 找到对应的右大括号
            size_t meta_colon = json_str.find(':', pos);
            size_t meta_brace = json_str.find('{', meta_colon);
            int brace_cnt = 1;
            size_t meta_end = meta_brace + 1;
            while (meta_end < json_str.size() && brace_cnt > 0) {
                if (json_str[meta_end] == '{') brace_cnt++;
                else if (json_str[meta_end] == '}') brace_cnt--;
                meta_end++;
            }
            pos = meta_end;
            continue;
        }

        // 查找这个tensor的数据
        // 格式: "tensor_name":{"dtype":"...","shape":[...],"data_offsets":[...]}

        // 找到冒号
        size_t colon_pos = json_str.find(':', pos);
        if (colon_pos == std::string::npos) break;

        // 找到左大括号
        size_t brace_start = json_str.find('{', colon_pos);
        if (brace_start == std::string::npos) break;

        // 找到对应的右大括号
        int brace_count = 1;
        size_t brace_end = brace_start + 1;
        while (brace_end < json_str.size() && brace_count > 0) {
            if (json_str[brace_end] == '{') brace_count++;
            else if (json_str[brace_end] == '}') brace_count--;
            brace_end++;
        }

        if (brace_count != 0) break;

        std::string tensor_info = json_str.substr(brace_start, brace_end - brace_start);

        // 解析dtype
        TensorInfo info;
        size_t dtype_pos = tensor_info.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dtype_val_start = tensor_info.find('"', dtype_pos + 7);
            size_t dtype_val_end = tensor_info.find('"', dtype_val_start + 1);
            info.dtype = tensor_info.substr(dtype_val_start + 1, dtype_val_end - dtype_val_start - 1);
        }

        // 解析shape
        size_t shape_pos = tensor_info.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t shape_start = tensor_info.find('[', shape_pos);
            size_t shape_end = tensor_info.find(']', shape_start);
            std::string shape_str = tensor_info.substr(shape_start + 1, shape_end - shape_start - 1);

            // 解析数字
            std::istringstream iss(shape_str);
            std::string num;
            while (std::getline(iss, num, ',')) {
                // 去除空格
                num.erase(0, num.find_first_not_of(" \t\n\r"));
                num.erase(num.find_last_not_of(" \t\n\r") + 1);
                if (!num.empty()) {
                    info.shape.push_back(std::stoull(num));
                }
            }
        }

        // 解析data_offsets
        size_t offsets_pos = tensor_info.find("\"data_offsets\"");
        if (offsets_pos != std::string::npos) {
            size_t offsets_start = tensor_info.find('[', offsets_pos);
            size_t offsets_end = tensor_info.find(']', offsets_start);
            std::string offsets_str = tensor_info.substr(offsets_start + 1, offsets_end - offsets_start - 1);

            // 解析两个数字
            std::istringstream iss(offsets_str);
            std::string num;
            std::vector<size_t> offsets;
            while (std::getline(iss, num, ',')) {
                num.erase(0, num.find_first_not_of(" \t\n\r"));
                num.erase(num.find_last_not_of(" \t\n\r") + 1);
                if (!num.empty()) {
                    offsets.push_back(std::stoull(num));
                }
            }

            if (offsets.size() == 2) {
                info.offset_begin = offsets[0];
                info.offset_end = offsets[1];
            }
        }

        // 存储tensor info
        tensors_[tensor_name] = info;

        pos = brace_end;
    }
}

/**
 * 解析JSON字符串
 */
std::string SafeTensorsLoader::parse_string(const std::string& json, size_t& pos) const {
    if (json[pos] != '"') {
        throw std::runtime_error("Expected '\"' at position " + std::to_string(pos));
    }
    pos++;  // 跳过开始的引号

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        result += json[pos];
        pos++;
    }

    if (pos >= json.size()) {
        throw std::runtime_error("Unterminated string");
    }

    pos++;  // 跳过结束的引号
    return result;
}

/**
 * 解析JSON数组
 */
std::vector<size_t> SafeTensorsLoader::parse_array(const std::string& json, size_t& pos) const {
    if (json[pos] != '[') {
        throw std::runtime_error("Expected '[' at position " + std::to_string(pos));
    }
    pos++;  // 跳过'['

    std::vector<size_t> result;

    skip_whitespace(json, pos);

    while (pos < json.size() && json[pos] != ']') {
        skip_whitespace(json, pos);

        // 读取数字
        size_t num_start = pos;
        while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '-')) {
            pos++;
        }

        std::string num_str = json.substr(num_start, pos - num_start);
        result.push_back(std::stoull(num_str));

        skip_whitespace(json, pos);

        if (json[pos] == ',') {
            pos++;
        }
    }

    if (pos >= json.size()) {
        throw std::runtime_error("Unterminated array");
    }

    pos++;  // 跳过']'
    return result;
}

/**
 * 跳过空白字符
 */
void SafeTensorsLoader::skip_whitespace(const std::string& json, size_t& pos) const {
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) {
        pos++;
    }
}

/**
 * 提取tensor数据
 */
std::vector<float> SafeTensorsLoader::extract_tensor_data(const TensorInfo& info) const {
    /*
     * 当前仅支持F32类型
     *
     * 未来可以扩展支持：
     * - F16 (half precision)
     * - I32, I64 (整数)
     * - BF16 (bfloat16)
     */
    if (info.dtype != "F32") {
        throw std::runtime_error("Unsupported dtype: " + info.dtype + " (only F32 supported)");
    }

    size_t num_elements = info.num_elements();
    size_t num_bytes = info.num_bytes();

    // 验证大小匹配
    if (num_bytes != num_elements * sizeof(float)) {
        throw std::runtime_error(
            "Size mismatch: expected " + std::to_string(num_elements * sizeof(float)) +
            " bytes, got " + std::to_string(num_bytes)
        );
    }

    // 提取数据
    std::vector<float> data(num_elements);
    const uint8_t* src = data_.data() + info.offset_begin;
    std::memcpy(data.data(), src, num_bytes);

    return data;
}

/**
 * 获取tensor并转换为Matrix
 */
Matrix SafeTensorsLoader::get_tensor(const std::string& name) const {
    if (!has_tensor(name)) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const TensorInfo& info = tensors_.at(name);
    std::vector<float> data = extract_tensor_data(info);

    // 转换为Matrix
    if (info.shape.size() == 1) {
        // 1D tensor -> [1, n] matrix
        return Matrix(1, info.shape[0], data);
    } else if (info.shape.size() == 2) {
        // 2D tensor -> [m, n] matrix
        return Matrix(info.shape[0], info.shape[1], data);
    } else {
        throw std::runtime_error(
            "Unsupported tensor shape (only 1D and 2D supported): " + name
        );
    }
}

/**
 * 获取1D tensor数据
 */
std::vector<float> SafeTensorsLoader::get_tensor_1d(const std::string& name) const {
    if (!has_tensor(name)) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const TensorInfo& info = tensors_.at(name);
    return extract_tensor_data(info);
}

/**
 * 检查tensor是否存在
 */
bool SafeTensorsLoader::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

/**
 * 列出所有tensor名称
 */
std::vector<std::string> SafeTensorsLoader::list_tensors() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

/**
 * 获取tensor信息
 */
const TensorInfo& SafeTensorsLoader::get_info(const std::string& name) const {
    if (!has_tensor(name)) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return tensors_.at(name);
}

} // namespace llm
