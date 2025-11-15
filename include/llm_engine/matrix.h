#ifndef LLM_ENGINE_MATRIX_H
#define LLM_ENGINE_MATRIX_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace llm {

class Matrix {
public:
    // 构造函数
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, float value);
    Matrix(size_t rows, size_t cols, const std::vector<float>& data);

    // 获取维度
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    // 访问元素
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;

    // 获取数据指针
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // 矩阵运算
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(float scalar) const;

    // 矩阵乘法
    Matrix matmul(const Matrix& other) const;

    // 转置
    Matrix transpose() const;

    // 工具函数
    void fill(float value);
    void print() const;

    // 静态工厂方法
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix randn(size_t rows, size_t cols, float mean = 0.0f, float stddev = 1.0f);

private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;

    size_t index(size_t row, size_t col) const {
        return row * cols_ + col;
    }
};

} // namespace llm

#endif // LLM_ENGINE_MATRIX_H
