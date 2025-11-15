#include "llm_engine/matrix.h"
#include <random>
#include <iomanip>

namespace llm {

// 构造函数
Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {
}

Matrix::Matrix(size_t rows, size_t cols, float value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {
}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<float>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size doesn't match matrix dimensions");
    }
}

// 访问元素
float& Matrix::at(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[index(row, col)];
}

const float& Matrix::at(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[index(row, col)];
}

float& Matrix::operator()(size_t row, size_t col) {
    return at(row, col);
}

const float& Matrix::operator()(size_t row, size_t col) const {
    return at(row, col);
}

// 矩阵加法
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// 矩阵减法
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

// 标量乘法
Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

// 矩阵乘法 (简单的三重循环实现)
Matrix Matrix::matmul(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix result(rows_, other.cols_, 0.0f);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_; ++k) {
                sum += at(i, k) * other.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }

    return result;
}

// 转置
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.at(j, i) = at(i, j);
        }
    }
    return result;
}

// 填充
void Matrix::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

// 打印矩阵
void Matrix::print() const {
    std::cout << "Matrix [" << rows_ << " x " << cols_ << "]:" << std::endl;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << std::setw(10) << std::setprecision(4) << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// 静态工厂方法
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0f);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0f);
}

Matrix Matrix::randn(size_t rows, size_t cols, float mean, float stddev) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);

    Matrix result(rows, cols);
    for (size_t i = 0; i < result.size(); ++i) {
        result.data_[i] = dist(gen);
    }
    return result;
}

} // namespace llm
