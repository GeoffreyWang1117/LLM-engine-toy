#include "llm_engine/matrix.h"
#include "llm_engine/layers.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>

using namespace emscripten;
using namespace llm;

// 包装函数，方便从JavaScript调用

// Matrix相关函数
val matrix_to_array(const Matrix& mat) {
    val result = val::array();
    for (size_t i = 0; i < mat.rows(); ++i) {
        val row = val::array();
        for (size_t j = 0; j < mat.cols(); ++j) {
            row.call<void>("push", mat.at(i, j));
        }
        result.call<void>("push", row);
    }
    return result;
}

Matrix array_to_matrix(const val& arr) {
    size_t rows = arr["length"].as<size_t>();
    if (rows == 0) {
        return Matrix(0, 0);
    }

    val first_row = arr[0];
    size_t cols = first_row["length"].as<size_t>();

    std::vector<float> data;
    data.reserve(rows * cols);

    for (size_t i = 0; i < rows; ++i) {
        val row = arr[i];
        for (size_t j = 0; j < cols; ++j) {
            data.push_back(row[j].as<float>());
        }
    }

    return Matrix(rows, cols, data);
}

// 包装类，用于提供更友好的JavaScript接口
class MatrixWrapper {
public:
    MatrixWrapper(size_t rows, size_t cols) : mat_(rows, cols) {}
    MatrixWrapper(const Matrix& mat) : mat_(mat) {}

    size_t rows() const { return mat_.rows(); }
    size_t cols() const { return mat_.cols(); }

    float get(size_t row, size_t col) const {
        return mat_.at(row, col);
    }

    void set(size_t row, size_t col, float value) {
        mat_.at(row, col) = value;
    }

    val toArray() const {
        return matrix_to_array(mat_);
    }

    MatrixWrapper add(const MatrixWrapper& other) const {
        return MatrixWrapper(mat_ + other.mat_);
    }

    MatrixWrapper matmul(const MatrixWrapper& other) const {
        return MatrixWrapper(mat_.matmul(other.mat_));
    }

    MatrixWrapper transpose() const {
        return MatrixWrapper(mat_.transpose());
    }

    static MatrixWrapper fromArray(const val& arr) {
        return MatrixWrapper(array_to_matrix(arr));
    }

    static MatrixWrapper zeros(size_t rows, size_t cols) {
        return MatrixWrapper(Matrix::zeros(rows, cols));
    }

    static MatrixWrapper ones(size_t rows, size_t cols) {
        return MatrixWrapper(Matrix::ones(rows, cols));
    }

    static MatrixWrapper randn(size_t rows, size_t cols) {
        return MatrixWrapper(Matrix::randn(rows, cols));
    }

    const Matrix& getMatrix() const { return mat_; }

private:
    Matrix mat_;
};

// Linear层包装
class LinearWrapper {
public:
    LinearWrapper(size_t in_features, size_t out_features)
        : layer_(in_features, out_features) {}

    MatrixWrapper forward(const MatrixWrapper& input) {
        return MatrixWrapper(layer_.forward(input.getMatrix()));
    }

private:
    Linear layer_;
};

// LayerNorm包装
class LayerNormWrapper {
public:
    LayerNormWrapper(size_t normalized_shape)
        : layer_(normalized_shape) {}

    MatrixWrapper forward(const MatrixWrapper& input) {
        return MatrixWrapper(layer_.forward(input.getMatrix()));
    }

private:
    LayerNorm layer_;
};

// SelfAttention包装
class SelfAttentionWrapper {
public:
    SelfAttentionWrapper(size_t embed_dim, size_t num_heads)
        : layer_(embed_dim, num_heads) {}

    MatrixWrapper forward(const MatrixWrapper& input) {
        return MatrixWrapper(layer_.forward(input.getMatrix()));
    }

private:
    SelfAttention layer_;
};

// FeedForward包装
class FeedForwardWrapper {
public:
    FeedForwardWrapper(size_t embed_dim, size_t hidden_dim)
        : layer_(embed_dim, hidden_dim) {}

    MatrixWrapper forward(const MatrixWrapper& input) {
        return MatrixWrapper(layer_.forward(input.getMatrix()));
    }

private:
    FeedForward layer_;
};

// Emscripten绑定
EMSCRIPTEN_BINDINGS(llm_engine) {
    // Matrix类
    class_<MatrixWrapper>("Matrix")
        .constructor<size_t, size_t>()
        .function("rows", &MatrixWrapper::rows)
        .function("cols", &MatrixWrapper::cols)
        .function("get", &MatrixWrapper::get)
        .function("set", &MatrixWrapper::set)
        .function("toArray", &MatrixWrapper::toArray)
        .function("add", &MatrixWrapper::add)
        .function("matmul", &MatrixWrapper::matmul)
        .function("transpose", &MatrixWrapper::transpose)
        .class_function("fromArray", &MatrixWrapper::fromArray)
        .class_function("zeros", &MatrixWrapper::zeros)
        .class_function("ones", &MatrixWrapper::ones)
        .class_function("randn", &MatrixWrapper::randn);

    // Linear层
    class_<LinearWrapper>("Linear")
        .constructor<size_t, size_t>()
        .function("forward", &LinearWrapper::forward);

    // LayerNorm层
    class_<LayerNormWrapper>("LayerNorm")
        .constructor<size_t>()
        .function("forward", &LayerNormWrapper::forward);

    // SelfAttention层
    class_<SelfAttentionWrapper>("SelfAttention")
        .constructor<size_t, size_t>()
        .function("forward", &SelfAttentionWrapper::forward);

    // FeedForward层
    class_<FeedForwardWrapper>("FeedForward")
        .constructor<size_t, size_t>()
        .function("forward", &FeedForwardWrapper::forward);
}
