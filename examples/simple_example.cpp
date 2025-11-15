#include "llm_engine/matrix.h"
#include "llm_engine/layers.h"
#include <iostream>

using namespace llm;

void test_matrix() {
    std::cout << "\n=== Testing Matrix Operations ===" << std::endl;

    // 创建矩阵
    Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix B(3, 2, {1, 2, 3, 4, 5, 6});

    std::cout << "\nMatrix A:" << std::endl;
    A.print();

    std::cout << "\nMatrix B:" << std::endl;
    B.print();

    // 矩阵乘法
    Matrix C = A.matmul(B);
    std::cout << "\nA * B:" << std::endl;
    C.print();

    // 转置
    Matrix A_T = A.transpose();
    std::cout << "\nA^T:" << std::endl;
    A_T.print();
}

void test_linear_layer() {
    std::cout << "\n=== Testing Linear Layer ===" << std::endl;

    // 创建Linear层: input_dim=4, output_dim=3
    Linear linear(4, 3);

    // 创建输入: [2, 4] (batch_size=2, features=4)
    Matrix input(2, 4);
    input.fill(1.0f);

    std::cout << "\nInput:" << std::endl;
    input.print();

    // 前向传播
    Matrix output = linear.forward(input);
    std::cout << "\nOutput:" << std::endl;
    output.print();
}

void test_layer_norm() {
    std::cout << "\n=== Testing LayerNorm ===" << std::endl;

    // 创建LayerNorm: normalized_shape=4
    LayerNorm ln(4);

    // 创建输入
    Matrix input(2, 4, {1, 2, 3, 4, 5, 6, 7, 8});

    std::cout << "\nInput:" << std::endl;
    input.print();

    // 前向传播
    Matrix output = ln.forward(input);
    std::cout << "\nNormalized output:" << std::endl;
    output.print();
}

void test_attention() {
    std::cout << "\n=== Testing Self-Attention ===" << std::endl;

    // 创建Self-Attention: embed_dim=8, num_heads=2
    SelfAttention attn(8, 2);

    // 创建输入: [seq_len=3, embed_dim=8]
    Matrix input = Matrix::randn(3, 8, 0.0f, 0.1f);

    std::cout << "\nInput (seq_len=3, embed_dim=8):" << std::endl;
    input.print();

    // 前向传播
    Matrix output = attn.forward(input);
    std::cout << "\nAttention output:" << std::endl;
    output.print();
}

void test_feedforward() {
    std::cout << "\n=== Testing FeedForward Network ===" << std::endl;

    // 创建FFN: embed_dim=8, hidden_dim=32
    FeedForward ffn(8, 32);

    // 创建输入: [seq_len=3, embed_dim=8]
    Matrix input = Matrix::randn(3, 8, 0.0f, 0.1f);

    std::cout << "\nInput:" << std::endl;
    input.print();

    // 前向传播
    Matrix output = ffn.forward(input);
    std::cout << "\nFFN output:" << std::endl;
    output.print();
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  LLM Engine Toy - Simple Example" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 测试各个组件
        test_matrix();
        test_linear_layer();
        test_layer_norm();
        test_attention();
        test_feedforward();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All tests completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
