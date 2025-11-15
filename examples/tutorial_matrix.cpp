#include "llm_engine/matrix.h"
#include <iostream>

using namespace llm;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Matrix类教学演示" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 1. 创建矩阵的几种方式
    std::cout << "【1】创建矩阵" << std::endl;
    std::cout << "方式1: 指定大小，默认初始化为0" << std::endl;
    Matrix A(2, 3);  // 2行3列，全0
    A.print();

    std::cout << "\n方式2: 指定大小和初始值" << std::endl;
    Matrix B(2, 3, 5.0f);  // 2行3列，全部填充5.0
    B.print();

    std::cout << "\n方式3: 从数组创建" << std::endl;
    Matrix C(2, 3, {1, 2, 3, 4, 5, 6});
    C.print();

    std::cout << "\n方式4: 使用工厂方法" << std::endl;
    Matrix zeros = Matrix::zeros(2, 2);
    std::cout << "zeros(2,2):" << std::endl;
    zeros.print();

    Matrix ones = Matrix::ones(2, 2);
    std::cout << "\nones(2,2):" << std::endl;
    ones.print();

    Matrix randn = Matrix::randn(2, 3, 0.0f, 1.0f);  // 均值0，标准差1
    std::cout << "\nrandn(2,3) - 正态分布随机数:" << std::endl;
    randn.print();

    // 2. 访问和修改元素
    std::cout << "\n========================================" << std::endl;
    std::cout << "【2】访问和修改矩阵元素" << std::endl;
    Matrix D(2, 2, {1, 2, 3, 4});
    std::cout << "原始矩阵D:" << std::endl;
    D.print();

    std::cout << "\n访问元素 D(0,1) = " << D(0, 1) << std::endl;
    std::cout << "访问元素 D(1,0) = " << D(1, 0) << std::endl;

    D(0, 1) = 10.0f;  // 修改元素
    std::cout << "\n修改 D(0,1) = 10.0 后:" << std::endl;
    D.print();

    // 3. 矩阵加法
    std::cout << "\n========================================" << std::endl;
    std::cout << "【3】矩阵加法" << std::endl;
    Matrix E(2, 2, {1, 2, 3, 4});
    Matrix F(2, 2, {5, 6, 7, 8});

    std::cout << "矩阵E:" << std::endl;
    E.print();
    std::cout << "\n矩阵F:" << std::endl;
    F.print();

    Matrix sum = E + F;
    std::cout << "\nE + F =" << std::endl;
    sum.print();

    // 4. 矩阵减法
    std::cout << "\n========================================" << std::endl;
    std::cout << "【4】矩阵减法" << std::endl;
    Matrix diff = F - E;
    std::cout << "F - E =" << std::endl;
    diff.print();

    // 5. 标量乘法
    std::cout << "\n========================================" << std::endl;
    std::cout << "【5】标量乘法" << std::endl;
    Matrix scaled = E * 2.0f;
    std::cout << "E * 2.0 =" << std::endl;
    scaled.print();

    // 6. 矩阵乘法（最重要！）
    std::cout << "\n========================================" << std::endl;
    std::cout << "【6】矩阵乘法 - LLM的核心运算" << std::endl;
    std::cout << "说明: 矩阵乘法用于所有的线性变换" << std::endl;
    std::cout << "     在LLM中，几乎所有层都需要矩阵乘法\n" << std::endl;

    Matrix G(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix H(3, 2, {1, 2, 3, 4, 5, 6});

    std::cout << "矩阵G [2x3]:" << std::endl;
    G.print();
    std::cout << "\n矩阵H [3x2]:" << std::endl;
    H.print();

    Matrix product = G.matmul(H);
    std::cout << "\nG × H = [2x2]" << std::endl;
    std::cout << "计算过程示例:" << std::endl;
    std::cout << "  第(0,0)元素 = 1*1 + 2*3 + 3*5 = 1+6+15 = 22" << std::endl;
    std::cout << "  第(0,1)元素 = 1*2 + 2*4 + 3*6 = 2+8+18 = 28" << std::endl;
    product.print();

    // 7. 矩阵转置
    std::cout << "\n========================================" << std::endl;
    std::cout << "【7】矩阵转置" << std::endl;
    std::cout << "说明: 转置在Attention机制中非常重要\n" << std::endl;

    Matrix I(2, 3, {1, 2, 3, 4, 5, 6});
    std::cout << "原矩阵I [2x3]:" << std::endl;
    I.print();

    Matrix I_T = I.transpose();
    std::cout << "\n转置后 I^T [3x2]:" << std::endl;
    std::cout << "行列互换" << std::endl;
    I_T.print();

    // 8. 实际应用示例：简单的线性变换
    std::cout << "\n========================================" << std::endl;
    std::cout << "【8】实际应用：线性变换 y = Wx" << std::endl;
    std::cout << "这就是神经网络中Linear层的核心\n" << std::endl;

    // 假设输入是一个向量 [1, 2, 3]
    Matrix input(1, 3, {1, 2, 3});
    std::cout << "输入向量x [1x3]:" << std::endl;
    input.print();

    // 权重矩阵 W [3x2]，将3维映射到2维
    Matrix weight(3, 2, {
        0.5, -0.5,
        1.0, -1.0,
        0.2,  0.8
    });
    std::cout << "\n权重矩阵W [3x2]:" << std::endl;
    weight.print();

    // y = x * W
    Matrix output = input.matmul(weight);
    std::cout << "\n输出 y = x·W [1x2]:" << std::endl;
    std::cout << "将3维特征转换为2维特征" << std::endl;
    output.print();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Matrix类演示完成！" << std::endl;
    std::cout << "这些运算是构建LLM的基础" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
