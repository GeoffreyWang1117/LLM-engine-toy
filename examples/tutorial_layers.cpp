#include "llm_engine/matrix.h"
#include "llm_engine/layers.h"
#include <iostream>

using namespace llm;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Transformer组件教学演示" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // ============================================================
    // 1. Linear层 - 全连接层
    // ============================================================
    std::cout << "【1】Linear层（全连接层）" << std::endl;
    std::cout << "说明：将输入从一个维度映射到另一个维度" << std::endl;
    std::cout << "     公式: y = x·W^T + b" << std::endl;
    std::cout << "     这是神经网络最基础的层\n" << std::endl;

    // 创建一个Linear层: 4维输入 -> 3维输出
    Linear linear(4, 3);
    std::cout << "创建Linear层: 输入4维 -> 输出3维\n" << std::endl;

    // 创建输入: batch_size=2, features=4
    Matrix input1(2, 4, {
        1.0, 2.0, 3.0, 4.0,  // 第1个样本
        5.0, 6.0, 7.0, 8.0   // 第2个样本
    });

    std::cout << "输入 [2 samples x 4 features]:" << std::endl;
    input1.print();

    Matrix output1 = linear.forward(input1);
    std::cout << "\n输出 [2 samples x 3 features]:" << std::endl;
    std::cout << "每个样本从4维特征变成了3维特征" << std::endl;
    output1.print();

    std::cout << "\n权重矩阵 W [3 x 4]:" << std::endl;
    std::cout << "这些权重在真实模型中是通过训练学习得到的" << std::endl;
    linear.weight().print();

    // ============================================================
    // 2. LayerNorm层 - 层归一化
    // ============================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "【2】LayerNorm层（层归一化）" << std::endl;
    std::cout << "说明：将每个样本的特征归一化为均值0、方差1" << std::endl;
    std::cout << "     作用：稳定训练过程，加速收敛" << std::endl;
    std::cout << "     公式: y = (x - mean) / sqrt(var + eps) * gamma + beta\n" << std::endl;

    LayerNorm ln(4);

    Matrix input2(3, 4, {
        1.0, 2.0, 3.0, 4.0,   // 样本1
        10.0, 20.0, 30.0, 40.0,  // 样本2 (数值较大)
        -5.0, -10.0, 5.0, 10.0   // 样本3 (有负数)
    });

    std::cout << "输入 [3 samples x 4 features]:" << std::endl;
    std::cout << "注意三个样本的数值范围差异很大" << std::endl;
    input2.print();

    // 手动计算第一个样本的统计信息
    float mean1 = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
    std::cout << "\n第1个样本的均值: " << mean1 << std::endl;

    Matrix output2 = ln.forward(input2);
    std::cout << "\n归一化后的输出:" << std::endl;
    std::cout << "每个样本内部都被归一化了，不同样本之间的尺度差异被消除" << std::endl;
    output2.print();

    // ============================================================
    // 3. Self-Attention - 自注意力机制
    // ============================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "【3】Self-Attention（自注意力机制）" << std::endl;
    std::cout << "说明：这是Transformer的核心！" << std::endl;
    std::cout << "     让序列中的每个位置都能关注到其他所有位置" << std::endl;
    std::cout << "     步骤：" << std::endl;
    std::cout << "     1. 通过Q,K,V三个矩阵投影得到Query, Key, Value" << std::endl;
    std::cout << "     2. 计算注意力分数: Attention(Q,K,V) = softmax(Q·K^T / √d) · V" << std::endl;
    std::cout << "     3. 通过输出投影得到最终结果\n" << std::endl;

    // embed_dim=8, num_heads=2（简化版本暂时不做真正的多头拆分）
    SelfAttention attn(8, 2);

    // 创建一个序列：3个token，每个token 8维特征
    Matrix input3 = Matrix::randn(3, 8, 0.0f, 0.5f);

    std::cout << "输入序列 [seq_len=3, embed_dim=8]:" << std::endl;
    std::cout << "可以理解为3个单词，每个单词用8维向量表示" << std::endl;
    input3.print();

    Matrix output3 = attn.forward(input3);
    std::cout << "\nAttention输出 [seq_len=3, embed_dim=8]:" << std::endl;
    std::cout << "每个位置的输出融合了序列中所有位置的信息" << std::endl;
    std::cout << "这就是'注意力'的含义：关注重要的信息" << std::endl;
    output3.print();

    // ============================================================
    // 4. FeedForward Network - 前馈网络
    // ============================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "【4】FeedForward Network（前馈网络）" << std::endl;
    std::cout << "说明：两层全连接网络，中间用GELU激活" << std::endl;
    std::cout << "     结构: x -> Linear -> GELU -> Linear -> y" << std::endl;
    std::cout << "     作用: 增加模型的非线性表达能力\n" << std::endl;

    // embed_dim=8, hidden_dim=32（通常hidden是embed的4倍）
    FeedForward ffn(8, 32);

    Matrix input4 = Matrix::randn(3, 8, 0.0f, 0.3f);

    std::cout << "输入 [seq_len=3, embed_dim=8]:" << std::endl;
    input4.print();

    Matrix output4 = ffn.forward(input4);
    std::cout << "\nFFN输出 [seq_len=3, embed_dim=8]:" << std::endl;
    std::cout << "维度保持不变，但经过了非线性变换" << std::endl;
    std::cout << "中间hidden层维度是32（4倍扩展）" << std::endl;
    output4.print();

    // ============================================================
    // 5. 组合使用：完整的Transformer流程演示
    // ============================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "【5】组合演示：模拟一个完整的Transformer Block" << std::endl;
    std::cout << "标准Transformer Block的结构：" << std::endl;
    std::cout << "  x -> LayerNorm -> SelfAttention -> 残差连接" << std::endl;
    std::cout << "    -> LayerNorm -> FeedForward -> 残差连接 -> output\n" << std::endl;

    int embed_dim = 8;
    int num_heads = 2;
    int hidden_dim = 32;

    // 创建所有组件
    LayerNorm ln1(embed_dim);
    SelfAttention attn_block(embed_dim, num_heads);
    LayerNorm ln2(embed_dim);
    FeedForward ffn_block(embed_dim, hidden_dim);

    // 输入序列
    Matrix x = Matrix::randn(3, 8, 0.0f, 0.3f);
    std::cout << "原始输入 x [3x8]:" << std::endl;
    x.print();

    // 第一个子层：Self-Attention + 残差
    std::cout << "\n--- 第1个子层：Self-Attention ---" << std::endl;
    Matrix normed1 = ln1.forward(x);
    std::cout << "LayerNorm后:" << std::endl;
    normed1.print();

    Matrix attn_out = attn_block.forward(normed1);
    std::cout << "\nAttention输出:" << std::endl;
    attn_out.print();

    Matrix x2 = x + attn_out;  // 残差连接
    std::cout << "\n加上残差（x + attn_out）:" << std::endl;
    x2.print();

    // 第二个子层：FeedForward + 残差
    std::cout << "\n--- 第2个子层：FeedForward ---" << std::endl;
    Matrix normed2 = ln2.forward(x2);
    std::cout << "LayerNorm后:" << std::endl;
    normed2.print();

    Matrix ffn_out = ffn_block.forward(normed2);
    std::cout << "\nFFN输出:" << std::endl;
    ffn_out.print();

    Matrix final_output = x2 + ffn_out;  // 残差连接
    std::cout << "\n最终输出（加上残差）:" << std::endl;
    final_output.print();

    // ============================================================
    // 总结
    // ============================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "【总结】Transformer的核心思想" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n1. Linear层：特征维度变换" << std::endl;
    std::cout << "   - 用于投影、降维、升维等操作\n" << std::endl;

    std::cout << "2. LayerNorm：归一化，稳定训练" << std::endl;
    std::cout << "   - 消除不同样本间的尺度差异\n" << std::endl;

    std::cout << "3. Self-Attention：序列建模的关键" << std::endl;
    std::cout << "   - 让每个位置关注到整个序列" << std::endl;
    std::cout << "   - 捕获长距离依赖关系\n" << std::endl;

    std::cout << "4. FeedForward：增加非线性" << std::endl;
    std::cout << "   - 提升模型的表达能力\n" << std::endl;

    std::cout << "5. 残差连接：训练深层网络的关键" << std::endl;
    std::cout << "   - 让梯度更好地流动" << std::endl;
    std::cout << "   - 避免退化问题\n" << std::endl;

    std::cout << "这些组件堆叠起来，就构成了GPT、BERT等" << std::endl;
    std::cout << "强大的语言模型！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
