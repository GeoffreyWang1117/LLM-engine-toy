// WebAssembly模块
let Module = null;

// 初始化WASM模块
createLLMEngine().then(instance => {
    Module = instance;
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
    console.log('LLM Engine WASM module loaded successfully!');
}).catch(err => {
    document.getElementById('loading').innerHTML = `
        <p style="color: red;">❌ 加载失败</p>
        <p>请确保已经编译WASM模块。运行: <code>./build_wasm.sh</code></p>
        <p>错误信息: ${err.message}</p>
    `;
    console.error('Failed to load WASM module:', err);
});

// 工具函数：格式化矩阵显示
function formatMatrix(matrixObj, name = "Matrix") {
    const arr = matrixObj.toArray();
    let result = `${name} [${matrixObj.rows()} x ${matrixObj.cols()}]:\n`;

    for (let i = 0; i < arr.length; i++) {
        result += '  [';
        for (let j = 0; j < arr[i].length; j++) {
            result += arr[i][j].toFixed(4).padStart(10);
            if (j < arr[i].length - 1) result += ', ';
        }
        result += ']\n';
    }

    return result;
}

// 矩阵运算演示
function runMatrixDemo() {
    const output = document.getElementById('matrix-output');
    output.textContent = '运行中...\n';

    try {
        // 创建矩阵A (2x3)
        const A = Module.Matrix.fromArray([
            [1, 2, 3],
            [4, 5, 6]
        ]);

        // 创建矩阵B (3x2)
        const B = Module.Matrix.fromArray([
            [1, 2],
            [3, 4],
            [5, 6]
        ]);

        let result = '=== 矩阵运算演示 ===\n\n';
        result += formatMatrix(A, 'Matrix A');
        result += '\n';
        result += formatMatrix(B, 'Matrix B');
        result += '\n';

        // 矩阵乘法
        const C = A.matmul(B);
        result += formatMatrix(C, 'A × B');
        result += '\n';

        // 转置
        const A_T = A.transpose();
        result += formatMatrix(A_T, 'A^T (transpose)');

        output.textContent = result;

        // 清理
        A.delete();
        B.delete();
        C.delete();
        A_T.delete();

    } catch (err) {
        output.textContent = `错误: ${err.message}`;
        console.error(err);
    }
}

// Linear层演示
function runLinearDemo() {
    const output = document.getElementById('linear-output');
    output.textContent = '运行中...\n';

    try {
        const inFeatures = parseInt(document.getElementById('linear-in').value);
        const outFeatures = parseInt(document.getElementById('linear-out').value);

        // 创建Linear层
        const linear = new Module.Linear(inFeatures, outFeatures);

        // 创建输入 (batch_size=2)
        const inputData = Array(2).fill(0).map(() =>
            Array(inFeatures).fill(1.0)
        );
        const input = Module.Matrix.fromArray(inputData);

        // 前向传播
        const output_mat = linear.forward(input);

        let result = '=== Linear层演示 ===\n\n';
        result += `配置: Linear(${inFeatures} -> ${outFeatures})\n\n`;
        result += formatMatrix(input, 'Input');
        result += '\n';
        result += formatMatrix(output_mat, 'Output');

        output.textContent = result;

        // 清理
        input.delete();
        output_mat.delete();
        linear.delete();

    } catch (err) {
        output.textContent = `错误: ${err.message}`;
        console.error(err);
    }
}

// LayerNorm演示
function runLayerNormDemo() {
    const output = document.getElementById('layernorm-output');
    output.textContent = '运行中...\n';

    try {
        const normalized_shape = 4;

        // 创建LayerNorm层
        const ln = new Module.LayerNorm(normalized_shape);

        // 创建输入
        const input = Module.Matrix.fromArray([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]);

        // 前向传播
        const output_mat = ln.forward(input);

        let result = '=== LayerNorm演示 ===\n\n';
        result += formatMatrix(input, 'Input');
        result += '\n';
        result += formatMatrix(output_mat, 'Normalized Output');
        result += '\n注: 每行均值为0，方差为1\n';

        output.textContent = result;

        // 清理
        input.delete();
        output_mat.delete();
        ln.delete();

    } catch (err) {
        output.textContent = `错误: ${err.message}`;
        console.error(err);
    }
}

// Self-Attention演示
function runAttentionDemo() {
    const output = document.getElementById('attention-output');
    output.textContent = '运行中...\n';

    try {
        const embedDim = parseInt(document.getElementById('attn-dim').value);
        const numHeads = parseInt(document.getElementById('attn-heads').value);

        if (embedDim % numHeads !== 0) {
            output.textContent = '错误: 嵌入维度必须能被注意力头数整除';
            return;
        }

        // 创建Self-Attention层
        const attn = new Module.SelfAttention(embedDim, numHeads);

        // 创建随机输入 (seq_len=3, embed_dim)
        const input = Module.Matrix.randn(3, embedDim);

        // 前向传播
        const output_mat = attn.forward(input);

        let result = '=== Self-Attention演示 ===\n\n';
        result += `配置: embed_dim=${embedDim}, num_heads=${numHeads}\n`;
        result += `序列长度: 3\n\n`;
        result += formatMatrix(input, 'Input');
        result += '\n';
        result += formatMatrix(output_mat, 'Attention Output');

        output.textContent = result;

        // 清理
        input.delete();
        output_mat.delete();
        attn.delete();

    } catch (err) {
        output.textContent = `错误: ${err.message}`;
        console.error(err);
    }
}

// FeedForward演示
function runFFNDemo() {
    const output = document.getElementById('ffn-output');
    output.textContent = '运行中...\n';

    try {
        const embedDim = 8;
        const hiddenDim = 32;

        // 创建FeedForward层
        const ffn = new Module.FeedForward(embedDim, hiddenDim);

        // 创建随机输入 (seq_len=3, embed_dim)
        const input = Module.Matrix.randn(3, embedDim);

        // 前向传播
        const output_mat = ffn.forward(input);

        let result = '=== FeedForward Network演示 ===\n\n';
        result += `配置: ${embedDim} -> ${hiddenDim} -> ${embedDim}\n`;
        result += `激活函数: GELU\n\n`;
        result += formatMatrix(input, 'Input');
        result += '\n';
        result += formatMatrix(output_mat, 'FFN Output');

        output.textContent = result;

        // 清理
        input.delete();
        output_mat.delete();
        ffn.delete();

    } catch (err) {
        output.textContent = `错误: ${err.message}`;
        console.error(err);
    }
}
