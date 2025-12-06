#!/usr/bin/env python3
"""
下载BERT预训练模型权重

使用Hugging Face Hub下载bert-base-uncased模型的safetensors权重
"""

import os
from pathlib import Path

def download_bert_weights():
    """下载bert-base-uncased权重"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("错误：需要安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        return False

    model_id = "bert-base-uncased"
    save_dir = Path(__file__).parent / "bert-base-uncased"
    save_dir.mkdir(exist_ok=True)

    print(f"正在下载 {model_id} 权重...")
    print(f"保存位置: {save_dir}")

    # 需要下载的文件
    files_to_download = [
        "model.safetensors",  # 权重文件（safetensors格式）
        "config.json",        # 模型配置
        "vocab.txt",          # 词表（阶段4需要）
        "tokenizer_config.json",  # tokenizer配置
    ]

    for filename in files_to_download:
        print(f"\n下载 {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=save_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ 成功: {filename}")

            # 显示文件大小
            file_size = os.path.getsize(downloaded_path)
            size_mb = file_size / (1024 * 1024)
            print(f"   大小: {size_mb:.2f} MB")

        except Exception as e:
            print(f"❌ 失败: {filename}")
            print(f"   错误: {e}")

    print("\n" + "="*60)
    print("下载完成！")
    print(f"权重文件位置: {save_dir}")
    print("="*60)

    return True

if __name__ == "__main__":
    download_bert_weights()
