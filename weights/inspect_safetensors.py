#!/usr/bin/env python3
"""
检查safetensors文件格式

帮助理解safetensors的内部结构，为C++实现提供参考
"""

import struct
import json

def inspect_safetensors(filepath):
    """检查safetensors文件结构"""
    print("="*70)
    print(f"检查文件: {filepath}")
    print("="*70)

    with open(filepath, 'rb') as f:
        # 读取header长度（前8字节，little-endian uint64）
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]

        print(f"\n【1】Header大小: {header_size} 字节 ({header_size / 1024:.2f} KB)")

        # 读取header（JSON格式）
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))

        print(f"\n【2】Header内容（JSON）:")
        print(f"    包含 {len(header)} 个项目")

        # 显示metadata（如果有）
        if '__metadata__' in header:
            print(f"\n【3】Metadata:")
            for key, value in header['__metadata__'].items():
                print(f"    {key}: {value}")
            tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        else:
            tensors = header
            print(f"\n【3】没有metadata")

        # 显示所有tensor信息
        print(f"\n【4】Tensor列表 (共 {len(tensors)} 个):")
        print("-"*70)

        total_params = 0
        for i, (name, info) in enumerate(sorted(tensors.items()), 1):
            dtype = info['dtype']
            shape = info['shape']
            data_offsets = info['data_offsets']

            # 计算参数数量
            num_params = 1
            for dim in shape:
                num_params *= dim
            total_params += num_params

            # 计算字节数
            offset_start, offset_end = data_offsets
            size_bytes = offset_end - offset_start

            print(f"{i:3d}. {name}")
            print(f"     dtype: {dtype}")
            print(f"     shape: {shape}")
            print(f"     参数数: {num_params:,}")
            print(f"     大小: {size_bytes:,} bytes ({size_bytes / 1024 / 1024:.2f} MB)")
            print(f"     offset: [{offset_start:,}, {offset_end:,})")

            if i < len(tensors):
                print()

        print("-"*70)
        print(f"总参数数: {total_params:,} ({total_params / 1e6:.2f}M)")

        # 显示文件总大小
        current_pos = f.tell()
        f.seek(0, 2)  # 移到文件末尾
        file_size = f.tell()

        print(f"\n【5】文件大小: {file_size:,} 字节 ({file_size / 1024 / 1024:.2f} MB)")
        print(f"    Header: {8 + header_size:,} 字节")
        print(f"    Data: {file_size - 8 - header_size:,} 字节")

def show_embedding_weights(filepath):
    """显示embeddings的具体数据"""
    print("\n" + "="*70)
    print("【6】查看word_embeddings的前几个向量")
    print("="*70)

    with open(filepath, 'rb') as f:
        # 读取header
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))

        # 获取word_embeddings信息
        emb_info = header['embeddings.word_embeddings.weight']
        offset_start, offset_end = emb_info['data_offsets']
        shape = emb_info['shape']

        print(f"Shape: {shape} (vocab_size={shape[0]}, hidden_size={shape[1]})")

        # 跳转到数据位置
        f.seek(8 + header_size + offset_start)

        # 读取前3个向量（每个768维）
        # F32 = 4 bytes
        num_show = 3
        hidden_size = shape[1]

        for i in range(num_show):
            # 读取一个向量（768个float32）
            vector_bytes = f.read(hidden_size * 4)
            vector = struct.unpack(f'<{hidden_size}f', vector_bytes)

            print(f"\nToken ID {i} 的embedding (前10维):")
            print(f"  {vector[:10]}")

if __name__ == "__main__":
    filepath = "bert-base-uncased/model.safetensors"
    inspect_safetensors(filepath)
    show_embedding_weights(filepath)
