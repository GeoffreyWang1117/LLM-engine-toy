#!/usr/bin/env python3
"""
列出BERT权重名称，帮助实现C++权重加载器
"""

import struct
import json

def list_bert_weights(filepath):
    """列出所有BERT相关的权重（排除MLM head）"""
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))

        # 过滤出BERT基础模型的权重（不包括cls.predictions等MLM专用层）
        bert_weights = {}
        for name, info in header.items():
            if name.startswith('bert.') and not name.startswith('bert.pooler'):
                # 去掉'bert.'前缀，更容易映射
                clean_name = name[5:]  # 移除'bert.'
                bert_weights[clean_name] = info

            # pooler也要包括
            if name.startswith('bert.pooler'):
                clean_name = name[5:]  # 移除'bert.'
                bert_weights[clean_name] = info

        # 按类别分组显示
        categories = {
            'Embeddings': [],
            'Encoder Layers': {},
            'Pooler': []
        }

        for name in sorted(bert_weights.keys()):
            if name.startswith('embeddings.'):
                categories['Embeddings'].append(name)
            elif name.startswith('encoder.layer.'):
                # 提取层号
                layer_num = int(name.split('.')[2])
                if layer_num not in categories['Encoder Layers']:
                    categories['Encoder Layers'][layer_num] = []
                categories['Encoder Layers'][layer_num].append(name)
            elif name.startswith('pooler.'):
                categories['Pooler'].append(name)

        # 打印
        print("="*70)
        print("BERT权重结构")
        print("="*70)

        print("\n【1】Embeddings层:")
        for name in categories['Embeddings']:
            shape = bert_weights[name]['shape']
            print(f"  {name:50s} {str(shape):20s}")

        print(f"\n【2】Encoder层 ({len(categories['Encoder Layers'])} 层):")
        # 只显示第0层作为示例
        if 0 in categories['Encoder Layers']:
            print("\n  Layer 0 (其他层结构相同):")
            for name in categories['Encoder Layers'][0]:
                shape = bert_weights[name]['shape']
                short_name = '.'.join(name.split('.')[3:])  # 移除'encoder.layer.0'
                print(f"    {short_name:45s} {str(shape):20s}")

        print("\n【3】Pooler层:")
        for name in categories['Pooler']:
            shape = bert_weights[name]['shape']
            print(f"  {name:50s} {str(shape):20s}")

        # C++映射指南
        print("\n" + "="*70)
        print("C++权重映射指南")
        print("="*70)

        mapping = {
            'embeddings.word_embeddings.weight': 'token_embeddings_.weights_',
            'embeddings.position_embeddings.weight': 'position_embeddings_.weights_',
            'embeddings.token_type_embeddings.weight': 'token_type_embeddings_.weights_',
            'embeddings.LayerNorm.gamma': 'embeddings_layer_norm_.gamma_',
            'embeddings.LayerNorm.beta': 'embeddings_layer_norm_.beta_',
            'encoder.layer.{i}.attention.self.query.weight': 'encoder_.layers_[{i}].attention_.q_proj_.weight_',
            'encoder.layer.{i}.attention.self.query.bias': 'encoder_.layers_[{i}].attention_.q_proj_.bias_',
            'encoder.layer.{i}.attention.self.key.weight': 'encoder_.layers_[{i}].attention_.k_proj_.weight_',
            'encoder.layer.{i}.attention.self.key.bias': 'encoder_.layers_[{i}].attention_.k_proj_.bias_',
            'encoder.layer.{i}.attention.self.value.weight': 'encoder_.layers_[{i}].attention_.v_proj_.weight_',
            'encoder.layer.{i}.attention.self.value.bias': 'encoder_.layers_[{i}].attention_.v_proj_.bias_',
            'encoder.layer.{i}.attention.output.dense.weight': 'encoder_.layers_[{i}].attention_.o_proj_.weight_',
            'encoder.layer.{i}.attention.output.dense.bias': 'encoder_.layers_[{i}].attention_.o_proj_.bias_',
            'encoder.layer.{i}.attention.output.LayerNorm.gamma': 'encoder_.layers_[{i}].ln1_.gamma_',
            'encoder.layer.{i}.attention.output.LayerNorm.beta': 'encoder_.layers_[{i}].ln1_.beta_',
            'encoder.layer.{i}.intermediate.dense.weight': 'encoder_.layers_[{i}].ffn_.fc1_.weight_',
            'encoder.layer.{i}.intermediate.dense.bias': 'encoder_.layers_[{i}].ffn_.fc1_.bias_',
            'encoder.layer.{i}.output.dense.weight': 'encoder_.layers_[{i}].ffn_.fc2_.weight_',
            'encoder.layer.{i}.output.dense.bias': 'encoder_.layers_[{i}].ffn_.fc2_.bias_',
            'encoder.layer.{i}.output.LayerNorm.gamma': 'encoder_.layers_[{i}].ln2_.gamma_',
            'encoder.layer.{i}.output.LayerNorm.beta': 'encoder_.layers_[{i}].ln2_.beta_',
            'pooler.dense.weight': 'pooler_dense_.weight_',
            'pooler.dense.bias': 'pooler_dense_.bias_',
        }

        print("\nSafetensors名称 → C++成员变量:")
        print("-"*70)
        for st_name, cpp_name in mapping.items():
            print(f"{st_name:55s} → {cpp_name}")

if __name__ == "__main__":
    filepath = "bert-base-uncased/model.safetensors"
    list_bert_weights(filepath)
