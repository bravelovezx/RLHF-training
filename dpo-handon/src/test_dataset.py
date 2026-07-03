#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 truthy-dpo-v0.1 数据集的内容
"""

from datasets import load_dataset
import json

def explore_dataset():
    """探索数据集的基本信息"""
    print("=" * 60)
    print("正在加载 truthy-dpo-v0.1 数据集...")
    print("=" * 60)
    
    try:
        # 加载数据集
        dataset = load_dataset('truthy-dpo-v0.1', split='train')
        
        print(f"\n�� 数据集基本信息:")
        print(f"   总样本数: {len(dataset)}")
        print(f"   数据集类型: {type(dataset)}")
        
        print(f"\n🔍 数据集结构 (features):")
        print(f"   {dataset.features}")
        
        print(f"\n📝 数据集列名:")
        for i, column in enumerate(dataset.column_names):
            print(f"   {i+1}. {column}")
        
        # 显示前5个样本
        print(f"\n" + "=" * 60)
        print("�� 前5个训练样本:")
        print("=" * 60)
        
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"\n🔸 样本 {i+1}:")
            print(f"   Prompt: {sample['prompt']}")
            print(f"   Chosen: {sample['chosen']}")
            print(f"   Rejected: {sample['rejected']}")
            print("-" * 40)
        
        # 统计信息
        print(f"\n" + "=" * 60)
        print("📈 数据统计信息:")
        print("=" * 60)
        
        # 计算文本长度统计
        prompt_lengths = [len(sample['prompt']) for sample in dataset]
        chosen_lengths = [len(sample['chosen']) for sample in dataset]
        rejected_lengths = [len(sample['rejected']) for sample in dataset]
        
        print(f"   Prompt 长度统计:")
        print(f"     平均长度: {sum(prompt_lengths)/len(prompt_lengths):.1f}")
        print(f"     最短长度: {min(prompt_lengths)}")
        print(f"     最长长度: {max(prompt_lengths)}")
        
        print(f"\n   Chosen 回答长度统计:")
        print(f"     平均长度: {sum(chosen_lengths)/len(chosen_lengths):.1f}")
        print(f"     最短长度: {min(chosen_lengths)}")
        print(f"     最长长度: {max(chosen_lengths)}")
        
        print(f"\n   Rejected 回答长度统计:")
        print(f"     平均长度: {sum(rejected_lengths)/len(rejected_lengths):.1f}")
        print(f"     最短长度: {min(rejected_lengths)}")
        print(f"     最长长度: {max(rejected_lengths)}")
        
        # 显示一些随机样本
        print(f"\n" + "=" * 60)
        print("🎲 随机样本 (用于了解数据多样性):")
        print("=" * 60)
        
        import random
        random.seed(42)  # 固定随机种子，确保结果可重现
        random_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
        
        for i, idx in enumerate(random_indices):
            sample = dataset[idx]
            print(f"\n🔸 随机样本 {i+1} (索引 {idx}):")
            print(f"   Prompt: {sample['prompt']}")
            print(f"   Chosen: {sample['chosen']}")
            print(f"   Rejected: {sample['rejected']}")
            print("-" * 40)
        
        # 保存一些样本到文件
        print(f"\n" + "=" * 60)
        print("💾 保存样本到文件...")
        print("=" * 60)
        
        sample_data = []
        for i in range(min(10, len(dataset))):
            sample_data.append({
                "index": i,
                "prompt": dataset[i]['prompt'],
                "chosen": dataset[i]['chosen'],
                "rejected": dataset[i]['rejected']
            })
        
        with open('dataset_samples.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"   已保存前10个样本到 'dataset_samples.json'")
        
        print(f"\n✅ 数据集探索完成!")
        
    except Exception as e:
        print(f"❌ 加载数据集时出错: {e}")
        print("请检查网络连接或数据集名称是否正确")

def show_specific_samples():
    """显示特定索引的样本"""
    print(f"\n" + "=" * 60)
    print("🔍 查看特定样本:")
    print("=" * 60)
    
    try:
        dataset = load_dataset('truthy-dpo-v0.1', split='train')
        
        # 显示索引为0, 100, 500的样本
        indices = [0, 100, 500]
        for idx in indices:
            if idx < len(dataset):
                sample = dataset[idx]
                print(f"\n🔸 样本 {idx}:")
                print(f"   Prompt: {sample['prompt']}")
                print(f"   Chosen: {sample['chosen']}")
                print(f"   Rejected: {sample['rejected']}")
                print("-" * 40)
            else:
                print(f"   索引 {idx} 超出数据集范围")
                
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    print("�� 开始探索 truthy-dpo-v0.1 数据集")
    print("=" * 60)
    
    # 探索数据集
    explore_dataset()
    
    # 显示特定样本
    show_specific_samples()
    
    print(f"\n🎉 所有操作完成!")
