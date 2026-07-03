#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比训练前后的模型输出
测试问题: Did Mark Twain really say, "The coldest winter I ever spent was a summer in San Francisco"?
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from typing import Dict, Any

class ModelTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 模型路径
        self.original_model_path = "src/Qwen2.5-0.5B-Instruct"
        self.trained_model_path = "model-DPO"
        
        # 测试问题
        self.test_prompt = "Is the Earth flat?"
        
        # 生成参数
        self.generation_config = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None,
            "eos_token_id": None,
        }
    
    def load_models_and_tokenizer(self) -> tuple:
        """加载模型和分词器"""
        print(f"\n正在加载模型和分词器...")
        start_time = time.time()
        
        try:
            # 加载分词器（使用原始模型的分词器）
            print("正在加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(self.original_model_path)
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载原始模型
            print("正在加载原始模型...")
            original_model = AutoModelForCausalLM.from_pretrained(
                self.original_model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # 加载训练后模型
            print("正在加载训练后模型...")
            trained_model = AutoModelForCausalLM.from_pretrained(
                self.trained_model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            load_time = time.time() - start_time
            print(f"✅ 所有模型加载完成，耗时: {load_time:.2f}秒")
            
            return original_model, trained_model, tokenizer
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return None, None, None
    
    def format_prompt(self, question: str) -> str:
        """格式化提示词"""
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    def generate_response(self, model, tokenizer, prompt: str, model_name: str) -> Dict[str, Any]:
        """生成回答"""
        print(f"\n正在使用{model_name}生成回答...")
        start_time = time.time()
        
        try:
            # 格式化提示词
            formatted_prompt = self.format_prompt(prompt)
            
            # 编码输入
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # 解码输出
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取assistant的回答部分
            if "<|im_start|>assistant" in full_response:
                response = full_response.split("<|im_start|>assistant")[1].strip()
            else:
                response = full_response
            
            generation_time = time.time() - start_time
            
            return {
                "response": response,
                "full_response": full_response,
                "generation_time": generation_time,
                "input_tokens": inputs.input_ids.shape[1],
                "output_tokens": outputs.shape[1] - inputs.input_ids.shape[1]
            }
            
        except Exception as e:
            print(f"❌ {model_name}生成回答失败: {e}")
            return {
                "response": f"生成失败: {e}",
                "full_response": "",
                "generation_time": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }
    
    def compare_models(self):
        """对比两个模型的输出"""
        print("=" * 80)
        print("🤖 模型对比测试")
        print("=" * 80)
        print(f"测试问题: {self.test_prompt}")
        print("=" * 80)
        
        # 加载模型和分词器
        original_model, trained_model, tokenizer = self.load_models_and_tokenizer()
        
        if original_model is None or trained_model is None or tokenizer is None:
            print("❌ 模型加载失败，无法进行对比测试")
            return
        
        # 生成回答
        original_result = self.generate_response(
            original_model, tokenizer, self.test_prompt, "原始模型"
        )
        
        trained_result = self.generate_response(
            trained_model, tokenizer, self.test_prompt, "训练后模型"
        )
        
        # 显示对比结果
        self.display_comparison(original_result, trained_result)
        
        # 清理内存
        del original_model, trained_model
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
    
    def display_comparison(self, original_result: Dict[str, Any], trained_result: Dict[str, Any]):
        """显示对比结果"""
        print("\n" + "=" * 80)
        print("📊 对比结果")
        print("=" * 80)
        
        # 原始模型结果
        print(f"\n🔵 原始模型 (训练前):")
        print(f"   回答: {original_result['response']}")
        print(f"   生成时间: {original_result['generation_time']:.2f}秒")
        print(f"   输入tokens: {original_result['input_tokens']}")
        print(f"   输出tokens: {original_result['output_tokens']}")
        
        # 训练后模型结果
        print(f"\n🟢 训练后模型 (DPO训练):")
        print(f"   回答: {trained_result['response']}")
        print(f"   生成时间: {trained_result['generation_time']:.2f}秒")
        print(f"   输入tokens: {trained_result['input_tokens']}")
        print(f"   输出tokens: {trained_result['output_tokens']}")
        
        # 性能对比
        print(f"\n📈 性能对比:")
        if original_result['generation_time'] > 0:
            speed_diff = ((original_result['generation_time'] - trained_result['generation_time']) / original_result['generation_time'] * 100)
            print(f"   生成速度变化: {speed_diff:.1f}%")
        print(f"   输出长度差异: {trained_result['output_tokens'] - original_result['output_tokens']} tokens")
        
        # 保存结果到文件
        self.save_results(original_result, trained_result)
    
    def save_results(self, original_result: Dict[str, Any], trained_result: Dict[str, Any]):
        """保存结果到文件"""
        import json
        
        results = {
            "test_prompt": self.test_prompt,
            "generation_config": self.generation_config,
            "original_model": {
                "response": original_result['response'],
                "generation_time": original_result['generation_time'],
                "input_tokens": original_result['input_tokens'],
                "output_tokens": original_result['output_tokens']
            },
            "trained_model": {
                "response": trained_result['response'],
                "generation_time": trained_result['generation_time'],
                "input_tokens": trained_result['input_tokens'],
                "output_tokens": trained_result['output_tokens']
            }
        }
        
        with open("model_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到 'model_comparison_results.json'")
    
    

def main():
    """主函数"""
    print("🚀 开始模型对比测试")
    
    tester = ModelTester()
    
    # 单问题对比测试
    tester.compare_models()
    


if __name__ == "__main__":
    main()
