"""
GRPO Training Script for Qwen2.5-0.5B-Instruct on GSM8K
复现 DeepSeek R1 的 GRPO 训练方法，提升小模型数学推理能力
"""

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "./Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/Qwen2.5-0.5B-reasoning-GRPO"
RUN_NAME = "Qwen2.5-0.5B-GRPO-gsm8k"

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# ============================================================
# 工具函数
# ============================================================

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# ============================================================
# 数据集加载 (兼容新旧版本 datasets 库)
# ============================================================

def get_gsm8k_questions(split="train") -> Dataset:
    """
    加载 GSM8K 数据集，兼容多种加载方式：
    1. 本地 ./gsm8k 目录（modelscope 下载）
    2. HuggingFace openai/gsm8k (需要 config 'main')
    3. 旧版 datasets 库直接 load_dataset('gsm8k')
    """
    data = None

    # 方式1: 从本地目录加载
    try:
        data = load_dataset("./gsm8k")[split]
        print(f"[INFO] 从本地 ./gsm8k 目录加载数据集成功")
    except Exception:
        pass

    # 方式2: 从 HuggingFace 加载 (新版 datasets 需要 config name)
    if data is None:
        try:
            data = load_dataset("openai/gsm8k", "main")[split]
            print(f"[INFO] 从 HuggingFace openai/gsm8k 加载数据集成功")
        except Exception:
            pass

    # 方式3: 旧版兼容
    if data is None:
        try:
            data = load_dataset("gsm8k", "main")[split]
            print(f"[INFO] 从 HuggingFace gsm8k 加载数据集成功")
        except Exception:
            pass

    if data is None:
        raise RuntimeError(
            "无法加载 GSM8K 数据集。请尝试以下方式之一：\n"
            "1. 运行: modelscope download --dataset modelscope/gsm8k --local_dir ./gsm8k\n"
            "2. 运行: huggingface-cli download openai/gsm8k --local-dir ./gsm8k\n"
            "3. 确保网络可以访问 HuggingFace"
        )

    # 映射为训练所需格式
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })

    # 过滤掉 answer 为 None 的样本
    data = data.filter(lambda x: x['answer'] is not None)

    return data


# ============================================================
# 奖励函数
# ============================================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """答案完全正确得2分"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20,
          f"Question:\n{q}",
          f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """答案是整数得0.5分"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格格式检查（含换行）得0.5分"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松格式检查得0.5分"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """根据XML标签结构打分 0~0.5"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """计算一个批次的xml得分"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ============================================================
# 主训练流程
# ============================================================

def main():
    # 1. 加载数据集
    print("=" * 50)
    print("加载数据集...")
    print("=" * 50)
    dataset = get_gsm8k_questions()
    print(f"训练集样本数: {len(dataset)}")

    # 2. 配置训练参数
    print("=" * 50)
    print("配置训练参数...")
    print("=" * 50)
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="none",  # 改为 "wandb" 如果需要 wandb 日志
    )

    # 3. 加载模型和 tokenizer
    print("=" * 50)
    print(f"加载模型: {MODEL_NAME}")
    print("=" * 50)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 4. 创建 Trainer
    print("=" * 50)
    print("创建 GRPOTrainer...")
    print("=" * 50)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # 5. 开始训练
    print("=" * 50)
    print("开始 GRPO 训练!")
    print("=" * 50)
    trainer.train()

    # 6. 保存模型
    trainer.save_model(OUTPUT_DIR)
    print(f"\n模型已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
