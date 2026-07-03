# 导入所需的库
import torch
import warnings
from datasets import load_dataset # 用于加载数据集
from transformers import (
    AutoModelForSequenceClassification, # 用于加载情感分析模型
    AutoTokenizer, # 用于加载分词器
    pipeline, # 用于创建情感分析 pipeline
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer # TRL 库的核心组件
from tqdm import tqdm # 用于显示进度条
# import wandb # 注释掉wandb导入
import os # 用于生成 wandb 运行名称等


# 抑制特定的用户警告 (例如来自 accelerate 或 bitsandbytes 的警告)
warnings.filterwarnings("ignore")

# --- 定义关键长度参数 ---
# 从原始评论中提取的 Prompt token 片段的最大长度
MAX_PROMPT_TOKEN_LEN = 10
# 模型生成回复（续写）的最大 token 长度
MAX_NEW_TOKENS = 128
# ----------------------------------


# --- 1. 配置 PPO 训练参数 ---
# 使用 TRL 的 PPOConfig 定义强化学习训练的超参数
config = PPOConfig(
    model_name="Qwen2.5-0.5B-Instruct", # 基础模型名称
    learning_rate=6e-6, # 学习率
    log_with=None,  # 将日志记录方式改为 None，不使用wandb
    batch_size=128,     # 每次优化步骤前处理的 Prompt 数量 (rollout batch size)
    mini_batch_size=16, # 优化期间每次前向传播的 Prompt 数量 (generation batch size)
    gradient_accumulation_steps=1, # 梯度累积步数
    optimize_cuda_cache=True, # 尝试优化 CUDA 缓存使用
    early_stopping=False, # 是否启用早停
    target_kl=0.03,      # PPO 算法中的 KL 散度目标阈值
    kl_penalty="kl",    # KL 惩罚类型
    seed=42, # 随机种子
    use_score_norm=True, # 是否标准化奖励分数
    score_clip=None,     # 是否裁剪奖励分数
    ppo_epochs=1,       # 每个数据批次在 PPO 优化阶段重复的 epoch 数量
)

# --- 2. 配置生成参数 ---
# 定义模型生成文本时的参数
generation_kwargs = {
    "min_length": -1, # 允许生成短回复，不忽略 EOS token (Qwen 特定)
    "top_k": 0, # 不使用 top-k 采样
    "top_p": 1.0, # 使用 top-p 采样，值为 1.0 表示不进行概率过滤
    "do_sample": True, # 启用采样，非确定性生成
    "pad_token_id": None, # 填充 token ID，稍后设置
    "max_new_tokens": MAX_NEW_TOKENS, # 模型生成回复的最大 token 长度，使用上方定义的变量
    "eos_token_id": 151643, # Qwen 的 EOS token ID，用于停止生成
}

# --- 3. 加载数据集和分词器 ---

# 加载 Qwen 分词器
# 对于 Qwen 模型通常需要设置 trust_remote_code=True
policy_tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
# 如果 Qwen 模型没有默认的填充 token，将其设置为 EOS token
if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
# 更新生成参数中的填充 token ID
generation_kwargs["pad_token_id"] = policy_tokenizer.pad_token_id


def build_dataset(tokenizer, dataset_name="imdb", split="train", max_len=MAX_PROMPT_TOKEN_LEN):
    """
    构建用于 PPO 训练的数据集。
    从指定的 Hugging Face 数据集加载数据，过滤，打乱，并对 Prompt 进行分词和格式化。

    参数:
        tokenizer: 用于编码文本的分词器 (Policy model's tokenizer)。
        dataset_name (str): Hugging Face 数据集名称 (例如, "stanfordnlp/imdb")。
        split (str): 数据集划分 (例如, "train", "test")。
        max_len (int): 从原始评论中提取的 Prompt token 片段的最大长度。

    返回:
        Dataset: 处理后的 Hugging Face Dataset 对象，包含 tokenized prompts。
    """
    # 加载数据集，指定 'plain_text' 配置
    ds = load_dataset(dataset_name, name='plain_text', split=split)
    # 过滤掉长度小于 100 的评论，确保 Prompt 片段有足够的上下文来源
    ds = ds.filter(lambda x: len(x["text"]) > 100, batched=False)
    # 打乱数据集
    ds = ds.shuffle(seed=42)

    def tokenize(sample):
        # 提取评论开头的指定数量的 token 作为 Prompt 片段
        original_text = sample["text"] # 获取原始完整文本

        # 先对原始文本进行分词
        # add_special_tokens=False 以免在原始文本开头意外加入特殊 token (如 CLS/SEP)
        original_tokens = tokenizer(original_text, add_special_tokens=False).input_ids

        # 截取前 max_len 个 token ID
        snippet_tokens = original_tokens[:max_len]

        # 将截取的 token ID 列表解码回文本字符串
        # skip_special_tokens=True 确保解码干净，不包含特殊 token 符号
        prompt_snippet_text = tokenizer.decode(snippet_tokens, skip_special_tokens=True)

        # 应用 Qwen 的对话模板，构建完整的 Prompt 消息结构
        # 这适用于 Qwen-Instruct 等指令遵循模型
        messages = [
            {"role": "system", "content": "You are a helpful assistant that continues to finish writing English movie reviews."}, # System 消息，设定助手角色和任务类型
            {"role": "user", "content": f"Continue to write the following English movie review snippet:\n\n{prompt_snippet_text}"} # User 消息，包含要续写的片段指令
        ]
        # 使用分词器将完整的 messages 结构转换为 token IDs
        # ensure_eos_token=False (transformers 默认行为)
        # add_generation_prompt=True 会在 User 消息后添加 token，告诉模型从这里开始生成助手的回复
        tokenized_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True, # 返回 token IDs
            add_generation_prompt=True, # 对 instruct 模型很重要
            return_tensors="pt" # 返回 PyTorch 张量
        ).squeeze(0) # 移除 apply_chat_template 自动添加的批次维度

        sample["input_ids"] = tokenized_prompt # 存储用于模型输入的 token ID 序列
        # 解码最终的 Prompt token ID 序列（包含模板和片段），用于后续拼接和调试
        sample["query"] = tokenizer.decode(tokenized_prompt)
        return sample

    # 对数据集中的每个样本应用 tokenize 函数
    ds = ds.map(tokenize, batched=False)
    # 将数据集格式设置为 PyTorch 张量
    ds.set_format(type="torch")
    return ds

# 构建数据集，使用上方定义的 Prompt token 长度
dataset = build_dataset(policy_tokenizer, max_len=MAX_PROMPT_TOKEN_LEN)


def collator(data):
    """
    批处理函数，将数据集中的单个样本（字典）列表转换为适合 PPO Trainer 的批次格式（字典）。
    将 input_ids, query 等键对应的列表转换为字典。
    """
    return dict((key, [d[key] for d in data]) for key in data[0])

# --- 4. 加载模型 ---

# 设置 PyTorch 使用的设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载策略模型 (Policy Model)
# 这是将要通过 PPO 训练的主模型，带有用于强化学习的价值头 (Value Head)
# 使用 bfloat16 精度加载以节省显存，同时保持一定的精度
# low_cpu_mem_usage=True 尝试在加载时减少 CPU 内存占用
# device_map="auto" 自动将模型层分配到可用设备上
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16, # 使用 bfloat16 以提高兼容 GPU 上的效率
    low_cpu_mem_usage=True,     # 尝试减少加载期间的 CPU 内存使用
    trust_remote_code=True, # 对于某些自定义模型（如 Qwen）可能需要信任远程代码
    device_map="auto" # 如果需要，自动分配到可用的 GPU 上
)
print("策略模型加载完成。")

# 加载参考模型 (Reference Model)
# 这是策略模型的原始拷贝，参数不会在训练中更新
# 用于计算 KL 散度，防止策略模型与原始模型偏离过远
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)
# 确保参考模型处于评估模式且参数不需要梯度更新
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
print("参考模型加载完成。")


# 加载奖励模型 (Reward Model)
# 这是一个预训练的情感分类模型，用于为生成的文本提供奖励信号
reward_model_name = "distilbert"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

# 获取之前确定的设备 (cuda 或 cpu)，确保奖励模型加载到正确的设备
current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载情感分类模型
# 注意：根据之前的经验，此模型在 pipeline 中处理 BF16 可能有问题，因此使用默认的 Float32 精度加载
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name,
    # torch_dtype=torch.bfloat16, # 移除或注释此行，使用默认的 Float32 精度加载奖励模型
    # device_map="auto" # 已移除 - 此模型不支持 device_map="auto"
)
# 将奖励模型显式移动到设备
reward_model.to(current_device)

# 将奖励模型设置为评估模式
reward_model.eval()
print(f"奖励模型 ({reward_model_name}) 加载完成。")

# 确保 pipeline 使用的 reward_device 变量设置正确
# 通过检查模型参数来找到奖励模型加载到的设备
reward_device = next(reward_model.parameters()).device # 再次获取一次，确保是最新的设备信息
print(f"奖励模型设备: {reward_device}")

# 创建情感分析 pipeline (简化奖励获取过程)
# 确保 pipeline 使用正确的模型、分词器和设备
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    tokenizer=reward_tokenizer,
    device=reward_device, # 为 pipeline 指定设备
    return_all_scores=True # 获取所有标签（POSITIVE, NEGATIVE）的分数
)
print("情感分析 pipeline 创建完成。")

# --- 5. 初始化 PPOTrainer ---
# 使用 TRL 的 PPOTrainer 协调整个 PPO 训练流程
ppo_trainer = PPOTrainer(
    config=config, # PPO 训练配置
    model=policy_model, # 策略模型
    ref_model=ref_model, # 参考模型
    tokenizer=policy_tokenizer, # 分词器
    dataset=dataset, # 训练数据集
    data_collator=collator, # 批处理函数
    # optimizer=optimizer, # PPOTrainer 默认创建自己的 Adam 优化器
)
print("PPOTrainer 初始化完成。")

# --- Wandb 初始化 ---
# 注释掉wandb初始化部分
# wandb_project = "Qwen-PPO-Sentiment-IMDB"
# wandb_run_name = f"{wandb_project}-{os.getpid()}"
# print(f"正在初始化 wandb 运行: {wandb_run_name}")
# wandb.init(
#     project=wandb_project,
#     name=wandb_run_name,
#     config={
#         **config.to_dict(),
#         "generation_kwargs": generation_kwargs,
#         "dataset_name": "stanfordnlp/imdb:plain_text",
#         "max_prompt_token_len": MAX_PROMPT_TOKEN_LEN,
#         "reward_model": reward_model_name,
#         "policy_model": config.model_name,
#         "device": str(device),
#         "policy_model_dtype": str(next(policy_model.parameters()).dtype),
#         "reward_model_dtype": str(next(reward_model.parameters()).dtype),
#     },
# )
# print("Wandb 初始化完成。")

# 添加训练配置打印
print("=== PPO 训练配置 ===")
print(f"模型名称: {config.model_name}")
print(f"学习率: {config.learning_rate}")
print(f"批次大小: {config.batch_size}")
print(f"迷你批次大小: {config.mini_batch_size}")
print(f"目标KL散度: {config.target_kl}")
print(f"PPO epochs: {config.ppo_epochs}")
print(f"设备: {device}")
print(f"策略模型数据类型: {next(policy_model.parameters()).dtype}")
print(f"奖励模型数据类型: {next(reward_model.parameters()).dtype}")
print("==================")

# --- 6. 训练循环 ---
# 定义总共要训练的 PPO Epoch 数量，从 config 中获取
total_ppo_epochs = int(config.ppo_epochs)
# 计算每个 epoch 需要处理的批次数量
steps_per_epoch = len(ppo_trainer.dataloader)
# print(f"数据集大小: {len(dataset)}")
# print(f"每个 epoch 的步数: {steps_per_epoch}")
# print(f"批次大小: {config.batch_size}")
# print(f"迷你批次大小: {config.mini_batch_size}")
# print(f"总共 PPO Epochs: {total_ppo_epochs}")

# 添加训练统计变量
total_steps = 0
epoch_rewards = []

# 开始训练循环
for epoch in range(total_ppo_epochs):
    print(f"\n--- 开始 Epoch {epoch + 1} ---")
    epoch_rewards = []  # 重置epoch奖励列表
    
    for step, batch in enumerate(ppo_trainer.dataloader):
        total_steps += 1
        
        # 检查是否达到最大步数限制
        if total_steps > 70:
            print(f"\n达到最大步数限制 (50步)，训练结束。")
            break
        
        # 检查批次内容 (可选的调试打印)
        # print(f"Step {step}: Batch keys: {batch.keys()}")
        # print(f"Step {step}: Number of queries in batch: {len(batch['query'])}")

        # 获取当前批次的 Prompt token IDs
        query_tensors = batch["input_ids"] # Prompt 的 token IDs 列表 (张量列表)

        # 从策略模型生成回复
        # Note: PPOTrainer.generate 期望一个 Prompt token IDs 的张量列表作为查询输入
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False, # 只获取生成的部分 (不包含 Prompt token IDs)
            batch_size=config.mini_batch_size, # 使用 mini_batch_size 控制生成时的显存
            **generation_kwargs, # 使用上方定义的生成参数 (包含 max_new_tokens)
        )
        # 将生成的回复 token IDs 解码为文本字符串
        # response_tensors 只包含生成的 token ID，不含 Prompt 部分
        batch["response"] = [policy_tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # 检查生成的回复 (可选的调试打印)
        # print(f"Step {step}: Example generated response: {batch['response'][0]}")

        # --- 计算情感奖励 ---
        # 将原始 Prompt 字符串 (包含模板和片段) 和模型生成的回复字符串拼接起来
        # 这是输入到奖励模型进行评分的完整文本
        texts_to_score = [q + r for q, r in zip(batch["query"], batch["response"])] # 对 Prompt + 回复进行评分

        # 打印第一个样本输入到奖励模型的完整文本 (可选)
        print(f"Step {step}: Example text for reward model (first 1):\n{texts_to_score[0]}")

        # 确保 pipeline 输入是字符串列表
        try:
            # 使用 mini_batch_size 对 pipeline 进行分批推理以节省 VRAM
            # pipe_outputs 是一个列表，每个元素是对应输入的评分结果（通常是 [{'label': '...', 'score': ...}, ...]）
            pipe_outputs = sentiment_pipe(texts_to_score, batch_size=config.mini_batch_size)
        except Exception as e:
            # 捕获情感 pipeline 中的错误，打印信息并跳过当前批次
            print(f"Error during sentiment pipeline: {e}")
            # 打印第一个 problematic text 即可，避免输出过多
            print(f"Problematic text (first): {texts_to_score[0] if texts_to_score else 'N/A'}")
            # 如果评分失败，跳过此批次，不进行 PPO 优化
            continue

        # 从 pipeline 输出中提取 'POSITIVE' 分数作为奖励
        rewards = []
        for output in pipe_outputs:
            positive_score = 0.0
            # pipeline 输出每个输入的得分列表，找到 POSITIVE 的分数
            for score_dict in output:
                if score_dict['label'] == 'POSITIVE':
                    positive_score = score_dict['score']
                    break
            # 将奖励分数转换为 PyTorch tensor 并移动到策略模型所在的设备
            rewards.append(torch.tensor(positive_score, device=device))

        # 检查奖励 (可选的调试打印)
        print(f"Step {step}: Example rewards (first 1): {[r.item() for r in rewards[:1]]}")

        # 计算当前批次的平均奖励
        batch_mean_reward = sum([r.item() for r in rewards]) / len(rewards)
        epoch_rewards.append(batch_mean_reward)

        # --- PPO 优化步骤 ---
        try:
            # ppo_trainer.step 会计算 Loss，执行反向传播和优化器更新
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # 直接打印训练统计信息到控制台
            print(f"Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}:")
            print(f"  - 平均奖励: {batch_mean_reward:.4f}")
            
            # 安全地获取和打印统计信息，处理可能的字符串值
            ppo_loss = stats.get('ppo/loss/total', 'N/A')
            kl_mean = stats.get('ppo/policy/kl_mean', 'N/A')
            policy_loss = stats.get('ppo/policy/policy_loss', 'N/A')
            value_loss = stats.get('ppo/policy/value_loss', 'N/A')
            entropy_loss = stats.get('ppo/policy/entropy_loss', 'N/A')
            learning_rate = stats.get('ppo/learning_rate', 'N/A')
            
            # 根据数据类型进行不同的格式化
            print(f"  - PPO Loss: {ppo_loss:.4f}" if isinstance(ppo_loss, (int, float)) else f"  - PPO Loss: {ppo_loss}")
            # print(f"  - KL散度: {kl_mean:.4f}" if isinstance(kl_mean, (int, float)) else f"  - KL散度: {kl_mean}")
            # print(f"  - 策略损失: {policy_loss:.4f}" if isinstance(policy_loss, (int, float)) else f"  - 策略损失: {policy_loss}")
            # print(f"  - 价值损失: {value_loss:.4f}" if isinstance(value_loss, (int, float)) else f"  - 价值损失: {value_loss}")
            # print(f"  - 熵损失: {entropy_loss:.4f}" if isinstance(entropy_loss, (int, float)) else f"  - 熵损失: {entropy_loss}")
            # print(f"  - 学习率: {learning_rate:.2e}" if isinstance(learning_rate, (int, float)) else f"  - 学习率: {learning_rate}")
            
            # 每10步打印一次详细统计
            if step % 10 == 0:
                if epoch_rewards:
                    current_epoch_avg = sum(epoch_rewards) / len(epoch_rewards)
                    print(f"  - 当前epoch平均奖励: {current_epoch_avg:.4f}")
                else:
                    print(f"  - 当前epoch平均奖励: N/A")
                print(f"  - 总步数: {total_steps}")
                print("  " + "-" * 50)

        except Exception as e:
            # 捕获 PPO step 中的错误，打印信息并跳过当前批次
            print(f"Error during PPO step: {e}")
            print(f"Sizes: queries={len(query_tensors)}, responses={len(response_tensors)}, rewards={len(rewards)}")
            # 如果错误持续发生，可以记录更多关于张量的细节 (可选)
            # Potentially log more details about the tensors if errors persist
            # 如果步骤失败，跳过批次
            continue


    # --- 可选：定期保存模型 ---
    # 在每个 epoch 结束时保存模型 checkpoint (例如用于中断后继续训练)
    if (epoch + 1) % 1 == 0: # 每 1 个 epoch 保存一次 (用于演示)
        save_path = f"./qwen_ppo_sentiment_epoch_{epoch+1}"
        print(f"正在保存模型 checkpoint 到 {save_path}...")
        # 使用 Accelerate 的 save_model 方法更安全，特别是使用 device_map 或分布式训练时
        ppo_trainer.accelerator.wait_for_everyone() # 等待所有进程完成
        if ppo_trainer.accelerator.is_main_process: # 只在主进程保存
            ppo_trainer.save_pretrained(save_path)
            policy_tokenizer.save_pretrained(save_path)
        print("模型 checkpoint 保存完成。")
        print(epoch_rewards)
    
    # 检查是否因为达到步数限制而提前结束训练
    if total_steps > 70:
        print(f"训练因达到步数限制而提前结束，总步数: {total_steps}")
        break


# --- 7. (可选) 保存最终模型 ---
final_save_path = "./qwen_ppo_sentiment_final"
print(f"训练完成。正在保存最终模型到 {final_save_path}...")
# 使用 Accelerate 的 save_model 方法更安全，特别是在分布式环境下
ppo_trainer.accelerator.wait_for_everyone() # 等待所有进程
if ppo_trainer.accelerator.is_main_process: # 只在主进程保存
    # 保存训练后的策略模型和分词器
    ppo_trainer.save_pretrained(final_save_path)
    policy_tokenizer.save_pretrained(final_save_path)
print("最终模型保存完成。")
print(epoch_rewards)

# --- Wandb 结束 ---
# 注释掉wandb结束
# wandb.finish()
print("训练完成，日志记录结束。")


# --- 8. (可选) 示例生成：训练前后的对比 ---
print("\n--- 示例生成：训练前后的对比 ---")

# 定义用于对比的示例 Prompt 文本
prompt_text = "This movie was really not good" # 示例 Prompt - 训练前后相同

# 准备用于示例生成的 Prompt 的 input_ids
# 使用 ppo_trainer.tokenizer（即 policy_tokenizer），它与原始 tokenizer 相同
# 注意：这里示例 Prompt 不使用数据集中的片段，而是固定的文本
messages = [
    {"role": "system", "content": "You are a helpful assistant that continues to finish writing English movie reviews."}, # 使用与训练时相同的 system 消息
    # 使用固定的 prompt_text 作为用户 Prompt 的一部分
    {"role": "user", "content": f"Continue to write the following English movie review snippet:\n\n{prompt_text}"}
]

# 应用对话模板并将完整的 Prompt 转换为 token IDs，移动到主要设备
input_ids = ppo_trainer.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True, # 添加生成 Prompt token
    return_tensors="pt" # 返回 PyTorch 张量
).to(device) # 移动到主要设备


# 使用相同的生成参数 (从训练配置中获取)
gen_kwargs = generation_kwargs.copy()
# 确保填充 token ID 是正确的 (使用 ppo_trainer.tokenizer 的)
gen_kwargs["pad_token_id"] = ppo_trainer.tokenizer.pad_token_id

print(f"\nPrompt (用于示例生成对比): {prompt_text}")

# --- 生成：训练前的原始模型 ---
print("\n--- 生成续写 (原始模型) ---")
# 再次加载训练前的原始模型，用于直接对比
# 注意：这里需要确保加载时的 torch_dtype 与训练前策略模型加载时一致 (即 BF16)
initial_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16, # <-- 使用与训练前策略模型相同的精度
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # 如果显存允许，可以直接加载到主设备
    # device_map="auto" # 如果显存紧张，可以尝试 auto
).to(device) # <-- 确保模型在要生成所在的设备上
initial_model.eval() # 设置为评估模式
print("原始模型加载完成。")

# 获取训练后的模型实例 (PPOTrainer 持有的就是训练后的模型)
trained_model = ppo_trainer.model
# 训练后的 tokenizer 已经在上面用于生成 input_ids 了


# 使用原始模型生成文本
with torch.no_grad(): # 生成过程中不需要计算梯度
    initial_output_ids = initial_model.generate(input_ids, **gen_kwargs)

# 从生成的 token IDs 中，只解码生成的部分 (排除 Prompt 的 token)
# output_ids 包含 Prompt + 生成，切片 [:, input_ids.shape[1]:] 排除 Prompt 部分
initial_generated_ids = initial_output_ids[:, input_ids.shape[1]:]
# 将生成的 token ID 解码为文本
initial_generated_text = ppo_trainer.tokenizer.decode(initial_generated_ids[0], skip_special_tokens=True) # 使用 tokenizer 解码

print(initial_generated_text)


# --- 生成：训练后的模型 ---
print("\n--- 生成续写 (训练后模型) ---")
# 使用训练后的模型生成文本
with torch.no_grad(): # 生成过程中不需要计算梯度
    trained_output_ids = trained_model.generate(input_ids, **gen_kwargs)

# 从生成的 token IDs 中，只解码生成的部分 (排除 Prompt 的 token)
trained_generated_ids = trained_output_ids[:, input_ids.shape[1]:]
# 将生成的 token ID 解码为文本
trained_generated_text = ppo_trainer.tokenizer.decode(trained_generated_ids[0], skip_special_tokens=True)

print(trained_generated_text)

# 为什么没有单独的value_model？
# 在TRL库的设计中，价值头被直接集成到策略模型中，这样做的好处是：
# 参数共享：语言模型和价值头可以共享底层的特征提取器
# 训练效率：同时训练策略和价值函数，提高训练效率
# 架构简化：不需要单独维护一个价值模型
# 所以在这个代码中，价值模型就是策略模型中的价值头部分，它会在PPO训练过程中自动学习和更新。
