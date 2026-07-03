# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import random  # 用于生成随机数
import numpy as np  # 用于数值计算
from functools import partial  # 用于创建偏函数

# 导入PyTorch相关库
import torch  # PyTorch核心库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
from torch.optim import AdamW  # AdamW优化器

# 导入数据处理相关库
from torch.utils.data import DataLoader  # 数据加载器
from datasets import load_dataset  # Hugging Face数据集加载
from transformers import AutoTokenizer, AutoModelForCausalLM  # 自动分词器和因果语言模型

# import wandb  # 注释掉wandb，不使用实验跟踪
from tqdm import tqdm  # 进度条显示

def seed_everything(seed=2003):
    """设置随机种子，确保实验可重现"""
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch GPU随机种子
    np.random.seed(seed)  # 设置NumPy随机种子
    random.seed(seed)  # 设置Python random随机种子
    torch.backends.cudnn.deterministic = True  # 确保CUDA操作确定性

def calculate_DPO_loss(model_preferred_logprob, model_dispreferred_logprob,
                       ref_preferred_logprob, ref_dispreferred_logprob,
                       beta=0.5):
    """计算DPO损失函数"""
    
    # 计算模型相对于参考模型的log概率差异
    preferred_relative_logprob = model_preferred_logprob - ref_preferred_logprob  # 偏好回答的相对log概率
    dispreferred_relative_logprob = model_dispreferred_logprob - ref_dispreferred_logprob  # 非偏好回答的相对log概率

    # 计算奖励准确率（偏好回答的log概率是否大于非偏好回答）
    reward_accuracies = (preferred_relative_logprob > dispreferred_relative_logprob).float().mean()
    # 计算奖励边际（偏好回答与非偏好回答的log概率差异）
    reward_margins = (preferred_relative_logprob - dispreferred_relative_logprob).mean()

    # 计算DPO损失：使用log-sigmoid函数
    loss = -F.logsigmoid(beta * (preferred_relative_logprob - dispreferred_relative_logprob)).mean()

    # 返回损失值和各种统计指标
    return loss, preferred_relative_logprob.mean(), dispreferred_relative_logprob.mean(), reward_accuracies, reward_margins

def get_log_prob(logits, labels, prompt_lengths):
    """计算回答部分的平均log概率"""
    log_probs = F.log_softmax(logits, dim=-1)  # 计算log softmax概率
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)  # 提取对应token的log概率
    
    batch_size, seq_len = labels.shape  # 获取批次大小和序列长度
    prompt_lengths = prompt_lengths.to(labels.device)  # 将prompt长度移到正确设备
    # 创建回答部分的掩码（prompt之后的部分）
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()  # 转换为浮点数
    
    # 计算回答部分的log概率总和
    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    # 计算回答部分的长度（至少为1）
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    return response_log_probs / response_lengths  # 返回平均log概率

def collate_fn(batch, tokenizer, max_length, device):
    """数据整理函数，将批次数据转换为模型输入格式"""
    # 对prompt进行编码
    prompt_encodings = tokenizer(
        ['Instruct: ' + item['prompt'] + '\n' for item in batch],  # 添加"Instruct:"前缀
        padding='max_length',  # 填充到最大长度
        truncation=True,  # 截断超长序列
        max_length=max_length,  # 最大长度
        return_tensors='pt'  # 返回PyTorch张量
    )
    
    # 对偏好回答进行编码
    chosen_encodings = tokenizer(
        ['Output: ' + item['chosen'] for item in batch],  # 添加"Output:"前缀
        padding='max_length',  # 填充到最大长度
        truncation=True,  # 截断超长序列
        max_length=max_length,  # 最大长度
        return_tensors='pt'  # 返回PyTorch张量
    )
    
    # 对非偏好回答进行编码
    rejected_encodings = tokenizer(
        ['Output: ' + item['rejected'] for item in batch],  # 添加"Output:"前缀
        padding='max_length',  # 填充到最大长度
        truncation=True,  # 截断超长序列
        max_length=max_length,  # 最大长度
        return_tensors='pt'  # 返回PyTorch张量
    )

    # 将prompt和偏好回答拼接
    prompt_preferred_ids = torch.cat([
        prompt_encodings.input_ids,  # prompt的token ID
        chosen_encodings.input_ids  # 偏好回答的token ID
    ], dim=-1).to(device)  # 在最后一个维度拼接并移到设备
    
    # 将prompt和非偏好回答拼接
    prompt_dispreferred_ids = torch.cat([
        prompt_encodings.input_ids,  # prompt的token ID
        rejected_encodings.input_ids  # 非偏好回答的token ID
    ], dim=-1).to(device)  # 在最后一个维度拼接并移到设备

    # 将prompt和偏好回答的注意力掩码拼接
    prompt_preferred_mask = torch.cat([
        prompt_encodings.attention_mask,  # prompt的注意力掩码
        chosen_encodings.attention_mask  # 偏好回答的注意力掩码
    ], dim=-1).to(device)  # 在最后一个维度拼接并移到设备
    
    # 将prompt和非偏好回答的注意力掩码拼接
    prompt_dispreferred_mask = torch.cat([
        prompt_encodings.attention_mask,  # prompt的注意力掩码
        rejected_encodings.attention_mask  # 非偏好回答的注意力掩码
    ], dim=-1).to(device)  # 在最后一个维度拼接并移到设备

    # 计算prompt的长度（用于后续计算回答部分的log概率）
    prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1)

    # 返回整理后的数据字典
    return {
        'prompt_preferred_ids': prompt_preferred_ids,  # prompt+偏好回答的token ID
        'prompt_dispreferred_ids': prompt_dispreferred_ids,  # prompt+非偏好回答的token ID
        'prompt_preferred_mask': prompt_preferred_mask,  # prompt+偏好回答的注意力掩码
        'prompt_dispreferred_mask': prompt_dispreferred_mask,  # prompt+非偏好回答的注意力掩码
        'prompt_lengths': prompt_lengths  # prompt的长度
    }

def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, beta=0.1):
    """训练函数"""
    model.train()  # 设置模型为训练模式
    ref_model.eval()  # 设置参考模型为评估模式
    
    # 添加步数计数器
    global_step = 0  # 全局步数
    total_batches = len(train_dataloader)  # 总批次数

    for epoch in range(epochs):  # 遍历每个epoch
        print(f"\n{'='*50}")  # 打印分隔线
        print(f"开始训练第 {epoch + 1}/{epochs} 个epoch")  # 打印当前epoch信息
        print(f"总批次数: {total_batches}")  # 打印总批次数
        print(f"{'='*50}")  # 打印分隔线
        
        # 初始化epoch统计变量
        epoch_loss = 0.0  # epoch总损失
        epoch_reward_accuracy = 0.0  # epoch总奖励准确率
        epoch_reward_margin = 0.0  # epoch总奖励边际
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):  # 遍历每个批次
            optimizer.zero_grad()  # 清空梯度

            # 前向传播：计算模型对偏好回答的logits
            model_preferred_logits = model(
                input_ids=batch['prompt_preferred_ids'],  # prompt+偏好回答的token ID
                attention_mask=batch['prompt_preferred_mask']  # 对应的注意力掩码
            ).logits  # 获取logits输出
            
            # 计算模型对偏好回答的log概率
            model_preferred_logprob = get_log_prob(
                model_preferred_logits,  # 模型的logits输出
                batch['prompt_preferred_ids'],  # 完整的token ID序列
                batch['prompt_lengths']  # prompt的长度
            )

            # 前向传播：计算模型对非偏好回答的logits
            model_dispreferred_logits = model(
                input_ids=batch['prompt_dispreferred_ids'],  # prompt+非偏好回答的token ID
                attention_mask=batch['prompt_dispreferred_mask']  # 对应的注意力掩码
            ).logits  # 获取logits输出
            
            # 计算模型对非偏好回答的log概率
            model_dispreferred_logprob = get_log_prob(
                model_dispreferred_logits,  # 模型的logits输出
                batch['prompt_dispreferred_ids'],  # 完整的token ID序列
                batch['prompt_lengths']  # prompt的长度
            )

            with torch.no_grad():  # 不计算梯度（参考模型不需要训练）
                # 前向传播：计算参考模型对偏好回答的logits
                ref_preferred_logits = ref_model(
                    input_ids=batch['prompt_preferred_ids'],  # prompt+偏好回答的token ID
                    attention_mask=batch['prompt_preferred_mask']  # 对应的注意力掩码
                ).logits  # 获取logits输出
                
                # 计算参考模型对偏好回答的log概率
                ref_preferred_logprob = get_log_prob(
                    ref_preferred_logits,  # 参考模型的logits输出
                    batch['prompt_preferred_ids'],  # 完整的token ID序列
                    batch['prompt_lengths']  # prompt的长度
                )

                # 前向传播：计算参考模型对非偏好回答的logits
                ref_dispreferred_logits = ref_model(
                    input_ids=batch['prompt_dispreferred_ids'],  # prompt+非偏好回答的token ID
                    attention_mask=batch['prompt_dispreferred_mask']  # 对应的注意力掩码
                ).logits  # 获取logits输出
                
                # 计算参考模型对非偏好回答的log概率
                ref_dispreferred_logprob = get_log_prob(
                    ref_dispreferred_logits,  # 参考模型的logits输出
                    batch['prompt_dispreferred_ids'],  # 完整的token ID序列
                    batch['prompt_lengths']  # prompt的长度
                )

            # 计算DPO损失和统计指标
            loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                model_preferred_logprob,  # 模型对偏好回答的log概率
                model_dispreferred_logprob,  # 模型对非偏好回答的log概率
                ref_preferred_logprob,  # 参考模型对偏好回答的log概率
                ref_dispreferred_logprob,  # 参考模型对非偏好回答的log概率
                beta=beta  # DPO的beta参数
            )

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            # 更新epoch统计
            epoch_loss += loss.item()  # 累加损失值
            epoch_reward_accuracy += reward_accuracies.item()  # 累加奖励准确率
            epoch_reward_margin += reward_margins.item()  # 累加奖励边际
            global_step += 1  # 增加全局步数

            # 每10步打印一次详细日志
            if global_step % 10 == 0:
                print(f"\n[Step {global_step}] "  # 打印当前步数
                      f"Loss: {loss.item():.6f} | "  # 打印当前损失
                      f"Reward Acc: {reward_accuracies.item():.4f} | "  # 打印奖励准确率
                      f"Reward Margin: {reward_margins.item():.4f} | "  # 打印奖励边际
                      f"Pref LogProb: {preferred_relative_logprob.item():.4f} | "  # 打印偏好回答的相对log概率
                      f"Dispref LogProb: {dispreferred_relative_logprob.item():.4f}")  # 打印非偏好回答的相对log概率

            # 每50步打印一次进度
            if global_step % 50 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)  # 计算平均损失
                avg_acc = epoch_reward_accuracy / (batch_idx + 1)  # 计算平均奖励准确率
                avg_margin = epoch_reward_margin / (batch_idx + 1)  # 计算平均奖励边际
                print(f"\n{'='*30}")  # 打印分隔线
                print(f"当前进度: {batch_idx+1}/{total_batches} 批次")  # 打印当前进度
                print(f"平均Loss: {avg_loss:.6f}")  # 打印平均损失
                print(f"平均Reward Accuracy: {avg_acc:.4f}")  # 打印平均奖励准确率
                print(f"平均Reward Margin: {avg_margin:.4f}")  # 打印平均奖励边际
                print(f"{'='*30}")  # 打印分隔线

        # 每个epoch结束后打印总结
        avg_epoch_loss = epoch_loss / total_batches  # 计算整个epoch的平均损失
        avg_epoch_acc = epoch_reward_accuracy / total_batches  # 计算整个epoch的平均奖励准确率
        avg_epoch_margin = epoch_reward_margin / total_batches  # 计算整个epoch的平均奖励边际
        
        print(f"\n{'='*50}")  # 打印分隔线
        print(f"Epoch {epoch + 1} 训练完成!")  # 打印epoch完成信息
        print(f"平均Loss: {avg_epoch_loss:.6f}")  # 打印平均损失
        print(f"平均Reward Accuracy: {avg_epoch_acc:.4f}")  # 打印平均奖励准确率
        print(f"平均Reward Margin: {avg_epoch_margin:.4f}")  # 打印平均奖励边际
        print(f"总步数: {global_step}")  # 打印总步数
        print(f"{'='*50}\n")  # 打印分隔线

        # 可选：仍然记录到wandb（如果你想要的话）
        if wandb.run is not None:  # 如果wandb运行存在
            wandb.log({  # 记录到wandb
                'epoch': epoch + 1,  # 当前epoch
                'epoch_loss': avg_epoch_loss,  # epoch平均损失
                'epoch_reward_accuracy': avg_epoch_acc,  # epoch平均奖励准确率
                'epoch_reward_margin': avg_epoch_margin,  # epoch平均奖励边际
                'global_step': global_step  # 全局步数
            })

def main():
    """主函数"""
    parser = argparse.ArgumentParser()  # 创建参数解析器

    # 添加各种命令行参数
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--beta", type=float, default=0.1)  # DPO的beta参数
    parser.add_argument("--batch_size", type=int, default=2)  # 批次大小
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)  # 梯度累积步数
    parser.add_argument("--max_length", type=int, default=512)  # 最大序列长度
    parser.add_argument("--lr", type=float, default=1e-6)  # 学习率
    parser.add_argument("--seed", type=int, default=2003)  # 随机种子
    parser.add_argument("--model_name", type=str, default="phi-2")  # 模型名称
    parser.add_argument("--dataset_name", type=str, default="truthy-dpo-v0.1")  # 数据集名称
    parser.add_argument("--wandb_project", type=str, default="truthy-dpo")  # wandb项目名称
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")  # 是否使用wandb

    args = parser.parse_args()  # 解析命令行参数

    seed_everything(args.seed)  # 设置随机种子

    # 只在需要时初始化wandb
    if args.use_wandb:  # 如果使用wandb
        wandb.login()  # 登录wandb
        wandb.init(project=args.wandb_project, config=args)  # 初始化wandb项目
        print("Wandb已启用")  # 打印启用信息
    else:  # 如果不使用wandb
        print("Wandb已禁用，只在终端显示日志")  # 打印禁用信息

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备（GPU或CPU）
    print(f"使用设备: {device}")  # 打印设备信息

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # 从预训练模型加载分词器
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充token为结束token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)  # 加载主模型并移到设备
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)  # 加载参考模型并移到设备

    ref_model.requires_grad_(False)  # 冻结参考模型的参数（不需要训练）

    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)  # 创建AdamW优化器
    print(f"学习率: {args.lr}")  # 打印学习率

    # 加载数据集
    dataset = load_dataset(args.dataset_name, split="train")  # 加载训练数据集
    print(f"数据集大小: {len(dataset)}")  # 打印数据集大小
    
    # 创建数据加载器
    collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length, device=device)  # 创建偏函数
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)  # 创建数据加载器

    # 打印训练配置信息
    print(f"\n开始训练...")  # 打印开始训练信息
    print(f"模型: {args.model_name}")  # 打印模型名称
    print(f"数据集: {args.dataset_name}")  # 打印数据集名称
    print(f"批次大小: {args.batch_size}")  # 打印批次大小
    print(f"最大长度: {args.max_length}")  # 打印最大长度
    print(f"Beta参数: {args.beta}")  # 打印beta参数

    # 开始训练
    train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=args.epochs, beta=args.beta)  # 调用训练函数

    # 保存训练后的模型
    print(f"\n训练完成! 模型已保存到 'model-DPO'")  # 打印完成信息
    model.save_pretrained("model-DPO")  # 保存模型到指定目录

if __name__ == "__main__":
    main()  # 运行主函数
