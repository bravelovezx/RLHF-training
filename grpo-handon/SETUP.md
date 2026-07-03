# GRPO 训练环境搭建与运行指南

使用 `trl` 复刻 DeepSeek R1 的 GRPO 方法，在 `Qwen2.5-0.5B-Instruct` 上训练数学推理能力。

---

## 1. 环境要求

- Python >= 3.10
- CUDA >= 11.8
- GPU 显存 >= 8GB（推荐 16GB+）
- AutoDL 推荐镜像：PyTorch 2.x + CUDA 12.x

## 2. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install transformers trl datasets
pip install modelscope wandb accelerate
```

> AutoDL 镜像一般已预装 torch，可跳过第一行。

## 3. 下载模型和数据集

```bash
# 下载 Qwen2.5-0.5B-Instruct 模型
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./Qwen2.5-0.5B-Instruct

# 下载 GSM8K 数据集（可选，脚本也会自动从 HuggingFace 下载）
modelscope download --dataset modelscope/gsm8k --local_dir ./gsm8k
```

## 4. 数据集加载兼容说明

旧版 `datasets` 库可以直接 `load_dataset('gsm8k')` 加载，新版必须指定完整路径和配置名：

```python
# 旧版（已失效）
load_dataset('gsm8k')

# 新版正确写法
load_dataset('openai/gsm8k', 'main')
```

`train.py` 已做兼容处理，会按顺序尝试：本地 `./gsm8k` 目录 → `openai/gsm8k` → `gsm8k`。

## 5. 配置 wandb（可选）

```bash
wandb login
```

然后将 `train.py` 中 `report_to="none"` 改为 `report_to="wandb"`。

不使用 wandb 则无需任何操作，默认已关闭。

## 6. 开始训练

```bash
python train.py
```

训练完成后模型保存在 `outputs/Qwen2.5-0.5B-reasoning-GRPO/`。

## 7. 训练参数说明

| 参数 | 值 | 说明 |
|------|------|------|
| `learning_rate` | 5e-6 | 学习率 |
| `per_device_train_batch_size` | 8 | 每卡 batch size |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `num_generations` | 8 | 每个问题采样回答数 (G) |
| `max_completion_length` | 200 | 最大生成长度 |
| `num_train_epochs` | 1 | 训练轮数 |
| `save_steps` | 100 | 每 100 步保存 checkpoint |
| `bf16` | True | 使用 bfloat16 混合精度 |

显存不足时可减小 `per_device_train_batch_size` 和 `num_generations`。

## 8. 项目结构

```
grpo/
├── grpo.ipynb                  # 原始 notebook（含 GRPO 原理讲解）
├── train.py                    # 训练脚本（可直接运行）
├── SETUP.md                    # 本文档
├── Qwen2.5-0.5B-Instruct/     # 模型目录（需下载）
├── gsm8k/                      # 数据集目录（可选）
└── outputs/                    # 训练输出目录（训练后生成）
```

## 9. 快速开始（AutoDL 一键流程）

```bash
# 1. 安装依赖
pip install transformers trl datasets modelscope wandb accelerate

# 2. 下载模型
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./Qwen2.5-0.5B-Instruct

# 3. 开始训练
python train.py
```
