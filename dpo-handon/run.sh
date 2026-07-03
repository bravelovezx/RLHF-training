#!/usr/bin/env sh

python src/train.py \
    --epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 32\
    --max_length 256 \
    --lr 1e-6 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "src/Qwen2.5-0.5B-Instruct" \
    --dataset_name "src/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo"
