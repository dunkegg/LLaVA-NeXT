#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=6

# 运行 Python 脚本
/home/zejinw/anaconda3/envs/llava/bin/python llava/serve/cli_eval_goat.py \
  --model-path model_output/llava-next-interleave-qwen-7b-2 \
  --json-file data/goat_test_13000.json \
  --temperature 0.5 \
  --top-p 0.8 \
  --max-new-tokens 65536 \
  --id 1000 \
  --torch-type bfloat16 \
  --result_name result_train_mm