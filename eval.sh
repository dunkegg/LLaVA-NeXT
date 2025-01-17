#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行 Python 脚本
/home/zejinw/anaconda3/envs/llava/bin/python llava/serve/cli_eval_goat.py \
  --model-path model_output/llava-next-interleave-qwen-7b-aug-reason2 \
  --json-file data/goat_aug_test.json \
  --temperature 0.5 \
  --top-p 0.8 \
  --max-new-tokens 131072 \
  --id 2000 \
  --torch-type bfloat16 \
  --result_name result_aug_reason_c2