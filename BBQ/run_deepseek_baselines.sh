#!/bin/bash

# Convert prompt format
python3 data/convert_format.py

# Baseline Test 1 - 7B Chat Model
python3 src/pred.py --model deepseek-ai/deepseek-llm-7b-base --file data/jsonl/eval_prompt_no_taskinst.jsonl

# Baseline Test 2 - 7B Distill Qwen Model
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_no_taskinst.jsonl

# Baseline Test 3 - 14B Distill Qwen Model
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --file data/jsonl/eval_prompt_no_taskinst.jsonl

# Results
python3 evaluation/eval_bbq.py --result_dir result