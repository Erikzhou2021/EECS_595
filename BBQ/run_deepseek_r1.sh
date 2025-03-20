#!/bin/bash

# Convert prompt format
python3 data/convert_format.py
python3 data/convert_format.py --few_shot

# Test 1 - Gender Plain Neg Without Debiasing
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_lower.jsonl --debias_prompt gender-plain-neg_without
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_upper.jsonl --debias_prompt gender-plain-neg_without
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt gender-plain-neg_without
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_upper.jsonl --debias_prompt gender-plain-neg_without

# Test 2 - Gender Plain Neg With Debiasing
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_lower.jsonl --debias_prompt gender-plain-neg_with
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_upper.jsonl --debias_prompt gender-plain-neg_with
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt gender-plain-neg_with
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_upper.jsonl --debias_prompt gender-plain-neg_with

# Test 3 - Gender Instruct Neg Without Debiasing
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_lower.jsonl --debias_prompt gender-instruct-neg_without
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_upper.jsonl --debias_prompt gender-instruct-neg_without
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt gender-instruct-neg_without
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_upper.jsonl --debias_prompt gender-instruct-neg_without

# Test 4 - Gender Instruct Neg With Debiasing
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_lower.jsonl --debias_prompt gender-instruct-neg_with
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_1_upper.jsonl --debias_prompt gender-instruct-neg_with
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt gender-instruct-neg_with
python3 src/pred.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --file data/jsonl/eval_prompt_fewshot_1_upper.jsonl --debias_prompt gender-instruct-neg_with

# Results
python3 evaluation/eval_bbq.py --result_dir result