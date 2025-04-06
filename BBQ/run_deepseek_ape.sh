#!/bin/bash

# Convert prompt format
python3 data/convert_format.py --few_shot


python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-1
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-2
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-3
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-4
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-5
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-6
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-7
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-8
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-9
python3 src/pred.py --model openai-community/gpt2 --file data/jsonl/eval_prompt_fewshot_1_lower.jsonl --debias_prompt ape-10


# Results
python3 evaluation/eval_bbq.py --result_dir result --ape