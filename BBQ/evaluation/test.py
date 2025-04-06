import json
import re
import string
import argparse
import random
import math
from pathlib import Path

p = Path('.')
paths = p.glob('**/*.py')
new_paths = list(paths)
print(new_paths[0].exists())



# reads ground truth from BBQ_Gender_identity_cyclic_permutation.jsonl into jsonl_data list
with open("eval_config.json", "r") as f:
    eval_config = json.load(f)
eval_base_file = eval_config["eval_base_file"]

with open(eval_base_file, "r") as f:
    jsonl_data = [json.loads(line) for line in f.readlines()]

jsonl_data_len = len(jsonl_data)
print(jsonl_data_len)


with open('./evaluation/hi.txt', "r") as f:
    lines = f.read().split("\n")
    
print(len(lines))


sample_indices = random.sample(range(jsonl_data_len), 4)
print(sample_indices)